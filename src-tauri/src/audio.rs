use crate::vad::VadModel;
use crate::whisper::{self, AccurateModel, WhisperModel};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

const TARGET_SAMPLE_RATE: u32 = 16000;
/// Silero VAD's exported graph expects exactly this many samples (32ms @ 16kHz) per call.
const FRAME_SAMPLES: usize = 512;
const FRAME_MS: u64 = 32;
const SPEECH_THRESHOLD: f32 = 0.5;
const MIN_SILENCE_MS: u64 = 600;
const SPEECH_PAD_MS: u64 = 300;
const PRE_ROLL_SAMPLES: usize = (TARGET_SAMPLE_RATE as usize / 1000) * SPEECH_PAD_MS as usize;
/// Hard cap on how long an utterance grows before it's force-committed even
/// without a pause. Continuous speech with few pauses (e.g. reading a script
/// aloud) otherwise builds one huge utterance: interim decodes re-transcribe the
/// whole in-progress buffer each time, so cost climbs with its length, and no
/// committed text appears until the cap. A tighter cap keeps the buffer — and
/// thus every interim decode — bounded, and commits text several times more
/// often. Whisper's mel window is 30s; staying well under it also avoids
/// decoding a full-length window. Pre-roll bridges the forced cut so the next
/// utterance doesn't clip the word it lands on.
const MAX_UTTERANCE_MS: u64 = 15_000;
/// How much new speech to accumulate before firing an interim ("partial")
/// decode of the in-progress utterance, so text appears mid-sentence instead
/// of only after the speaker pauses. Kept short (500ms) so the live preview
/// updates frequently; the fast model is cheap enough to keep up. Interim
/// decodes are additionally gated by an in-flight guard, so on a slow machine
/// they simply arrive less often rather than piling up.
const PARTIAL_INTERVAL_SAMPLES: usize = (TARGET_SAMPLE_RATE as usize / 1000) * 500;
/// Interim decodes keep no state across calls (Whisper re-transcribes from
/// scratch), so decoding the whole in-progress utterance makes each partial cost
/// grow with the utterance: a long, pause-free stretch re-decodes its entire
/// buffer every `PARTIAL_INTERVAL_SAMPLES`, and the live preview stalls while
/// that ever-longer decode runs. Interim decodes instead transcribe only this
/// trailing window of the utterance, bounding every partial to a fixed cost (and,
/// via the audio-context scaling in `whisper::run`, a fixed encoder pass too).
/// The final (commit) decode still runs on the full utterance, so committed text
/// is complete; the trade-off is that on utterances longer than the window the
/// partial preview shows only the most recent words until commit fills the rest.
/// Kept to a short 5s window so each interim decode stays cheap and the preview
/// lands fast.
const PARTIAL_WINDOW_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * 5;

/// Shortest below-threshold run *inside* an utterance that counts as a
/// hesitation pause. Natural inter-word gaps in fluent speech are ~0-150ms, so
/// 250ms filters those out and keeps only deliberate/hesitation silences. Runs
/// that reach `MIN_SILENCE_MS` end the utterance, so internal pauses are bounded
/// to roughly 250-600ms; longer gaps show up as separate utterances instead.
const MIN_PAUSE_MS: u64 = 250;
/// Cap on the number of amplitude points sent per utterance for the waveform
/// sparkline. The buffer is bucketed down to at most this many RMS values, so
/// the payload and the frontend render stay bounded regardless of utterance
/// length (a 15s utterance is ~470 frames; 160 points is plenty to draw).
const ENVELOPE_POINTS: usize = 160;

#[derive(Clone, Serialize)]
pub struct TranscriptSegment {
    /// Monotonic id of the utterance within a recording session. Interim and
    /// final results for the same utterance share an index so the UI can
    /// update one line in place and then commit it.
    pub index: u64,
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub is_final: bool,
    /// True only for the deferred accurate-model correction of an already
    /// committed line. A refined segment is also `is_final`, but this flag lets
    /// the UI overwrite the earlier fast final (which it otherwise locks) and
    /// re-tally its stats for the corrected text.
    pub refined: bool,
}

/// A silent gap detected *within* a committed utterance (a hesitation pause).
/// Offsets are milliseconds relative to the utterance's own start, so the UI can
/// place the shaded region directly on that utterance's waveform.
#[derive(Clone, Serialize)]
pub struct Pause {
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Per-utterance acoustic analysis emitted once when an utterance commits, on a
/// separate `utterance_analysis` event (correlated to the transcript line by
/// `index`). Kept separate from `TranscriptSegment` so interim and refined text
/// updates stay unchanged; the frontend draws the waveform and pause metrics
/// from this. Timing/pauses come from the Silero VAD trace (a trained speech
/// detector — more reliable than an energy threshold, which mistakes unvoiced
/// fricatives and stop-closures for gaps); the waveform is the amplitude
/// envelope, used only for display.
#[derive(Clone, Serialize)]
pub struct UtteranceAnalysis {
    pub index: u64,
    /// Total utterance length in ms, so pause offsets map onto the waveform.
    pub duration_ms: u64,
    /// RMS amplitude per bucket, normalized to 0-255; `len() <= ENVELOPE_POINTS`.
    pub envelope: Vec<u8>,
    pub pauses: Vec<Pause>,
    pub pause_count: u32,
    pub total_pause_ms: u64,
    /// Absolute RMS loudness of the whole utterance (0-1). Unlike `envelope`
    /// (self-normalized per utterance), this is comparable across utterances, so
    /// the frontend can score loudness consistency and flag too-quiet speech.
    pub rms_level: f32,
    /// Median voiced pitch in Hz (0.0 if unvoiced).
    pub f0_median: f32,
    /// Within-utterance pitch spread in semitones (the inflection range; ~0 = monotone).
    pub f0_range_semitones: f32,
    /// True if the utterance's pitch rises at the end (uptalk).
    pub f0_terminal_rising: bool,
}

/// A committed utterance queued for the accurate-model correction pass. Carries
/// the full utterance audio (re-decoded from scratch by the medium model) plus
/// the index/timing needed to emit a segment that replaces the fast final.
struct CorrectionJob {
    index: u64,
    start_ms: u64,
    end_ms: u64,
    audio: Vec<f32>,
}

/// Monotonic recording-session counter, bumped on every `start_recording`. The
/// correction worker captures the generation it was spawned for and drops any
/// job (or result) once this moves past it, so a slow medium decode from a
/// previous session can't emit into a new one (whose utterance indices restart
/// and would collide).
#[derive(Default)]
pub struct SessionGen(pub AtomicU64);

pub struct RecordingHandle {
    stop_tx: Sender<()>,
    capture_thread: Option<std::thread::JoinHandle<()>>,
    processing_thread: Option<std::thread::JoinHandle<()>>,
}

#[derive(Default)]
pub struct RecordingState(pub Mutex<Option<RecordingHandle>>);

#[tauri::command]
pub fn start_recording(app: AppHandle) -> Result<(), String> {
    let recording_state = app.state::<RecordingState>();
    let mut guard = recording_state.0.lock().map_err(|e| e.to_string())?;
    if guard.is_some() {
        return Err("already recording".into());
    }

    // Only probed here to learn the native sample rate; the capture thread
    // builds its own Device/Stream since cpal's stream types aren't Send on
    // Linux and must be created and live entirely on one thread.
    let probe_device = cpal::default_host()
        .default_input_device()
        .ok_or("no input device available")?;
    let sample_rate = probe_device
        .default_input_config()
        .map_err(|e| e.to_string())?
        .sample_rate()
        .0;
    drop(probe_device);

    // Claim a fresh session generation. The correction worker stamps its
    // results with this so any decode still running from a prior session is
    // discarded rather than emitted into this one.
    let generation = app
        .state::<SessionGen>()
        .0
        .fetch_add(1, Ordering::AcqRel)
        + 1;

    let (audio_tx, audio_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();

    // Set true while the user is mid-utterance (VAD triggered). Both models share
    // the one GPU, so the correction worker watches this and only decodes during
    // speech gaps — letting live interims own the GPU while the user is talking.
    let speaking = Arc::new(AtomicBool::new(false));

    // Accurate-model correction worker. It owns the receiving end; the sender
    // lives in the processing loop, so when capture stops (loop returns, sender
    // dropped) the worker drains its backlog and exits on its own — no join, so
    // stopping is instant even if a medium decode is still in flight.
    let (correction_tx, correction_rx) = std::sync::mpsc::channel::<CorrectionJob>();
    {
        let worker_app = app.clone();
        let worker_speaking = speaking.clone();
        std::thread::spawn(move || {
            run_correction_worker(worker_app, correction_rx, generation, worker_speaking)
        });
    }

    let capture_thread = std::thread::spawn(move || {
        if let Err(e) = run_capture(audio_tx, stop_rx) {
            eprintln!("audio capture failed: {e}");
        }
    });

    if let Ok(mut vad) = app.state::<VadModel>().0.lock() {
        vad.reset();
    }

    let app_handle = app.clone();
    let processing_thread = std::thread::spawn(move || {
        run_processing_loop(app_handle, audio_rx, sample_rate, correction_tx, speaking)
    });

    *guard = Some(RecordingHandle {
        stop_tx,
        capture_thread: Some(capture_thread),
        processing_thread: Some(processing_thread),
    });

    Ok(())
}

#[tauri::command]
pub fn stop_recording(app: AppHandle) -> Result<(), String> {
    let recording_state = app.state::<RecordingState>();
    let handle = recording_state.0.lock().map_err(|e| e.to_string())?.take();
    let Some(mut handle) = handle else {
        return Err("not recording".into());
    };

    let _ = handle.stop_tx.send(());
    if let Some(t) = handle.capture_thread.take() {
        let _ = t.join();
    }
    if let Some(t) = handle.processing_thread.take() {
        let _ = t.join();
    }
    Ok(())
}

fn run_capture(audio_tx: Sender<Vec<f32>>, stop_rx: Receiver<()>) -> Result<(), String> {
    let device = cpal::default_host()
        .default_input_device()
        .ok_or("no input device available")?;
    let config = device.default_input_config().map_err(|e| e.to_string())?;
    let sample_format = config.sample_format();
    let channels = config.channels();
    let stream_config: cpal::StreamConfig = config.into();
    let err_fn = |e| eprintln!("audio stream error: {e}");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &_| {
                let _ = audio_tx.send(mixdown(data, channels));
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _: &_| {
                let floats: Vec<f32> = data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                let _ = audio_tx.send(mixdown(&floats, channels));
            },
            err_fn,
            None,
        ),
        other => return Err(format!("unsupported input sample format: {other:?}")),
    }
    .map_err(|e| e.to_string())?;

    stream.play().map_err(|e| e.to_string())?;
    let _ = stop_rx.recv();
    Ok(())
}

fn mixdown(data: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        data.to_vec()
    } else {
        data.chunks(channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    }
}

/// Resamples a continuous stream of chunks via linear interpolation,
/// carrying the fractional phase across calls so chunk boundaries don't
/// introduce clicks or long-run drift.
struct Resampler {
    from_rate: u32,
    to_rate: u32,
    buffer: Vec<f32>,
    pos: f64,
}

impl Resampler {
    fn new(from_rate: u32, to_rate: u32) -> Self {
        Self { from_rate, to_rate, buffer: Vec::new(), pos: 0.0 }
    }

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.from_rate == self.to_rate {
            return input.to_vec();
        }

        self.buffer.extend_from_slice(input);
        let ratio = self.from_rate as f64 / self.to_rate as f64;
        let mut out = Vec::new();
        while (self.pos as usize + 1) < self.buffer.len() {
            let idx = self.pos as usize;
            let frac = (self.pos - idx as f64) as f32;
            out.push(self.buffer[idx] + (self.buffer[idx + 1] - self.buffer[idx]) * frac);
            self.pos += ratio;
        }

        let drop_n = self.pos as usize;
        if drop_n > 0 {
            self.buffer.drain(0..drop_n.min(self.buffer.len()));
            self.pos -= drop_n as f64;
        }
        out
    }
}

/// One-pole DC-blocking high-pass filter. Some capture devices (e.g. the
/// laptop's digital mic array) deliver samples with a large constant DC
/// offset; left in, that bias dominates the waveform and makes the VAD
/// score every frame as non-speech. Running `y[n] = x[n] - x[n-1] + R*y[n-1]`
/// removes the offset while leaving the speech band intact.
struct DcBlocker {
    prev_in: f32,
    prev_out: f32,
}

impl DcBlocker {
    const R: f32 = 0.995;

    fn new() -> Self {
        Self { prev_in: 0.0, prev_out: 0.0 }
    }

    fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let y = *s - self.prev_in + Self::R * self.prev_out;
            self.prev_in = *s;
            self.prev_out = y;
            *s = y;
        }
    }
}

/// Buckets the utterance's raw samples into at most `ENVELOPE_POINTS` RMS values,
/// normalized to 0-255 by the loudest bucket, for the waveform sparkline. Purely
/// a function of the committed buffer (not the VAD frames), so it stays aligned
/// regardless of the fractional-frame pre-roll seam.
fn compute_envelope(audio: &[f32]) -> Vec<u8> {
    if audio.is_empty() {
        return Vec::new();
    }
    let points = ENVELOPE_POINTS.min(audio.len());
    let bucket = audio.len().div_ceil(points);
    let mut rms: Vec<f32> = Vec::with_capacity(points);
    let mut peak = 0.0f32;
    for chunk in audio.chunks(bucket) {
        let mean_sq = chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32;
        let r = mean_sq.sqrt();
        peak = peak.max(r);
        rms.push(r);
    }
    rms.into_iter()
        .map(|r| {
            if peak > 0.0 {
                (r / peak * 255.0).round().clamp(0.0, 255.0) as u8
            } else {
                0
            }
        })
        .collect()
}

/// Finds hesitation pauses in the per-frame VAD probability trace: runs of frames
/// below `SPEECH_THRESHOLD` lasting at least `MIN_PAUSE_MS`. Only runs bracketed
/// by speech on *both* sides count — the leading run (pre-roll / onset silence)
/// starts at frame 0 and is skipped, and the trailing run (the silence that ended
/// the utterance) never closes with a following speech frame, so it's excluded
/// too. That leaves genuine mid-utterance gaps, not the utterance boundaries.
fn detect_pauses(probs: &[f32]) -> Vec<Pause> {
    let min_frames = MIN_PAUSE_MS.div_ceil(FRAME_MS) as usize;
    let mut pauses = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, &p) in probs.iter().enumerate() {
        if p < SPEECH_THRESHOLD {
            run_start.get_or_insert(i);
        } else if let Some(start) = run_start.take() {
            // Closed by a speech frame at `i` (bracketed on the right); `start > 0`
            // means it was also preceded by speech (bracketed on the left).
            if start > 0 && i - start >= min_frames {
                pauses.push(Pause {
                    start_ms: start as u64 * FRAME_MS,
                    end_ms: i as u64 * FRAME_MS,
                });
            }
        }
    }
    pauses
}

/// Builds the analysis payload for a committed utterance from its raw buffer
/// (waveform) and its per-frame VAD trace (pauses).
fn analyze_utterance(index: u64, audio: &[f32], probs: &[f32]) -> UtteranceAnalysis {
    let pauses = detect_pauses(probs);
    let total_pause_ms = pauses.iter().map(|p| p.end_ms - p.start_ms).sum();
    let rms_level = if audio.is_empty() {
        0.0
    } else {
        (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt()
    };
    let pitch = crate::pitch::analyze(audio);
    UtteranceAnalysis {
        index,
        duration_ms: audio.len() as u64 * 1000 / TARGET_SAMPLE_RATE as u64,
        envelope: compute_envelope(audio),
        pause_count: pauses.len() as u32,
        total_pause_ms,
        pauses,
        rms_level,
        f0_median: pitch.median_hz,
        f0_range_semitones: pitch.range_semitones,
        f0_terminal_rising: pitch.terminal_rising,
    }
}

fn run_processing_loop(
    app: AppHandle,
    audio_rx: Receiver<Vec<f32>>,
    native_sample_rate: u32,
    correction_tx: Sender<CorrectionJob>,
    speaking: Arc<AtomicBool>,
) {
    let mut resampler = Resampler::new(native_sample_rate, TARGET_SAMPLE_RATE);
    let mut dc_blocker = DcBlocker::new();
    let mut pending: Vec<f32> = Vec::new();
    let mut pre_roll: VecDeque<f32> = VecDeque::with_capacity(PRE_ROLL_SAMPLES);
    let mut utterance: Vec<f32> = Vec::new();
    // Per-frame VAD probabilities parallel to `utterance`, used at commit to
    // locate hesitation pauses. `pre_roll_probs` mirrors `pre_roll` (one prob per
    // frame) so the trace is seeded to match the pre-roll samples prepended to
    // each utterance.
    let mut utterance_probs: Vec<f32> = Vec::new();
    let pre_roll_prob_cap = PRE_ROLL_SAMPLES / FRAME_SAMPLES + 1;
    let mut pre_roll_probs: VecDeque<f32> = VecDeque::with_capacity(pre_roll_prob_cap);
    let mut triggered = false;
    let mut silence_run_ms: u64 = 0;
    let mut total_samples: u64 = 0;
    let mut utterance_start_sample: u64 = 0;
    let mut utterance_index: u64 = 0;
    let mut samples_since_partial: usize = 0;
    // Live input level (peak RMS) for the UI ribbon visualizer, emitted ~every
    // other frame (~64ms) so IPC stays cheap; the frontend smooths it.
    let mut level_peak: f32 = 0.0;
    let mut level_frames: u32 = 0;
    // Ensures at most one interim decode runs at a time; a new one is skipped
    // while the previous is still in the model, which self-throttles to the
    // machine's decode speed.
    let partial_in_flight = Arc::new(AtomicBool::new(false));

    while let Ok(chunk) = audio_rx.recv() {
        let mut resampled = resampler.process(&chunk);
        dc_blocker.process(&mut resampled);
        pending.extend(resampled);

        while pending.len() >= FRAME_SAMPLES {
            let frame: Vec<f32> = pending.drain(0..FRAME_SAMPLES).collect();

            let energy: f32 = frame.iter().map(|s| s * s).sum();
            level_peak = level_peak.max((energy / frame.len() as f32).sqrt());
            level_frames += 1;
            if level_frames >= 2 {
                let _ = app.emit("audio_level", level_peak);
                level_peak = 0.0;
                level_frames = 0;
            }

            let vad_state = app.state::<VadModel>();
            let prob = match vad_state.0.lock() {
                Ok(mut vad) => vad.process(&frame).unwrap_or_else(|e| {
                    eprintln!("VAD inference failed: {e}");
                    0.0
                }),
                Err(_) => 0.0,
            };

            pre_roll.extend(frame.iter().copied());
            while pre_roll.len() > PRE_ROLL_SAMPLES {
                pre_roll.pop_front();
            }
            pre_roll_probs.push_back(prob);
            while pre_roll_probs.len() > pre_roll_prob_cap {
                pre_roll_probs.pop_front();
            }

            if !triggered {
                if prob > SPEECH_THRESHOLD {
                    triggered = true;
                    // Speech started: hold off GPU corrections (see `speaking`).
                    speaking.store(true, Ordering::Release);
                    silence_run_ms = 0;
                    utterance.clear();
                    utterance.extend(pre_roll.iter().copied());
                    // Seed the trace to match the pre-roll samples just prepended,
                    // so pause offsets line up with the utterance buffer.
                    utterance_probs.clear();
                    utterance_probs.extend(pre_roll_probs.iter().copied());
                    utterance_start_sample = total_samples.saturating_sub(pre_roll.len() as u64);
                    utterance_index += 1;
                    samples_since_partial = 0;
                }
            } else {
                utterance.extend_from_slice(&frame);
                utterance_probs.push(prob);
                if prob > SPEECH_THRESHOLD {
                    silence_run_ms = 0;
                } else {
                    silence_run_ms += FRAME_MS;
                }

                let utterance_ms = utterance.len() as u64 * 1000 / TARGET_SAMPLE_RATE as u64;
                let start_ms = utterance_start_sample * 1000 / TARGET_SAMPLE_RATE as u64;
                if silence_run_ms >= MIN_SILENCE_MS || utterance_ms >= MAX_UTTERANCE_MS {
                    triggered = false;
                    // Utterance done: a correction may now run in this gap.
                    speaking.store(false, Ordering::Release);
                    samples_since_partial = 0;
                    let end_ms = start_ms + utterance_ms;
                    let audio = std::mem::take(&mut utterance);
                    // Emit the acoustic analysis (waveform + hesitation pauses)
                    // for this committed utterance before the text decode kicks
                    // off; the frontend correlates it to the line by index.
                    let probs = std::mem::take(&mut utterance_probs);
                    let _ = app.emit(
                        "utterance_analysis",
                        analyze_utterance(utterance_index, &audio, &probs),
                    );
                    // Queue the accurate-model correction before the fast decode:
                    // it re-decodes the same audio in the background and later
                    // replaces the fast final.
                    let _ = correction_tx.send(CorrectionJob {
                        index: utterance_index,
                        start_ms,
                        end_ms,
                        audio: audio.clone(),
                    });
                    spawn_transcription(
                        app.clone(),
                        audio,
                        utterance_index,
                        start_ms,
                        end_ms,
                        true,
                        None,
                    );
                } else {
                    // Not done yet: emit an interim decode of what we have so far.
                    samples_since_partial += FRAME_SAMPLES;
                    if samples_since_partial >= PARTIAL_INTERVAL_SAMPLES
                        && !partial_in_flight.load(Ordering::Acquire)
                    {
                        samples_since_partial = 0;
                        partial_in_flight.store(true, Ordering::Release);
                        // Decode only the trailing window (see PARTIAL_WINDOW_SAMPLES).
                        let partial_audio = if utterance.len() > PARTIAL_WINDOW_SAMPLES {
                            utterance[utterance.len() - PARTIAL_WINDOW_SAMPLES..].to_vec()
                        } else {
                            utterance.clone()
                        };
                        spawn_transcription(
                            app.clone(),
                            partial_audio,
                            utterance_index,
                            start_ms,
                            start_ms + utterance_ms,
                            false,
                            Some(partial_in_flight.clone()),
                        );
                    }
                }
            }

            total_samples += FRAME_SAMPLES as u64;
        }
    }

    // Capture has stopped; let the correction worker drain its backlog.
    speaking.store(false, Ordering::Release);

    if triggered && !utterance.is_empty() {
        let start_ms = utterance_start_sample * 1000 / TARGET_SAMPLE_RATE as u64;
        let end_ms = start_ms + utterance.len() as u64 * 1000 / TARGET_SAMPLE_RATE as u64;
        let _ = app.emit(
            "utterance_analysis",
            analyze_utterance(utterance_index, &utterance, &utterance_probs),
        );
        let _ = correction_tx.send(CorrectionJob {
            index: utterance_index,
            start_ms,
            end_ms,
            audio: utterance.clone(),
        });
        spawn_transcription(
            app,
            utterance,
            utterance_index,
            start_ms,
            end_ms,
            true,
            None,
        );
    }
    // Dropping `correction_tx` here closes the worker's channel, so it drains any
    // remaining jobs and exits.
}

/// Background correction worker. Serially re-decodes each committed utterance
/// with the accurate medium model on its own state/mutex — deferring to speech
/// gaps so it doesn't stall the fast path on the shared GPU — and emits a `refined` segment
/// that replaces the fast final. Serial (one decode at a time) so a slow medium
/// pass can't oversubscribe the machine; if speech outruns it, corrections just
/// trail. Exits when the sender is dropped (capture stopped).
fn run_correction_worker(
    app: AppHandle,
    rx: Receiver<CorrectionJob>,
    generation: u64,
    speaking: Arc<AtomicBool>,
) {
    let gen_matches = |app: &AppHandle| app.state::<SessionGen>().0.load(Ordering::Acquire) == generation;
    while let Ok(job) = rx.recv() {
        // A newer session started: its indices restart from 1 and would collide,
        // so skip the decode entirely (this also drains a backlog fast on stop).
        if !gen_matches(&app) {
            continue;
        }
        // Defer to the live fast path: both models share the one GPU, so wait for
        // a speech gap before starting a (slow) medium decode. During continuous
        // speech this backlogs corrections until the user pauses, keeping interim
        // previews snappy; on stop `speaking` is cleared so the backlog drains.
        // Once we pass this gate the decode still takes the GPU lock, so if the
        // user resumes mid-decode one interim may wait — the rare, bounded hitch.
        while speaking.load(Ordering::Acquire) && gen_matches(&app) {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        if !gen_matches(&app) {
            continue;
        }
        let result = app
            .state::<AccurateModel>()
            .0
            .lock()
            .map_err(|e| e.to_string())
            .and_then(|mut state| whisper::run(&mut state, &job.audio, true));
        // Re-check after the (possibly long) decode; the session may have ended.
        if !gen_matches(&app) {
            continue;
        }
        match result {
            // Empty correction: keep the fast result rather than blanking a line.
            Ok(text) if !text.is_empty() => {
                let _ = app.emit(
                    "transcript_segment",
                    TranscriptSegment {
                        index: job.index,
                        text,
                        start_ms: job.start_ms,
                        end_ms: job.end_ms,
                        is_final: true,
                        refined: true,
                    },
                );
            }
            Ok(_) => {}
            Err(e) => eprintln!("correction failed: {e}"),
        }
    }
}

/// Decodes `samples` with the fast (tiny) model on a background thread and emits
/// the result. `is_final` distinguishes a committed utterance from an interim
/// update. `in_flight`, if present, is cleared once the model is free again so
/// the next interim decode can start.
fn spawn_transcription(
    app: AppHandle,
    samples: Vec<f32>,
    index: u64,
    start_ms: u64,
    end_ms: u64,
    is_final: bool,
    in_flight: Option<Arc<AtomicBool>>,
) {
    std::thread::spawn(move || {
        let result = {
            let whisper_state = app.state::<WhisperModel>();
            whisper_state
                .0
                .lock()
                .map_err(|e| e.to_string())
                .and_then(|mut state| whisper::run(&mut state, &samples, false))
        };

        if let Some(flag) = &in_flight {
            flag.store(false, Ordering::Release);
        }

        match result {
            // Finals always emit (even when empty) so the UI can commit or drop
            // the line; interim updates only emit once there's something to show.
            Ok(text) if is_final || !text.is_empty() => {
                let _ = app.emit(
                    "transcript_segment",
                    TranscriptSegment { index, text, start_ms, end_ms, is_final, refined: false },
                );
            }
            Ok(_) => {}
            Err(e) => {
                if is_final {
                    eprintln!("transcription failed: {e}");
                    let _ = app.emit("transcription_error", e);
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // A low-probability gap of at least MIN_PAUSE_MS, bracketed by speech, is one pause.
    #[test]
    fn detects_internal_pause() {
        let mut probs = vec![0.9; 5];
        probs.extend(vec![0.1; 10]); // 10 frames = 320ms >= 250ms
        probs.extend(vec![0.9; 5]);
        let pauses = detect_pauses(&probs);
        assert_eq!(pauses.len(), 1);
        assert_eq!(pauses[0].start_ms, 5 * FRAME_MS);
        assert_eq!(pauses[0].end_ms, 15 * FRAME_MS);
    }

    // A gap shorter than MIN_PAUSE_MS is a natural inter-word gap, not a pause.
    #[test]
    fn ignores_short_gap() {
        let mut probs = vec![0.9; 5];
        probs.extend(vec![0.1; 5]); // 5 frames = 160ms < 250ms
        probs.extend(vec![0.9; 5]);
        assert!(detect_pauses(&probs).is_empty());
    }

    // Leading (onset/pre-roll) and trailing (commit) silence are utterance
    // boundaries, not hesitations, so neither is counted.
    #[test]
    fn excludes_boundary_silence() {
        let mut leading = vec![0.1; 10];
        leading.extend(vec![0.9; 5]);
        assert!(detect_pauses(&leading).is_empty());

        let mut trailing = vec![0.9; 5];
        trailing.extend(vec![0.1; 10]);
        assert!(detect_pauses(&trailing).is_empty());
    }

    #[test]
    fn counts_multiple_pauses() {
        let mut probs = vec![0.9; 3];
        probs.extend(vec![0.1; 10]);
        probs.extend(vec![0.9; 3]);
        probs.extend(vec![0.1; 9]);
        probs.extend(vec![0.9; 3]);
        assert_eq!(detect_pauses(&probs).len(), 2);
    }

    #[test]
    fn continuous_speech_has_no_pauses() {
        assert!(detect_pauses(&vec![0.9; 40]).is_empty());
    }

    #[test]
    fn envelope_is_bounded_and_normalized() {
        assert!(compute_envelope(&[]).is_empty());

        // Constant amplitude: every bucket equals the peak, so all map to 255.
        let flat = vec![0.5f32; 4000];
        let env = compute_envelope(&flat);
        assert!(!env.is_empty());
        assert!(env.len() <= ENVELOPE_POINTS);
        assert!(env.iter().all(|&v| v == 255));

        // Rising amplitude: the envelope tracks it (louder tail than head).
        let ramp: Vec<f32> = (0..4000).map(|i| i as f32 / 4000.0).collect();
        let env = compute_envelope(&ramp);
        assert!(env.len() <= ENVELOPE_POINTS);
        assert!(env[0] < env[env.len() - 1]);
        assert_eq!(*env.iter().max().unwrap(), 255);
    }

    #[test]
    fn analysis_reports_duration_and_pauses() {
        // 16000 samples @ 16kHz = 1000ms.
        let audio = vec![0.2f32; TARGET_SAMPLE_RATE as usize];
        let mut probs = vec![0.9; 5];
        probs.extend(vec![0.1; 10]);
        probs.extend(vec![0.9; 5]);
        let a = analyze_utterance(7, &audio, &probs);
        assert_eq!(a.index, 7);
        assert_eq!(a.duration_ms, 1000);
        assert_eq!(a.pause_count, 1);
        assert_eq!(a.total_pause_ms, 10 * FRAME_MS);
        assert_eq!(a.total_pause_ms, a.pauses[0].end_ms - a.pauses[0].start_ms);
    }
}

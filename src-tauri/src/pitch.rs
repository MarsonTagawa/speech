//! Autocorrelation F0 (pitch) tracker for one committed utterance. Runs on the
//! same 16kHz mono buffer `analyze_utterance` already has, so it needs no new
//! model or dependency — it's the only signal the grader's "pitch" dimension
//! (monotone / inflection / uptalk) had none of. Kept deliberately small: a
//! per-frame normalized-autocorrelation peak search over the human voice band.

/// Human voice F0 band. Below ~75Hz is sub-voice rumble/hum; above ~400Hz is
/// past normal *speaking* pitch even for high voices (sustained speech sits
/// well under it). These are the physical calibration knobs — a particular
/// mic/room, or an unusually high or low voice, may want them nudged.
const F0_MIN_HZ: f32 = 75.0;
const F0_MAX_HZ: f32 = 400.0;
/// A frame must clear this RMS to be treated as voiced; below it we're in a
/// breath/gap where pitch is meaningless. Tune to the mic's noise floor.
/// ponytail: fixed energy gate, fine for a headset/laptop mic; a proper voicing
/// classifier only pays off if noisy far-field capture becomes a target.
const VOICING_RMS: f32 = 0.01;
/// The best normalized-autocorrelation peak must reach this for the frame to
/// count as voiced-and-pitched; unvoiced fricatives/noise have no clear peak
/// and fall below it.
const VOICING_AC: f32 = 0.3;

const FRAME_LEN: usize = 640; // 40ms @ 16kHz
const HOP: usize = 320; // 20ms @ 16kHz
/// Trailing span examined for uptalk (a statement whose pitch rises at the end).
const TERMINAL_MS: usize = 300;
/// How much the trailing pitch must rise over the utterance median to read as
/// uptalk, in semitones. 2 semitones is an audible, deliberate rise.
const TERMINAL_RISE_SEMITONES: f32 = 2.0;

/// Per-utterance pitch summary. All fields are 0/false when the utterance has no
/// voiced frames (e.g. a whisper or pure noise).
pub struct PitchStats {
    /// Median voiced F0 in Hz (0.0 if none).
    pub median_hz: f32,
    /// Within-utterance pitch spread (10th–90th percentile) in semitones — the
    /// "inflection range". Near 0 = monotone.
    pub range_semitones: f32,
    /// True if the last ~300ms of voiced pitch sits a couple semitones above the
    /// utterance median (uptalk).
    pub terminal_rising: bool,
}

impl PitchStats {
    fn empty() -> Self {
        Self { median_hz: 0.0, range_semitones: 0.0, terminal_rising: false }
    }
}

/// One voiced frame: its center sample (for the trailing-window check) and F0.
struct Voiced {
    center: usize,
    hz: f32,
}

pub fn analyze(audio: &[f32]) -> PitchStats {
    let sr = 16_000.0f32;
    let lag_min = (sr / F0_MAX_HZ) as usize; // 40
    let lag_max = (sr / F0_MIN_HZ) as usize; // ~213
    if audio.len() < FRAME_LEN {
        return PitchStats::empty();
    }

    let mut voiced: Vec<Voiced> = Vec::new();
    let mut start = 0;
    while start + FRAME_LEN <= audio.len() {
        let frame = &audio[start..start + FRAME_LEN];
        if let Some(hz) = frame_f0(frame, lag_min, lag_max, sr) {
            voiced.push(Voiced { center: start + FRAME_LEN / 2, hz });
        }
        start += HOP;
    }

    if voiced.is_empty() {
        return PitchStats::empty();
    }

    let mut hzs: Vec<f32> = voiced.iter().map(|v| v.hz).collect();
    hzs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = percentile(&hzs, 0.5);
    let p10 = percentile(&hzs, 0.10);
    let p90 = percentile(&hzs, 0.90);
    let range_semitones = if p10 > 0.0 { 12.0 * (p90 / p10).log2() } else { 0.0 };

    // Uptalk: median F0 of the trailing window vs the whole-utterance median.
    let terminal_start = audio.len().saturating_sub(TERMINAL_MS * 16); // 16 samples/ms
    let mut tail: Vec<f32> = voiced
        .iter()
        .filter(|v| v.center >= terminal_start)
        .map(|v| v.hz)
        .collect();
    let terminal_rising = if tail.len() >= 2 {
        tail.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let tail_median = percentile(&tail, 0.5);
        median > 0.0 && 12.0 * (tail_median / median).log2() >= TERMINAL_RISE_SEMITONES
    } else {
        false
    };

    PitchStats { median_hz: median, range_semitones, terminal_rising }
}

/// Estimates one frame's F0 via normalized autocorrelation, or None if the frame
/// is too quiet or has no clear periodic peak (unvoiced).
fn frame_f0(frame: &[f32], lag_min: usize, lag_max: usize, sr: f32) -> Option<f32> {
    let energy: f32 = frame.iter().map(|s| s * s).sum();
    let rms = (energy / frame.len() as f32).sqrt();
    if rms < VOICING_RMS || energy == 0.0 {
        return None;
    }

    // r(lag) = sum x[n]*x[n+lag]; normalized by r(0)=energy so the peak is in
    // [0,1] and comparable across frames regardless of loudness.
    let mut best_lag = 0usize;
    let mut best_norm = 0.0f32;
    let mut prev = 0.0f32;
    let mut peak_neighbors = (0.0f32, 0.0f32); // (r[best-1], r[best+1])
    for lag in lag_min..=lag_max.min(frame.len() - 1) {
        let mut sum = 0.0f32;
        for n in 0..frame.len() - lag {
            sum += frame[n] * frame[n + lag];
        }
        let norm = sum / energy;
        if norm > best_norm {
            best_norm = norm;
            best_lag = lag;
            peak_neighbors = (prev, 0.0); // r[best+1] filled next iteration
        } else if best_lag != 0 && lag == best_lag + 1 {
            peak_neighbors.1 = norm;
        }
        prev = norm;
    }

    if best_lag == 0 || best_norm < VOICING_AC {
        return None;
    }

    // Parabolic interpolation around the integer peak lag for sub-sample F0.
    let (rm, rp) = peak_neighbors;
    let denom = rm - 2.0 * best_norm + rp;
    let offset = if denom.abs() > 1e-9 { 0.5 * (rm - rp) / denom } else { 0.0 };
    let lag = best_lag as f32 + offset.clamp(-1.0, 1.0);
    Some(sr / lag)
}

/// Linear-interpolated percentile of an already-sorted slice.
fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f32;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(hz: f32, secs: f32) -> Vec<f32> {
        let n = (16_000.0 * secs) as usize;
        (0..n).map(|i| (i as f32 * hz * TAU / 16_000.0).sin() * 0.5).collect()
    }

    #[test]
    fn detects_steady_tone() {
        let stats = analyze(&sine(200.0, 0.5));
        assert!((stats.median_hz - 200.0).abs() < 5.0, "median {}", stats.median_hz);
        // A pure tone barely inflects.
        assert!(stats.range_semitones < 1.0, "range {}", stats.range_semitones);
        assert!(!stats.terminal_rising);
    }

    #[test]
    fn silence_is_unvoiced() {
        let stats = analyze(&vec![0.0f32; 16_000]);
        assert_eq!(stats.median_hz, 0.0);
        assert_eq!(stats.range_semitones, 0.0);
        assert!(!stats.terminal_rising);
    }

    #[test]
    fn detects_terminal_rise() {
        // Low pitch for 0.7s, then a clearly higher pitch for the last 0.4s.
        let mut audio = sine(150.0, 0.7);
        audio.extend(sine(260.0, 0.4));
        let stats = analyze(&audio);
        assert!(stats.terminal_rising, "median {} range {}", stats.median_hz, stats.range_semitones);
        // Two distinct pitches → a wide inflection range.
        assert!(stats.range_semitones > 3.0, "range {}", stats.range_semitones);
    }

    #[test]
    fn short_buffer_is_empty() {
        let stats = analyze(&vec![0.1f32; 100]);
        assert_eq!(stats.median_hz, 0.0);
    }
}

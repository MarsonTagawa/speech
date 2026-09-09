use whisper_rs::{WhisperState, FullParams, SamplingStrategy};
use std::sync::Mutex;

/// Serializes every GPU decode across both models. The fast (tiny) and accurate
/// (medium) models both run on the Vulkan backend, which shares one device/queue
/// across contexts — two `state.full` calls submitting to it concurrently abort
/// the process (see [`AccurateModel`] and examples/concurrency.rs). Holding this
/// lock for the span of each decode guarantees only one runs at a time. It's a
/// process-global (one GPU), acquired inside [`run`] so every call site — interim,
/// commit, correction, warmup — serializes automatically. The correction worker
/// additionally *defers* to speech gaps (see `audio::run_correction_worker`) so
/// this serialization rarely makes a live interim wait behind a slow medium decode.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Holds one reusable decode state. It's reused across every interim and final
/// decode so we don't reallocate the KV-cache/decode buffers per utterance;
/// decodes serialize on the mutex (which they already did).
///
/// This is the *fast* model (tiny.en-q5_1), on the **GPU** (Vulkan): it drives
/// the live preview and the immediate commit so text appears as soon as possible
/// (~160-500ms per decode on GPU vs ~2x that on CPU).
pub struct WhisperModel(pub Mutex<WhisperState>);

/// The *accurate* model (medium.en-q5_0), also on the **GPU**, used only for the
/// background correction pass. It has its own context/state; the correction
/// worker (see `audio::run_correction_worker`) re-decodes each committed
/// utterance and emits a corrected segment that replaces the fast one.
///
/// Both models share the single Vulkan device/queue, which ggml can't drive from
/// two threads at once (concurrent `state.full` calls abort the process — the
/// "crashes when I speak" bug; see examples/concurrency.rs). Two things keep that
/// safe: [`GPU_LOCK`] serializes all decodes, and the correction worker defers to
/// speech gaps so a slow medium decode almost never blocks a live interim.
pub struct AccurateModel(pub Mutex<WhisperState>);

/// Transcribes 16kHz mono f32 samples. Shared by the manual `transcribe`
/// command and the recording pipeline's per-utterance calls.
///
/// `prime_fillers` seeds the decoder with a hesitation-sound prompt so it keeps
/// "um"/"uh"s instead of cleaning them up (see the prompt below). Only the
/// accurate (medium) correction pass sets it — that pass produces the text the
/// filler tally is scored from. The fast (tiny) live preview leaves it off: tiny
/// is too weak to resist the prompt bias and, primed, collapses a whole spoken
/// sentence into a string of "um"s. The preview just needs to be legible; the
/// correction pass re-decodes and re-tallies fillers anyway.
pub fn run(state: &mut WhisperState, audio: &[f32], prime_fillers: bool) -> Result<String, String> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Latency tuning. Each call transcribes one already-segmented utterance, so:
    // - Pin the language to skip Whisper's auto-detect pass (a whole extra decode).
    // - Use all physical cores instead of the library default (4).
    // - Drop prior-segment context; utterances are independent here.
    // Whisper's matmuls are memory-bandwidth bound, so past the physical core
    // count the extra SMT threads mostly contend for cache and slow the decode
    // down. std only exposes logical CPUs, so cap at 8 (never raising the old
    // count, only trimming hyperthreads on machines that have more).
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().min(8) as i32)
        .unwrap_or(4);
    params.set_n_threads(n_threads);
    params.set_language(Some("en"));
    params.set_translate(false);
    params.set_no_context(true);

    // Repetition-loop handling. Greedy decoding can fall into a loop, emitting
    // the same phrase over and over (in the token trace, result_len never
    // advances). The per-segment token cap (set below) is the bound: it truncates
    // any single decode — looping or not — instead of letting it run out to the
    // model's full context.
    //
    // Whisper's other loop guard, temperature fallback, is deliberately DISABLED
    // here (increment 0 → exactly one decode per segment). It re-decodes a segment
    // at a higher temperature whenever the output looks degenerate — either too
    // repetitive (entropy check, >32 tokens) or low-confidence (avg-logprob
    // check). The filler-priming prompt below trips both on ordinary speech:
    // priming every segment to expect "um"/"uh" lowers decode confidence on the
    // many segments that have none, and makes the ones that do more repetitive.
    // With fallback on, that meant 2–3 full re-decodes on a large fraction of
    // utterances — the dominant cost in the pipeline. Worse, the escape it offers
    // works against us: re-decoding at higher temperature cleans up disfluencies,
    // dropping the very fillers this grader exists to catch. So we keep the single
    // greedy decode and let the token cap bound the rare hard loop.
    params.set_temperature_inc(0.0);

    // Filler retention. Unprompted, the model cleans up disfluencies and drops
    // many "um"/"uh"s — bad for a fluency grader whose job is to flag them. Prime
    // it with a SHORT, non-narrative filler prompt. An earlier attempt used a
    // first-person narrative prompt (e.g. "...it's kind of hard to explain"),
    // which biased the decoder to *continue* the narrative and seeded repetition
    // loops ("it's a little, it's a little, ..."); a bare list of hesitation
    // sounds instead nudges the token distribution toward emitting fillers with
    // no narrative to loop on. The temperature fallback and per-segment token cap
    // still bound any runaway. (suppress_nst already defaults off in whisper.cpp,
    // so non-speech tokens aren't stripped — the prompt is the only lever here.)
    //
    // Only the accurate pass is primed (see `prime_fillers`): on the tiny fast
    // model this prompt overwhelms the decode, so a full sentence comes back as
    // just "um, um". The fast preview stays unprompted and legible.
    if prime_fillers {
        params.set_initial_prompt("Um, uh, hmm, er, um, uh.");
    }

    // Whisper's encoder otherwise always processes a full 30s mel window
    // regardless of clip length. Most utterances are far shorter, so scale the
    // audio context to the actual audio (1 encoder frame = 20ms → 50/sec) plus
    // a margin so trailing speech isn't clipped. Clamped to the full 1500-frame
    // context. This is the largest per-decode saving for short utterances.
    let audio_ctx = (((audio.len() as f32 / 16_000.0) * 50.0).ceil() as i32 + 64)
        .clamp(128, 1500);
    params.set_audio_ctx(audio_ctx);

    // Cap tokens per segment so a repetition loop (or any runaway decode) is
    // bounded instead of generating hundreds of tokens out to the model's full
    // text context. ~10 tokens/s comfortably covers fast speech (≈3-4 words/s);
    // anything beyond the margin is a loop and gets truncated. Scales with clip
    // length, clamped so short interim clips still get a usable budget.
    let max_tokens = (((audio.len() as f32 / 16_000.0) * 10.0).ceil() as i32 + 24)
        .clamp(48, 224);
    params.set_max_tokens(max_tokens);

    let started = std::time::Instant::now();
    {
        // Only one GPU decode at a time across both models (see GPU_LOCK).
        let _gpu = GPU_LOCK.lock().map_err(|e| e.to_string())?;
        state.full(params, audio).map_err(|e| e.to_string())?;
    }
    let elapsed = started.elapsed();

    // Timing log for comparing backends (CPU vs Vulkan) and model sizes.
    // audio_s = clip length, decode = wall-clock spent in Whisper, rtf =
    // decode / audio_s (below 1.0 is faster than real time).
    let audio_s = audio.len() as f32 / 16_000.0;
    let decode_s = elapsed.as_secs_f32();
    let rtf = if audio_s > 0.0 { decode_s / audio_s } else { 0.0 };
    eprintln!(
        "[whisper] audio={audio_s:.2}s decode={:.0}ms rtf={rtf:.2} threads={n_threads}",
        decode_s * 1000.0
    );

    let num_segments = state.full_n_segments().map_err(|e| e.to_string())?;
    let mut text = String::new();
    for i in 0..num_segments {
        // Use the lossy accessor: the per-segment token cap can end a decode
        // mid-token, and a BPE token may split a multi-byte UTF-8 character, so
        // the raw bytes can be incomplete. The strict accessor errors out the
        // whole utterance on that; lossy substitutes a replacement char for the
        // stray byte and keeps the rest of the text.
        text.push_str(&state.full_get_segment_text_lossy(i).map_err(|e| e.to_string())?);
    }
    Ok(collapse_repeats(text.trim()))
}

/// Collapses runaway repetition-loop hallucinations. Greedy Whisper decoding —
/// especially on the small/tiny models — can get stuck re-emitting the same
/// phrase until it hits the per-segment token cap, e.g. "Hello, test. Hello,
/// test. Hello, test. ..." a dozen times for a single spoken "Hello, test". The
/// per-segment token cap only bounds how long the loop runs; it doesn't remove
/// it. Temperature fallback (Whisper's built-in loop breaker) is deliberately
/// off here (see the note above), so this is the backstop.
///
/// It finds the shortest word-block that repeats 3+ times back-to-back and keeps
/// a single copy. Requiring 3 consecutive exact repeats makes this safe for
/// ordinary speech; the trade-off is that a genuine 3x+ verbal repetition
/// ("no, no, no") also collapses to one — an acceptable price versus showing a
/// dozen hallucinated copies, and it keeps the loop from inflating word counts.
fn collapse_repeats(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();
    if n < 3 {
        return text.to_string();
    }
    let mut out: Vec<&str> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let mut collapsed = false;
        // Shortest block first, so a period-2 loop collapses to one 2-word copy
        // rather than to a 4-word (two-period) copy. Need room for >=3 repeats.
        for l in 1..=(n - i) / 3 {
            let mut reps = 1;
            let mut j = i + l;
            while j + l <= n && words[i..i + l] == words[j..j + l] {
                reps += 1;
                j += l;
            }
            if reps >= 3 {
                out.extend_from_slice(&words[i..i + l]); // keep one copy
                i = j; // skip the rest of the run
                collapsed = true;
                break;
            }
        }
        if !collapsed {
            out.push(words[i]);
            i += 1;
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::collapse_repeats;

    #[test]
    fn collapses_phrase_loop() {
        let input = "Hello, test. ".repeat(12);
        assert_eq!(collapse_repeats(input.trim()), "Hello, test.");
    }

    #[test]
    fn collapses_single_word_loop() {
        assert_eq!(collapse_repeats("you you you you you"), "you");
    }

    #[test]
    fn keeps_lead_in_before_loop() {
        assert_eq!(
            collapse_repeats("I think Hello test Hello test Hello test"),
            "I think Hello test"
        );
    }

    #[test]
    fn leaves_normal_speech_alone() {
        let s = "the quick brown fox jumps over the lazy dog";
        assert_eq!(collapse_repeats(s), s);
    }

    #[test]
    fn keeps_a_double() {
        // Only 2 repeats: below the 3x threshold, left untouched.
        assert_eq!(collapse_repeats("bye bye now"), "bye bye now");
    }
}

#[tauri::command]
pub fn transcribe(state: tauri::State<WhisperModel>, audio: Vec<f32>) -> Result<String, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    run(&mut guard, &audio, false)
}

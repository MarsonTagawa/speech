// Quick latency comparison for the fast (tiny) model on GPU vs CPU, with the
// same decode caps whisper::run uses, to size up whether moving tiny back onto
// the GPU is worth the concurrency complexity.
// Run: cargo run --release --example timing --manifest-path src-tauri/Cargo.toml
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

const TINY: &str = "/home/marsont/speech/src-tauri/resources/ggml-tiny.en-q5_1.bin";

fn load(path: &str, use_gpu: bool) -> WhisperState {
    let mut p = WhisperContextParameters::default();
    p.use_gpu(use_gpu);
    p.flash_attn(true);
    WhisperContext::new_with_params(path, p).unwrap().create_state().unwrap()
}

fn decode(state: &mut WhisperState, audio: &[f32]) {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_n_threads(8);
    params.set_language(Some("en"));
    params.set_no_context(true);
    params.set_temperature_inc(0.0);
    params.set_initial_prompt("Um, uh, hmm, er, um, uh.");
    let secs = audio.len() as f32 / 16_000.0;
    params.set_audio_ctx(((secs * 50.0).ceil() as i32 + 64).clamp(128, 1500));
    params.set_max_tokens(((secs * 10.0).ceil() as i32 + 24).clamp(48, 224));
    state.full(params, audio).unwrap();
}

fn bench(label: &str, state: &mut WhisperState, audio: &[f32]) {
    decode(state, audio); // warmup
    let mut best = u128::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        decode(state, audio);
        best = best.min(t.elapsed().as_millis());
    }
    eprintln!("[bench] {label}: best {best} ms over 5 runs");
}

fn main() {
    let a3: Vec<f32> = (0..16_000 * 3).map(|i| ((i as f32) * 0.02).sin() * 0.05).collect();
    let a5: Vec<f32> = (0..16_000 * 5).map(|i| ((i as f32) * 0.02).sin() * 0.05).collect();
    eprintln!("loading tiny on GPU...");
    let mut gpu = load(TINY, true);
    bench("tiny GPU 3s", &mut gpu, &a3);
    bench("tiny GPU 5s", &mut gpu, &a5);
    drop(gpu);
    eprintln!("loading tiny on CPU...");
    let mut cpu = load(TINY, false);
    bench("tiny CPU 3s", &mut cpu, &a3);
    bench("tiny CPU 5s", &mut cpu, &a5);
}

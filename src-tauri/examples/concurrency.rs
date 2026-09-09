// Headless reproduction of the "crashes when I speak" report. During recording,
// the fast (tiny) model runs interim/commit decodes while the accurate (medium)
// model runs a correction decode on its own thread — so both hit the GPU at the
// same time. ggml's Vulkan backend shares one device/queue across contexts, so
// this checks whether concurrent decodes across the two contexts are safe.
//
// Run: cargo run --example concurrency --manifest-path src-tauri/Cargo.toml
use std::sync::{Arc, Mutex};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

const TINY: &str = "/home/marsont/speech/src-tauri/resources/ggml-tiny.en-q5_1.bin";
const MEDIUM: &str = "/home/marsont/speech/src-tauri/resources/ggml-medium.en-q5_0.bin";

// Mirrors whisper::GPU_LOCK: serializes GPU submission across both contexts.
// Without it, both-on-GPU concurrent decodes abort (exit 134); with it they must
// complete cleanly.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn load(path: &str, use_gpu: bool) -> WhisperState {
    let mut p = WhisperContextParameters::default();
    p.use_gpu(use_gpu);
    p.flash_attn(true);
    let ctx = WhisperContext::new_with_params(path, p).expect("load ctx");
    ctx.create_state().expect("create state")
}

fn decode(state: &mut WhisperState, audio: &[f32], threads: i32) {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_n_threads(threads);
    params.set_language(Some("en"));
    params.set_no_context(true);
    params.set_temperature_inc(0.0);
    params.set_initial_prompt("Um, uh, hmm, er, um, uh.");
    // Mirror whisper::run's caps so timing reflects real usage (short audio_ctx +
    // bounded tokens), not a full 30s-window decode on noise.
    let secs = audio.len() as f32 / 16_000.0;
    params.set_audio_ctx(((secs * 50.0).ceil() as i32 + 64).clamp(128, 1500));
    params.set_max_tokens(((secs * 10.0).ceil() as i32 + 24).clamp(48, 224));
    let _gpu = GPU_LOCK.lock().unwrap();
    state.full(params, audio).expect("decode");
}

fn main() {
    eprintln!("loading models (both on GPU, serialized by GPU_LOCK)...");
    let tiny = Arc::new(Mutex::new(load(TINY, true)));
    let medium = Arc::new(Mutex::new(load(MEDIUM, true)));
    // A little non-zero audio so it isn't a trivial silence graph.
    let audio: Vec<f32> = (0..16_000 * 3).map(|i| ((i as f32) * 0.001).sin() * 0.05).collect();

    eprintln!("starting concurrent decodes (tiny GPU x6 || medium GPU x3), serialized...");

    let t = {
        let m = tiny.clone();
        let a = audio.clone();
        std::thread::spawn(move || {
            for i in 0..6 {
                decode(&mut m.lock().unwrap(), &a, 6);
                eprintln!("tiny decode {i} ok");
            }
        })
    };
    let md = {
        let m = medium.clone();
        let a = audio.clone();
        std::thread::spawn(move || {
            for i in 0..3 {
                decode(&mut m.lock().unwrap(), &a, 6);
                eprintln!("medium decode {i} ok");
            }
        })
    };
    t.join().unwrap();
    md.join().unwrap();
    println!("ALL DONE — no crash, concurrent GPU decode is safe");
}

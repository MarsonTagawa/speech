mod audio;
mod history;
mod pitch;
mod vad;
mod whisper;

use audio::RecordingState;
use std::sync::Mutex;
use tauri::Manager;
use vad::{SileroVad, VadModel};
use whisper::{AccurateModel, WhisperModel};
use whisper_rs::{WhisperContext, WhisperContextParameters, WhisperState};

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Loads a bundled ggml model and creates its reusable decode state.
            // Flash attention lowers attention memory and speeds decoding
            // (notably on the Vulkan backend); we don't use DTW timestamps, so
            // its DTW incompatibility doesn't apply. The state holds an Arc to
            // the context, so the context stays alive after the wrapper is
            // dropped; reusing one state avoids reallocating decode buffers per
            // utterance.
            let load_model = |resource: &str, use_gpu: bool| -> Result<WhisperState, Box<dyn std::error::Error>> {
                let path = app.path().resolve(resource, tauri::path::BaseDirectory::Resource)?;
                let mut params = WhisperContextParameters::default();
                params.use_gpu(use_gpu);
                params.flash_attn(true);
                let ctx = WhisperContext::new_with_params(&path.to_string_lossy(), params)?;
                Ok(ctx.create_state()?)
            };

            // Both models run on the GPU (Vulkan). They can't decode at the same
            // time — ggml shares one Vulkan device/queue across contexts and
            // concurrent `state.full` calls abort the process — but that's handled
            // downstream: whisper::GPU_LOCK serializes decodes, and the correction
            // worker defers to speech gaps so the fast path effectively owns the
            // GPU while the user is talking (see AccurateModel / audio.rs).
            //
            //   Fast model — English-only tiny (q5_1). Language is pinned to "en"
            //   in whisper::run so the multilingual heads are dead weight; tiny on
            //   GPU decodes in ~160-500ms, keeping the live preview and immediate
            //   commit fast. Its lower accuracy doesn't matter — the medium model
            //   re-decodes and replaces this draft (see AccurateModel).
            app.manage(WhisperModel(Mutex::new(load_model("resources/ggml-tiny.en-q5_1.bin", true)?)));

            //   Accurate model — English-only medium (q5_0). Slower (~2.7s/3s clip
            //   on GPU) but far more accurate; runs the deferred correction pass.
            app.manage(AccurateModel(Mutex::new(load_model("resources/ggml-medium.en-q5_0.bin", true)?)));

            // Warm both models up on a background thread. The first decode after
            // load pays one-time costs — Vulkan shader compilation, compute-graph
            // setup, buffer allocation — that would otherwise stall the user's
            // first spoken utterance (fast model) or first correction (accurate
            // model) for seconds. A throwaway decode on a short silence buffer
            // pays it during launch instead. Backgrounded so it doesn't hold up
            // the window; each model's mutex serializes the warmup against real
            // decodes, so if recording starts first the first decode just waits
            // for the warmup (the cost it would pay anyway).
            let warmup_handle = app.handle().clone();
            std::thread::spawn(move || {
                // 1s of silence (16kHz mono) still runs the full encoder+decoder
                // graph, which is what triggers the shader/allocation work we
                // want cached before the first real utterance.
                let silence = vec![0.0f32; 16_000];
                let fast = warmup_handle
                    .state::<WhisperModel>()
                    .0
                    .lock()
                    .map_err(|e| e.to_string())
                    .and_then(|mut state| whisper::run(&mut state, &silence, false));
                match fast {
                    Ok(_) => eprintln!("[whisper] fast warmup complete"),
                    Err(e) => eprintln!("[whisper] fast warmup failed: {e}"),
                }
                let accurate = warmup_handle
                    .state::<AccurateModel>()
                    .0
                    .lock()
                    .map_err(|e| e.to_string())
                    .and_then(|mut state| whisper::run(&mut state, &silence, true));
                match accurate {
                    Ok(_) => eprintln!("[whisper] accurate warmup complete"),
                    Err(e) => eprintln!("[whisper] accurate warmup failed: {e}"),
                }
            });

            let vad_model_path = app.path().resolve(
                "resources/silero_vad.onnx",
                tauri::path::BaseDirectory::Resource,
            )?;
            let vad = SileroVad::new(&vad_model_path.to_string_lossy())?;
            app.manage(VadModel(Mutex::new(vad)));

            app.manage(RecordingState::default());
            app.manage(audio::SessionGen::default());

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            whisper::transcribe,
            audio::start_recording,
            audio::stop_recording,
            history::save_session,
            history::list_sessions
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

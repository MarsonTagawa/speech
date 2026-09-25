mod audio;
mod history;
mod models;
mod pitch;
mod vad;
mod whisper;

use audio::RecordingState;
use std::sync::Mutex;
use tauri::Manager;
use vad::{SileroVad, VadModel};
use whisper::{AccurateModel, WhisperModel};

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
            // Both models run on the GPU (Vulkan). They can't decode at the same
            // time — ggml shares one Vulkan device/queue across contexts and
            // concurrent `state.full` calls abort the process — but that's handled
            // downstream: whisper::GPU_LOCK serializes decodes, and corrections
            // only run after recording stops, so the fast path owns the GPU
            // while the user is talking (see AccurateModel / audio.rs).
            //
            //   Fast model — English-only tiny (q5_1). Language is pinned to "en"
            //   in whisper::run so the multilingual heads are dead weight; tiny on
            //   GPU decodes in ~160-500ms, keeping the live preview and immediate
            //   commit fast. Its lower accuracy doesn't matter — the medium model
            //   re-decodes and replaces this draft (see AccurateModel).
            app.manage(WhisperModel(Mutex::new(whisper::load_model(app.handle(), "tiny.en")?)));

            //   The frontend swaps in the user's chosen live model at startup
            //   (whisper::set_live_model); tiny.en is bundled so there's always one.
            //
            //   Accurate model — medium.en by default, chosen in settings. Slower
            //   (~1.6s for an 11s clip on GPU) but far more accurate; runs the
            //   deferred correction pass. Loaded per pass by the correction
            //   worker, not here.
            app.manage(AccurateModel(Mutex::new(None)));

            // Warm the fast model up on a background thread. The first decode after
            // load pays one-time costs — Vulkan shader compilation, compute-graph
            // setup, buffer allocation — that would otherwise stall the user's
            // first spoken utterance for seconds. A throwaway decode on a short silence buffer
            // pays it during launch instead. Backgrounded so it doesn't hold up
            // the window; the model's mutex serializes the warmup against real
            // decodes, so if recording starts first the first decode just waits
            // for the warmup (the cost it would pay anyway).
            let warmup_handle = app.handle().clone();
            std::thread::spawn(move || {
                let fast = warmup_handle
                    .state::<WhisperModel>()
                    .0
                    .lock()
                    .map_err(|e| e.to_string())
                    .and_then(|mut state| whisper::warm_up(&mut state));
                match fast {
                    Ok(_) => eprintln!("[whisper] fast warmup complete"),
                    Err(e) => eprintln!("[whisper] fast warmup failed: {e}"),
                }
            });

            let vad_model_path = app.path().resolve(
                "resources/silero_vad.onnx",
                tauri::path::BaseDirectory::Resource,
            )?;
            let vad = SileroVad::new(&vad_model_path.to_string_lossy())?;
            app.manage(VadModel(Mutex::new(vad)));

            app.manage(RecordingState::default());
            app.manage(audio::MicTestState::default());
            app.manage(audio::SessionGen::default());

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            whisper::transcribe,
            whisper::set_live_model,
            models::list_models,
            models::download_model,
            models::delete_model,
            audio::start_recording,
            audio::stop_recording,
            audio::start_mic_test,
            audio::stop_mic_test,
            history::save_session,
            history::set_session_saved,
            history::load_session_detail,
            history::list_sessions,
            history::clear_sessions
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

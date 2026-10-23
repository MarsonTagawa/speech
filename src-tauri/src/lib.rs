mod audio;
mod corners;
mod drive;
mod history;
mod models;
mod notch;
mod pitch;
mod screen;
mod screenshot;
mod tray;
mod vad;
mod vision;

/// For `examples/vision_probe.rs` only.
pub mod vision_probe {
    pub use crate::vision::Pipelines;
}
mod whisper;

use audio::RecordingState;
use std::sync::Mutex;
use tauri::Manager;
use vad::{SileroVad, VadModel};
use whisper::{AccurateModel, WhisperModel};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Screen OCR (rten, screen/ocr.rs) would take every core from whisper and
    // the camera. Set before any thread starts; rten reads it once.
    if std::env::var_os("RTEN_NUM_THREADS").is_none() {
        std::env::set_var("RTEN_NUM_THREADS", "2");
    }
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
            app.manage(vision::VisionState::default());
            app.manage(screen::ScreenState::default());

            if let Some(main) = app.get_webview_window("main") {
                corners::apply(&main);
            }
            tray::build(app.handle())?;

            Ok(())
        })
        // Release the camera when the main window closes (vision threads
        // otherwise run until the process exits), and take the notch with it
        // so the app can quit. The notch closing (notch.rs) touches nothing.
        .on_window_event(|window, event| {
            // Entering/leaving fullscreen arrives as a resize: square the
            // corners there, round them again after.
            if let (tauri::WindowEvent::Resized(_), Some(main)) = (event, window.app_handle().get_webview_window("main")) {
                if window.label() == "main" {
                    corners::apply(&main);
                }
            }
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                if window.label() != "main" {
                    return;
                }
                window.state::<vision::VisionState>().stop();
                window.state::<screen::ScreenState>().release();
                if let Some(notch) = window.app_handle().get_webview_window("notch") {
                    let _ = notch.destroy();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            whisper::transcribe,
            whisper::set_live_model,
            models::list_models,
            models::download_model,
            models::delete_model,
            audio::start_recording,
            audio::stop_recording,
            audio::start_mic_test,
            audio::stop_mic_test,
            audio::list_mics,
            history::save_session,
            history::set_session_saved,
            history::load_session_detail,
            history::session_clip,
            history::list_sessions,
            history::clear_sessions,
            screenshot::screenshot,
            notch::notch_show,
            tray::tray_recording,
            screen::screen_prepare,
            screen::screen_release,
            screen::screen_displays,
            screen::session_frame,
            screen::session_has_frames,
            screen::screen_text,
            screen::export::export_session,
            vision::start_vision,
            vision::stop_vision,
            vision::list_cameras,
            vision::save_gesture_report,
            drive::fetch_google_pdf,
            drive::google_sign_in,
            drive::google_sign_in_cancel,
            drive::google_sign_out,
            drive::google_status,
            drive::google_list_files
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

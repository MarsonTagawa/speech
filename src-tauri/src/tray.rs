//! Menu bar item: a record ring that fills in while recording, with a menu to
//! start/stop recording, bring the window back, or quit. Recording itself stays
//! in the frontend (toggleRecording in main.ts): the menu emits "tray-record"
//! and main.ts reports the new state back through `tray_recording`.

use tauri::image::Image;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{TrayIcon, TrayIconBuilder};
use tauri::{AppHandle, Emitter, Manager, Wry};

const SIZE: u32 = 36; // 18pt at 2x; macOS scales the tray image to the bar
const RING_OUT: f32 = 13.0;
const RING_IN: f32 = 10.0;
const DOT: f32 = 6.5;

/// Kept so `tray_recording` can relabel the item and swap the icon.
pub struct Tray {
    icon: TrayIcon,
    record: MenuItem<Wry>,
}

pub fn build(app: &AppHandle) -> tauri::Result<()> {
    let record = MenuItem::with_id(app, "record", "Start Recording", true, None::<&str>)?;
    let show = MenuItem::with_id(app, "show", "Show Speech", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Speech", true, Some("CmdOrCtrl+Q"))?;
    let menu = Menu::with_items(app, &[&record, &show, &PredefinedMenuItem::separator(app)?, &quit])?;
    let icon = TrayIconBuilder::with_id("main")
        .icon(glyph(false))
        .icon_as_template(true) // macOS tints it for light/dark menu bars
        .tooltip("Speech")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app, e| match e.id().as_ref() {
            "record" => {
                let _ = app.emit_to("main", "tray-record", ());
            }
            "show" => show_main(app),
            // Closing main runs the CloseRequested cleanup in lib.rs (camera,
            // screen capture, notch) and the app exits with its last window.
            "quit" => match app.get_webview_window("main") {
                Some(main) => {
                    let _ = main.close();
                }
                None => app.exit(0),
            },
            _ => {}
        })
        .build(app)?;
    app.manage(Tray { icon, record });
    Ok(())
}

fn show_main(app: &AppHandle) {
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.unminimize();
        let _ = main.show();
        let _ = main.set_focus();
    }
}

#[tauri::command]
pub fn tray_recording(tray: tauri::State<'_, Tray>, on: bool) -> Result<(), String> {
    let label = if on { "Stop Recording" } else { "Start Recording" };
    tray.record.set_text(label).map_err(|e| e.to_string())?;
    tray.icon.set_icon(Some(glyph(on))).map_err(|e| e.to_string())?;
    // set_icon drops the template flag on macOS; put it back.
    tray.icon.set_icon_as_template(true).map_err(|e| e.to_string())
}

/// The ring (idle) or ring and dot (recording), black on clear; as a template
/// image only the alpha matters. 4x4 supersampled for smooth edges.
fn glyph(recording: bool) -> Image<'static> {
    let c = SIZE as f32 / 2.0;
    let mut rgba = vec![0u8; (SIZE * SIZE * 4) as usize];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let mut hits = 0;
            for sy in 0..4 {
                for sx in 0..4 {
                    let dx = x as f32 + (sx as f32 + 0.5) / 4.0 - c;
                    let dy = y as f32 + (sy as f32 + 0.5) / 4.0 - c;
                    let d = (dx * dx + dy * dy).sqrt();
                    if (RING_IN..=RING_OUT).contains(&d) || (recording && d <= DOT) {
                        hits += 1;
                    }
                }
            }
            rgba[((y * SIZE + x) * 4 + 3) as usize] = (hits * 255 / 16) as u8;
        }
    }
    Image::new_owned(rgba, SIZE, SIZE)
}

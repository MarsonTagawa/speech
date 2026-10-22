//! Live notch (macOS): a click-through overlay wrapped around the MacBook
//! notch, showing the live cues the main window pushes to it (src/notch.ts,
//! notch.html). A no-op on other platforms.

// Keep in step with --wing / --row in notch.html.
const WING: f64 = 110.0; // each side of the notch gap
const ROW: f64 = 28.0; // the script row under the bar
const PILL_W: f64 = 360.0; // no notch: a pill this wide
const PILL_H: f64 = 32.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// The overlay's frame in AppKit's global, bottom-left-origin points,
/// top-aligned on `screen`. `notch` is the gap's x-range (global); None centres
/// a pill instead. `bar_h` is the menu bar height beside the notch.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))] // used by the macOS placement and tests
pub fn notch_frame(screen: Rect, notch: Option<(f64, f64)>, bar_h: f64, script: bool) -> Rect {
    let row = if script { ROW } else { 0.0 };
    let (x, w, h) = match notch {
        Some((l, r)) => (l - WING, r - l + 2.0 * WING, bar_h + row),
        None => (screen.x + (screen.w - PILL_W) / 2.0, PILL_W, PILL_H + row),
    };
    Rect { x, y: screen.y + screen.h - h, w, h }
}

/// Opens (or re-places, e.g. after the Script toggle changes its height) or
/// closes the notch window.
#[tauri::command]
pub async fn notch_show(app: tauri::AppHandle, on: bool, script: bool) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    return mac::show(&app, on, script);
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, on, script);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use super::{notch_frame, Rect};
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSScreen, NSStatusWindowLevel, NSWindow, NSWindowCollectionBehavior as B};
    use objc2_foundation::{NSPoint, NSRect, NSSize};
    use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

    pub fn show(app: &tauri::AppHandle, on: bool, script: bool) -> Result<(), String> {
        let existing = app.get_webview_window("notch");
        if !on {
            if let Some(w) = existing {
                w.close().map_err(|e| e.to_string())?;
            }
            return Ok(());
        }
        let win = match existing {
            Some(w) => w,
            // Built hidden; `place` orders it front without activating the app,
            // so Keynote (or whatever is presenting) keeps focus.
            None => WebviewWindowBuilder::new(app, "notch", WebviewUrl::App("notch.html".into()))
                .title("Notch")
                .transparent(true)
                .decorations(false)
                .focusable(false)
                .shadow(false)
                .resizable(false)
                .skip_taskbar(true)
                .visible_on_all_workspaces(true)
                .always_on_top(true)
                .visible(false)
                .build()
                .map_err(|e| e.to_string())?,
        };
        win.set_ignore_cursor_events(true).map_err(|e| e.to_string())?;
        let ns = win.ns_window().map_err(|e| e.to_string())? as usize; // raw pointers aren't Send
        // AppKit only on the main thread. The window outlives this call: Tauri owns it.
        win.run_on_main_thread(move || unsafe { place(&*(ns as *const NSWindow), script) })
            .map_err(|e| e.to_string())
    }

    // Above the menu bar (Tauri's always-on-top level sits below it, and macOS
    // pushes the frame down), on every Space and over fullscreen apps, around
    // the notch of the built-in display, else a pill on the main screen.
    // ponytail: placed once per open; toggle off/on after a display change.
    unsafe fn place(w: &NSWindow, script: bool) {
        let mtm = MainThreadMarker::new_unchecked();
        let notched = NSScreen::screens(mtm).iter().find(|s| s.safeAreaInsets().top > 0.0);
        let Some(screen) = notched.or_else(|| NSScreen::mainScreen(mtm)) else { return };
        let f = screen.frame();
        let bar_h = screen.safeAreaInsets().top;
        let gap = (bar_h > 0.0).then(|| {
            let (l, r) = (screen.auxiliaryTopLeftArea(), screen.auxiliaryTopRightArea());
            (l.origin.x + l.size.width, r.origin.x)
        });
        let screen_rect = Rect { x: f.origin.x, y: f.origin.y, w: f.size.width, h: f.size.height };
        let r = notch_frame(screen_rect, gap, bar_h, script);
        w.setLevel(NSStatusWindowLevel);
        w.setCollectionBehavior(B::CanJoinAllSpaces | B::Stationary | B::FullScreenAuxiliary | B::IgnoresCycle);
        w.setHasShadow(false);
        w.setIgnoresMouseEvents(true);
        w.setFrame_display(NSRect::new(NSPoint::new(r.x, r.y), NSSize::new(r.w, r.h)), true);
        w.orderFrontRegardless();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 14" MacBook Pro: 1512x982 pt, notch gap 656..856, menu bar 38 pt.
    const MBP: Rect = Rect { x: 0.0, y: 0.0, w: 1512.0, h: 982.0 };

    #[test]
    fn wings_straddle_the_notch() {
        assert_eq!(
            notch_frame(MBP, Some((656.0, 856.0)), 38.0, false),
            Rect { x: 546.0, y: 944.0, w: 420.0, h: 38.0 }
        );
        // The script row hangs below, so the frame grows downwards (AppKit y is up).
        assert_eq!(
            notch_frame(MBP, Some((656.0, 856.0)), 38.0, true),
            Rect { x: 546.0, y: 916.0, w: 420.0, h: 66.0 }
        );
    }

    #[test]
    fn pill_centred_without_a_notch() {
        // External display left of the main one: negative global x.
        let ext = Rect { x: -1920.0, y: 0.0, w: 1920.0, h: 1080.0 };
        assert_eq!(notch_frame(ext, None, 0.0, false), Rect { x: -1140.0, y: 1048.0, w: 360.0, h: 32.0 });
        assert_eq!(notch_frame(ext, None, 0.0, true), Rect { x: -1140.0, y: 1020.0, w: 360.0, h: 60.0 });
    }
}

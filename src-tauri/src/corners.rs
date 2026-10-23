//! Rounded corners for the main window on macOS. With decorations off it's a
//! borderless NSWindow, which AppKit draws square. Clipping the content view's
//! layer and clearing the window background rounds it; the shadow then follows
//! the rounded shape. Square in fullscreen, where corners would show black.

const RADIUS: f64 = 12.0;

#[cfg(target_os = "macos")]
pub fn apply(window: &tauri::WebviewWindow) {
    use objc2_app_kit::{NSColor, NSWindow};
    let Ok(ptr) = window.ns_window() else { return };
    // SAFETY: tauri hands back the live NSWindow; we're on the main thread
    // (setup and window events run there).
    let ns_window = unsafe { &*(ptr as *const NSWindow) };
    ns_window.setOpaque(false);
    ns_window.setBackgroundColor(Some(&NSColor::clearColor()));
    let r = if window.is_fullscreen().unwrap_or(false) { 0.0 } else { RADIUS };
    if let Some(view) = ns_window.contentView() {
        view.setWantsLayer(true);
        unsafe {
            let layer: *mut objc2::runtime::AnyObject = objc2::msg_send![&*view, layer];
            if !layer.is_null() {
                let _: () = objc2::msg_send![layer, setCornerRadius: r];
                let _: () = objc2::msg_send![layer, setMasksToBounds: true];
            }
        }
    }
    ns_window.invalidateShadow();
}

#[cfg(not(target_os = "macos"))]
pub fn apply(_window: &tauri::WebviewWindow) {
    let _ = RADIUS;
}

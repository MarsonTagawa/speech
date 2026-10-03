//! Report screenshot: the webview's own snapshot of the page (so canvases and
//! fonts come out exactly as drawn), cropped to the report's rect, then saved
//! to Downloads and/or copied to the clipboard.
//!
//! Each platform's webview has its own snapshot call: WebKitGTK's
//! `get_snapshot`, WKWebView's `takeSnapshot`, WebView2's DevTools
//! `Page.captureScreenshot`. The clipboard is written natively on Linux and
//! macOS; on Windows the PNG goes back to the page, whose ClipboardItem
//! (Chromium) also writes the bitmap formats Windows apps paste.

use std::sync::mpsc::{channel, Sender};
use tauri::{ipc::Response, AppHandle, Manager, WebviewWindow};

type Png = Result<Vec<u8>, String>;

/// The crop, in CSS px, plus the page's CSS width (gives the device-pixel ratio).
#[derive(Clone, Copy)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))] // macOS/Windows crop in CSS px natively
    view_w: f64,
}

/// Saves to Downloads when given a file `name`; copies when `copy`. Returns the
/// PNG only when the page has to copy it itself (Windows), else nothing.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn screenshot(
    app: AppHandle,
    window: WebviewWindow,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    view_w: f64,
    name: Option<String>,
    copy: bool,
) -> Result<Response, String> {
    let r = Rect { x, y, w, h, view_w };
    let (tx, rx) = channel();
    window.with_webview(move |wv| snap(wv, r, tx)).map_err(|e| e.to_string())?;
    let png = recv(rx).await??;
    if let Some(n) = name {
        // File name only: nothing outside Downloads.
        let file = std::path::Path::new(&n).file_name().ok_or("bad file name")?.to_owned();
        let dir = app.path().download_dir().map_err(|e| e.to_string())?;
        std::fs::write(dir.join(file), &png).map_err(|e| e.to_string())?;
    }
    #[cfg(windows)]
    return Ok(Response::new(if copy { png } else { Vec::new() }));
    #[cfg(not(windows))]
    {
        if copy {
            // The clipboard belongs to the UI thread on Linux and macOS.
            let (tx, rx) = channel();
            app.run_on_main_thread(move || {
                let _ = tx.send(copy_png(&png));
            })
            .map_err(|e| e.to_string())?;
            recv(rx).await??;
        }
        Ok(Response::new(Vec::new()))
    }
}

async fn recv<T: Send + 'static>(rx: std::sync::mpsc::Receiver<T>) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(move || rx.recv())
        .await
        .map_err(|e| e.to_string())?
        .map_err(|_| "screenshot was cancelled".to_string())
}

#[cfg(target_os = "linux")]
fn snap(wv: tauri::webview::PlatformWebview, r: Rect, tx: Sender<Png>) {
    use webkit2gtk::{SnapshotOptions, SnapshotRegion, WebViewExt};
    wv.inner().snapshot(SnapshotRegion::Visible, SnapshotOptions::NONE, None::<&gtk::gio::Cancellable>, move |res| {
        let _ = tx.send(res.map_err(|e| e.to_string()).and_then(|s| crop(s, r)));
    });
}

#[cfg(target_os = "linux")]
fn crop(src: gtk::cairo::Surface, r: Rect) -> Png {
    use gtk::{cairo, gdk};
    let err = |e: cairo::Error| e.to_string();
    let src = cairo::ImageSurface::try_from(src).map_err(|_| "unexpected snapshot format")?;
    src.set_device_scale(1.0, 1.0); // work in raw pixels
    let k = src.width() as f64 / r.view_w.max(1.0);
    let (pw, ph) = ((r.w * k).round() as i32, (r.h * k).round() as i32);
    let out = cairo::ImageSurface::create(cairo::Format::ARgb32, pw, ph).map_err(err)?;
    let cr = cairo::Context::new(&out).map_err(err)?;
    cr.set_source_surface(&src, -r.x * k, -r.y * k).map_err(err)?;
    cr.paint().map_err(err)?;
    drop(cr);
    let pb = gdk::pixbuf_get_from_surface(&out, 0, 0, pw, ph).ok_or("couldn't convert the screenshot")?;
    pb.save_to_bufferv("png", &[]).map_err(|e| e.to_string())
}

#[cfg(target_os = "linux")]
fn copy_png(png: &[u8]) -> Result<(), String> {
    use gtk::{gdk, gdk_pixbuf::Pixbuf, gio, glib};
    let stream = gio::MemoryInputStream::from_bytes(&glib::Bytes::from(png));
    let pb = Pixbuf::from_stream(&stream, None::<&gio::Cancellable>).map_err(|e| e.to_string())?;
    gtk::Clipboard::get(&gdk::SELECTION_CLIPBOARD).set_image(&pb);
    Ok(())
}

#[cfg(target_os = "macos")]
fn snap(wv: tauri::webview::PlatformWebview, r: Rect, tx: Sender<Png>) {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSBitmapImageFileType, NSBitmapImageRep, NSImage};
    use objc2_core_foundation::{CGPoint, CGRect, CGSize};
    use objc2_foundation::{NSDictionary, NSError};
    use objc2_web_kit::{WKSnapshotConfiguration, WKWebView};
    // with_webview runs this on the main thread; WKWebView is flipped, so the
    // rect is in the page's own top-left-origin points.
    unsafe {
        let view = &*wv.inner().cast::<WKWebView>();
        let config = WKSnapshotConfiguration::new(MainThreadMarker::new_unchecked());
        config.setRect(CGRect::new(CGPoint::new(r.x, r.y), CGSize::new(r.w, r.h)));
        let done = block2::RcBlock::new(move |img: *mut NSImage, _: *mut NSError| {
            let png = img
                .as_ref()
                .and_then(|img| img.TIFFRepresentation())
                .and_then(|tiff| NSBitmapImageRep::imageRepWithData(&tiff))
                .and_then(|rep| rep.representationUsingType_properties(NSBitmapImageFileType::PNG, &NSDictionary::new()))
                .map(|data| data.to_vec())
                .ok_or_else(|| "couldn't take the screenshot".to_string());
            let _ = tx.send(png);
        });
        view.takeSnapshotWithConfiguration_completionHandler(Some(&config), &done);
    }
}

#[cfg(target_os = "macos")]
fn copy_png(png: &[u8]) -> Result<(), String> {
    use objc2_app_kit::{NSPasteboard, NSPasteboardTypePNG};
    use objc2_foundation::NSData;
    unsafe {
        let pb = NSPasteboard::generalPasteboard();
        pb.clearContents();
        if pb.setData_forType(Some(&NSData::with_bytes(png)), NSPasteboardTypePNG) {
            Ok(())
        } else {
            Err("couldn't write to the clipboard".into())
        }
    }
}

#[cfg(windows)]
fn snap(wv: tauri::webview::PlatformWebview, r: Rect, tx: Sender<Png>) {
    use base64::Engine;
    use webview2_com::CallDevToolsProtocolMethodCompletedHandler;
    use windows::core::HSTRING;
    let params = serde_json::json!({
        "format": "png",
        "clip": { "x": r.x, "y": r.y, "width": r.w, "height": r.h, "scale": 1 },
    })
    .to_string();
    let fail = tx.clone();
    let handler = CallDevToolsProtocolMethodCompletedHandler::create(Box::new(move |res, json| {
        let png = res.map_err(|e| e.to_string()).and_then(|()| {
            let v: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
            let data = v["data"].as_str().ok_or("couldn't take the screenshot")?;
            base64::engine::general_purpose::STANDARD.decode(data).map_err(|e| e.to_string())
        });
        let _ = tx.send(png);
        Ok(())
    }));
    let call = unsafe {
        wv.controller().CoreWebView2().and_then(|core| {
            core.CallDevToolsProtocolMethod(&HSTRING::from("Page.captureScreenshot"), &HSTRING::from(params), &handler)
        })
    };
    if let Err(e) = call {
        let _ = fail.send(Err(e.to_string()));
    }
}

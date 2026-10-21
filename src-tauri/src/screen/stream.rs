//! Screen capture behind one small interface: open a display, pull RGB frames.
//! Everything platform-specific lives in this file.
//!
//! Linux: the xdg-desktop-portal ScreenCast portal (ashpd) picks the display
//! and hands over a PipeWire stream, read on a thread of its own. Neither
//! capture crate worked on Wayland as published (scap times the picker out
//! after 10 s; xcap races Start ahead of the pick and its PipeWire format
//! request didn't match Hyprland's), so this path is ours.
//! macOS/Windows: xcap's recorder (ScreenCaptureKit / Windows Graphics Capture).
use serde::Serialize;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::time::{Duration, Instant};

pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

#[derive(Serialize)]
pub struct Display {
    pub id: String,
    pub name: String,
}

/// Frames arrive over a one-slot channel; the capture side drops frames
/// while it's full instead of queueing them.
pub struct Stream {
    rx: Receiver<Result<Frame, String>>,
    #[cfg(target_os = "linux")]
    quit: pipewire::channel::Sender<()>,
    #[cfg(not(target_os = "linux"))]
    rec: xcap::VideoRecorder,
}

impl Stream {
    /// `Ok(None)` on timeout; `Err` once the stream has ended.
    pub fn next(&self, timeout: Duration) -> Result<Option<Frame>, String> {
        match self.rx.recv_timeout(timeout) {
            Ok(r) => r.map(Some),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err("the screen share ended".into()),
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        let _ = self.quit.send(()); // ends the PipeWire loop; the thread closes the portal session
        #[cfg(not(target_os = "linux"))]
        let _ = self.rec.stop();
    }
}

/// Opens `display` (an id from `displays`; ignored on Linux, where the portal
/// asks). Blocks until the user picks or permission resolves.
pub fn open(display: Option<String>) -> Result<Stream, String> {
    imp::open(display)
}

/// Displays to choose from on macOS/Windows. Linux's portal has its own picker.
pub fn displays() -> Vec<Display> {
    imp::displays()
}

/// The newest frame is all the recorder needs; it samples every 250 ms.
const MIN_GAP: Duration = Duration::from_millis(200);

/// The newest raw frame from the capture, converted and sent at most every
/// MIN_GAP. A frame held back by the gap isn't lost: `flush` sends it later
/// (each path calls it on a timer too), because a compositor that only sends
/// frames on change may never send another (a slide that has settled).
// ponytail: copies every raw frame (~8 MB at 1080p; ~0.5 GB/s while the screen animates at 60 Hz); convert in place if that shows up.
#[derive(Default)]
struct Newest {
    raw: Vec<u8>,
    w: usize,
    h: usize,
    order: [usize; 3],
    dirty: bool,
    last: Option<Instant>,
}

impl Newest {
    fn put(&mut self, w: usize, h: usize, raw: &[u8], order: [usize; 3]) {
        self.raw.clear();
        self.raw.extend_from_slice(raw);
        (self.w, self.h, self.order, self.dirty) = (w, h, order, true);
    }

    /// Sends the held frame if it's new and the gap has passed. Returns false
    /// once the receiving Stream is gone.
    fn flush(&mut self, tx: &SyncSender<Result<Frame, String>>) -> bool {
        if !self.dirty || self.last.is_some_and(|t| t.elapsed() < MIN_GAP) {
            return true;
        }
        match tx.try_send(Ok(Frame { w: self.w, h: self.h, rgb: pack(self.w, self.h, &self.raw, self.order) })) {
            Ok(()) => {
                self.dirty = false;
                self.last = Some(Instant::now());
                true
            }
            Err(TrySendError::Full(_)) => true, // still dirty: retried on the next call
            Err(TrySendError::Disconnected(_)) => false,
        }
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::{Display, Frame, Newest, Stream, MIN_GAP};
    use ashpd::desktop::screencast::{CursorMode, Screencast, SelectSourcesOptions, SourceType};
    use ashpd::desktop::PersistMode;
    use pipewire::spa::param::format::{FormatProperties, MediaSubtype, MediaType};
    use pipewire::spa::param::video::{VideoFormat, VideoInfoRaw};
    use pipewire::spa::param::{format_utils, ParamType};
    use pipewire::spa::pod::{self, serialize::PodSerializer, Pod};
    use pipewire::spa::utils::{Direction, SpaTypes};
    use pipewire::stream::{StreamFlags, StreamRc, StreamState};
    use std::cell::RefCell;
    use std::io::Cursor;
    use std::os::fd::OwnedFd;
    use std::rc::Rc;
    use std::sync::mpsc::{self, SyncSender};

    pub fn displays() -> Vec<Display> {
        vec![]
    }

    /// One runtime for every portal call, for the app's lifetime: ashpd shares
    /// one D-Bus connection process-wide, and its tasks live on the runtime that
    /// first made it, so a per-open runtime left later opens hanging forever.
    /// A worker thread keeps that connection serviced between calls.
    static RT: std::sync::LazyLock<tokio::runtime::Runtime> = std::sync::LazyLock::new(|| {
        tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build().expect("tokio runtime for the screen portal")
    });

    pub fn open(_display: Option<String>) -> Result<Stream, String> {
        let (tx, rx) = mpsc::sync_channel(1);
        let (opened_tx, opened) = mpsc::channel();
        let (quit, quit_rx) = pipewire::channel::channel();
        std::thread::spawn(move || {
            RT.block_on(async move {
                let proxy = match Screencast::new().await {
                    Ok(p) => p,
                    Err(e) => return void(opened_tx.send(Err(e.to_string()))),
                };
                let session = match proxy.create_session(Default::default()).await {
                    Ok(s) => s,
                    Err(e) => return void(opened_tx.send(Err(e.to_string()))),
                };
                let started = async {
                    let sources = SelectSourcesOptions::default()
                        .set_cursor_mode(CursorMode::Embedded)
                        .set_sources(ashpd::enumflags2::BitFlags::from(SourceType::Monitor))
                        .set_multiple(false)
                        .set_persist_mode(PersistMode::DoNot);
                    proxy.select_sources(&session, sources).await?.response()?; // waits for the pick
                    let streams = proxy.start(&session, None, Default::default()).await?.response()?;
                    let node = streams.streams().first().map(|s| s.pipe_wire_node_id()).ok_or(ashpd::Error::NoResponse)?;
                    let fd = proxy.open_pipe_wire_remote(&session, Default::default()).await?;
                    Ok::<_, ashpd::Error>((node, fd))
                }
                .await;
                match started {
                    Ok((node, fd)) => {
                        // Runs until the stream ends or the Stream is dropped, off the
                        // runtime's worker so the portal connection stays serviced.
                        let pumped = tokio::task::spawn_blocking(move || pump(node, fd, tx, quit_rx, opened_tx)).await;
                        if let Ok(Err(e)) = pumped {
                            eprintln!("[screen] pipewire: {e}");
                        }
                    }
                    Err(ashpd::Error::Response(ashpd::desktop::ResponseError::Cancelled)) => {
                        let _ = opened_tx.send(Err("you cancelled the screen picker".into()));
                    }
                    Err(e) => {
                        let _ = opened_tx.send(Err(e.to_string()));
                    }
                }
                let _ = session.close().await;
            });
        });
        opened.recv().map_err(|_| "screen capture failed to start".to_string())??;
        Ok(Stream { rx, quit })
    }

    fn void<T>(_: T) {}

    /// Every 4-byte RGB layout a compositor might hand out, with where R, G
    /// and B sit in each pixel. Size and frame rate are left to the producer.
    const FORMATS: [(VideoFormat, [usize; 3]); 8] = [
        (VideoFormat::BGRx, [2, 1, 0]),
        (VideoFormat::BGRA, [2, 1, 0]),
        (VideoFormat::RGBx, [0, 1, 2]),
        (VideoFormat::RGBA, [0, 1, 2]),
        (VideoFormat::xRGB, [1, 2, 3]),
        (VideoFormat::ARGB, [1, 2, 3]),
        (VideoFormat::xBGR, [3, 2, 1]),
        (VideoFormat::ABGR, [3, 2, 1]),
    ];

    fn pump(
        node: u32,
        fd: OwnedFd,
        tx: SyncSender<Result<Frame, String>>,
        quit_rx: pipewire::channel::Receiver<()>,
        opened_tx: mpsc::Sender<Result<(), String>>,
    ) -> Result<(), pipewire::Error> {
        pipewire::init();
        let main_loop = pipewire::main_loop::MainLoopRc::new(None)?;
        let context = pipewire::context::ContextRc::new(&main_loop, None)?;
        let core = context.connect_fd_rc(fd, None)?;
        let stream = StreamRc::new(
            core,
            "speech-screen",
            pipewire::properties::properties! {
                *pipewire::keys::MEDIA_TYPE => "Video",
                *pipewire::keys::MEDIA_CATEGORY => "Capture",
                *pipewire::keys::MEDIA_ROLE => "Screen",
            },
        )?;

        let newest = Rc::new(RefCell::new(Newest::default()));
        let (newest_p, tx_p) = (newest.clone(), tx.clone());
        let err_tx = tx.clone();
        let ended = main_loop.clone();
        let _listener = stream
            .add_local_listener_with_user_data(VideoInfoRaw::default())
            .state_changed(move |_, _, _, new| {
                let why = match new {
                    StreamState::Error(e) => e,
                    StreamState::Unconnected => "the screen share ended".into(),
                    _ => return,
                };
                let _ = err_tx.send(Err(why));
                ended.quit();
            })
            .param_changed(|_, info, id, param| {
                let Some(param) = param else { return };
                if id != ParamType::Format.as_raw() {
                    return;
                }
                if let Ok((MediaType::Video, MediaSubtype::Raw)) = format_utils::parse_format(param) {
                    let _ = info.parse(param);
                }
            })
            .process(move |stream, info| {
                let Some(mut buffer) = stream.dequeue_buffer() else { return };
                let Some(&(_, order)) = FORMATS.iter().find(|(f, _)| *f == info.format()) else { return };
                let (w, h) = (info.size().width as usize, info.size().height as usize);
                let datas = buffer.datas_mut();
                let Some(d) = datas.first_mut() else { return };
                let (offset, size) = (d.chunk().offset() as usize, d.chunk().size() as usize);
                let Some(bytes) = d.data() else { return }; // unmapped (DMA-BUF): we only ask for memory buffers
                let bytes = if size > 0 && offset + size <= bytes.len() { &bytes[offset..offset + size] } else { bytes };
                if w == 0 || h == 0 || bytes.len() < w * h * 4 {
                    return;
                }
                let mut newest = newest_p.borrow_mut();
                newest.put(w, h, bytes, order);
                newest.flush(&tx_p);
            })
            .register()?;

        let obj = pod::object!(
            SpaTypes::ObjectParamFormat,
            ParamType::EnumFormat,
            pod::property!(FormatProperties::MediaType, Id, MediaType::Video),
            pod::property!(FormatProperties::MediaSubtype, Id, MediaSubtype::Raw),
            pod::property!(
                FormatProperties::VideoFormat,
                Choice,
                Enum,
                Id,
                VideoFormat::BGRx, // the default; then every alternative
                VideoFormat::BGRx,
                VideoFormat::BGRA,
                VideoFormat::RGBx,
                VideoFormat::RGBA,
                VideoFormat::xRGB,
                VideoFormat::ARGB,
                VideoFormat::xBGR,
                VideoFormat::ABGR,
            ),
        );
        let bytes = PodSerializer::serialize(Cursor::new(Vec::new()), &pod::Value::Object(obj))
            .map_err(|_| pipewire::Error::CreationFailed)?
            .0
            .into_inner();
        let mut params = [Pod::from_bytes(&bytes).ok_or(pipewire::Error::CreationFailed)?];
        stream.connect(Direction::Input, Some(node), StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS, &mut params)?;

        // Sends a frame the gap held back once no newer one has come.
        let timer = main_loop.loop_().add_timer(move |_| {
            newest.borrow_mut().flush(&tx);
        });
        let _ = timer.update_timer(Some(MIN_GAP), Some(MIN_GAP));

        let quit_loop = main_loop.clone();
        let _quit = quit_rx.attach(main_loop.loop_(), move |()| quit_loop.quit());
        let _ = opened_tx.send(Ok(()));
        main_loop.run();
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::{Display, Newest, Stream, MIN_GAP};
    use std::sync::mpsc::{self, RecvTimeoutError};
    use xcap::Monitor;

    pub fn displays() -> Vec<Display> {
        Monitor::all()
            .unwrap_or_default()
            .iter()
            .filter_map(|m| Some(Display { id: m.id().ok()?.to_string(), name: m.name().ok()? }))
            .collect()
    }

    pub fn open(display: Option<String>) -> Result<Stream, String> {
        let monitors = Monitor::all().map_err(|e| e.to_string())?;
        let pick = |m: &&Monitor| match &display {
            Some(id) => m.id().is_ok_and(|i| i.to_string() == *id),
            None => m.is_primary().unwrap_or(false),
        };
        let monitor = monitors.iter().find(pick).or(monitors.first()).ok_or("no display found")?;
        let (rec, frames) = monitor.video_recorder().map_err(|e| e.to_string())?;
        let (tx, rx) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut newest = Newest::default();
            loop {
                match frames.recv_timeout(MIN_GAP) {
                    Ok(f) => {
                        let (w, h) = (f.width as usize, f.height as usize);
                        if w > 0 && h > 0 && f.raw.len() >= w * h * 4 {
                            newest.put(w, h, &f.raw, [0, 1, 2]);
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
                if !newest.flush(&tx) {
                    break;
                }
            }
        });
        rec.start().map_err(|e| e.to_string())?;
        Ok(Stream { rx, rec })
    }
}

/// 4-byte pixels → packed RGB, taking channels at `order`. Rows may be padded:
/// the stride is `data.len() / h`.
fn pack(w: usize, h: usize, data: &[u8], order: [usize; 3]) -> Vec<u8> {
    let stride = data.len() / h;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for row in data.chunks_exact(stride).take(h) {
        for p in row[..w * 4].chunks_exact(4) {
            rgb.extend([p[order[0]], p[order[1]], p[order[2]]]);
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    // Review #1: a compositor that only sends frames on change may never send
    // another, so a frame held back by the gap must still go out later.
    #[test]
    fn a_frame_held_back_by_the_gap_is_sent_later() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut n = Newest::default();
        n.put(1, 1, &[1, 1, 1, 0], [0, 1, 2]);
        n.flush(&tx);
        assert_eq!(rx.try_recv().unwrap().unwrap().rgb, vec![1, 1, 1]);
        n.put(1, 1, &[2, 2, 2, 0], [0, 1, 2]); // inside the gap: held, not sent
        n.flush(&tx);
        assert!(rx.try_recv().is_err());
        std::thread::sleep(MIN_GAP);
        n.flush(&tx); // the timer's call, with no new frame
        assert_eq!(rx.try_recv().unwrap().unwrap().rgb, vec![2, 2, 2]);
        n.flush(&tx);
        assert!(rx.try_recv().is_err()); // sent once
    }

    #[test]
    fn packs_bgra_with_padded_rows() {
        // 2×2 BGRA, rows padded to 12 bytes (stride 12, 4 bytes junk each row).
        let data = [
            3, 2, 1, 255, 6, 5, 4, 255, 9, 9, 9, 9, //
            12, 11, 10, 255, 15, 14, 13, 255, 9, 9, 9, 9,
        ];
        assert_eq!(pack(2, 2, &data, [2, 1, 0]), vec![1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]);
    }
}

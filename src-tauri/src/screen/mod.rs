//! Screen replay: records one display (plus the camera when it's on) during a
//! session as change-driven JPEG frames in `sessions/<ts>.frames`, replayed in
//! the report in sync with the session audio. See
//! docs/superpowers/specs/2026-10-20-screen-replay-design.md.
mod frames;
pub mod stream;

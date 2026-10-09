//! Camera capture and face/pose/hand inference for presence grading, run in
//! Rust so the webview's UI thread stays free (see
//! docs/superpowers/specs/2026-10-08-rust-vision-design.md).
pub mod geometry;
pub mod face;
pub mod pose;
pub mod hands;
mod model;
#[cfg(test)]
mod golden_util;

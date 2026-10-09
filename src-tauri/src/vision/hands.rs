//! Hands pipeline: MediaPipe's palm detector → rotated ROI per palm → 21
//! landmarks; tracks follow the landmarks, and the detector re-runs about once
//! a second while fewer than 2 hands are tracked. Constants from
//! hand_detector_graph.cc, hand_landmarks_detector_graph.cc and
//! hand_landmarks_to_rect_calculator.cc.

use super::geometry::{crop, decode, hand_track_roi, ssd_anchors, weighted_nms, Anchor, Norm, Roi};
use super::model::Model;
use std::path::Path;

/// Re-run the palm detector this often (frames, ~1 s) while < 2 hands are tracked.
const REDETECT_EVERY: u32 = 12;

pub struct Hands {
    det: Model,
    lm: Model,
    anchors: Vec<Anchor>,
    tracks: Vec<Roi>,
    since_detect: u32,
}

impl Hands {
    pub fn load(dir: &Path) -> Result<Hands, String> {
        Ok(Hands {
            det: Model::load(dir, "hand_detector")?,
            lm: Model::load(dir, "hand_landmarks_detector")?,
            anchors: ssd_anchors(192, &[8, 16, 16, 16]),
            tracks: Vec::new(),
            since_detect: REDETECT_EVERY,
        })
    }

    pub fn process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<Vec<[f32; 2]>>, String> {
        if self.tracks.len() < 2 && (self.tracks.is_empty() || self.since_detect >= REDETECT_EVERY) {
            self.since_detect = 0;
            for roi in self.detect(rgb, w, h)? {
                if self.tracks.len() < 2 && !self.tracks.iter().any(|t| t.iou(&roi) > 0.5) {
                    self.tracks.push(roi);
                }
            }
        } else {
            self.since_detect += 1;
        }
        let mut hands = Vec::new();
        let mut next: Vec<Roi> = Vec::new();
        for roi in std::mem::take(&mut self.tracks) {
            let out = self.lm.run(&[1, 224, 224, 3], crop(rgb, w, h, &roi, 224, Norm::ZeroToOne), &["Identity", "Identity_1"])?;
            if out[1][0] < 0.5 {
                continue; // lost this hand
            }
            let pts: Vec<[f32; 2]> = out[0].chunks(3).map(|p| roi.to_frame(p[0] / 224.0, p[1] / 224.0)).collect();
            let r = hand_track_roi(&pts);
            if next.iter().any(|t| t.iou(&r) > 0.5) {
                continue; // two tracks converged on one hand
            }
            next.push(r);
            hands.push(pts);
        }
        self.tracks = next;
        Ok(hands)
    }

    fn detect(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<Roi>, String> {
        let whole = Roi::whole(w as f32, h as f32);
        let out = self.det.run(&[1, 192, 192, 3], crop(rgb, w, h, &whole, 192, Norm::ZeroToOne), &["Identity", "Identity_1"])?;
        let dets = weighted_nms(decode(&out[0], &out[1], &self.anchors, 18, 7, 192.0, 0.5), 0.3);
        Ok(dets
            .iter()
            .take(2)
            .map(|d| {
                let [cx, cy] = whole.to_frame(d.cx, d.cy);
                let kp = |k: usize| whole.to_frame(d.kps[k][0], d.kps[k][1]);
                // wrist (0) → middle-finger MCP (2); scale 2.6, shift −0.5. The target
                // is 90 *radians*: hand_detector_graph.cc sets
                // rotation_vector_target_angle(90) (the radians field, not _degrees),
                // and DetectionsToRects uses it as-is — matching MediaPipe means
                // matching that.
                Roi::from_box(cx, cy, d.w * whole.size, d.h * whole.size, kp(0), kp(2), 90.0, 2.6, -0.5)
            })
            .collect())
    }
}

#[cfg(test)]
mod golden {
    use super::*;
    use crate::vision::golden_util::*;

    #[test]
    #[ignore = "needs scripts/convert-vision-models.sh --goldens"]
    fn hands_match_mediapipe() {
        let (rgb, w, h) = image("thumb_up.jpg");
        let want = goldens()["thumb_up.jpg"]["hands"].as_array().unwrap().clone();
        assert!(!want.is_empty());
        let mut hd = Hands::load(&dir()).unwrap();
        let got = hd.process(&rgb, w, h).unwrap();
        assert_eq!(got.len(), want.len(), "hand count");
        for e in &want {
            // match each expected hand to our nearest by wrist
            let ew = [e[0][0].as_f64().unwrap() as f32 * w as f32, e[0][1].as_f64().unwrap() as f32 * h as f32];
            let g = got.iter().min_by(|a, b| (a[0][0] - ew[0]).hypot(a[0][1] - ew[1]).total_cmp(&(b[0][0] - ew[0]).hypot(b[0][1] - ew[1]))).unwrap();
            let err = mean_err(g, e, w, h);
            eprintln!("hands golden: landmark err {err:.4} of width");
            assert!(err < 0.01, "mean landmark error {err} of width");
        }
    }
}

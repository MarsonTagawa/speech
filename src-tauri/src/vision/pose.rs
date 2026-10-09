//! Pose pipeline: MediaPipe's pose detector (only when there's no track) →
//! alignment-point ROI → 33 landmarks; the model's auxiliary points 33/34 give
//! the next frame's ROI. Constants from pose_detector_graph.cc and
//! pose_landmarks_detector_graph.cc.

use super::geometry::{crop, decode, ssd_anchors, weighted_nms, Anchor, Norm, Roi};
use super::model::Model;
use std::f32::consts::PI;
use std::path::Path;

pub struct Pose {
    det: Model,
    lm: Model,
    anchors: Vec<Anchor>,
    roi: Option<Roi>,
}

/// AlignmentPointsRects: centred on `a`, sized 2·|a→b|, rotated so a→b points
/// up (90°); then scale 1.25, square_long.
fn alignment_roi(a: [f32; 2], b: [f32; 2]) -> Roi {
    let size = 2.0 * (b[0] - a[0]).hypot(b[1] - a[1]);
    Roi::from_box(a[0], a[1], size, size, a, b, PI / 2.0, 1.25, 0.0)
}

impl Pose {
    pub fn load(dir: &Path) -> Result<Pose, String> {
        Ok(Pose {
            det: Model::load(dir, "pose_detector")?,
            lm: Model::load(dir, "pose_landmarks_detector")?,
            anchors: ssd_anchors(224, &[8, 16, 32, 32, 32]),
            roi: None,
        })
    }

    pub fn process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Option<Vec<[f32; 2]>>, String> {
        if self.roi.is_none() {
            self.roi = self.detect(rgb, w, h)?;
        }
        let Some(roi) = self.roi else { return Ok(None) };
        let out = self.lm.run(&[1, 256, 256, 3], crop(rgb, w, h, &roi, 256, Norm::ZeroToOne), &["Identity", "Identity_1"])?;
        if out[1][0] < 0.5 {
            self.roi = None; // presence is already a probability (no activation in MediaPipe)
            return Ok(None);
        }
        // 39 landmarks × (x, y, z, visibility, presence) in input pixels
        let pts: Vec<[f32; 2]> = out[0].chunks(5).map(|p| roi.to_frame(p[0] / 256.0, p[1] / 256.0)).collect();
        self.roi = Some(alignment_roi(pts[33], pts[34]));
        Ok(Some(pts[..33].to_vec()))
    }

    fn detect(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Option<Roi>, String> {
        let whole = Roi::whole(w as f32, h as f32);
        let out = self.det.run(&[1, 224, 224, 3], crop(rgb, w, h, &whole, 224, Norm::MinusOneToOne), &["Identity", "Identity_1"])?;
        let dets = weighted_nms(decode(&out[0], &out[1], &self.anchors, 12, 4, 224.0, 0.5), 0.3);
        Ok(dets.first().map(|d| {
            let kp = |k: usize| whole.to_frame(d.kps[k][0], d.kps[k][1]);
            alignment_roi(kp(0), kp(1))
        }))
    }
}

#[cfg(test)]
mod golden {
    use super::*;
    use crate::vision::golden_util::*;

    #[test]
    #[ignore = "needs scripts/convert-vision-models.sh --goldens"]
    fn pose_matches_mediapipe() {
        let (rgb, w, h) = image("pose.jpg");
        let mut p = Pose::load(&dir()).unwrap();
        let pts = p.process(&rgb, w, h).unwrap().expect("a pose");
        assert_eq!(pts.len(), 33);
        let err = mean_err(&pts, &goldens()["pose.jpg"]["pose"], w, h);
        eprintln!("pose golden: landmark err {err:.4} of width");
        assert!(err < 0.01, "mean landmark error {err} of width");
    }
}

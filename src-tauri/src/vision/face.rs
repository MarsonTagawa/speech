//! Face pipeline: MediaPipe's face detector (only when there's no track) →
//! rotated ROI → 478 landmarks → 52 blendshapes, plus head yaw/pitch from the
//! landmarks. Constants from face_detector_graph.cc,
//! face_landmarks_detector_graph.cc and face_blendshapes_graph.cc.

/// The 146 landmarks the blendshape model reads (kLandmarksSubsetIdxs).
pub const BLENDSHAPE_SUBSET: [usize; 146] = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40,
    46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82,
    84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145,
    146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168,
    172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269,
    270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311,
    312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373,
    374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397,
    398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474, 475,
    476, 477,
];

/// Blendshape names in the model's output order (kBlendshapeNames).
pub const BLENDSHAPE_NAMES: [&str; 52] = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
];

use super::geometry::{bounds, crop, decode, sigmoid, ssd_anchors, weighted_nms, Anchor, Norm, Roi};
use super::model::Model;
use std::f32::consts::PI;
use std::path::Path;

pub struct FaceResult {
    pub pts: Vec<[f32; 3]>,
    pub blend: Vec<f32>,
    pub yaw: f32,
    pub pitch: f32,
}

pub struct Face {
    det: Model,
    lm: Model,
    blend: Model,
    anchors: Vec<Anchor>,
    pub(crate) roi: Option<Roi>,
}

impl Face {
    pub fn load(dir: &Path) -> Result<Face, String> {
        Ok(Face {
            det: Model::load(dir, "face_detector")?,
            lm: Model::load(dir, "face_landmarks_detector")?,
            blend: Model::load(dir, "face_blendshapes")?,
            anchors: ssd_anchors(128, &[8, 16, 16, 16]),
            roi: None,
        })
    }

    pub fn process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Option<FaceResult>, String> {
        if self.roi.is_none() {
            self.roi = self.detect(rgb, w, h)?;
        }
        let Some(roi) = self.roi else { return Ok(None) };
        let out = self.lm.run(&[1, 256, 256, 3], crop(rgb, w, h, &roi, 256, Norm::ZeroToOne), &["Identity", "Identity_1"])?;
        if sigmoid(out[1][0]) < 0.5 {
            self.roi = None; // lost: detect again next frame
            return Ok(None);
        }
        let pts: Vec<[f32; 3]> = out[0]
            .chunks(3)
            .map(|p| {
                let [x, y] = roi.to_frame(p[0] / 256.0, p[1] / 256.0);
                [x, y, p[2] / 256.0 * roi.size]
            })
            .collect();
        let xy: Vec<[f32; 2]> = pts.iter().map(|p| [p[0], p[1]]).collect();
        let (cx, cy, bw, bh) = bounds(&xy);
        // Next frame's ROI from these landmarks (eye corners 33 → 263, target 0°, scale 1.5).
        self.roi = Some(Roi::from_box(cx, cy, bw, bh, xy[33], xy[263], 0.0, 1.5, 0.0));
        // The blendshape model takes the subset in image pixel coordinates.
        let sub: Vec<f32> = BLENDSHAPE_SUBSET.iter().flat_map(|&i| [pts[i][0], pts[i][1]]).collect();
        let blend = self.blend.run(&[1, 146, 2], sub, &["StatefulPartitionedCall:0"])?.remove(0);
        let (yaw, pitch) = head_angles(&pts);
        Ok(Some(FaceResult { pts, blend, yaw, pitch }))
    }

    fn detect(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Option<Roi>, String> {
        let whole = Roi::whole(w as f32, h as f32);
        let out = self.det.run(&[1, 128, 128, 3], crop(rgb, w, h, &whole, 128, Norm::MinusOneToOne), &["regressors", "classificators"])?;
        let dets = weighted_nms(decode(&out[0], &out[1], &self.anchors, 16, 6, 128.0, 0.5), 0.3);
        Ok(dets.first().map(|d| {
            let [cx, cy] = whole.to_frame(d.cx, d.cy);
            let eye = |k: usize| whole.to_frame(d.kps[k][0], d.kps[k][1]);
            Roi::from_box(cx, cy, d.w * whole.size, d.h * whole.size, eye(0), eye(1), 0.0, 1.5, 0.0)
        }))
    }
}

/// Head yaw/pitch in degrees from 3D face landmarks: the normal of the plane
/// spanned by the outer eye corners (33 → 263) and forehead → chin (10 → 152).
/// Replaces MediaPipe's facial transformation matrix (a 3D Procrustes fit).
pub fn head_angles(p: &[[f32; 3]]) -> (f32, f32) {
    let d = |a: [f32; 3], b: [f32; 3]| [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let (a, b) = (d(p[33], p[263]), d(p[10], p[152]));
    let n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let deg = 180.0 / PI;
    (n[0].atan2(n[2]) * deg, n[1].atan2(n[2]) * deg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(rot_y_deg: f32) -> Vec<[f32; 3]> {
        let (s, c) = rot_y_deg.to_radians().sin_cos();
        let mut p = vec![[0.0f32; 3]; 478];
        // outer eye corners 33 (image left) → 263 (image right), forehead 10 → chin 152
        for (i, [x, y, z]) in [(33, [-1.0f32, 0.0, 0.0]), (263, [1.0, 0.0, 0.0]), (10, [0.0, -1.0, 0.0]), (152, [0.0, 1.0, 0.0])] {
            p[i] = [x * c + z * s, y, -x * s + z * c];
        }
        p
    }

    #[test]
    fn head_angles_frontal_and_turned() {
        let (yaw, pitch) = head_angles(&face(0.0));
        assert!(yaw.abs() < 0.01 && pitch.abs() < 0.01, "{yaw} {pitch}");
        let (yaw, pitch) = head_angles(&face(30.0));
        assert!((yaw.abs() - 30.0).abs() < 0.01, "{yaw}");
        assert!(pitch.abs() < 0.01);
    }
}

#[cfg(test)]
mod golden {
    use super::*;
    use crate::vision::golden_util::*;

    #[test]
    #[ignore = "needs scripts/convert-vision-models.sh --goldens"]
    fn face_matches_mediapipe() {
        let (rgb, w, h) = image("portrait.jpg");
        let want = &goldens()["portrait.jpg"]["face"];
        let mut f = Face::load(&dir()).unwrap();
        let r = f.process(&rgb, w, h).unwrap().expect("a face");
        let xy: Vec<[f32; 2]> = r.pts.iter().map(|p| [p[0], p[1]]).collect();
        let err = mean_err(&xy, &want["landmarks"], w, h);
        assert!(err < 0.01, "mean landmark error {err} of width");
        let diffs: Vec<f32> = BLENDSHAPE_NAMES.iter().zip(&r.blend).map(|(n, v)| (v - want["blend"][*n].as_f64().unwrap() as f32).abs()).collect();
        let mean = diffs.iter().sum::<f32>() / 52.0;
        assert!(mean < 0.05, "mean blendshape diff {mean}");
        let (wy, wp) = (want["yaw"].as_f64().unwrap() as f32, want["pitch"].as_f64().unwrap() as f32);
        eprintln!("face golden: landmark err {err:.4} of width, blendshape mean diff {mean:.3}, yaw {:.1} vs {wy:.1}, pitch {:.1} vs {wp:.1}", r.yaw, r.pitch);
        assert!((r.yaw.abs() - wy.abs()).abs() < 5.0, "yaw {} vs {wy}", r.yaw);
        assert!((r.pitch.abs() - wp.abs()).abs() < 5.0, "pitch {} vs {wp}", r.pitch);
    }

    #[test]
    #[ignore = "needs scripts/convert-vision-models.sh"]
    fn blank_frame_has_no_face_and_no_track() {
        let mut f = Face::load(&dir()).unwrap();
        let black = vec![0u8; 640 * 360 * 3];
        assert!(f.process(&black, 640, 360).unwrap().is_none());
        assert!(f.roi.is_none());
    }
}

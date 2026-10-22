//! Hands pipeline: MediaPipe's palm detector → rotated ROI per palm → 21
//! landmarks; tracks follow the landmarks (looking where each hand is heading),
//! and the detector re-runs the moment a track is lost, or every few frames
//! while fewer than 2 hands are tracked. Constants from hand_detector_graph.cc,
//! hand_landmarks_detector_graph.cc and hand_landmarks_to_rect_calculator.cc.

use super::geometry::{crop, decode, hand_track_roi, ssd_anchors, weighted_nms, Anchor, Norm, Roi};
use super::model::Model;
use std::path::Path;

/// Re-run the palm detector this often (frames, ~0.25 s at 25 Hz) while < 2
/// hands are tracked, so a second hand coming into view is picked up quickly
/// (~8 ms a run).
const REDETECT_EVERY: u32 = 6;
/// Two hands are the same one when both their wrists and middle knuckles are
/// within this many hand sizes. From landmarks (tight), or from the palm
/// detector's rougher keypoints (looser). Not box overlap: the ROIs are 2–2.6×
/// the hand, so two hands close together overlapped and the second was dropped.
const SAME_LM: f32 = 0.3;
const SAME_DET: f32 = 0.6;

/// Wrist and middle knuckle, px: where a hand is, for telling hands apart.
type Anchor2 = [[f32; 2]; 2];

fn same_hand(a: &Anchor2, b: &Anchor2, tol: f32) -> bool {
    let d = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).hypot(p[1] - q[1]);
    let size = d(a[0], a[1]).max(1.0);
    d(a[0], b[0]) < tol * size && d(a[1], b[1]) < tol * size
}

/// Landmark bounding box, px: x0, y0, x1, y1.
type Bbox = [f32; 4];

fn bbox(xy: &[[f32; 2]]) -> Bbox {
    xy.iter().fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])
}

/// One hand read twice can come out shifted, or with its wrist and knuckles in
/// other places (so `same_hand` misses it, and it's tracked as the other hand
/// too); either way it covers the same patch. This much of the smaller box
/// inside the other: two real hands touching (fingertips, a chop onto a palm)
/// overlap far less…
const SAME_BOX: f32 = 0.6;
/// …except interlocked fingers, which cover one patch too, but from wrists on
/// either side of it: same patch counts only with the wrists this close (hand sizes).
const SAME_WRIST: f32 = 1.0;

fn same_box(a: &Bbox, b: &Bbox, wa: &Anchor2, wb: &Anchor2) -> bool {
    let area = |b: &Bbox| (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let inter = area(&[a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])]);
    let d = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).hypot(p[1] - q[1]);
    let size = d(wa[0], wa[1]).max(d(wb[0], wb[1])).max(1.0);
    inter >= SAME_BOX * area(a).min(area(b)).max(1.0) && d(wa[0], wb[0]) < SAME_WRIST * size
}

/// One tracked hand: 21 landmarks in frame pixels, z (depth from the wrist,
/// negative = nearer the camera) on the same scale as x; the same 21 as the
/// model's metric 3D "world" landmarks (metres, frame axes), whose depth is
/// real where the image z is squashed ~7× — so hand orientation comes from
/// these; and the model's handedness score (MediaPipe's "Left" label, which
/// assumes a mirrored image).
pub struct HandOut {
    pub pts: Vec<[f32; 3]>,
    pub world: Vec<[f32; 3]>,
    pub handed: f32,
}

/// A followed hand: the ROI its last landmarks gave, and how far that moved
/// since the frame before (px per processed frame).
struct Track {
    roi: Roi,
    vel: [f32; 2],
    at: Anchor2,
    bbox: Bbox,
}

impl Track {
    /// The hand with these landmarks is this track's hand.
    fn has(&self, at: &Anchor2, bbox: &Bbox) -> bool {
        same_hand(&self.at, at, SAME_LM) || same_box(&self.bbox, bbox, &self.at, at)
    }
}

pub struct Hands {
    det: Model,
    lm: Model,
    anchors: Vec<Anchor>,
    tracks: Vec<Track>,
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

    pub fn process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<HandOut>, String> {
        let mut hands = Vec::new();
        let mut next: Vec<Track> = Vec::new();
        let mut lost = false;
        for t in std::mem::take(&mut self.tracks) {
            // Look where the hand is heading: at ~12 fps a fast hand has left
            // last frame's ROI. (Constant velocity, capped at half the ROI.)
            let k = (0.5 * t.roi.size / t.vel[0].hypot(t.vel[1]).max(1e-3)).min(1.0);
            let ahead = Roi { cx: t.roi.cx + k * t.vel[0], cy: t.roi.cy + k * t.vel[1], ..t.roi };
            match self.landmarks(rgb, w, h, &ahead)? {
                // lost it, or it slid onto the hand another track has: look again
                None => lost = true,
                Some((_, _, at, bbox)) if next.iter().any(|n| n.has(&at, &bbox)) => lost = true,
                Some((hand, r, at, bbox)) => {
                    next.push(Track { vel: [r.cx - t.roi.cx, r.cy - t.roi.cy], roi: r, at, bbox });
                    hands.push(hand);
                }
            }
        }
        // Look for hands now if one was just lost (it moved too fast to follow)
        // or none are tracked, so it's picked up again this frame, not a second later.
        self.since_detect += 1;
        if next.len() < 2 && (lost || next.is_empty() || self.since_detect >= REDETECT_EVERY) {
            self.since_detect = 0;
            for (roi, seen) in self.detect(rgb, w, h)? {
                if next.len() >= 2 || next.iter().any(|n| same_hand(&n.at, &seen, SAME_DET)) {
                    continue; // already tracked
                }
                if let Some((hand, r, at, bbox)) = self.landmarks(rgb, w, h, &roi)? {
                    if !next.iter().any(|n| n.has(&at, &bbox)) {
                        next.push(Track { roi: r, vel: [0.0, 0.0], at, bbox });
                        hands.push(hand);
                    }
                }
            }
        }
        self.tracks = next;
        Ok(hands)
    }

    /// Landmarks in one ROI, the ROI they give for the next frame, and where
    /// the hand is and what it covers; None if there's no hand there.
    fn landmarks(&mut self, rgb: &[u8], w: usize, h: usize, roi: &Roi) -> Result<Option<(HandOut, Roi, Anchor2, Bbox)>, String> {
        let out = self.lm.run(&[1, 224, 224, 3], crop(rgb, w, h, roi, 224, Norm::ZeroToOne), &["Identity", "Identity_1", "Identity_2", "Identity_3"])?;
        if out[1][0] < 0.5 {
            return Ok(None);
        }
        let xy: Vec<[f32; 2]> = out[0].chunks(3).map(|p| roi.to_frame(p[0] / 224.0, p[1] / 224.0)).collect();
        // the crop is rotated in-plane only, so z just scales with it
        let pts = xy.iter().zip(out[0].chunks(3)).map(|(&[x, y], p)| [x, y, p[2] / 224.0 * roi.size]).collect();
        // world landmarks are in the crop's axes; turn them by its rotation
        // into the frame's (as MediaPipe's WorldLandmarkProjectionCalculator)
        let (s, c) = roi.rot.sin_cos();
        let world = out[3].chunks(3).map(|p| [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]]).collect();
        Ok(Some((HandOut { pts, world, handed: out[2][0] }, hand_track_roi(&xy), [xy[0], xy[9]], bbox(&xy))))
    }

    /// Palms in the whole frame: each one's ROI, and its wrist and middle knuckle.
    fn detect(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<(Roi, Anchor2)>, String> {
        let whole = Roi::whole(w as f32, h as f32);
        let out = self.det.run(&[1, 192, 192, 3], crop(rgb, w, h, &whole, 192, Norm::ZeroToOne), &["Identity", "Identity_1"])?;
        let dets = weighted_nms(decode(&out[0], &out[1], &self.anchors, 18, 7, 192.0, 0.5), 0.3);
        Ok(dets
            .iter()
            .take(3) // a spare, in case one is the hand already tracked
            .map(|d| {
                let [cx, cy] = whole.to_frame(d.cx, d.cy);
                let kp = |k: usize| whole.to_frame(d.kps[k][0], d.kps[k][1]);
                // wrist (0) → middle-finger MCP (2); scale 2.6, shift −0.5. The target
                // is 90 *radians*: hand_detector_graph.cc sets
                // rotation_vector_target_angle(90) (the radians field, not _degrees),
                // and DetectionsToRects uses it as-is — matching MediaPipe means
                // matching that.
                (Roi::from_box(cx, cy, d.w * whole.size, d.h * whole.size, kp(0), kp(2), 90.0, 2.6, -0.5), [kp(0), kp(2)])
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hands_side_by_side_are_two_hands() {
        // wrist → knuckle 60 px; hands 90 px apart (close: their ROIs overlap)
        let a: Anchor2 = [[100.0, 200.0], [100.0, 140.0]];
        let b: Anchor2 = [[190.0, 200.0], [190.0, 140.0]];
        assert!(!same_hand(&a, &b, SAME_LM) && !same_hand(&a, &b, SAME_DET));
        // the same hand, jittered by a few px
        let a2: Anchor2 = [[104.0, 197.0], [103.0, 143.0]];
        assert!(same_hand(&a, &a2, SAME_LM));
        // the detector's rough keypoints on that hand still match it
        let rough: Anchor2 = [[112.0, 215.0], [95.0, 160.0]];
        assert!(same_hand(&a, &rough, SAME_DET) && !same_hand(&a, &rough, SAME_LM));
    }

    #[test]
    fn one_hand_read_twice_is_one_hand() {
        let hand = [100.0, 100.0, 160.0, 200.0];
        let at: Anchor2 = [[130.0, 200.0], [130.0, 140.0]];
        // shifted by a quarter of its width, wrist misread 40 px over: the same patch
        assert!(same_box(&hand, &[115.0, 110.0, 175.0, 210.0], &at, &[[160.0, 175.0], [140.0, 150.0]]));
        // a second hand beside it, close enough to touch
        assert!(!same_box(&hand, &[155.0, 100.0, 215.0, 200.0], &at, &[[185.0, 200.0], [185.0, 140.0]]));
        // fingertips touching in a steeple: corners overlap
        assert!(!same_box(&hand, &[140.0, 60.0, 200.0, 130.0], &at, &[[170.0, 130.0], [170.0, 80.0]]));
        // fingers interlocked: one patch, wrists on either side of it
        let (l, r) = ([100.0, 100.0, 210.0, 170.0], [110.0, 95.0, 220.0, 165.0]);
        let (wl, wr): (Anchor2, Anchor2) = ([[100.0, 150.0], [150.0, 130.0]], [[220.0, 145.0], [170.0, 125.0]]);
        assert!(!same_box(&l, &r, &wl, &wr));
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
            let xy = |hd: &HandOut| hd.pts.iter().map(|p| [p[0], p[1]]).collect::<Vec<_>>();
            let g = got.iter().map(xy).min_by(|a, b| (a[0][0] - ew[0]).hypot(a[0][1] - ew[1]).total_cmp(&(b[0][0] - ew[0]).hypot(b[0][1] - ew[1]))).unwrap();
            let err = mean_err(&g, e, w, h);
            eprintln!("hands golden: landmark err {err:.4} of width");
            assert!(err < 0.01, "mean landmark error {err} of width");
        }
    }
}

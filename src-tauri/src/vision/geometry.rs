//! Pure geometry shared by the face, pose and hand pipelines: MediaPipe's SSD
//! anchors, detection decoding, weighted NMS, and the square rotated ROI that
//! maps between the camera frame and a model's input. Constants and formulas
//! follow MediaPipe's calculators (ssd_anchors, tensors_to_detections,
//! non_max_suppression, detections_to_rects, rect_transformation,
//! hand_landmarks_to_rect).
use std::f32::consts::PI;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Wraps an angle into [-π, π).
pub fn normalize_radians(a: f32) -> f32 {
    a - 2.0 * PI * ((a + PI) / (2.0 * PI)).floor()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub x: f32,
    pub y: f32,
}

/// SsdAnchorsCalculator as MediaPipe's detectors configure it: fixed-size
/// anchors, aspect ratio 1.0 plus the interpolated one (2 per cell per layer);
/// consecutive layers with the same stride share one grid.
pub fn ssd_anchors(input: usize, strides: &[usize]) -> Vec<Anchor> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < strides.len() {
        let mut j = i;
        while j < strides.len() && strides[j] == strides[i] {
            j += 1;
        }
        let grid = input.div_ceil(strides[i]);
        for y in 0..grid {
            for x in 0..grid {
                for _ in 0..2 * (j - i) {
                    out.push(Anchor { x: (x as f32 + 0.5) / grid as f32, y: (y as f32 + 0.5) / grid as f32 });
                }
            }
        }
        i = j;
    }
    out
}

/// A detection, normalised to the model input's square (0..1).
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub score: f32,
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub kps: Vec<[f32; 2]>,
}

/// TensorsToDetectionsCalculator with reverse_output_order (x, y, w, h first),
/// fixed anchors (w = h = 1) and sigmoid scores clipped to ±100.
pub fn decode(boxes: &[f32], scores: &[f32], anchors: &[Anchor], num_coords: usize, num_kps: usize, scale: f32, min_score: f32) -> Vec<Detection> {
    anchors
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            let score = sigmoid(scores[i].clamp(-100.0, 100.0));
            if score < min_score {
                return None;
            }
            let r = &boxes[i * num_coords..(i + 1) * num_coords];
            let kps = (0..num_kps).map(|k| [r[4 + 2 * k] / scale + a.x, r[5 + 2 * k] / scale + a.y]).collect();
            Some(Detection { score, cx: r[0] / scale + a.x, cy: r[1] / scale + a.y, w: r[2] / scale, h: r[3] / scale, kps })
        })
        .collect()
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix = ((a.cx + a.w / 2.0).min(b.cx + b.w / 2.0) - (a.cx - a.w / 2.0).max(b.cx - b.w / 2.0)).max(0.0);
    let iy = ((a.cy + a.h / 2.0).min(b.cy + b.h / 2.0) - (a.cy - a.h / 2.0).max(b.cy - b.h / 2.0)).max(0.0);
    let union = a.w * a.h + b.w * b.h - ix * iy;
    if union <= 0.0 { 0.0 } else { ix * iy / union }
}

/// NonMaxSuppressionCalculator, WEIGHTED: each cluster (IoU > `thresh` with
/// its best member) becomes one detection whose box corners and keypoints are
/// the score-weighted mean, keeping the best score.
pub fn weighted_nms(mut dets: Vec<Detection>, thresh: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out = Vec::new();
    while !dets.is_empty() {
        let top = dets[0].clone();
        let (cluster, rest): (Vec<_>, Vec<_>) = dets.into_iter().enumerate().partition(|(i, d)| *i == 0 || iou(&top, d) > thresh);
        dets = rest.into_iter().map(|(_, d)| d).collect();
        let cluster: Vec<Detection> = cluster.into_iter().map(|(_, d)| d).collect();
        let total: f32 = cluster.iter().map(|d| d.score).sum();
        let avg = |f: &dyn Fn(&Detection) -> f32| cluster.iter().map(|d| f(d) * d.score).sum::<f32>() / total;
        let (x0, y0) = (avg(&|d| d.cx - d.w / 2.0), avg(&|d| d.cy - d.h / 2.0));
        let (x1, y1) = (avg(&|d| d.cx + d.w / 2.0), avg(&|d| d.cy + d.h / 2.0));
        let kps = (0..top.kps.len()).map(|k| [avg(&|d| d.kps[k][0]), avg(&|d| d.kps[k][1])]).collect();
        out.push(Detection { score: top.score, cx: (x0 + x1) / 2.0, cy: (y0 + y1) / 2.0, w: x1 - x0, h: y1 - y0, kps });
    }
    out
}

/// A square, rotated region of the camera frame in pixels (MediaPipe's
/// NormalizedRect after square_long) that a model looks at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roi {
    pub cx: f32,
    pub cy: f32,
    pub size: f32,
    pub rot: f32,
}

impl Roi {
    /// The whole frame letterboxed into a square — how the detectors see it
    /// (ImageToTensor keep_aspect_ratio).
    pub fn whole(w: f32, h: f32) -> Roi {
        Roi { cx: w / 2.0, cy: h / 2.0, size: w.max(h), rot: 0.0 }
    }

    /// A point in the model input (normalised 0..1) → frame pixels.
    pub fn to_frame(&self, u: f32, v: f32) -> [f32; 2] {
        let (dx, dy) = ((u - 0.5) * self.size, (v - 0.5) * self.size);
        let (s, c) = self.rot.sin_cos();
        [self.cx + dx * c - dy * s, self.cy + dx * s + dy * c]
    }

    /// DetectionsToRects (rotation from the `from`→`to` vector, in pixels) then
    /// RectTransformation (shift along the rotated y axis, square_long, scale).
    pub fn from_box(cx: f32, cy: f32, w: f32, h: f32, from: [f32; 2], to: [f32; 2], target: f32, scale: f32, shift_y: f32) -> Roi {
        let rot = normalize_radians(target - (-(to[1] - from[1])).atan2(to[0] - from[0]));
        Roi::from_rotated(cx, cy, w, h, rot, scale, shift_y)
    }

    pub fn from_rotated(cx: f32, cy: f32, w: f32, h: f32, rot: f32, scale: f32, shift_y: f32) -> Roi {
        let (s, c) = rot.sin_cos();
        Roi { cx: cx - h * shift_y * s, cy: cy + h * shift_y * c, size: w.max(h) * scale, rot }
    }

    /// Overlap of the two squares, ignoring rotation (for deduplicating tracks).
    pub fn iou(&self, o: &Roi) -> f32 {
        let d = |r: &Roi| Detection { score: 0.0, cx: r.cx, cy: r.cy, w: r.size, h: r.size, kps: vec![] };
        iou(&d(self), &d(o))
    }
}

#[derive(Clone, Copy)]
pub enum Norm {
    MinusOneToOne,
    ZeroToOne,
}

/// ImageToTensor: samples `roi` from an RGB frame into a size×size NHWC f32
/// tensor (bilinear; outside the frame is black, MediaPipe's BORDER_ZERO).
pub fn crop(rgb: &[u8], w: usize, h: usize, roi: &Roi, size: usize, norm: Norm) -> Vec<f32> {
    let (lo, span) = match norm {
        Norm::MinusOneToOne => (-1.0, 2.0),
        Norm::ZeroToOne => (0.0, 1.0),
    };
    let mut out = vec![0f32; size * size * 3];
    for j in 0..size {
        for i in 0..size {
            let [x, y] = roi.to_frame((i as f32 + 0.5) / size as f32, (j as f32 + 0.5) / size as f32);
            let px = sample(rgb, w, h, x - 0.5, y - 0.5);
            let o = (j * size + i) * 3;
            for k in 0..3 {
                out[o + k] = lo + span * px[k] / 255.0;
            }
        }
    }
    out
}

fn sample(rgb: &[u8], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let mut acc = [0f32; 3];
    for (dx, dy, wt) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
        let (xi, yi) = (x0 as i64 + dx, y0 as i64 + dy);
        if xi < 0 || yi < 0 || xi >= w as i64 || yi >= h as i64 {
            continue;
        }
        let o = (yi as usize * w + xi as usize) * 3;
        for k in 0..3 {
            acc[k] += wt * rgb[o + k] as f32;
        }
    }
    acc
}

/// Axis-aligned bounds of points → (cx, cy, w, h).
pub fn bounds(p: &[[f32; 2]]) -> (f32, f32, f32, f32) {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &[x, y] in p {
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
    }
    ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)
}

/// HandLandmarksToRectCalculator (legacy palm subset): rotation from the wrist
/// toward the middle of the palm, the rotated bounding box of the subset, then
/// scale 2.0, shift −0.1, square_long. `lm` is 21 points in frame pixels.
pub fn hand_track_roi(lm: &[[f32; 2]]) -> Roi {
    const SUB: [usize; 12] = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
    let p: Vec<[f32; 2]> = SUB.iter().map(|&i| lm[i]).collect();
    let [x0, y0] = lm[0];
    let x1 = ((lm[5][0] + lm[13][0]) / 2.0 + lm[9][0]) / 2.0;
    let y1 = ((lm[5][1] + lm[13][1]) / 2.0 + lm[9][1]) / 2.0;
    let rot = normalize_radians(PI / 2.0 - (-(y1 - y0)).atan2(x1 - x0));
    let (acx, acy, _, _) = bounds(&p);
    let (s, c) = (-rot).sin_cos();
    let q: Vec<[f32; 2]> = p
        .iter()
        .map(|&[x, y]| {
            let (dx, dy) = (x - acx, y - acy);
            [dx * c - dy * s, dx * s + dy * c]
        })
        .collect();
    let (pcx, pcy, w, h) = bounds(&q);
    let (s, c) = rot.sin_cos();
    Roi::from_rotated(pcx * c - pcy * s + acx, pcx * s + pcy * c + acy, w, h, rot, 2.0, -0.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn near(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-3, "{a} != {b}");
    }

    #[test]
    fn anchor_counts_match_the_models() {
        assert_eq!(ssd_anchors(128, &[8, 16, 16, 16]).len(), 896); // face detector
        assert_eq!(ssd_anchors(192, &[8, 16, 16, 16]).len(), 2016); // palm detector
        assert_eq!(ssd_anchors(224, &[8, 16, 32, 32, 32]).len(), 2254); // pose detector
        let a = ssd_anchors(128, &[8, 16, 16, 16]);
        assert_eq!(a[0], Anchor { x: 0.5 / 16.0, y: 0.5 / 16.0 });
        assert_eq!(a[1], a[0]); // two anchors per cell
        assert_eq!(a[512], Anchor { x: 0.5 / 8.0, y: 0.5 / 8.0 }); // second grid starts after 16·16·2
    }

    #[test]
    fn decode_applies_anchor_scale_and_sigmoid() {
        let anchors = [Anchor { x: 0.5, y: 0.5 }];
        // x, y, w, h, kp0x, kp0y in input pixels (scale 128)
        let boxes = [12.8, -12.8, 25.6, 12.8, 0.0, 6.4];
        let d = decode(&boxes, &[2.0], &anchors, 6, 1, 128.0, 0.5);
        assert_eq!(d.len(), 1);
        near(d[0].cx, 0.6);
        near(d[0].cy, 0.4);
        near(d[0].w, 0.2);
        near(d[0].h, 0.1);
        near(d[0].kps[0][1], 0.55);
        near(d[0].score, sigmoid(2.0));
        assert!(decode(&boxes, &[-3.0], &anchors, 6, 1, 128.0, 0.5).is_empty()); // below threshold
    }

    #[test]
    fn weighted_nms_merges_overlaps_and_keeps_separate_boxes() {
        let det = |score, cx| Detection { score, cx, cy: 0.5, w: 0.2, h: 0.2, kps: vec![[cx, 0.5]] };
        let out = weighted_nms(vec![det(0.6, 0.52), det(0.9, 0.5), det(0.8, 0.1)], 0.3);
        assert_eq!(out.len(), 2);
        near(out[0].score, 0.9);
        near(out[0].cx, (0.5 * 0.9 + 0.52 * 0.6) / 1.5); // score-weighted
        near(out[1].cx, 0.1);
        // degenerate (zero-size) boxes must not loop forever
        let z = Detection { score: 0.9, cx: 0.5, cy: 0.5, w: 0.0, h: 0.0, kps: vec![] };
        assert_eq!(weighted_nms(vec![z.clone(), z], 0.3).len(), 2);
    }

    #[test]
    fn roi_maps_model_points_to_the_frame() {
        let r = Roi { cx: 300.0, cy: 200.0, size: 100.0, rot: 0.0 };
        assert_eq!(r.to_frame(0.5, 0.5), [300.0, 200.0]);
        assert_eq!(r.to_frame(1.0, 0.0), [350.0, 150.0]);
        // rotated 90°: the input's +u axis points down the frame
        let r = Roi { rot: PI / 2.0, ..r };
        let [x, y] = r.to_frame(1.0, 0.5);
        near(x, 300.0);
        near(y, 250.0);
        // whole frame letterboxed into a square
        let w = Roi::whole(640.0, 360.0);
        assert_eq!(w.to_frame(0.0, 0.5), [0.0, 180.0]);
        assert_eq!(w.to_frame(0.5, 0.0), [320.0, -140.0]);
    }

    #[test]
    fn roi_from_box_rotation_shift_and_scale() {
        // vector pointing right, target 0 → no rotation
        near(Roi::from_box(0.0, 0.0, 10.0, 10.0, [0.0, 0.0], [10.0, 0.0], 0.0, 1.0, 0.0).rot, 0.0);
        // vector pointing up the frame, target 90° → no rotation (an upright hand)
        near(Roi::from_box(0.0, 0.0, 10.0, 10.0, [0.0, 0.0], [0.0, -10.0], PI / 2.0, 1.0, 0.0).rot, 0.0);
        // vector pointing right, target 90° → rotated a quarter turn
        near(Roi::from_box(0.0, 0.0, 10.0, 10.0, [0.0, 0.0], [10.0, 0.0], PI / 2.0, 1.0, 0.0).rot, PI / 2.0);
        // shift −0.5 of height moves an upright box up; size = long side × scale
        let r = Roi::from_box(100.0, 100.0, 40.0, 20.0, [0.0, 0.0], [0.0, -1.0], PI / 2.0, 2.0, -0.5);
        near(r.cx, 100.0);
        near(r.cy, 90.0);
        near(r.size, 80.0);
        near(normalize_radians(1.5 * PI), -0.5 * PI);
    }

    #[test]
    fn crop_samples_inside_and_pads_outside() {
        // 4×2 frame, every pixel (255, 0, 0)
        let rgb: Vec<u8> = std::iter::repeat([255u8, 0, 0]).take(8).flatten().collect();
        let t = crop(&rgb, 4, 2, &Roi { cx: 2.0, cy: 1.0, size: 2.0, rot: 0.0 }, 2, Norm::ZeroToOne);
        assert_eq!(t.len(), 2 * 2 * 3);
        near(t[0], 1.0);
        near(t[1], 0.0);
        let t = crop(&rgb, 4, 2, &Roi::whole(4.0, 2.0), 4, Norm::MinusOneToOne);
        near(t[0], -1.0); // top row is letterbox padding (outside the frame)
        near(t[4 * 3 + 0], 1.0); // second row is inside
    }

    #[test]
    fn hand_track_roi_upright_hand_has_no_rotation() {
        // 21 points: wrist at the bottom, fingers straight up, x spread by finger
        let mut lm = [[0.0f32; 2]; 21];
        lm[0] = [100.0, 200.0];
        for f in 0..5 {
            for j in 0..4 {
                lm[1 + f * 4 + j] = [80.0 + 10.0 * f as f32, 180.0 - 20.0 * j as f32];
            }
        }
        let r = hand_track_roi(&lm);
        assert!(r.rot.abs() < 0.2, "rot {}", r.rot);
        assert!(r.size > 100.0);
    }
}

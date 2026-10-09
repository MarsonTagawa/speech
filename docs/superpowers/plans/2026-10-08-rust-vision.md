# Rust Vision Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move camera capture and all face/pose/hand inference out of the webview into Rust, so the UI thread stays at ~60 fps with the camera on.

**Architecture:** A capture thread (`nokhwa`, YUYV 640×360@30) keeps the latest frame. An inference thread (~12.5 Hz) runs MediaPipe's detect-then-track pipelines on ONNX conversions of MediaPipe's own models via `ort`. It streams results (JSON) and a JPEG preview to the webview over two Tauri channels. The webview's `camera.ts` becomes a thin client feeding the unchanged `presence.ts` scoring.

**Tech Stack:** Rust (tauri 2, ort 2.0.0-rc.13, nokhwa 0.10, jpeg-encoder 0.7), TypeScript/Vite, Python 3.11 venv via `uv` (one-time model conversion only).

**Spec:** `docs/superpowers/specs/2026-10-08-rust-vision-design.md`

## Global Constraints

- Commit messages: short, lowercase, repo style. **No `Co-Authored-By` or any Claude attribution** (user preference).
- The UI thread does no inference: per frame it does at most one channel message, one JPEG decode via `createImageBitmap`, and one `drawImage`.
- Models live in git-ignored `src-tauri/resources/vision/` (bundled because `resources/` is bundled), produced by `scripts/convert-vision-models.sh`.
- `presence.ts` scoring (`summarize`, `movingHands`, `liveCue`, `coverMap`, `PRESENCE_TUNING`) is unchanged. Only `RawResults.matrix` → `head` changes.
- MediaPipe constants are as listed in this plan (copied from MediaPipe's source and the models' metadata). Don't "tune" them. The golden tests are the arbiter.
- Rust work runs inside the flake: prefix cargo with `nix develop -c`. Shell commands run from the repo root `/home/marsont/speech` unless a step says otherwise.
- Linux is the only tested platform; `nokhwa`'s other backends must still compile.

## Review Focus

1. **Tilted head or hand (non-zero ROI rotation).** The golden images are near-upright, so a sign error in rotation or back-projection would pass them but put landmarks in the wrong place on a tilted subject. Expected: landmarks stay on the subject. Task 2 pins `to_frame`/`from_box` rotation with explicit cases.
2. **Subject leaves and re-enters the frame.** Expected: tracking drops (presence < 0.5) and the detector runs again, rather than tracking a stale ROI forever. Task 3 adds a blank-frame test that must return `None` and clear the ROI.
3. **Camera busy (another app holds it).** Expected: `start_vision` returns an error quickly, and the UI shows "Camera unavailable" and turns the toggle off. Covered by a Task 7 manual step.
4. **Rapid off/on toggling.** Expected: the threads are joined, the camera is released, and the second start succeeds with no EBUSY. Covered by a Task 7 manual step.
5. **Inference error mid-run.** Expected: the frame is skipped, and 30 consecutive failures end the session with `{ended: true}` rather than a silent freeze. Task 6 implements this; Task 7's manual check confirms that unplugging the camera sends `ended`.

## File map

| File | Responsibility |
|---|---|
| `scripts/convert-vision-models.sh` (new) | One-time: venv, download `.task`, run `convert.py`; `--goldens` also runs `goldens.py` |
| `scripts/vision/convert.py` (new) | Unzip, densify the pose detector, tf2onnx all 7 models, verify against TFLite |
| `scripts/vision/goldens.py` (new) | Official MediaPipe Python on 3 test images → `goldens.json` |
| `src-tauri/src/vision/mod.rs` (new) | `VisionMsg`, `Pipelines`, `start_vision`/`stop_vision`, threads, `VisionState` |
| `src-tauri/src/vision/geometry.rs` (new) | Pure: anchors, decode, weighted NMS, `Roi`, crop, hand tracking ROI, bounds |
| `src-tauri/src/vision/model.rs` (new) | `ort` session wrapper |
| `src-tauri/src/vision/face.rs` (new) | Face detect/track, landmarks, blendshapes, head angles |
| `src-tauri/src/vision/pose.rs` (new) | Pose detect/track, 33 landmarks |
| `src-tauri/src/vision/hands.rs` (new) | Palm detect, ≤2 hand tracks, 21 landmarks each |
| `src-tauri/src/vision/capture.rs` (new) | Camera thread, latest-frame slot |
| `src-tauri/src/lib.rs` | Register vision; remove webview camera workarounds; stop on window close |
| `src-tauri/Cargo.toml` | `nokhwa`, `jpeg-encoder`; dev `image` |
| `src/camera.ts` | Thin client over the channels |
| `src/presence.ts`, `src/presence.check.ts` | `matrix` → `head`; delete `headAngles` |
| `src/main.ts`, `index.html`, `src/styles.css` | `<video>` → `<canvas>` self-view |
| `package.json`, `.gitignore`, `README.md` | Drop MediaPipe JS and `public/mediapipe`; add vision setup |

---

### Task 1: Model conversion script

**Files:**
- Create: `scripts/convert-vision-models.sh`, `scripts/vision/convert.py`, `scripts/vision/goldens.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `src-tauri/resources/vision/{face_detector,face_landmarks_detector,face_blendshapes,pose_detector,pose_landmarks_detector,hand_detector,hand_landmarks_detector}.onnx`. With `--goldens`, also `src-tauri/resources/vision/golden/{portrait.jpg,pose.jpg,thumb_up.jpg,goldens.json}`.
- `goldens.json` shape: `{ "<image>": { "w": int, "h": int, "face"?: { "landmarks": [[x,y]×478], "blend": {name: score}, "yaw": deg, "pitch": deg }, "pose"?: [[x,y]×33], "hands": [[[x,y]×21], …] } }`. Coordinates are normalised to 0..1.

- [ ] **Step 1: Ignore the output dir**

Append to `.gitignore` after the `src-tauri/resources/*.onnx` line:

```
# ONNX vision models + golden test data — produced by scripts/convert-vision-models.sh
src-tauri/resources/vision/
```

- [ ] **Step 2: Write `scripts/convert-vision-models.sh`**

```bash
#!/usr/bin/env bash
# Converts MediaPipe's face, pose and hand models to ONNX for the Rust vision
# pipeline (src-tauri/src/vision). One-time setup; output is git-ignored.
#   scripts/convert-vision-models.sh            # models
#   scripts/convert-vision-models.sh --goldens  # + test images and expected outputs
# nixpkgs' TensorFlow doesn't build on Python 3.14/3.12, so this uses a uv venv
# with upstream wheels (needs nix-ld, which this machine has).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/src-tauri/resources/vision
WORK=$ROOT/src-tauri/target/vision-convert
mkdir -p "$OUT" "$WORK"
UV="nix run nixpkgs#uv --"
PY=$WORK/venv/bin/python
if [ ! -x "$PY" ]; then
  $UV venv --python 3.11 "$WORK/venv"
  $UV pip install --python "$PY" "tensorflow-cpu==2.15.*" "tf2onnx==1.16.1" onnxruntime "numpy<2" "mediapipe==0.10.21"
  # mediapipe pulls a desktop OpenCV that needs libGL; the headless build doesn't.
  $UV pip uninstall --python "$PY" opencv-contrib-python
  $UV pip install --python "$PY" "opencv-python-headless<4.11"
fi
M=https://storage.googleapis.com/mediapipe-models
fetch() { [ -f "$2" ] || curl -fsSL -o "$2" "$1"; }
fetch $M/face_landmarker/face_landmarker/float16/1/face_landmarker.task "$WORK/face_landmarker.task"
fetch $M/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task "$WORK/pose_landmarker_lite.task"
fetch $M/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task "$WORK/hand_landmarker.task"
"$PY" "$ROOT/scripts/vision/convert.py" "$WORK" "$OUT"
if [ "${1:-}" = "--goldens" ]; then
  A=https://storage.googleapis.com/mediapipe-assets
  mkdir -p "$OUT/golden"
  for f in portrait.jpg pose.jpg thumb_up.jpg; do fetch "$A/$f" "$OUT/golden/$f"; done
  "$PY" "$ROOT/scripts/vision/goldens.py" "$WORK" "$OUT/golden"
fi
echo "vision models ready in $OUT"
```

Then `chmod +x scripts/convert-vision-models.sh`.

- [ ] **Step 3: Write `scripts/vision/convert.py`**

```python
# Unzips MediaPipe's .task bundles, densifies the pose detector (its sparse
# DENSIFY weights crash tf2onnx), converts all 7 models to ONNX, and checks each
# against its TFLite original. Usage: convert.py <work dir> <output dir>
import os, sys, zipfile
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.tools import flatbuffer_utils as fu

work, out = sys.argv[1], sys.argv[2]
MODELS = {
    "face_landmarker.task": ["face_detector", "face_landmarks_detector", "face_blendshapes"],
    "pose_landmarker_lite.task": ["pose_detector", "pose_landmarks_detector"],
    "hand_landmarker.task": ["hand_detector", "hand_landmarks_detector"],
}


def densify(src, dst):
    """Replace DENSIFY ops with plain dense constant buffers (same weights)."""
    interp = tf.lite.Interpreter(model_path=src, experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    for d in interp.get_input_details():
        interp.set_tensor(d["index"], np.zeros(d["shape"], dtype=d["dtype"]))
    interp.invoke()
    model = fu.read_model(src)
    g = model.subgraphs[0]
    codes = {i for i, c in enumerate(model.operatorCodes)
             if max(c.builtinCode, c.deprecatedBuiltinCode) == schema.BuiltinOperator.DENSIFY}
    kept = []
    for op in g.operators:
        if op.opcodeIndex not in codes:
            kept.append(op)
            continue
        dense = interp.get_tensor(op.outputs[0])
        buf = type(model.buffers[0])()
        buf.data = np.frombuffer(dense.tobytes(), dtype=np.uint8)
        model.buffers.append(buf)
        t = g.tensors[op.outputs[0]]
        t.buffer, t.sparsity = len(model.buffers) - 1, None
        src_t = g.tensors[op.inputs[0]]  # now unused; converters must not read it
        src_t.buffer, src_t.sparsity = 0, None
    g.operators = kept
    fu.write_model(model, dst)


def to_onnx(tflite_path, onnx_path, stub_details):
    # get_tensor_details() segfaults on the densified model's orphaned tensors;
    # tf2onnx only uses it for shape hints and falls back to the model's shapes.
    orig = tf.lite.Interpreter.get_tensor_details
    if stub_details:
        tf.lite.Interpreter.get_tensor_details = lambda self: []
    try:
        tf2onnx.convert.from_tflite(tflite_path, opset=17, output_path=onnx_path)
    finally:
        tf.lite.Interpreter.get_tensor_details = orig


def verify(tflite_path, onnx_path):
    it = tf.lite.Interpreter(model_path=tflite_path)
    it.allocate_tensors()
    d = it.get_input_details()[0]
    x = np.random.RandomState(0).rand(*d["shape"]).astype(np.float32)
    it.set_tensor(d["index"], x)
    it.invoke()
    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = s.run(None, {s.get_inputs()[0].name: x})
    for o in it.get_output_details():
        ref = it.get_tensor(o["index"]).ravel()
        if ref.size == 256 * 256:  # pose segmentation mask: unused, differs after conversion
            continue
        diff = min(float(np.abs(g.ravel() - ref).max()) for g in got if g.size == ref.size)
        if diff > 0.01:
            sys.exit(f"{os.path.basename(onnx_path)}: output {o['name']} differs by {diff}")


for task, names in MODELS.items():
    zipfile.ZipFile(os.path.join(work, task)).extractall(work)
    for name in names:
        src = os.path.join(work, f"{name}.tflite")
        if name == "pose_detector":
            dense = os.path.join(work, "pose_detector_dense.tflite")
            densify(src, dense)
            src = dense
        dst = os.path.join(out, f"{name}.onnx")
        to_onnx(src, dst, stub_details=(name == "pose_detector"))
        verify(src, dst)
        print(f"ok {name}")
```

- [ ] **Step 4: Write `scripts/vision/goldens.py`**

```python
# Expected outputs from the official MediaPipe Python package on MediaPipe's own
# test images, for the Rust golden tests. Usage: goldens.py <work dir> <golden dir>
import json, math, sys
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision as V

work, gold = sys.argv[1], sys.argv[2]
base = lambda f: mpt.BaseOptions(model_asset_path=f"{work}/{f}")
face = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
    base_options=base("face_landmarker.task"), output_face_blendshapes=True,
    output_facial_transformation_matrixes=True))
pose = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(base_options=base("pose_landmarker_lite.task")))
hand = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(base_options=base("hand_landmarker.task"), num_hands=2))
out = {}
for name in ["portrait.jpg", "pose.jpg", "thumb_up.jpg"]:
    img = mp.Image.create_from_file(f"{gold}/{name}")
    f, p, h = face.detect(img), pose.detect(img), hand.detect(img)
    r = {"w": img.width, "h": img.height}
    if f.face_landmarks:
        R = f.facial_transformation_matrixes[0]  # 4x4, row-major math layout
        r["face"] = {
            "landmarks": [[l.x, l.y] for l in f.face_landmarks[0]],
            "blend": {c.category_name: c.score for c in f.face_blendshapes[0]},
            "yaw": math.degrees(math.asin(max(-1.0, min(1.0, -R[2][0])))),
            "pitch": math.degrees(math.atan2(R[2][1], R[2][2])),
        }
    if p.pose_landmarks:
        r["pose"] = [[l.x, l.y] for l in p.pose_landmarks[0]]
    r["hands"] = [[[l.x, l.y] for l in hl] for hl in h.hand_landmarks]
    out[name] = r
json.dump(out, open(f"{gold}/goldens.json", "w"), indent=1)
for k, v in out.items():
    print(k, "face" in v, "pose" in v, len(v["hands"]), "hands")
```

- [ ] **Step 5: Run it**

Run: `scripts/convert-vision-models.sh --goldens` (the first run downloads ~250 MB; it may take several minutes)
Expected: seven `ok <name>` lines and `vision models ready in …`. Then the goldens summary, which must include `portrait.jpg True …` (face found), `pose.jpg … True …` (pose found) and `thumb_up.jpg … 1 hands` or more. Then:
`ls src-tauri/resources/vision/*.onnx | wc -l` → `7`.

If a summary line lacks the expected detection (e.g. no face in `portrait.jpg`), stop and report. The goldens are unusable without it.

- [ ] **Step 6: Commit**

```bash
git add .gitignore scripts/convert-vision-models.sh scripts/vision/convert.py scripts/vision/goldens.py
git commit -m "vision: script to convert mediapipe models to onnx"
```

---

### Task 2: Geometry primitives

**Files:**
- Create: `src-tauri/src/vision/geometry.rs`, `src-tauri/src/vision/mod.rs` (initially just `pub mod geometry;`)
- Modify: `src-tauri/src/lib.rs` (add `mod vision;`)

**Interfaces (produced, exact):**
- `pub fn sigmoid(x: f32) -> f32`
- `pub fn normalize_radians(a: f32) -> f32`, wrapping into [-π, π)
- `pub struct Anchor { pub x: f32, pub y: f32 }`; `pub fn ssd_anchors(input: usize, strides: &[usize]) -> Vec<Anchor>`
- `pub struct Detection { pub score: f32, pub cx: f32, pub cy: f32, pub w: f32, pub h: f32, pub kps: Vec<[f32; 2]> }`, normalised to the model's square input
- `pub fn decode(boxes: &[f32], scores: &[f32], anchors: &[Anchor], num_coords: usize, num_kps: usize, scale: f32, min_score: f32) -> Vec<Detection>`
- `pub fn weighted_nms(dets: Vec<Detection>, thresh: f32) -> Vec<Detection>`
- `pub struct Roi { pub cx: f32, pub cy: f32, pub size: f32, pub rot: f32 }` (pixels, radians) with:
  - `Roi::whole(w: f32, h: f32) -> Roi`
  - `Roi::to_frame(&self, u: f32, v: f32) -> [f32; 2]`
  - `Roi::from_box(cx, cy, w, h: f32, from: [f32; 2], to: [f32; 2], target: f32, scale: f32, shift_y: f32) -> Roi`
  - `Roi::from_rotated(cx, cy, w, h, rot, scale, shift_y: f32) -> Roi`
  - `Roi::iou(&self, other: &Roi) -> f32`
- `pub enum Norm { MinusOneToOne, ZeroToOne }`; `pub fn crop(rgb: &[u8], w: usize, h: usize, roi: &Roi, size: usize, norm: Norm) -> Vec<f32>` (NHWC)
- `pub fn bounds(p: &[[f32; 2]]) -> (f32, f32, f32, f32)`, giving (cx, cy, w, h)
- `pub fn hand_track_roi(lm: &[[f32; 2]]) -> Roi`

- [ ] **Step 1: Register the module**

`src-tauri/src/vision/mod.rs`:

```rust
//! Camera capture and face/pose/hand inference for presence grading, run in
//! Rust so the webview's UI thread stays free (see
//! docs/superpowers/specs/2026-10-08-rust-vision-design.md).
pub mod geometry;
```

In `src-tauri/src/lib.rs`, add `mod vision;` after `mod vad;`.

- [ ] **Step 2: Write the failing tests**

Create `src-tauri/src/vision/geometry.rs` with only the test module:

```rust
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
```

- [ ] **Step 3: Run to verify it fails**

Run: `nix develop -c cargo test --manifest-path src-tauri/Cargo.toml vision::geometry 2>&1 | tail -5`
Expected: compile errors (`cannot find function ssd_anchors` …).

- [ ] **Step 4: Implement (prepend to `geometry.rs`, above the tests)**

```rust
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
```

- [ ] **Step 5: Run to verify it passes**

Run: `nix develop -c cargo test --manifest-path src-tauri/Cargo.toml vision::geometry 2>&1 | tail -5`
Expected: `test result: ok. 7 passed`. If one fails, fix the implementation, not the test. The test values are hand-derived from MediaPipe's formulas.

- [ ] **Step 6: Commit**

```bash
git add src-tauri/src/vision/ src-tauri/src/lib.rs
git commit -m "vision: anchors, detection decode, nms and roi geometry"
```

---

### Task 3: Model wrapper and face pipeline (with golden test)

**Files:**
- Create: `src-tauri/src/vision/model.rs`, `src-tauri/src/vision/face.rs`
- Modify: `src-tauri/src/vision/mod.rs`, `src-tauri/Cargo.toml` (dev-dependency `image`)

**Interfaces:**
- Consumes: Task 2's geometry.
- Produces:
  - `model::Model::load(dir: &Path, name: &str) -> Result<Model, String>`
  - `Model::run(&mut self, shape: &[i64], data: Vec<f32>, outputs: &[&str]) -> Result<Vec<Vec<f32>>, String>`
  - `face::Face::load(dir: &Path) -> Result<Face, String>`
  - `Face::process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Option<FaceResult>, String>`
  - `pub struct FaceResult { pub pts: Vec<[f32; 3]>, pub blend: Vec<f32>, pub yaw: f32, pub pitch: f32 }`. `pts` is 478 points in frame pixels (z in the same scale).
  - `face::BLENDSHAPE_NAMES: [&str; 52]`; `face::head_angles(p: &[[f32; 3]]) -> (f32, f32)`

- [ ] **Step 1: Dev dependency for decoding test images**

In `src-tauri/Cargo.toml`, add after the `[build-dependencies]` section:

```toml
[dev-dependencies]
# Golden tests decode MediaPipe's test images (src/vision/*_golden tests).
image = { version = "0.25", default-features = false, features = ["jpeg"] }
```

- [ ] **Step 2: Write `model.rs`**

```rust
//! One ONNX model (converted from MediaPipe's TFLite by
//! scripts/convert-vision-models.sh) run through ONNX Runtime on the CPU.
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub struct Model {
    session: Session,
    input: String,
}

impl Model {
    pub fn load(dir: &Path, name: &str) -> Result<Model, String> {
        let path = dir.join(format!("{name}.onnx"));
        if !path.exists() {
            return Err(format!("Camera models missing ({name}.onnx) — run scripts/convert-vision-models.sh"));
        }
        // 2 threads each so whisper and audio aren't starved.
        let session = Session::builder().map_err(err)?.with_intra_threads(2).map_err(err)?.commit_from_file(&path).map_err(err)?;
        let input = session.inputs()[0].name().to_string();
        Ok(Model { session, input })
    }

    /// Runs one input tensor; returns each named output, flattened.
    pub fn run(&mut self, shape: &[i64], data: Vec<f32>, outputs: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        let t = Tensor::from_array((shape.to_vec(), data)).map_err(err)?;
        let out = self.session.run(ort::inputs![self.input.as_str() => t]).map_err(err)?;
        outputs.iter().map(|n| out[*n].try_extract_tensor::<f32>().map(|(_, v)| v.to_vec()).map_err(err)).collect()
    }
}
```

If `Tensor::from_array` rejects `Vec<i64>` as a shape, use `shape.iter().map(|&d| d as usize).collect::<Vec<usize>>()`. Ledger the change.

- [ ] **Step 3: Write `face.rs` with the tests first**

Create `src-tauri/src/vision/face.rs` containing only the constants and tests below (the implementation goes in Step 5):

```rust
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
```

Add a shared test helper module, `src-tauri/src/vision/golden_util.rs`:

```rust
//! Helpers for the golden tests: MediaPipe's test images and the official
//! MediaPipe Python output on them (scripts/convert-vision-models.sh --goldens).
use std::path::PathBuf;

pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/vision")
}

pub fn image(name: &str) -> (Vec<u8>, usize, usize) {
    let img = image::open(dir().join("golden").join(name)).expect("run scripts/convert-vision-models.sh --goldens").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    (img.into_raw(), w, h)
}

pub fn goldens() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join("golden/goldens.json")).unwrap()).unwrap()
}

/// Mean distance between our pixel points and the expected normalised points,
/// as a fraction of the image width.
pub fn mean_err(got: &[[f32; 2]], want: &serde_json::Value, w: usize, h: usize) -> f32 {
    let want = want.as_array().unwrap();
    got.iter()
        .zip(want)
        .map(|(g, e)| {
            let (ex, ey) = (e[0].as_f64().unwrap() as f32 * w as f32, e[1].as_f64().unwrap() as f32 * h as f32);
            (g[0] - ex).hypot(g[1] - ey) / w as f32
        })
        .sum::<f32>()
        / got.len() as f32
}
```

In `src-tauri/src/vision/mod.rs`, add:

```rust
pub mod face;
mod model;
#[cfg(test)]
mod golden_util;
```

- [ ] **Step 4: Run to verify it fails**

Run: `nix develop -c cargo test --manifest-path src-tauri/Cargo.toml vision::face -- --include-ignored 2>&1 | tail -5`
Expected: compile errors (`cannot find function head_angles`, `Face`).

- [ ] **Step 5: Implement the face pipeline (insert above the test modules in `face.rs`)**

```rust
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
```

- [ ] **Step 6: Run the unit and golden tests**

Run: `nix develop -c cargo test --manifest-path src-tauri/Cargo.toml vision::face -- --include-ignored 2>&1 | grep -E 'test |test result'`
Expected: `head_angles_frontal_and_turned ... ok`, `face_matches_mediapipe ... ok`, `blank_frame_has_no_face_and_no_track ... ok`.

If `face_matches_mediapipe` fails:
- **Landmark error ≫ 1%:** the crop or back-projection is wrong. Check `crop`'s `Norm` (the face landmark model takes 0..1) and that `to_frame` divides by 256.
- **Only yaw/pitch fail, and by a constant offset:** the matrix-derived angles have a different zero. Record the measured offset in the ledger as a ruling and assert on angle *differences* between two images rather than absolute values. Don't loosen the landmark tolerance.

- [ ] **Step 7: Commit**

```bash
git add src-tauri/Cargo.toml src-tauri/Cargo.lock src-tauri/src/vision/
git commit -m "vision: face landmarks, blendshapes and head pose in rust"
```

---

### Task 4: Pose pipeline (with golden test)

**Files:**
- Create: `src-tauri/src/vision/pose.rs`
- Modify: `src-tauri/src/vision/mod.rs` (`pub mod pose;`)

**Interfaces:**
- Produces: `pose::Pose::load(dir: &Path) -> Result<Pose, String>`; `Pose::process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Option<Vec<[f32; 2]>>, String>`, returning 33 points in frame pixels.

- [ ] **Step 1: Write the golden test first (`pose.rs`)**

```rust
//! Pose pipeline: MediaPipe's pose detector (only when there's no track) →
//! alignment-point ROI → 33 landmarks; the model's auxiliary points 33/34 give
//! the next frame's ROI. Constants from pose_detector_graph.cc and
//! pose_landmarks_detector_graph.cc.

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
        assert!(err < 0.01, "mean landmark error {err} of width");
    }
}
```

Add `pub mod pose;` to `mod.rs`.

- [ ] **Step 2: Run to verify it fails**

Run: `nix develop -c cargo test --manifest-path src-tauri/Cargo.toml vision::pose -- --include-ignored 2>&1 | tail -3`
Expected: compile error (`Pose` not found).

- [ ] **Step 3: Implement (above the test module)**

```rust
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `nix develop -c cargo test --manifest-path src-tauri/Cargo.toml vision::pose -- --include-ignored 2>&1 | grep -E 'test |test result'`
Expected: `pose_matches_mediapipe ... ok`.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/vision/
git commit -m "vision: pose landmarks in rust"
```

---

### Task 5: Hands pipeline (with golden test)

**Files:**
- Create: `src-tauri/src/vision/hands.rs`
- Modify: `src-tauri/src/vision/mod.rs` (`pub mod hands;`)

**Interfaces:**
- Produces: `hands::Hands::load(dir: &Path) -> Result<Hands, String>`; `Hands::process(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<Vec<[f32; 2]>>, String>`, returning ≤2 hands × 21 points in frame pixels.

- [ ] **Step 1: Write the golden test first (`hands.rs`)**

```rust
//! Hands pipeline: MediaPipe's palm detector → rotated ROI per palm → 21
//! landmarks; tracks follow the landmarks, and the detector re-runs about once
//! a second while fewer than 2 hands are tracked. Constants from
//! hand_detector_graph.cc, hand_landmarks_detector_graph.cc and
//! hand_landmarks_to_rect_calculator.cc.

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
            assert!(err < 0.01, "mean landmark error {err} of width");
        }
    }
}
```

Add `pub mod hands;` to `mod.rs`.

- [ ] **Step 2: Run to verify it fails**

Run: `nix develop -c cargo test --manifest-path src-tauri/Cargo.toml vision::hands -- --include-ignored 2>&1 | tail -3`
Expected: compile error (`Hands` not found).

- [ ] **Step 3: Implement (above the test module)**

```rust
use super::geometry::{crop, decode, hand_track_roi, ssd_anchors, weighted_nms, Anchor, Norm, Roi};
use super::model::Model;
use std::f32::consts::PI;
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
                // wrist (0) → middle-finger MCP (2), upright = 90°; scale 2.6, shift −0.5
                Roi::from_box(cx, cy, d.w * whole.size, d.h * whole.size, kp(0), kp(2), PI / 2.0, 2.6, -0.5)
            })
            .collect())
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `nix develop -c cargo test --manifest-path src-tauri/Cargo.toml vision::hands -- --include-ignored 2>&1 | grep -E 'test |test result'`
Expected: `hands_match_mediapipe ... ok`.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/vision/
git commit -m "vision: hand landmarks in rust"
```

---

### Task 6: Camera capture, inference loop, commands

**Files:**
- Create: `src-tauri/src/vision/capture.rs`, `src-tauri/examples/vision_probe.rs`
- Modify: `src-tauri/src/vision/mod.rs`, `src-tauri/src/lib.rs`, `src-tauri/Cargo.toml`

**Interfaces:**
- Consumes: `Face`, `Pose`, `Hands`, `BLENDSHAPE_NAMES`.
- Produces (used by Task 7's TS):
  - command `start_vision(results: Channel<VisionMsg>, preview: Channel<InvokeResponseBody>) -> Result<(), String>`, async;
  - command `stop_vision()`, async;
  - `VisionMsg` serialises as either `{ t, blend, head, pose, hands }` (frame) or `{ ended: true }`;
  - `pub struct VisionState` (managed).
- Preview delivery: every processed frame's JPEG is sent (~12/s × ~30 KB). The spec's "drop if the previous one hasn't been sent" isn't possible because Tauri channels give no delivery feedback, and the volume is negligible. Note this as a ruling.

- [ ] **Step 1: Dependencies**

In `src-tauri/Cargo.toml` `[dependencies]`, add:

```toml
nokhwa = { version = "0.10", features = ["input-native"] } # camera (v4l2 on Linux)
jpeg-encoder = "0.7" # camera preview frames for the webview
```

- [ ] **Step 2: Write `capture.rs`**

```rust
//! Camera thread: YUYV 640×360@30 via nokhwa, decoded to RGB, kept in a
//! one-frame slot so the inference thread always takes the newest frame.
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

pub const W: usize = 640;
pub const H: usize = 360;

#[derive(Default)]
pub struct Latest {
    pub slot: Mutex<Option<(u64, Vec<u8>)>>, // (ms since start, RGB)
    pub ready: Condvar,
}

/// Runs until `stop`; sets `ended` if the camera fails mid-stream. Reports the
/// open result on `opened` first so start_vision can fail fast (busy/missing).
pub fn run(latest: Arc<Latest>, stop: Arc<AtomicBool>, ended: Arc<AtomicBool>, opened: Sender<Result<(), String>>) {
    let fmt = CameraFormat::new(Resolution::new(W as u32, H as u32), FrameFormat::YUYV, 30);
    let cam = Camera::new(CameraIndex::Index(0), RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt))).and_then(|mut c| c.open_stream().map(|_| c));
    let mut cam = match cam {
        Ok(c) => {
            let _ = opened.send(Ok(()));
            c
        }
        Err(e) => {
            let _ = opened.send(Err(format!("Camera unavailable — {e}")));
            return;
        }
    };
    let start = Instant::now();
    while !stop.load(Relaxed) {
        match cam.frame().and_then(|f| f.decode_image::<RgbFormat>()) {
            Ok(img) => {
                *latest.slot.lock().unwrap() = Some((start.elapsed().as_millis() as u64, img.into_raw()));
                latest.ready.notify_one();
            }
            Err(e) => {
                eprintln!("[vision] camera read failed: {e}");
                ended.store(true, Relaxed);
                break;
            }
        }
    }
    let _ = cam.stop_stream();
    latest.ready.notify_one();
}
```

- [ ] **Step 3: Write the rest of `mod.rs`**

Replace `src-tauri/src/vision/mod.rs` with:

```rust
//! Camera capture and face/pose/hand inference for presence grading, run in
//! Rust so the webview's UI thread stays free (see
//! docs/superpowers/specs/2026-10-08-rust-vision-design.md). The webview gets
//! one JSON result and one JPEG preview per processed frame (~12.5 Hz).
mod capture;
pub mod face;
pub mod geometry;
#[cfg(test)]
mod golden_util;
pub mod hands;
mod model;
pub mod pose;

use capture::{Latest, H, W};
use face::{Face, BLENDSHAPE_NAMES};
use hands::Hands;
use pose::Pose;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tauri::ipc::{Channel, InvokeResponseBody};
use tauri::{AppHandle, Manager, State};

const TICK: Duration = Duration::from_millis(80);
const MAX_FAILS: u32 = 30;

#[derive(Serialize, Clone, Copy)]
pub struct Pt {
    x: f32,
    y: f32,
}

#[derive(Serialize)]
pub struct Head {
    yaw: f32,
    pitch: f32,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum VisionMsg {
    Frame { t: u64, blend: Option<BTreeMap<&'static str, f32>>, head: Option<Head>, pose: Option<Vec<Pt>>, hands: Vec<Vec<Pt>> },
    Ended { ended: bool },
}

pub struct Pipelines {
    face: Face,
    pose: Pose,
    hands: Hands,
}

impl Pipelines {
    pub fn load(dir: &Path) -> Result<Pipelines, String> {
        Ok(Pipelines { face: Face::load(dir)?, pose: Pose::load(dir)?, hands: Hands::load(dir)? })
    }

    pub fn frame(&mut self, t: u64, rgb: &[u8]) -> Result<VisionMsg, String> {
        let norm = |p: &[f32; 2]| Pt { x: p[0] / W as f32, y: p[1] / H as f32 };
        let face = self.face.process(rgb, W, H)?;
        let pose = self.pose.process(rgb, W, H)?;
        let hands = self.hands.process(rgb, W, H)?;
        Ok(VisionMsg::Frame {
            t,
            blend: face.as_ref().map(|f| BLENDSHAPE_NAMES.iter().copied().zip(f.blend.iter().copied()).collect()),
            head: face.as_ref().map(|f| Head { yaw: f.yaw, pitch: f.pitch }),
            pose: pose.map(|p| p.iter().map(norm).collect()),
            hands: hands.iter().map(|hd| hd.iter().map(norm).collect()).collect(),
        })
    }
}

fn jpeg(rgb: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = jpeg_encoder::Encoder::new(&mut buf, 70).encode(rgb, W as u16, H as u16, jpeg_encoder::ColorType::Rgb);
    buf
}

/// Inference thread: ~12.5 Hz on the newest frame; one result + one preview per frame.
fn infer(mut pipes: Pipelines, latest: Arc<Latest>, stop: Arc<AtomicBool>, ended: Arc<AtomicBool>, results: Channel<VisionMsg>, preview: Channel<InvokeResponseBody>) {
    let mut fails = 0;
    let mut last = Instant::now() - TICK;
    while !stop.load(Relaxed) && !ended.load(Relaxed) {
        thread::sleep(TICK.saturating_sub(last.elapsed()));
        last = Instant::now();
        let frame = {
            let mut g = latest.slot.lock().unwrap();
            while g.is_none() && !stop.load(Relaxed) && !ended.load(Relaxed) {
                g = latest.ready.wait_timeout(g, Duration::from_millis(200)).unwrap().0;
            }
            g.take()
        };
        let Some((t, rgb)) = frame else { continue };
        match pipes.frame(t, &rgb) {
            Ok(msg) => {
                fails = 0;
                let _ = results.send(msg);
                let _ = preview.send(InvokeResponseBody::Raw(jpeg(&rgb)));
            }
            Err(e) => {
                eprintln!("[vision] frame failed: {e}");
                fails += 1;
                if fails >= MAX_FAILS {
                    ended.store(true, Relaxed);
                }
            }
        }
    }
    if ended.load(Relaxed) && !stop.load(Relaxed) {
        let _ = results.send(VisionMsg::Ended { ended: true });
    }
}

struct Running {
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

#[derive(Default)]
pub struct VisionState(Mutex<Option<Running>>);

impl VisionState {
    /// Signals both threads and waits for them; the camera is released when
    /// the capture thread returns.
    pub fn stop(&self) {
        if let Some(r) = self.0.lock().unwrap().take() {
            r.stop.store(true, Relaxed);
            for t in r.threads {
                let _ = t.join();
            }
        }
    }
}

// async: model loading and opening the camera take a few hundred ms and must
// not block the main thread (sync Tauri commands run on it).
#[tauri::command]
pub async fn start_vision(app: AppHandle, state: State<'_, VisionState>, results: Channel<VisionMsg>, preview: Channel<InvokeResponseBody>) -> Result<(), String> {
    state.stop();
    let dir = app.path().resolve("resources/vision", tauri::path::BaseDirectory::Resource).map_err(|e| e.to_string())?;
    let pipes = Pipelines::load(&dir)?;
    let latest = Arc::new(Latest::default());
    let (stop, ended) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let (tx, rx) = mpsc::channel();
    let cap = {
        let (latest, stop, ended) = (latest.clone(), stop.clone(), ended.clone());
        thread::spawn(move || capture::run(latest, stop, ended, tx))
    };
    if let Err(e) = rx.recv().map_err(|e| e.to_string()).and_then(|r| r) {
        let _ = cap.join();
        return Err(e);
    }
    let inf = {
        let stop = stop.clone();
        thread::spawn(move || infer(pipes, latest, stop, ended, results, preview))
    };
    *state.0.lock().unwrap() = Some(Running { stop, threads: vec![cap, inf] });
    Ok(())
}

#[tauri::command]
pub async fn stop_vision(state: State<'_, VisionState>) -> Result<(), String> {
    state.stop();
    Ok(())
}
```

- [ ] **Step 4: Register in `lib.rs`; remove the webview camera workarounds; stop on close**

In `src-tauri/src/lib.rs`:
1. Delete the `#[cfg(target_os = "linux")] std::env::set_var("WEBKIT_GST_DMABUF_SINK_DISABLED", "1");` statement and its comment at the top of `run()`.
2. Delete the whole `// Camera for presence grading (camera.ts). wry leaves WebKitGTK's …` block, i.e. the `#[cfg(target_os = "linux")] app.get_webview_window("main")…with_webview(…)?;` statement.
3. After `app.manage(audio::SessionGen::default());`, add `app.manage(vision::VisionState::default());`.
4. Add `vision::start_vision,` and `vision::stop_vision,` to `generate_handler![…]` (after `screenshot::screenshot`, adding a comma to that line).
5. Before `.invoke_handler(`, add:

```rust
        // Release the camera when the window closes (vision threads otherwise
        // run until the process exits).
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                window.state::<vision::VisionState>().stop();
            }
        })
```

- [ ] **Step 5: Write a probe example that exercises the real camera without the UI**

`src-tauri/examples/vision_probe.rs`:

```rust
// Runs the vision pipelines on the live camera for 5 s without the UI and prints
// throughput: `cargo run --example vision_probe --manifest-path src-tauri/Cargo.toml`
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera;
use speech_lib::vision_probe::Pipelines;
use std::time::Instant;

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/vision");
    let mut pipes = Pipelines::load(&dir).expect("models");
    let fmt = CameraFormat::new(Resolution::new(640, 360), FrameFormat::YUYV, 30);
    let mut cam = Camera::new(CameraIndex::Index(0), RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(fmt))).expect("camera");
    cam.open_stream().expect("stream");
    let (start, mut n, mut ms) = (Instant::now(), 0, Vec::new());
    while start.elapsed().as_secs() < 5 {
        let rgb = cam.frame().unwrap().decode_image::<RgbFormat>().unwrap().into_raw();
        let t = Instant::now();
        let msg = serde_json::to_string(&pipes.frame(0, &rgb).unwrap()).unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
        n += 1;
        if n % 10 == 0 {
            println!("{}", &msg[..msg.len().min(160)]);
        }
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{n} frames, inference p50 {:.1} ms, p95 {:.1} ms", ms[ms.len() / 2], ms[ms.len() * 95 / 100]);
}
```

For the example to reach `Pipelines`, add to `src-tauri/src/lib.rs` after the `mod vision;` line:

```rust
/// For `examples/vision_probe.rs` only.
pub mod vision_probe {
    pub use crate::vision::Pipelines;
}
```

- [ ] **Step 6: Build and run all Rust tests**

Run: `nix develop -c cargo test --manifest-path src-tauri/Cargo.toml -- --include-ignored 2>&1 | grep -E 'test result|FAILED|panicked'`
Expected: every `test result: ok` (the earlier 22 plus the vision tests), no `FAILED`.

- [ ] **Step 7: Run the probe on the real camera**

The camera must be free: no app instance with Video on. Run:
`nix develop -c cargo run --release --example vision_probe --manifest-path src-tauri/Cargo.toml 2>&1 | tail -8`
Expected:
- JSON lines containing `"blend":{`, `"head":{` and `"pose":[` while a person is in front of the camera; `"hands":[[` when a hand is raised;
- a last line with inference p50 ≤ 40 ms and p95 ≤ 70 ms.

If nobody is in front of the camera, expect `"blend":null` and record that only throughput was verified.

- [ ] **Step 8: Commit**

```bash
git add src-tauri/Cargo.toml src-tauri/Cargo.lock src-tauri/src/vision/ src-tauri/src/lib.rs src-tauri/examples/vision_probe.rs
git commit -m "vision: camera thread, inference loop and start/stop commands"
```

---

### Task 7: Webview switch-over

**Files:**
- Modify: `src/camera.ts` (rewrite), `src/presence.ts`, `src/presence.check.ts`, `src/main.ts`, `index.html`, `src/styles.css`, `package.json`/`bun.lock`, `README.md`
- Delete: `public/mediapipe/` (ignored files; just remove the directory)

**Interfaces:**
- Consumes: the `start_vision` / `stop_vision` commands and the `VisionMsg` shape from Task 6.
- Produces: unchanged `camera.ts` exports used by `main.ts` (`startCamera`, `stopCamera`, `FRAME_W`, `FRAME_H`, `HAND_BONES`), except that `startCamera` now takes an `HTMLCanvasElement`.

- [ ] **Step 1: Change `presence.check.ts` first (RED)**

In `src/presence.check.ts`:
- Import line: remove `headAngles, ` from the `./presence` import.
- Delete the helper lines `const rad = …`, `const mat = … };` (the whole 6-line function), `const yawM = …`, `const pitchM = …`, `const transpose = …`.
- Delete the block from `// headAngles: magnitudes are right in either flattening order.` through `near(Math.abs(headAngles(transpose(yawM(30))).yaw), 30);`.
- Replace `matrix: yawM(5)` with `head: { yaw: 5, pitch: 0 }`.
- Replace `toFrame(raw({ blend: null, matrix: null, pose: null })` with `toFrame(raw({ blend: null, head: null, pose: null })`.
- Replace `raw({ matrix: yawM(30) })` with `raw({ head: { yaw: 30, pitch: 0 } })`.
- Replace `raw({ matrix: null })` with `raw({ head: null })`, and that line's comment `// no matrix → can't tell → not looking` with `// no head pose → can't tell → not looking`.
- Replace `matrix: yawM(0)` (in `stillRaw`) with `head: { yaw: 0, pitch: 0 }`.

Run: `bun src/presence.check.ts`
Expected: FAIL. `RawResults` has no `head` (a TS error from tsc), or the assertion `looking` is false for the frontal case, because `toFrame` still reads `matrix`.

- [ ] **Step 2: Change `presence.ts` (GREEN)**

In `src/presence.ts`:
- In `interface RawResults`, replace `matrix: number[] | null; // 4×4 facial transformation, flattened` with `head: { yaw: number; pitch: number } | null; // degrees, from Rust (vision/face.rs)`.
- Delete the `headAngles` function and its 3-line comment above it.
- In `toFrame`, replace

```ts
  if (b && r.matrix) {
    const { yaw, pitch } = headAngles(r.matrix);
    looking =
```

with

```ts
  if (b && r.head) {
    const { yaw, pitch } = r.head;
    looking =
```

Run: `bun src/presence.check.ts`
Expected: `presence ok`.

- [ ] **Step 3: Rewrite `src/camera.ts`**

```ts
// Camera presence: the camera and all vision inference run in Rust
// (src-tauri/src/vision) so the UI thread stays free — doing it here blocked
// animations for ~55 ms a tick. This file turns Rust's messages into Frames
// and paints its JPEG preview.
import { Channel, invoke } from "@tauri-apps/api/core";
import { toFrame, type Frame, type Pt } from "./presence";

export const FRAME_W = 640;
export const FRAME_H = 360;
// MediaPipe's 21-landmark hand skeleton (HandLandmarker.HAND_CONNECTIONS).
export const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
].map(([start, end]) => ({ start, end }));

interface VisionFrame {
  t: number;
  blend: Record<string, number> | null;
  head: { yaw: number; pitch: number } | null;
  pose: Pt[] | null;
  hands: Pt[][];
}
type VisionMsg = VisionFrame | { ended: true };

let gen = 0; // bumps on every start/stop so a stale session's messages are ignored

// onFrame gets the scored Frame plus each hand's 21 normalised landmarks (for
// the live highlight only; never stored).
export async function startCamera(view: HTMLCanvasElement, onFrame: (f: Frame, hands: Pt[][]) => void, onEnded: () => void): Promise<void> {
  const my = ++gen;
  const results = new Channel<VisionMsg>();
  results.onmessage = (m) => {
    if (my !== gen) return;
    if ("ended" in m) return onEnded();
    onFrame(toFrame({ blend: m.blend, head: m.head, pose: m.pose, wrists: m.hands.map((h) => h[0]) }, FRAME_W, FRAME_H, m.t), m.hands);
  };
  const preview = new Channel<ArrayBuffer>();
  const cx = view.getContext("2d")!;
  preview.onmessage = async (buf) => {
    if (my !== gen) return;
    const bmp = await createImageBitmap(new Blob([buf], { type: "image/jpeg" })); // decoded off the main thread
    if (view.width !== bmp.width || view.height !== bmp.height) [view.width, view.height] = [bmp.width, bmp.height];
    cx.drawImage(bmp, 0, 0);
    bmp.close();
  };
  await invoke("start_vision", { results, preview });
}

export function stopCamera() {
  gen++;
  void invoke("stop_vision");
}
```

- [ ] **Step 4: `<video>` → `<canvas>`**

- `index.html`: replace `<video id="self-view" autoplay playsinline muted></video>` with `<canvas id="self-view"></canvas>`.
- `src/styles.css`, in the camera block:
  - `.cam-view video,\n.cam-view canvas {` → `.cam-view canvas {`;
  - `.cam-view video {` → `#self-view {` (keeps `object-fit: cover`, which works on canvas);
  - `body.cam-scope .cam-view video {` → `body.cam-scope #self-view {`.
- `src/main.ts`:
  - in `setCamera`: `const video = $<HTMLVideoElement>("self-view");` → `const video = $<HTMLCanvasElement>("self-view");`;
  - in `drawHands`: delete `const video = $<HTMLVideoElement>("self-view");`, change `if (!cv || !video || !cx) return;` to `if (!cv || !cx) return;`, and replace the `// Landmarks are relative …` comment plus the `const map = video.videoWidth ? … : …;` line with:

```ts
  // Landmarks are normalised to the 640×360 camera frame the preview shows.
  const map = coverMap(FRAME_W, FRAME_H, bw, bh);
```

- [ ] **Step 5: Remove the webview MediaPipe path**

```bash
bun remove @mediapipe/tasks-vision
rm -rf public/mediapipe
```

In `README.md`, replace the paragraph starting `The camera's presence grading (optional) needs MediaPipe's wasm` and its code block, through `works audio-only.`, with:

````markdown
The camera's presence grading (optional) runs MediaPipe's face, pose and hand
models in Rust as ONNX. Convert them once (downloads ~250 MB of TensorFlow into
`src-tauri/target/vision-convert/` the first time):

```sh
scripts/convert-vision-models.sh            # models → src-tauri/resources/vision/
scripts/convert-vision-models.sh --goldens  # + golden test data for `cargo test -- --ignored`
```

Without them the Video toggle reports "Camera models missing" and the app
works audio-only.
````

- [ ] **Step 6: Build and run all checks**

Run: `for f in src/*.check.ts; do bun "$f" || echo "FAIL $f"; done; bun run build 2>&1 | grep -E 'error|✓ built'`
Expected: 7 `… ok` lines, no `FAIL`, `✓ built`.
Run: `grep -rn "mediapipe/tasks-vision\|getUserMedia\|headAngles" src/ index.html; echo "exit $?"`
Expected: no matches (`exit 1`).

- [ ] **Step 7: Manual checks in the real app**

Make sure no other instance is using the camera, then run `nix develop -c bun run tauri dev`.
1. Toggle Video on. The self-view shows your mirrored camera within ~1 s, and the Presence card reacts during a recording. Hand highlights line up with your hands in both the small view and "Camera in scope".
2. Toggle Video off, then on, five times quickly. It comes back each time with no "busy" error (Review Focus 4).
3. With Video on, start a second app instance and toggle its Video. It shows "Camera unavailable — …" and its toggle turns off (Review Focus 3).
4. Tilt your head and a hand about 30°. The highlight stays on the hand (Review Focus 1).
5. Leave the frame for 3 s and come back. The face and pose are picked up again within about a second (Review Focus 2).
6. Close the window with Video on, then relaunch. The camera is free (no EBUSY).

- [ ] **Step 8: Commit**

```bash
git add src/camera.ts src/presence.ts src/presence.check.ts src/main.ts index.html src/styles.css package.json bun.lock README.md
git commit -m "vision: webview uses the rust camera pipeline; drop mediapipe js"
```

---

### Task 8: Performance verification

**Files:** none committed. Temporary instrumentation only, removed in Step 4.

- [ ] **Step 1: Temporary frame-time logger**

In `src/main.ts` (temporary, do not commit):
- next to `let presenceDrawnAt = 0;`, add `let perfVis = 0; // PERF`;
- make `perfVis++; // PERF` the first line of `onPresenceFrame`'s body;
- at the end of the init function (after the `bindToggle("set-cam-scope", …)` line), add:

```ts
  // PERF (temporary): rAF frame intervals + vision results per second, every 5 s
  {
    let last = 0, n = 0, jank = 0;
    const tick = (t: number) => {
      if (last) {
        n++;
        if (t - last > 25) jank++;
      }
      last = t;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    setInterval(() => {
      void fetch("http://127.0.0.1:8765/", { method: "POST", mode: "no-cors", body: JSON.stringify({ fps: n / 5, jank, visPerSec: perfVis / 5 }) });
      n = jank = perfVis = 0;
    }, 5000);
  }
```

Create `src-tauri/target/perfsrv.py` (ignored directory):

```python
import http.server, sys, time
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        with open(sys.argv[1], "a") as f: f.write(time.strftime("%H:%M:%S ") + body + "\n")
        self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", 8765), H).serve_forever()
```

Run it in the background: `python3 src-tauri/target/perfsrv.py src-tauri/target/perf.log`.

- [ ] **Step 2: Measure**

Run the app (`nix develop -c bun run tauri dev`), keep the window visible, turn Video on, start a recording, and wait 30 s.
Expected in `src-tauri/target/perf.log`: `fps` ≥ 58, `jank` < 10 per 5 s, `visPerSec` ≥ 10.

- [ ] **Step 3: Whisper impact**

Record 20 s of speech with Video on and again with it off. Compare the `[whisper] audio=… decode=…ms` lines in the terminal for the live (tiny.en) decodes.
Expected: median decode with Video on ≤ 1.15 × with it off.

- [ ] **Step 4: Remove the instrumentation**

Remove the three PERF additions (`git diff src/main.ts` must show nothing), stop the log server, and delete `src-tauri/target/perfsrv.py` and `src-tauri/target/perf.log`.

- [ ] **Step 5: Record results**

No commit. Put the measured fps / jank / vis rate / whisper ratio in the final report. If any target is missed, report it with the numbers and don't tune `PRESENCE_TUNING` or MediaPipe constants to hit it.

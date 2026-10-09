# Rust vision pipeline — design

Date: 2026-10-08
Status: approved in conversation, pending spec review
Supersedes: the capture and inference parts of `2026-10-07-video-presence-design.md`.
The scoring, UI and report parts of that spec are unchanged.

## Problem

With the camera on, the app's animations lag. Measured in the real Tauri window,
MediaPipe's `detectForVideo` runs synchronously on the webview's main thread:
face, pose and hands together block for about 50–57 ms every 80 ms tick. That drops
a 60 fps `requestAnimationFrame` loop to about 35 fps, with ~60 janky frames per 5 s.

Moving inference to a Web Worker isn't possible on this platform. WebKitGTK has
no WebGL in workers, and MediaPipe's vision graphs need WebGL even on the CPU
delegate and even with `ImageData` input. All three ways of trying it failed.
Rotating one model per tick on the main thread still leaves hitches.

**Decision:** move camera capture and all vision inference into Rust. The
webview only draws a preview and runs the existing pure scoring.

## Feasibility (spike, 2026-10-08)

- MediaPipe's `.task` files are zips of plain TFLite models: `face_detector`,
  `face_landmarks_detector`, `face_blendshapes`, `pose_detector`,
  `pose_landmarks_detector`, `hand_detector`, `hand_landmarks_detector`.
  Blendshapes are therefore available natively.
- All 7 convert to ONNX with `tf2onnx` and match their TFLite originals to within
  0.0016 on every output we use. The pose model's segmentation-mask output differs,
  but we don't use it.
- `pose_detector` uses a `DENSIFY` op (sparse weights) that `tf2onnx` can't
  handle. Expanding those weights to dense constants beforehand gives
  bit-identical outputs (see Setup).
- ONNX Runtime CPU cost per model, from the Python benchmark (same engine as `ort`):
  - face detector 2.0 ms, face landmarks 8.1 ms, blendshapes 1.7 ms;
  - pose detector 8.1 ms, pose landmarks 9.4 ms;
  - hand detector 8.0 ms, hand landmarks 6.6 ms per hand.
- Camera: `nokhwa` 0.10 (`input-native`) opens the integrated camera at exactly
  YUYV 640×360 at 30 fps. It delivers about 27 fps, and YUYV→RGB takes 3 ms. Only one
  process can stream at a time (EBUSY), so the webview and Rust can't share the camera.

## Goals and success criteria

- With the camera on, the live view's animations hold **≥ 58 fps** in the real app,
  measured with a rAF frame-interval logger as in the investigation. Today it's ~35.
- Vision results arrive at **≥ 10 fps**.
- Face, pose and hand outputs match the official MediaPipe Python package on
  the same test images, within the tolerances in Testing.
- whisper `tiny.en` decode times stay within ~15% of current with the camera on.
- Scoring, report, history, the Video button, "Camera in scope" and the hand
  highlight all behave as before from the user's point of view.

## Architecture

```
camera (v4l2 via nokhwa)
   │ YUYV 640×360 @30
   ▼
capture thread ── keeps only the latest frame (Mutex<Option<Frame>> + Condvar)
   │
   ▼
inference thread (~12.5 Hz, skips to newest frame if behind)
   ├─ face pipeline  → blendshapes(52) + head {yaw, pitch}
   ├─ pose pipeline  → 33 landmarks (normalised)
   ├─ hand pipeline  → ≤2 × 21 landmarks (normalised)
   └─ JPEG encode of the same RGB frame
   │
   ▼  tauri::ipc::Channel ×2  (results: JSON, preview: raw bytes)
webview: camera.ts (thin client) → presence.ts (unchanged scoring) + preview canvas
```

### Rust: `src-tauri/src/vision/`

| File | Responsibility |
|---|---|
| `mod.rs` | `start_vision` / `stop_vision` commands, thread lifecycle, `VisionState` (managed); resolves `resources/vision/*.onnx` with a clear error if missing |
| `capture.rs` | `nokhwa` camera at YUYV 640×360@30, YUYV→RGB, latest-frame slot |
| `geometry.rs` | Pure: SSD anchors, box/keypoint decode, sigmoid, weighted NMS, rotated ROI, crop with bilinear sampling, inverse projection |
| `face.rs` | Face detect → ROI → landmarks (478) → blendshapes (52) + head yaw/pitch; tracking |
| `pose.rs` | Pose detect → ROI → landmarks (33); tracking |
| `hands.rs` | Palm detect → ROI per hand → landmarks (21); tracking, ≤2 hands |

**Sessions:** 7 `ort` sessions, created once per `start_vision` and dropped on
stop. Each uses `with_intra_threads(2)`.

**Inputs:** each model receives its input range per MediaPipe's graph config
(detectors may expect [-1, 1], landmark models [0, 1]). Each constant is copied
from the MediaPipe source file below, not guessed:
- face detector (short range, 128×128, 896 anchors):
  `mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc`
  and `mediapipe/modules/face_detection/face_detection_short_range*.pbtxt`;
- face landmarks (256×256, 478×3) and ROI (eye-line rotation, scale 1.5):
  `mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_detector_graph.cc`,
  `face_landmarker_graph.cc`;
- blendshapes (146-landmark subset, pixel coords, 52 outputs in MediaPipe's name
  order): `mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc`;
- pose detector (224×224, 2254 anchors) and ROI (hip centre and scale keypoint,
  scale 1.25): `mediapipe/tasks/cc/vision/pose_detector/pose_detector_graph.cc`,
  `pose_landmarker/pose_landmarker_graph.cc`;
- pose landmarks (256×256, 195 = 39×5, first 33 used):
  `pose_landmarker/pose_landmarks_detector_graph.cc`;
- palm detector (192×192, 2016 anchors) and ROI (wrist → middle-MCP rotation,
  scale 2.6, shift −0.5): `mediapipe/tasks/cc/vision/hand_detector/hand_detector_graph.cc`,
  `hand_landmarker/hand_landmarker_graph.cc`;
- hand landmarks (224×224, 21×3, presence, handedness):
  `hand_landmarker/hand_landmarks_detector_graph.cc`.

The anchor counts 896 / 2016 / 2254 must match the model output shapes the spike
observed. That's a unit test.

**Tracking (detect-then-track, as MediaPipe does):**
- Each frame's landmarks give the next frame's ROI. A detector runs only when
  there's no track, or when the landmark model's presence score is < 0.5.
- Hands: the palm detector also re-runs about once a second while fewer than 2
  hands are tracked, so a second hand gets picked up. Duplicate tracks (ROI IoU > 0.5) are merged.

**Head pose:** MediaPipe's facial transformation matrix comes from a 3D
Procrustes fit and isn't reproduced. Instead, yaw and pitch come from the 3D face
landmarks: the normal of the plane through both outer eye corners and the chin,
with pitch also using the forehead→chin direction. Indices are from MediaPipe's
canonical face mesh. The golden test checks these against the angles Python
MediaPipe derives from its matrix (Testing).

**Message to the webview (results channel, JSON), per processed frame:**
```ts
interface VisionFrame {
  t: number;                                  // ms since start_vision, monotonic
  blend: Record<string, number> | null;       // 52 blendshapes; null = no face
  head: { yaw: number; pitch: number } | null;// degrees
  pose: { x: number; y: number }[] | null;    // 33, normalised to the 640×360 frame
  hands: { x: number; y: number }[][];        // up to 2 × 21, normalised
}
```
Preview channel: the same frame's RGB, JPEG-encoded (quality ~70, `jpeg-encoder`).
Only one is in flight at a time; a frame is dropped if the previous one hasn't been sent.

**Commands:**
- `start_vision(results: Channel<VisionFrame>, preview: Channel<Vec<u8>>) -> Result<(), String>`
  starts the capture and inference threads. It errors if the camera can't be opened
  (busy, missing) or models are missing ("Camera models missing — run
  scripts/convert-vision-models.sh"). Starting while running restarts.
- `stop_vision()` sets the stop flag, joins both threads, and releases the camera.
- If the camera ends mid-stream (unplugged or read error), the inference thread
  sends one final `{ ended: true }` on the results channel and stops.

### TypeScript

- `camera.ts` becomes a thin client with the same exported interface:
  - `startCamera(view, onFrame, onEnded)` and `stopCamera()`;
  - `FRAME_W` / `FRAME_H` stay 640×360;
  - `HAND_BONES` becomes a local constant: MediaPipe's 21-landmark connection list, copied.
  - It invokes `start_vision` with two `Channel`s:
    - results → `toFrame(…)` → `onFrame(frame, hands)`;
    - `{ended}` → `onEnded()`;
    - preview bytes → `createImageBitmap(new Blob([bytes], {type: "image/jpeg"}))`
      → draw onto the preview canvas → `bitmap.close()`.
- `#self-view` becomes `<canvas id="self-view">`. The `.cam-view` CSS (small
  mode, "Camera in scope" mode, mirroring, overlay alignment) is unchanged. `coverMap`
  uses 640×360 as the source size.
- `presence.ts`:
  - `RawResults.matrix: number[] | null` → `head: { yaw: number; pitch: number } | null`;
  - `toFrame` uses `head` directly;
  - `headAngles` and its checks are deleted;
  - `RawResults.wrists` is derived from `hands` (landmark 0).
  - Everything else (`summarize`, `movingHands`, `liveCue`, scoring) is unchanged.
- `main.ts`: no logic change beyond the `<video>` → `<canvas>` element type.

### Deleted

- `@mediapipe/tasks-vision` dependency; `public/mediapipe/` and its README steps.
- `lib.rs`: the WebKitGTK user-media permission handler and
  `WEBKIT_GST_DMABUF_SINK_DISABLED`, since the webview no longer touches the camera.
- `getUserMedia` / MediaPipe code in `camera.ts`; `headAngles` in `presence.ts`.

## Setup: models

- Add `scripts/convert-vision-models.sh`:
  1. Create a `uv` Python 3.11 venv (`nix run nixpkgs#uv`) with
     `tensorflow-cpu==2.15.*`, `tf2onnx==1.16.1`, `onnxruntime`, `numpy<2`
     (runs via nix-ld). Nixpkgs' own TensorFlow doesn't build on Python 3.14 or 3.12.
  2. Download the three `.task` files from `storage.googleapis.com/mediapipe-models`
     (same URLs as today's README) and unzip them.
  3. Densify `pose_detector.tflite`:
     - run the TFLite interpreter with `experimental_preserve_all_tensors`;
     - write each `DENSIFY` output as a dense constant buffer;
     - clear `sparsity`, delete the `DENSIFY` ops, and empty the orphaned sparse
       source buffers.
  4. Convert all 7 with `tf2onnx --opset 17`. For the pose detector, convert through a
     wrapper that stubs `tf.lite.Interpreter.get_tensor_details` → `[]`, because it
     segfaults on the orphaned tensors.
  5. Verify each ONNX against its TFLite original on random input
     (max |diff| < 0.01 on used outputs) and fail loudly otherwise.
  6. Write the results to `src-tauri/resources/vision/*.onnx` (~22 MB, git-ignored, bundled).
- The README's "Models (required, not in git)" section replaces the MediaPipe
  steps with this script.

## Error handling

| Situation | Behaviour |
|---|---|
| Camera busy or missing | `start_vision` errors; UI shows "Camera unavailable — …", toggle off (existing path) |
| Models missing | Same path, message names the script |
| Camera unplugged mid-session | `{ended:true}` → `onEnded` → toggle off; frames so far still scored (existing) |
| Inference error on one frame | Log it, skip the frame, keep running; 30 consecutive errors → `{ended:true}` |
| Inference slower than the tick | Process the newest frame; never queue |
| App quits while vision runs | Threads hold no locks the app needs; the stop flag is set on window close |

## Testing

- **Rust unit tests (`cargo test`):**
  - anchor counts 896 / 2016 / 2254;
  - sigmoid and weighted NMS on hand-made boxes;
  - rotated ROI → crop → inverse projection round-trip (a point maps back to
    itself within 0.5 px);
  - YUYV→RGB on a known 2×2 pattern.
- **Golden tests (`cargo test -- --ignored`, need models plus test data):**
  - run each pipeline on MediaPipe's test images (`portrait.jpg`, `pose.jpg`, and
    `thumb_up.jpg`, all from the `mediapipe-assets` bucket);
  - compare against the official MediaPipe Python package's output on the same images.
    That output is produced once by `scripts/convert-vision-models.sh --goldens`
    and saved as JSON next to the models;
  - tolerances: mean landmark error < 1% of the image width; blendshape scores within
    0.05; head yaw/pitch within 5° of the matrix-derived angles from Python MediaPipe.
- **TS:** `presence.check.ts` updated for `head` (`headAngles` checks removed); all
  other checks unchanged.
- **Performance (manual, in the real app):**
  - a temporary rAF frame-interval logger with the camera on must show ≥ 58 fps and
    < 10 janky frames per 5 s;
  - a vision-rate logger must show ≥ 10 results/s;
  - whisper decode times are compared with the camera on vs off.

## Out of scope

- Testing on macOS/Windows. `nokhwa` compiles there; it's just untested.
- GPU execution providers for `ort` (CPU is fast enough per the spike).
- Iris landmarks, segmentation masks, pose world landmarks, handedness use.
- Hosting pre-converted ONNX files (only if the conversion script becomes a burden).

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

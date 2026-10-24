#!/usr/bin/env bash
# Converts MediaPipe's face, pose and hand models to ONNX for the Rust vision
# pipeline (src-tauri/src/vision). One-time setup; output is git-ignored.
#   scripts/convert-vision-models.sh            # models
#   scripts/convert-vision-models.sh --goldens  # + test images and expected outputs
# Runs on x86_64 Linux (any glibc distro) and macOS; mediapipe has no ARM Linux
# wheels. nixpkgs' TensorFlow doesn't build on Python 3.14/3.12, so this uses a
# uv venv with upstream wheels; on NixOS those need nix-ld. uv comes from PATH,
# then Nix, then astral.sh's installer (into the work dir, leaving PATH and
# shell profiles alone).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/src-tauri/resources/vision
WORK=$ROOT/src-tauri/target/vision-convert
mkdir -p "$OUT" "$WORK"
if command -v uv >/dev/null; then
  UV=uv
elif command -v nix >/dev/null; then
  UV="nix run nixpkgs#uv --"
else
  UV=$WORK/uv/uv
  [ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh |
    UV_UNMANAGED_INSTALL="$WORK/uv" sh
fi
PY=$WORK/venv/bin/python
if [ ! -x "$PY" ]; then
  $UV venv --python 3.11 "$WORK/venv"
  # tf2onnx pins protobuf<4 but runs on 4.x, which mediapipe needs. onnx 1.18
  # wants a newer ml_dtypes than TensorFlow 2.15 allows.
  echo "protobuf>=4.25.3,<5" > "$WORK/overrides.txt"
  # tensorflow-cpu has no ARM wheels; on Apple Silicon the full build is CPU-only anyway.
  $UV pip install --python "$PY" --override "$WORK/overrides.txt" \
    "tensorflow-cpu==2.15.*; platform_machine == 'x86_64'" \
    "tensorflow==2.15.*; platform_machine != 'x86_64'" \
    "tf2onnx==1.16.1" "onnx<1.18" onnxruntime "numpy<2" "mediapipe==0.10.21"
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
  # Outside resources/, which is bundled into the app.
  GOLD=$ROOT/src-tauri/test-data/vision-golden
  mkdir -p "$GOLD"
  for f in portrait.jpg pose.jpg thumb_up.jpg; do fetch "$A/$f" "$GOLD/$f"; done
  "$PY" "$ROOT/scripts/vision/goldens.py" "$WORK" "$GOLD"
fi
echo "vision models ready in $OUT"

#!/usr/bin/env bash
# Fetches the models, builds the app and installs it. Safe to re-run: models
# already in src-tauri/resources/ are skipped.
#   scripts/install.sh                # models + build + install
#   scripts/install.sh --models-only  # just the models, for `bun run tauri dev`
#   scripts/install.sh --no-medium    # skip medium.en (539 MB); download it from Settings later
# macOS installs Speech.app into /Applications (~/Applications if that isn't
# writable). Elsewhere it prints where the bundle landed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RES=$ROOT/src-tauri/resources
MODELS_ONLY=0
MEDIUM=1
for arg in "$@"; do
  case $arg in
    --models-only) MODELS_ONLY=1 ;;
    --no-medium) MEDIUM=0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Downloads to a .part file first so an interrupted download isn't mistaken
# for a finished one on the next run.
fetch() {
  if [ -f "$RES/$2" ]; then echo "have $2"; return; fi
  echo "fetching $2"
  curl -fL --progress-bar -o "$RES/$2.part" "$1"
  mv "$RES/$2.part" "$RES/$2"
}
mkdir -p "$RES"
HF=https://huggingface.co/ggerganov/whisper.cpp/resolve/main
fetch $HF/ggml-tiny.en-q5_1.bin ggml-tiny.en-q5_1.bin
fetch https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx silero_vad.onnx
[ $MEDIUM = 1 ] && fetch $HF/ggml-medium.en-q5_0.bin ggml-medium.en-q5_0.bin

# Camera models are optional and their converter needs Nix.
if [ -d "$RES/vision" ]; then
  echo "have vision/"
elif command -v nix >/dev/null; then
  "$ROOT/scripts/convert-vision-models.sh"
else
  echo "skipping camera models: scripts/convert-vision-models.sh needs Nix (the app works audio-only without them)"
fi

[ $MODELS_ONLY = 1 ] && exit 0

for tool in bun cargo; do
  command -v $tool >/dev/null || { echo "$tool not found; see Install in README.md" >&2; exit 1; }
done
cd "$ROOT"
bun install
BUNDLE=$ROOT/src-tauri/target/release/bundle
case $(uname -s) in
  Darwin)
    bun run tauri build --bundles app
    DEST=/Applications
    [ -w "$DEST" ] || { DEST=$HOME/Applications; mkdir -p "$DEST"; }
    rm -rf "$DEST/Speech.app"
    cp -R "$BUNDLE/macos/Speech.app" "$DEST/"
    echo "installed $DEST/Speech.app"
    ;;
  Linux)
    bun run tauri build --bundles appimage
    echo "built $(ls "$BUNDLE"/appimage/*.AppImage)"
    ;;
  *)
    bun run tauri build --bundles nsis
    echo "built $(ls "$BUNDLE"/nsis/*.exe); run it to install"
    ;;
esac

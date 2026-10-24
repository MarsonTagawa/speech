#!/usr/bin/env bash
# Fetches the models, builds the app and installs it. Safe to re-run: models
# already in src-tauri/resources/ are skipped.
#   scripts/install.sh                # models + build + install
#   scripts/install.sh --models-only  # just the models, for `bun run tauri dev`
#   scripts/install.sh --no-medium    # skip medium.en (539 MB); download it from Settings later
#   scripts/install.sh --install-deps # first install the build tools (Homebrew on macOS,
#                                     # apt/dnf/pacman/zypper on Linux; asks for sudo)
# macOS installs Speech.app into /Applications (~/Applications if that isn't
# writable). Elsewhere it prints where the bundle landed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RES=$ROOT/src-tauri/resources
MODELS_ONLY=0
MEDIUM=1
DEPS=0
for arg in "$@"; do
  case $arg in
    --models-only) MODELS_ONLY=1 ;;
    --no-medium) MEDIUM=0 ;;
    --install-deps) DEPS=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# From rustup's own installer on both platforms: Homebrew's rustup is keg-only,
# so it wouldn't be on PATH.
install_rust() {
  command -v cargo >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  [ ! -f "$HOME/.cargo/env" ] || . "$HOME/.cargo/env"
}

deps_macos() {
  if ! xcode-select -p >/dev/null 2>&1; then
    xcode-select --install
    echo "waiting for the Command Line Tools installer to finish..."
    until xcode-select -p >/dev/null 2>&1; do sleep 5; done
  fi
  if ! command -v brew >/dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
  fi
  command -v cmake >/dev/null || brew install cmake
  command -v bun >/dev/null || brew install oven-sh/bun/bun
  install_rust
}

# whisper.cpp's Vulkan shaders need a newer glslc and Vulkan headers than some
# distros ship (Ubuntu 24.04 has shaderc 2023.8; 2025 and later work). Take
# both from the Vulkan SDK into /usr/local, as scripts/build-linux.sh does; the
# runtime loader stays the system's.
VULKAN_SDK_VERSION=1.4.363.0
vulkan_sdk_tools() {
  local year
  year=$(glslc --version 2>/dev/null | sed -n '1s/^shaderc v\{0,1\}\([0-9]*\).*/\1/p') || true
  [ "${year:-0}" -lt 2025 ] || return 0
  if [ "$(uname -m)" != x86_64 ]; then
    echo "glslc is older than shaderc 2025; install a newer one for whisper.cpp's GPU backend" >&2
    return
  fi
  echo "fetching glslc and Vulkan headers from the Vulkan SDK $VULKAN_SDK_VERSION"
  local v=$VULKAN_SDK_VERSION
  curl -fsSL "https://sdk.lunarg.com/sdk/download/$v/linux/vulkansdk-linux-x86_64-$v.tar.xz" |
    sudo tar -xJ -C /usr/local --strip-components=2 \
      "$v/x86_64/bin/glslc" "$v/x86_64/include/vulkan" "$v/x86_64/include/vk_video"
  hash -r
}

# The same libraries flake.nix provides, plus Tauri's own prerequisites
# (the AppImage bundler needs xdg-open).
deps_linux() {
  if [ -e /etc/NIXOS ]; then
    echo "on NixOS, run this inside \`nix develop\` with Rust and Bun from nixpkgs instead"
    return
  elif command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential curl wget file unzip xz-utils pkg-config cmake clang \
      libclang-dev libssl-dev libgtk-3-dev libwebkit2gtk-4.1-dev libsoup-3.0-dev \
      libayatana-appindicator3-dev librsvg2-dev libxdo-dev libasound2-dev libvulkan-dev glslc \
      vulkan-tools libpipewire-0.3-dev libspa-0.2-dev xdg-utils
  elif command -v dnf >/dev/null; then
    sudo dnf install -y gcc gcc-c++ make curl wget file unzip xz pkgconf-pkg-config cmake clang \
      clang-devel openssl-devel gtk3-devel webkit2gtk4.1-devel libsoup3-devel \
      libappindicator-gtk3-devel librsvg2-devel libxdo-devel alsa-lib-devel vulkan-loader-devel \
      vulkan-headers glslc vulkan-tools pipewire-devel xdg-utils
  elif command -v pacman >/dev/null; then
    sudo pacman -S --needed --noconfirm base-devel curl wget file unzip xz cmake clang \
      openssl gtk3 webkit2gtk-4.1 libsoup3 libayatana-appindicator librsvg xdotool alsa-lib \
      vulkan-icd-loader vulkan-headers shaderc vulkan-tools pipewire xdg-utils
  elif command -v zypper >/dev/null; then
    sudo zypper --non-interactive install gcc gcc-c++ make curl wget file unzip xz pkgconf cmake \
      clang clang-devel libopenssl-devel gtk3-devel webkitgtk3-devel libsoup-devel \
      libayatana-appindicator3-devel librsvg-devel xdotool-devel alsa-devel vulkan-devel \
      vulkan-headers shaderc vulkan-tools pipewire-devel xdg-utils
  else
    echo "no apt, dnf, pacman or zypper; install the libraries flake.nix lists by hand" >&2
  fi
  vulkan_sdk_tools
  install_rust
  if ! command -v bun >/dev/null; then
    curl -fsSL https://bun.sh/install | bash
    export PATH=$HOME/.bun/bin:$PATH
  fi
}

if [ $DEPS = 1 ]; then
  case $(uname -s) in
    Darwin) deps_macos ;;
    Linux) deps_linux ;;
    *) echo "on Windows, use scripts/install.ps1 -InstallDeps" >&2; exit 1 ;;
  esac
fi

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

# Camera models are optional. They come pre-converted from a GitHub release;
# converting them locally (about 250 MB of TensorFlow) is only the fallback.
# errexit is off inside an `if` condition, hence the explicit returns.
VISION_ZIP=https://github.com/MarsonTagawa/speech/releases/download/vision-models-1/vision-models.zip
VISION_SHA=207c1f73308fc50f841b1bef0f5c8c3f5c080f45e227deb2c6e60960783d3887
fetch_vision() {
  command -v unzip >/dev/null || { echo "unzip not found" >&2; return 1; }
  echo "fetching vision/"
  curl -fL --progress-bar -o "$RES/vision-models.zip" "$VISION_ZIP" || return 1
  sum=$( (sha256sum "$RES/vision-models.zip" 2>/dev/null || shasum -a 256 "$RES/vision-models.zip") | cut -d' ' -f1)
  [ "$sum" = "$VISION_SHA" ] || { echo "vision-models.zip has the wrong checksum" >&2; return 1; }
  rm -rf "$RES/vision.part"
  unzip -q "$RES/vision-models.zip" -d "$RES/vision.part" || return 1
  mv "$RES/vision.part" "$RES/vision"
  rm "$RES/vision-models.zip"
}
if [ -d "$RES/vision" ]; then
  echo "have vision/"
elif fetch_vision; then
  :
elif ! "$ROOT/scripts/convert-vision-models.sh"; then
  echo "skipping camera models (the app works audio-only without them)"
fi

[ $MODELS_ONLY = 1 ] && exit 0

for tool in bun cargo cmake; do
  command -v $tool >/dev/null || { echo "$tool not found; run with --install-deps, or see Install in README.md" >&2; exit 1; }
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

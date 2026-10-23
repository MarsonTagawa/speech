#!/usr/bin/env bash
# Builds the Linux bundles inside Ubuntu 24.04 so they run on other distros.
# A build on NixOS links against /nix/store and only runs on NixOS. The floor is
# glibc 2.38+ (Ubuntu 24.04, Fedora 39, Debian 13): ort's prebuilt ONNX Runtime needs it.
#   scripts/build-linux.sh                      # deb, rpm, AppImage
#   scripts/build-linux.sh --bundles appimage   # args go to `tauri build`
# Output: src-tauri/target/docker/release/bundle/
# Uses the host's node_modules and src-tauri/resources (models) as they are.
set -euo pipefail
cd "$(dirname "$0")/.."

# Runs inside the container after `tauri build`. The AppImage bundles Ubuntu 24.04's
# libwayland, which shadows the host's; newer Mesa Vulkan drivers need a newer one
# (wl_fixes_interface), fail to load, and whisper.cpp aborts with no GPU. Drop it so
# the host's is used (every desktop with GTK has it), keeping the original runtime.
if [ "${1:-}" = --fix-appimage ]; then
  for a in "$CARGO_TARGET_DIR"/release/bundle/appimage/*.AppImage; do
    [ -e "$a" ] || continue
    d=$(mktemp -d)
    off=$("$a" --appimage-offset)
    unsquashfs -q -no-progress -o "$off" -d "$d/root" "$a"
    if ls "$d"/root/usr/lib/libwayland-*.so* >/dev/null 2>&1; then
      rm "$d"/root/usr/lib/libwayland-*.so*
      head -c "$off" "$a" > "$d/out"
      mksquashfs "$d/root" "$d/fs" -root-owned -noappend -comp zstd -b 131072 -quiet -no-progress
      cat "$d/fs" >> "$d/out" && chmod +x "$d/out" && mv "$d/out" "$a"
      echo "Removed bundled libwayland from $a"
    fi
    rm -rf "$d"
  done
  exit
fi

docker build -t speech-build - <<'EOF'
FROM oven/bun:1 AS bun
FROM ubuntu:24.04
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ca-certificates pkg-config xz-utils \
      libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev libxdo-dev \
      libssl-dev libasound2-dev libpipewire-0.3-dev libvulkan-dev \
      cmake clang libclang-dev xdg-utils file squashfs-tools \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs | CARGO_HOME=/usr/local/cargo sh -s -- -y --profile minimal \
    && chmod -R a+w /usr/local/rustup /usr/local/cargo
# whisper.cpp's Vulkan backend wants newer glslc and headers than distros ship (old glslc
# fails the coopmat shaders, and vulkan-shaders-gen deadlocks after 16 failed compiles).
# Both are build-time only, so take them from the Vulkan SDK; the runtime loader is the system's.
RUN curl -fsSL https://sdk.lunarg.com/sdk/download/1.4.363.0/linux/vulkansdk-linux-x86_64-1.4.363.0.tar.xz \
    | tar -xJ -C /usr/local --strip-components=2 \
      1.4.363.0/x86_64/bin/glslc 1.4.363.0/x86_64/include/vulkan 1.4.363.0/x86_64/include/vk_video
EOF

# Everything the build writes stays under target/docker, owned by you; the
# cargo registry and the AppImage tools are cached there between runs.
out=/src/src-tauri/target/docker
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD":/src -w /src \
  -e HOME=/tmp \
  -e CARGO_TARGET_DIR="$out" \
  -e CARGO_HOME="$out/cargo-home" \
  -e XDG_CACHE_HOME="$out/cache" \
  -e APPIMAGE_EXTRACT_AND_RUN=1 \
  speech-build sh -c 'bun tauri build "$@" && exec bash /src/scripts/build-linux.sh --fix-appimage' sh "$@"

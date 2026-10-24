# Speech

A local speech-fluency trainer — Monkeytype for talking. Speak, and it transcribes
you live, flags filler words, and scores your pace, pauses, and delivery. Runs
entirely on-device: no audio ever leaves your machine.

Tauri (Rust) + vanilla TypeScript, with whisper.cpp for transcription and Silero
VAD for voice detection.

## What it does

- **Live transcription** as you speak, using a two-pass scheme: a fast tiny.en
  model shows text immediately, then a slower medium.en model re-decodes in the
  background and silently corrects each committed line.
- **Filler detection** — hesitations (`um`, `uh`) flagged verbatim; ambiguous
  discourse markers (`so`, `like`, `you know`) disambiguated per-occurrence with
  an offline POS tagger (compromise) so real fillers get caught but legitimate
  usage doesn't.
- **Delivery metrics** — words-per-minute over voiced time, hesitation pauses,
  loudness, and pitch (median, inflection range, uptalk) from per-utterance
  acoustic analysis.
- **Speeches** — save scripts you're practising and read along while you talk;
  each speech tracks your attempts and quotes the sentences you skip most.
  Optional timer with a big countdown.
- **History** — every session saved locally with its full report (recent and
  starred ones keep per-line detail), plus a profile with progress over time,
  personal bests and streaks.
- **Model choice** — pick the live and correction models in Settings (tiny.en
  up to large-v3-turbo); non-bundled ones download on demand from Hugging Face.

## Install

### Linux

Download the AppImage from the [latest release](https://github.com/MarsonTagawa/speech/releases/latest)
and run it:

```sh
chmod +x Speech_*_amd64.AppImage
./Speech_*_amd64.AppImage
```

On NixOS: `nix run nixpkgs#appimage-run -- Speech_*_amd64.AppImage`. If it
won't start with a FUSE error, install `libfuse2` (`libfuse2t64` on Ubuntu
24.04) or run it with `--appimage-extract-and-run`.

Needs x86_64 with glibc 2.38 or newer (Ubuntu 24.04+, Fedora 39+, Debian 13+).
The speech models are bundled. A Vulkan GPU driver makes transcription fast;
without one it falls back to the CPU, which is slower.

### macOS, Windows, or from source

There's no prebuilt package for macOS or Windows yet, so build it yourself
(macOS needs 13.4 or newer). You'll need:

- [Rust](https://rustup.rs) and [Bun](https://bun.sh)
- Tauri's [platform prerequisites](https://v2.tauri.app/start/prerequisites/)
  (Xcode Command Line Tools on macOS; MSVC build tools and WebView2 on Windows;
  WebKitGTK and friends on Linux)
- CMake and Clang, which build whisper.cpp
- On Linux and Windows, the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
  (headers, loader and `glslc`) for whisper.cpp's GPU backend; macOS uses
  Metal and needs nothing extra

Or let the install script get them: one command installs the build tools,
fetches the models, builds the app and installs it:

```sh
git clone https://github.com/MarsonTagawa/speech && cd speech && scripts/install.sh --install-deps
```

`--install-deps` uses Homebrew on macOS (installing it and the Xcode Command
Line Tools if needed) and apt, dnf or pacman on Linux, which asks for sudo;
Rust comes from rustup either way. It skips what you already have. Leave it
off to only build. On NixOS, `nix develop` provides everything except Rust and
Bun.

On Windows, use the PowerShell version, which builds the setup `.exe` and runs
it. `-InstallDeps` first installs everything above with winget (skipping what
you already have):

```powershell
git clone https://github.com/MarsonTagawa/speech; cd speech; powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -InstallDeps
```

On macOS that puts `Speech.app` in `/Applications`. On Linux it prints where
the AppImage landed. Pass `--no-medium` to skip
the 539 MB correction model (you can download it from Settings later). Re-run
it to rebuild; models you already have aren't fetched again. For every bundle
format (`.dmg`, `.msi`, `.deb`, `.rpm`, …), run `bun run tauri build`; they land
in `src-tauri/target/release/bundle/`.

A Linux build links against the host's libraries, and one made on NixOS only
runs on NixOS. For packages that run on other distros, build inside Ubuntu
24.04 with Docker: `scripts/build-linux.sh` (output in
`src-tauri/target/docker/release/bundle/`).

## Models (required, not in git)

The whisper + VAD weights are large binaries, so they're gitignored. Fetch them
into `src-tauri/resources/` before building (anything in there is bundled).
`scripts/install.sh --models-only` fetches all of them, including the camera
models below if you have Nix. To fetch them by hand:

```sh
cd src-tauri/resources
curl -L -O https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin
curl -L -o silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
# optional: default correction model; otherwise download it from Settings
curl -L -O https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en-q5_0.bin
```

tiny.en and the VAD are required. Models downloaded from Settings go to the app
data dir (`<app data>/models/`). All whisper models run on the GPU (Metal on
macOS, Vulkan elsewhere); the
live model is warmed up at launch, the correction model is loaded per pass and
freed when it's done.

The camera's presence grading (optional) runs MediaPipe's face, pose and hand
models in Rust as ONNX. Convert them once (downloads ~250 MB of TensorFlow into
`src-tauri/target/vision-convert/` the first time). The script needs Nix, which
it uses to run `uv`:

```sh
scripts/convert-vision-models.sh            # models → src-tauri/resources/vision/
scripts/convert-vision-models.sh --goldens  # + golden test data for `cargo test -- --ignored`
```

Without them the Video toggle reports "Camera models missing" and the app
works audio-only.

## Develop

```sh
scripts/install.sh --models-only
bun install
bun run tauri dev
```

A Nix flake (`flake.nix`) provides the native build tools and libraries, but
not Rust or Bun; `nix develop` for a shell.

## Build

```sh
bun run tauri build
```

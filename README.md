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
- **Session history** saved locally so you can track progress over time.

## Models (required, not in git)

The whisper + VAD weights are large binaries, so they're gitignored. Fetch them
into `src-tauri/resources/` before building:

```sh
cd src-tauri/resources
curl -L -O https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en-q5_0.bin
curl -L -O https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin
curl -L -o silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

Both whisper models run on the GPU (Vulkan) and are warmed up at launch.

## Develop

```sh
bun install
bun run tauri dev
```

A Nix flake (`flake.nix`) provides the toolchain; `nix develop` for a shell.

## Build

```sh
bun run tauri build
```

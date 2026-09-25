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

## Models (required, not in git)

The whisper + VAD weights are large binaries, so they're gitignored. Fetch them
into `src-tauri/resources/` before building (anything in there is bundled):

```sh
cd src-tauri/resources
curl -L -O https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin
curl -L -o silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
# optional: default correction model; otherwise download it from Settings
curl -L -O https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en-q5_0.bin
```

tiny.en and the VAD are required. Models downloaded from Settings go to the app
data dir (`<app data>/models/`). All whisper models run on the GPU (Vulkan); the
live model is warmed up at launch, the correction model is loaded per pass and
freed when it's done.

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

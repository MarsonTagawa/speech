# Reading off the slides — design

Date: 2026-10-21
Status: approved in conversation, pending spec review

## Problem

A presenter who reads their slides aloud word for word loses the room, and
nothing in a session report says they did it. A first cut (`src/reading.ts`)
already catches it for decks opened in the app: each utterance is compared to
the text of the slide that was showing, and the report gets one "Read off the
slides: N%" line and a tip. Two things are missing:

- **Decks presented in another app** (PowerPoint, Keynote, Google Slides in a
  browser) are invisible to it. Screen replay (`src-tauri/src/screen/`) already
  records what's on that display, but only as JPEG frames: nobody reads them.
- **Where it happened.** One session-wide percentage doesn't say which slides
  were read and which were talked through.

## Goals and success criteria

- With **Record screen** on, a session presented from another app gets the same
  reading line, tip and breakdown as one presented from the in-app deck.
- A **per-slide breakdown** in the report: per slide a thumbnail, its number, a
  "% read" bar and how long the speaker spoke on it. A row at or above
  `READ_FRAC` (40%) is marked as read. Clicking a row jumps the transcript and
  the replay to the first utterance on that slide.
- Reading a slide aloud verbatim scores ≥ 80% on that slide; talking around the
  same slide scores ≤ 15% (checked against one real deck per source).
- The report isn't held up: OCR runs during the session, and saving waits at
  most 60 s after the correction pass for any OCR still running.
- OCR doesn't make whisper or the live view stutter (capped threads, see Risks).
- Works on Linux, macOS and Windows with nothing to install. Local-only: the OCR
  models are downloaded once, and nothing else leaves the machine.

## Non-goals

- Catching close paraphrase ("grew forty percent" vs "40%"). Matching stays
  verbatim 3-word runs.
- Live feedback while speaking. Report only (decided in conversation).
- Changing any score. The reading result is a tip and a panel, not a weighted
  dimension, until real numbers say how to weight it.
- OCR for the in-app deck's image slides. Image files dropped into the deck stay
  textless (they score nothing).
- Screen OCR in sessions that used a read-along script: the script is on screen
  and already graded by script accuracy.
- Keeping OCR text after the report is computed. Only the per-slide numbers
  (summary) and thumbnails (detail / `.frames`) persist.

## Approach

**Text source per session**, decided once at summary time:
1. If the in-app deck was showing for any utterance → deck text only (as now).
2. Else, if the session has screen OCR results and no read-along script →
   screen text.
3. Else → no reading result.

Mixing sources in one session is not supported: a recording of the app showing
its own deck is redundant, and the two numbering schemes would collide.

**OCR engine: `ocrs`** (pure Rust, `rten` runtime), chosen in conversation.

Rejected:
- **Tesseract.** Accurate on clean slide text, but a system install on every
  machine (and not on this one), plus a `flake.nix` change.
- **Platform OCR** (macOS Vision, `Windows.Media.Ocr`). Fast and free, but Linux,
  the main platform, has no built-in equivalent.
- **OCR after Stop only.** Simpler (no worker), but a 10-minute talk can have
  20–40 distinct screens at 0.5–2 s each, stacked on top of the medium.en
  correction pass. Doing it during the session hides most of that.

## Phase 1 — breakdown and thumbnails for the in-app deck

Ships on its own; no Rust.

### `reading.ts`

`Said` gains the utterance's timing:

```ts
interface Said { index: number; slide: number; ms: number; dur: number; said: string[]; shown: string[] }
```

`Reading.slides` changes from `number[]` to rows, in slide order:

```ts
interface SlideRead {
  slide: number;   // 1-based
  frac: number;    // 0–1 of this slide's spoken 3-word runs found on it; 0 when too few runs (< 3)
  runs: number;    // spoken runs counted on it (lets the UI show "—" for too few)
  index: number;   // the first utterance on it (transcript line)
  ms: number;      // session ms of that utterance
  spokeMs: number; // summed utterance durations on it
}
interface Reading { frac: number; slides: SlideRead[] }
```

Every slide spoken on gets a row, including slides with no text (`runs: 0`).
"Read" = `runs >= 3 && frac >= READ_FRAC`. The tip names the read slides as
it does today. The session-level `MIN_RUNS` gate is unchanged: below it,
`reading()` returns `null` and no panel is shown.

Older saved sessions have `reading.slides` as `number[]`. The report treats a
non-object entry as "no breakdown" and shows only the summary line.

### `main.ts`

- `slideByIndex` entries are already recorded per utterance. At summary time
  `ms`/`dur` come from `timingByIndex` (`start`, `end - start`).
- **Thumbnails.** When an utterance lands on a slide not yet thumbnailed this
  session, render that slide at 160 px wide to a JPEG data URL (quality 0.7)
  and keep it in `slideThumbs: Map<number, string>` (cleared in `clearLive`).
  `sessionDetail()` writes it as `thumbs: [slide, dataUrl][]`; `parseDetail`
  reads it back (absent → empty).
- **Panel.** A "Slides" panel in the report's bottom row, after Fillers, shown
  when `s.reading?.slides` holds row objects. Each row:
  `[thumb] N  [bar frac]  NN%  m:ss`; "—" instead of a percentage when
  `runs < 3`; a read row gets a marker. Rows carry `data-index` and
  `data-ms` from `index`/`ms`, and the existing Key moments click
  handler is widened to them, so a click jumps the transcript and replay.
- The "Slides" sub-line under Word habits is replaced by the panel's header
  ("Slides · read N%").

### `deck.ts`

`thumbDeck(n, width): Promise<string>` renders slide `n` to an off-screen
canvas and returns a JPEG data URL. It uses its own render task, not the shared
`task` that `paintDeck` cancels, so a slide change never cancels a thumbnail
(and vice versa).

## Phase 2 — screen OCR

### OCR worker (`src-tauri/src/screen/ocr.rs`)

- **Feed.** `screen_sample` already calls `Recorder::screen`, which returns
  whether the frame changed and was written. On `Ok(true)` it also hands
  `(ms, w, h, rgb)` (the capture's full-resolution frame, before the recorder's
  1280-px downscale) to the session's OCR worker through an unbounded channel.
  The recorder is change-driven at ≤ 4 Hz, so the queue only grows while
  the screen is busy.
- **Skip transients.** Before running a frame, the worker drains the channel;
  a frame with a newer one queued less than 1 s after it is dropped
  (scrolling, animations, slide transitions). Pure helper `keep(ms: &[u32]) ->
  Vec<bool>`, unit-tested.
- **Run.** ocrs detects and recognises text; the worker keeps
  `Vec<(u32 ms, String text)>` for the session (lines joined with "\n").
- **Lifecycle.** `begin` spawns the worker when a capture is live and the models
  are on disk; `end` closes its channel, and the worker finishes what's queued
  and exits. Results go to a `static OCR: Mutex<Option<(i64 ts, JoinHandle)>>`.
- **Command.** `screen_text(ts) -> Vec<(u32, String)>`: joins that session's
  worker, waiting at most 60 s, and returns its results; empty for any other
  `ts`, with no worker, or on timeout (whatever finished in time is kept: the
  worker pushes into a shared `Arc<Mutex<Vec<…>>>`).
- **Failure.** A frame that fails OCR is logged (`eprintln!`) and skipped. A
  worker panic leaves an empty result. Neither touches recording.

### Models

- `text-detection.rten` and `text-recognition.rten` (~12 MB total) from the
  ocrs model bucket into `<app data>/models/`. The exact URLs and the `ocrs`
  version are pinned in plan Task 1 (crates.io and the bucket weren't reachable
  from the design session).
- `models.rs`'s `fetch` is generalised to take a URL and destination; the
  whisper catalog keeps using it unchanged.
- Downloaded in the background by `screen_prepare` when missing (the toggle
  turning on). A failed download is logged; sessions run without OCR and retry
  on the next toggle.

### JS side

- `saveSession()` already waits for `corrections_done`. Before
  `computeSummary()` it awaits `invoke("screen_text", { ts })` when the session
  recorded the screen, and stores the result in `screenTexts` (cleared in
  `clearLive`). The live dock's `computeSummary()` calls simply don't have it.
- **Remove the app's own transcript.** If the app shares the recorded display,
  its transcript is on screen. For each frame, drop runs that also appear in
  the transcript spoken before that frame's `ms`. This keeps the app's lines
  from both matching speech and making every frame look like one slide.
- **Group frames into slides** (pure, in `reading.ts`):
  `groupScreens(frames: { ms: number; text: string[] }[]): { ms: number; slide: number }[]`.
  A frame with < 3 tokens is blank (no slide). A frame joins the earlier slide
  whose runs it shares most with, if that share (|A∩B| / min(|A|, |B|)) is
  ≥ 0.5; otherwise it starts a new slide, numbered by first appearance.
  A slide's text is its first frame's text.
- **Per utterance:** the last frame at or before the utterance's `start` (so
  its own transcript can't be on screen yet) gives `slide` and `shown`; blank or
  no frame → not scored. Then `reading()` as in Phase 1.
- **Thumbnails** for screen rows come from `session_frame(ts, SCREEN, row.ms)`,
  loaded lazily when the panel renders. A missing `.frames` file (pruned) →
  rows without images.

## Error handling summary

| Situation | Result |
|---|---|
| Models not downloaded / download failed | No screen reading; report as without screen |
| OCR still running after 60 s | Partial results used |
| A frame fails OCR | Skipped, logged |
| Deck thumbnail render fails | Row without image |
| Detail or `.frames` pruned (old session) | Rows without images; numbers from the summary |
| Old summary with `slides: number[]` | Summary line only, no panel |

## Testing

- `src/reading.check.ts` (bun):
  - per-slide rows: `frac`, `runs`, `ms`, `spokeMs`, slides with no text get
    `runs: 0`;
  - `groupScreens`: same text twice → one slide; a different slide → a new one;
    returning to slide 1 → number 1 again; blank frames skipped;
  - frame lookup at or before an utterance start;
  - transcript-run removal.
- `screen/ocr.rs` unit tests:
  - `keep` drops frames superseded within 1 s and keeps the last one;
  - one OCR of a small rendered text image, `#[ignore]`d unless the models are
    on disk (`cargo test -- --ignored`).
- Rust builds and tests run with `CARGO_TARGET_DIR` in the scratchpad, so the
  user's running `tauri dev` keeps linking.
- **Manual, per phase:**
  1. In-app deck: read two slides verbatim and talk around two; check the panel,
     thumbnails, click-to-jump, and that a reopened session shows the same.
  2. Google Slides in a browser on the recorded display, app on another
     display; then the same with the app on the recorded display.

## Risks and tuning

- **CPU contention.** `rten` uses all cores by default. Cap OCR to 2 threads
  (Task 1 finds the knob: `RTEN_NUM_THREADS` or an explicit thread pool) and
  check whisper's live latency doesn't move with OCR busy.
- **OCR quality at slide sizes.** Small body text on a 4K display may still be
  fine at full resolution; verify on a real deck before tuning.
- **Thresholds are guesses**: `READ_FRAC` 0.4, `MIN_RUNS` 10, grouping share 0.5,
  transient 1 s. Marked with `ponytail:` comments; tune from real sessions.
- **Detail size.** Deck thumbnails add roughly 5 KB per slide to
  `sessions/<ts>.json`, which is kept only for recent and starred sessions.

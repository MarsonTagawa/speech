# Reading off the Slides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A per-slide "read off the slides" breakdown with thumbnails in the session report, fed by the in-app deck's text or, for decks shown in other apps, by OCR of the screen recording.

**Architecture:** `src/reading.ts` (pure) scores each utterance against the text that was on screen and groups the results per slide. Phase 1 feeds it from the in-app deck (`src/deck.ts`) and adds thumbnails and the report panel. Phase 2 adds an `ocrs` worker beside the screen recorder (`src-tauri/src/screen/ocr.rs`) whose results JS fetches at Stop and feeds through the same scorer.

**Tech Stack:** TypeScript (Vite, bun for checks), pdf.js, Rust (Tauri 2), `ocrs` + `rten` (new), existing `ureq` downloader.

**Spec:** `docs/superpowers/specs/2026-10-21-screen-reading-design.md`

## Global Constraints

- Report only: no live cue, no change to any score or `SCORE_WEIGHTS`.
- `READ_FRAC = 0.4`, `MIN_RUNS = 10`, a row needs `runs >= 3` to be judged, grouping share `0.5`, transient `1000` ms, OCR wait cap `60` s, deck thumbnails `160` px wide JPEG quality `0.7`.
- One text source per session: deck if it showed for any utterance; else screen OCR if there are results and no read-along script; else none.
- Matching is verbatim 3-word runs with hesitations skipped (`isHesitation` from `src/crutch.ts`).
- Local-only: the only network use is downloading `text-detection.rten` and `text-recognition.rten` from `https://ocrs-models.s3-accelerate.amazonaws.com/` (verified 200 OK, 2.5 MB + 9.7 MB, 2026-10-21).
- OCR never fails or slows a recording: errors are logged with `eprintln!("[ocr] …")` and dropped.
- Rust builds/tests: `nix develop -c bash -c 'cd src-tauri && CARGO_TARGET_DIR=$SCRATCH/target cargo …'` where `SCRATCH=/tmp/claude-1000/-home-marsont-speech/2c47080e-67db-4b13-b4e4-b4a013e0d711/scratchpad`. Never build into `src-tauri/target`: the user's `tauri dev` links from it.
- TS checks: `bun src/<name>.check.ts`; typecheck: `bunx tsc --noEmit -p .`.
- Commits: the working tree holds the user's unrelated uncommitted work in the same files (`main.ts`, `deck.ts`, `styles.css`, `Cargo.toml`…). Commit only if the user asks, and then never add a `Co-Authored-By` or other Claude attribution line. Each task's last step is a checkpoint, not a commit.
- Match surrounding style: comment density, `ponytail:` comments on tuning guesses, `$()`/`escapeHtml` helpers in `main.ts`.

## Review Focus

1. **Reopening a session after the deck was closed or a new one loaded.** Thumbnails and numbers must come from what was saved (summary + detail), never from the live deck. Pinned in Task 3 (`slidesPanelHtml` takes thumbs from `Detail`).
2. **Summaries saved by the first cut**, where `reading.slides` is `number[]`. The report must show the line and no panel, never throw. Pinned in Task 1 (`slideRows` test).
3. **The app on the recorded display.** Its transcript must neither count as slide text nor merge every frame into one slide. Pinned in Task 6 (`withoutSpoken` + grouping test with a shared transcript).
4. **Stop, then start a new session before OCR finishes.** The old session's results must not leak into the new one, and the old session still saves. Pinned in Task 5 (`results` for another ts is empty) and Task 6 (ts guard).
5. **A busy screen (video playing) for minutes.** OCR memory stays bounded and the slide you land on is still read. Pinned in Task 5 (`push` caps the queue and keeps the newest).

---

## Phase 1: in-app deck

### Task 1: Per-slide rows in `reading()`

**Files:**
- Modify: `src/reading.ts` (whole file)
- Modify: `src/reading.check.ts` (whole file)
- Modify: `src/main.ts`: `computeSummary` (the `reading:` property), `slidesList`, the reading tip in `generateTips`, the `habits` line in `reportBodyHtml`

**Interfaces:**
- Produces:
  - `interface Said { index: number; slide: number; ms: number; dur: number; said: string[]; shown: string[] }`
  - `interface SlideRead { slide: number; frac: number; runs: number; index: number; ms: number; spokeMs: number }`
  - `interface Reading { frac: number; slides: SlideRead[]; screen?: true }`
  - `reading(utts: Said[]): Reading | null`
  - `isRead(r: SlideRead): boolean`
  - `slideRows(r: Reading): SlideRead[]`: `[]` for the old `number[]` shape
  - `runsOf(ws: string[]): string[]`
  - `READ_FRAC`

- [ ] **Step 1: Write the failing check**

Replace `src/reading.check.ts` with:

```ts
// Run: bun src/reading.check.ts
import assert from "node:assert";
import { type Reading, isRead, reading, slideRows } from "./reading";

const w = (s: string) => s.toLowerCase().split(/\s+/).filter(Boolean);
const shown = w("Revenue grew 40 percent in Q3 driven by enterprise renewals");
const verbatim = w("revenue grew 40 um percent in q3 driven by enterprise renewals");
const around = w("so the big story this quarter is that our biggest customers kept coming back");
const at = (index: number, slide: number, ms: number, said: string[], s = shown) => ({ index, slide, ms, dur: 2000, said, shown: s });

// Slide 2 read verbatim (an "um" doesn't break a run) over two lines; slide 3 talked around.
const r = reading([at(1, 1, 0, verbatim), at(2, 1, 2500, verbatim), at(3, 2, 6000, around)]);
assert.ok(r);
assert.equal(r.frac, 16 / (16 + 12));
assert.deepStrictEqual(r.slides, [
  { slide: 2, frac: 1, runs: 16, index: 1, ms: 0, spokeMs: 4000 },
  { slide: 3, frac: 0, runs: 12, index: 3, ms: 6000, spokeMs: 2000 },
]);
assert.deepStrictEqual(r.slides.map(isRead), [true, false]);

// A slide with no text (image) still gets a row, with no runs; it isn't judged.
const img = reading([at(1, 1, 0, verbatim), at(2, 1, 2500, verbatim), at(3, 4, 6000, around, [])]);
assert.deepStrictEqual(img?.slides[1], { slide: 5, frac: 0, runs: 0, index: 3, ms: 6000, spokeMs: 2000 });
assert.equal(isRead(img!.slides[1]), false);

// Too few runs on a slide: frac 0, not judged.
const few = reading([at(1, 1, 0, verbatim), at(2, 1, 2500, verbatim), at(3, 2, 6000, w("revenue grew 40 percent"))]);
assert.deepStrictEqual(few?.slides[1], { slide: 3, frac: 0, runs: 2, index: 3, ms: 6000, spokeMs: 2000 });

// Too little speech overall, or none over text slides: no result.
assert.equal(reading([at(1, 0, 0, w("a b c d e f g h i j k l m n"), [])]), null);
assert.equal(reading([at(1, 1, 0, w("revenue grew 40 percent"))]), null);

// The first cut saved `slides` as slide numbers: no rows, no throw.
assert.deepStrictEqual(slideRows({ frac: 0.5, slides: [2, 3] } as unknown as Reading), []);
assert.equal(slideRows(r).length, 2);
console.log("reading ok");
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `bun src/reading.check.ts`
Expected: FAIL: `isRead`/`slideRows` not exported (SyntaxError), or a deepStrictEqual mismatch on `r.slides`.

- [ ] **Step 3: Implement**

Replace `src/reading.ts` with:

```ts
// Reading off the slides: how much of what was said while a slide showed is
// that slide's own text, word for word, per slide and overall. Pure; checked
// by reading.check.ts.
//
// ponytail: verbatim 3-word runs only, so a close paraphrase ("grew forty
// percent" vs "grew 40%") isn't caught, and image slides (no text) never count.
// Thresholds are guesses; tune against real decks.

import { isHesitation } from "./crutch";

export const READ_FRAC = 0.4; // a slide with at least this share of its runs on it was read
const MIN_RUNS = 10; // fewer runs said over text slides isn't enough to judge
const MIN_SLIDE_RUNS = 3; // nor on one slide

export interface Said {
  index: number; // utterance (transcript line)
  slide: number; // 0-based slide shown when it started
  ms: number; // its start, session ms
  dur: number; // its length, ms
  said: string[]; // normalized spoken tokens
  shown: string[]; // that slide's normalized text tokens
}

export interface SlideRead {
  slide: number; // 1-based
  frac: number; // share of its spoken runs found on it; 0 when too few to judge
  runs: number; // spoken runs counted on it
  index: number; // first utterance on it
  ms: number; // that utterance's start
  spokeMs: number; // summed utterance lengths on it
}

export interface Reading {
  frac: number; // share of runs said over text slides that are on the slide
  slides: SlideRead[]; // every slide spoken on, in slide order
  screen?: true; // slides are recorded screens (OCR), not the in-app deck
}

export const isRead = (r: SlideRead) => r.runs >= MIN_SLIDE_RUNS && r.frac >= READ_FRAC;

// The rows of `r`; none for summaries saved before rows existed (slide numbers).
export const slideRows = (r: Reading) => r.slides.filter((x): x is SlideRead => typeof x === "object");

export function runsOf(ws: string[]): string[] {
  const t = ws.filter((w) => !isHesitation(w));
  return t.slice(2).map((w, i) => `${t[i]} ${t[i + 1]} ${w}`);
}

export function reading(utts: Said[]): Reading | null {
  const per = new Map<number, { hits: number; runs: number; index: number; ms: number; spokeMs: number }>();
  for (const u of utts) {
    const p = per.get(u.slide) ?? { hits: 0, runs: 0, index: u.index, ms: u.ms, spokeMs: 0 };
    per.set(u.slide, p);
    if (u.ms < p.ms) [p.index, p.ms] = [u.index, u.ms];
    p.spokeMs += u.dur;
    if (u.shown.length < 3) continue;
    const onSlide = new Set(runsOf(u.shown));
    const said = runsOf(u.said);
    p.hits += said.filter((r) => onSlide.has(r)).length;
    p.runs += said.length;
  }
  const all = [...per.values()];
  const total = all.reduce((a, p) => a + p.runs, 0);
  if (total < MIN_RUNS) return null;
  return {
    frac: all.reduce((a, p) => a + p.hits, 0) / total,
    slides: [...per]
      .sort((a, b) => a[0] - b[0])
      .map(([n, p]) => ({
        slide: n + 1,
        frac: p.runs >= MIN_SLIDE_RUNS ? p.hits / p.runs : 0,
        runs: p.runs,
        index: p.index,
        ms: p.ms,
        spokeMs: p.spokeMs,
      })),
  };
}
```

- [ ] **Step 4: Run the check**

Run: `bun src/reading.check.ts && bun src/crutch.check.ts`
Expected: `reading ok` and `crutch ok`.

- [ ] **Step 5: Update `main.ts` to the new shape**

Change the import:

```ts
import { type Reading, isRead, reading, slideRows } from "./reading";
```

In `computeSummary()` replace the `reading:` property with:

```ts
    reading: readingResult() ?? undefined,
```

and add above `computeSummary`:

```ts
// Read off the slides (reading.ts), from the in-app deck's text.
function readingResult(): Reading | null {
  return reading(
    [...slideByIndex].map(([index, u]) => {
      const t = timingByIndex.get(index);
      return { ...u, index, ms: t?.start ?? 0, dur: t ? t.end - t.start : 0, said: tokensByIndex.get(index) ?? [] };
    }),
  );
}
```

Replace `slidesList` with:

```ts
// " (slides 3, 4)" for the slides read from, or nothing.
function slidesList(r: Reading): string {
  const read = slideRows(r).filter(isRead).map((x) => x.slide);
  return read.length ? ` (slide${read.length === 1 ? "" : "s"} ${read.join(", ")})` : "";
}
```

In `generateTips` change the reading tip's condition from `s.reading.frac >= READ_FRAC` to:

```ts
  if (s.reading && (s.reading.frac >= READ_FRAC || slideRows(s.reading).some(isRead)))
```

and add `READ_FRAC` back to the import list (`import { type Reading, isRead, reading, READ_FRAC, slideRows } from "./reading";`). Leave the `habits` "Slides" line as is for now (Task 3 replaces it).

- [ ] **Step 6: Typecheck**

Run: `bunx tsc --noEmit -p .`
Expected: no output.

- [ ] **Step 7: Checkpoint**

`git diff --stat -- src/reading.ts src/reading.check.ts src/main.ts`. Commit only if the user asked (no attribution trailer).

---

### Task 2: Deck thumbnails saved with the session

**Files:**
- Modify: `src/deck.ts`: `Slide.draw`, `pdfSlides`, `imageSlide`, `paintDeck`, new `thumbDeck`
- Modify: `src/main.ts`: deck import, `slideByIndex` recording in `updateScriptFromSegment`, `clearLive`, `sessionDetail`, `Detail`, `parseDetail`

**Interfaces:**
- Consumes: `slideByIndex` (already in `main.ts`)
- Produces:
  - `thumbDeck(n: number, width?: number): Promise<string>` (JPEG data URL, `""` if no such slide)
  - `Detail.thumbs: Map<number, string>`, keyed by 0-based slide
  - `slideThumbs: Map<number, string>` (live session)

- [ ] **Step 1: Make thumbnail renders independent of `paintDeck`**

In `src/deck.ts` change the `Slide.draw` signature and the PDF slide so only tracked renders are cancellable:

```ts
interface Slide {
  w: number;
  h: number;
  text: string; // the slide's own words (PDF text layer); "" for images
  // `track`: a render paintDeck may cancel (thumbnails aren't).
  draw: (cx: CanvasRenderingContext2D, scale: number, track: boolean) => Promise<void>;
}
```

```ts
      draw: async (cx, scale, track) => {
        const t = page.render({ canvas: cx.canvas, canvasContext: cx, viewport: page.getViewport({ scale }) });
        if (track) task = t;
        await t.promise;
      },
```

In `paintDeck` change `await slide.draw(cx, s);` to `await slide.draw(cx, s, true);`. The image slide's `draw: async (cx, s) => …` needs no change (the extra argument is ignored).

- [ ] **Step 2: Add `thumbDeck`**

After `deckText` in `src/deck.ts`:

```ts
// Slide `n` at `width` px wide as a JPEG data URL, for the report's slide list.
export async function thumbDeck(n: number, width = 160): Promise<string> {
  const slide = slides[n];
  if (!slide) return "";
  const s = width / slide.w;
  const c = document.createElement("canvas");
  [c.width, c.height] = [width, Math.max(1, Math.round(slide.h * s))];
  const cx = c.getContext("2d");
  if (!cx) return "";
  cx.fillStyle = "#fff"; // JPEG has no alpha; transparent images would go black
  cx.fillRect(0, 0, c.width, c.height);
  await slide.draw(cx, s, false);
  return c.toDataURL("image/jpeg", 0.7);
}
```

- [ ] **Step 3: Thumbnail each slide the first time it's spoken over**

In `src/main.ts` add `thumbDeck` to the deck import. Next to `slideByIndex`:

```ts
// This session's deck thumbnails by slide (data URLs), saved in its detail.
// Replaced, not cleared, by clearLive, so a render finishing late lands in the
// old session's map.
let slideThumbs = new Map<number, string>();
```

In `updateScriptFromSegment`, replace the `slideByIndex.set(...)` statement with:

```ts
  if (!slideByIndex.has(segment.index) && deckSize() && b.contains("deck-on") && !b.contains("no-ribbon")) {
    const slide = deckAt();
    slideByIndex.set(segment.index, { slide, shown: tokenizeWithOffsets(deckText()).map((t) => t.norm) });
    if (!slideThumbs.has(slide)) {
      const thumbs = slideThumbs;
      thumbs.set(slide, "");
      thumbDeck(slide).then((url) => thumbs.set(slide, url), () => {}); // no image → row without one
    }
  }
```

In `clearLive` after `slideByIndex.clear();` add `slideThumbs = new Map();`.

- [ ] **Step 4: Save and load the thumbnails**

In `sessionDetail()` add after `lines: liveLinesHtml(),`:

```ts
    thumbs: [...slideThumbs].filter(([, url]) => url),
```

Extend `Detail` and `parseDetail`:

```ts
interface Detail {
  ts: number;
  L: Lines;
  lines: Array<[number, string]>;
  thumbs: Map<number, string>; // deck slide → thumbnail data URL; empty on older sessions
}
```

```ts
    return { ts, L, lines: d.lines ?? [], thumbs: new Map(d.thumbs ?? []) };
```

- [ ] **Step 5: Typecheck and run checks**

Run: `bunx tsc --noEmit -p . && bun src/reading.check.ts`
Expected: no tsc output; `reading ok`.

- [ ] **Step 6: Checkpoint**

`git diff --stat -- src/deck.ts src/main.ts`. Commit only if asked.

---

### Task 3: The Slides panel in the report

**Files:**
- Modify: `src/main.ts`: new `slidesPanelHtml`, `reportBodyHtml` (signature, `habits`, bottom row), its two callers (`renderReport`, `sessionPageHtml`), the summary-only branch of `sessionPageHtml`, `bindReport` click handler
- Modify: `src/styles.css`: after the `.moment span.note` rule; inside the `@media (max-width: 900px)` block that has `.rep-bottom > :last-child`

**Interfaces:**
- Consumes: `slideRows`, `isRead` (Task 1); `Detail.thumbs` (Task 2); `formatTimestamp`, `escapeHtml` (existing)
- Produces: `slidesPanelHtml(s: SessionSummary, thumbs: Map<number, string>): string`; rows are `.slide-row` buttons with `data-index`, `data-ms`, and (screen sessions) `img.slide-thumb[data-frame]` that Task 6 fills.

- [ ] **Step 1: Add `slidesPanelHtml`**

Next to `crutchHtml` in `src/main.ts`:

```ts
// Per-slide reading breakdown: thumbnail, slide number, % read, time spoken.
// Deck thumbnails come from the session's detail; recorded screens' are
// fetched after render (data-frame, see frameThumbs). Nothing without rows.
function slidesPanelHtml(s: SessionSummary, thumbs: Map<number, string>): string {
  const rows = s.reading ? slideRows(s.reading) : [];
  if (!rows.length) return "";
  const img = (r: SlideRead) => {
    if (s.reading?.screen) return `<img class="slide-thumb" data-frame="${r.ms}" alt="">`;
    const url = thumbs.get(r.slide - 1);
    return url ? `<img class="slide-thumb" src="${url}" alt="">` : `<span class="slide-thumb"></span>`;
  };
  const list = rows
    .map((r, i) => {
      const pct = Math.round(r.frac * 100);
      const judged = r.runs >= 3;
      return (
        `<button type="button" class="slide-row${isRead(r) ? " read" : ""}" style="--i:${i}" data-index="${r.index}" data-ms="${r.ms}" ` +
        `data-tip="Slide ${r.slide} — ${judged ? `${pct}% read word for word` : "too little said to judge"} · jump to ${formatTimestamp(r.ms)}">` +
        `${img(r)}<span class="n">${r.slide}</span><div class="fw-bar"><div style="width:${judged ? pct : 0}%"></div></div>` +
        `<span class="pct">${judged ? `${pct}%` : "—"}</span><span class="t">${formatTimestamp(r.spokeMs)}</span></button>`
      );
    })
    .join("");
  return `<div class="panel slides-panel"><span class="label">Slides · read ${Math.round(s.reading!.frac * 100)}%</span><div class="slide-list">${list}</div></div>`;
}
```

Add `type SlideRead` to the reading import.

- [ ] **Step 2: Put it in the report**

Change `reportBodyHtml(s: SessionSummary, L: Lines, pace: PaceData)` to take `thumbs: Map<number, string>` as a 4th parameter. Replace the `habits` definition with the original:

```ts
  const habits = s.crutch ? crutchHtml(s.crutch) : "";
```

In its return, after the Fillers panel line, add:

```ts
    slidesPanelHtml(s, thumbs) +
```

Update the callers: in `renderReport`, `reportBodyHtml(s, d.L, s.pace!, d.thumbs)`; in `sessionPageHtml`, `reportBodyHtml(s, detail.L, s.pace ?? paceData(detail.L), detail.thumbs)`. In `sessionPageHtml`'s summary-only branch (pruned detail), add `slidesPanelHtml(s, new Map()) +` after the "Score breakdown" panel line, so the numbers still show.

- [ ] **Step 3: Click a row to jump**

In `bindReport`, change

```ts
    const m = t.closest<HTMLElement>(".moment");
```

to

```ts
    const m = t.closest<HTMLElement>(".moment, .slide-row");
```

- [ ] **Step 4: Styles**

After `.moment span.note { … }` in `src/styles.css`:

```css
.rep-bottom:has(> .slides-panel) {
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr) minmax(0, 1.1fr) minmax(0, 1.2fr);
}
.slide-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow-y: auto;
}
.slide-row {
  display: grid;
  grid-template-columns: 48px 18px 1fr 34px 34px;
  align-items: center;
  gap: 8px;
  padding: 0;
  border: none;
  background: none;
  font: inherit;
  font-size: 12px;
  color: inherit;
  text-align: left;
  cursor: pointer;
}
.slide-thumb {
  width: 48px;
  aspect-ratio: 16 / 9;
  object-fit: contain;
  border-radius: 3px;
  background: var(--edge);
}
.slide-row .n,
.slide-row .pct,
.slide-row .t {
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.slide-row .t {
  color: var(--ink-2);
}
.slide-row .fw-bar div {
  background: var(--blue);
}
.slide-row.read .fw-bar div {
  background: var(--orange);
}
.slide-row.read .pct {
  color: var(--orange);
  font-weight: 600;
}
```

Inside the `@media (max-width: 900px)` block, after `.rep-bottom > :last-child { grid-column: 1 / -1; }`:

```css
  .rep-bottom:has(> .slides-panel) {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .rep-bottom:has(> .slides-panel) > :last-child {
    grid-column: auto;
  }
```

(The `max-width: 600px` rule already makes `.rep-bottom` one column.)

- [ ] **Step 5: Typecheck**

Run: `bunx tsc --noEmit -p .`
Expected: no output.

- [ ] **Step 6: Manual check in the app (user runs `tauri dev`)**

Ask the user to: open a PDF deck in the app, record about a minute reading two slides verbatim and talking around two others, then Stop. Expect: a "Slides · read N%" panel with four rows and thumbnails, the two read rows marked in orange; clicking a row flashes that slide's first transcript line. Then open the session from History: same panel. Then load a different deck and reopen it: thumbnails unchanged. Narrow the window below 900 px: panels in a 2×2 grid.

- [ ] **Step 7: Checkpoint**

Commit only if asked. **Phase 1 is shippable here.**

---

## Phase 2: screen OCR

### Task 4: OCR engine and model download

**Files:**
- Modify: `src-tauri/Cargo.toml`: add `ocrs`, `rten` under `[dependencies]`
- Create: `src-tauri/src/screen/ocr.rs` (engine part only in this task)
- Modify: `src-tauri/src/screen/mod.rs`: `mod ocr;`; `screen_prepare` starts the model download
- Modify: `src-tauri/src/models.rs`: `download_model`/`fetch` generalised; new `ocr_models`
- Modify: `src-tauri/src/lib.rs`: OCR thread cap at the top of `run()`
- Create: `src-tauri/resources/ocr-test/slide.jpg` (test fixture)

**Interfaces:**
- Produces:
  - `ocr::DETECTION = "text-detection.rten"`, `ocr::RECOGNITION = "text-recognition.rten"`, `ocr::MODEL_URL`
  - `ocr::Engine::load(dir: &Path) -> Result<Engine, String>`
  - `Engine::text(&self, w: usize, h: usize, rgb: &[u8]) -> Result<String, String>`
  - `models::ocr_dir(app: &AppHandle) -> Result<PathBuf, String>` (where the models live; no download)
  - `models::ocr_models(app: &AppHandle) -> Result<PathBuf, String>` (blocking; downloads missing ones)

- [ ] **Step 1: Pin the crates and read their API**

Run (network needed; crates.io was unreachable from the design session's sandbox, so run unsandboxed if it fails):

```bash
nix develop -c bash -c 'cd src-tauri && cargo add ocrs rten'
grep -A2 '^name = "ocrs"\|^name = "rten"$' src-tauri/Cargo.lock
```

Then pin exact versions in `Cargo.toml` with a comment, e.g.:

```toml
# Screen OCR (screen/ocr.rs): reading-off-the-slides from the screen replay.
ocrs = "=X.Y.Z"
rten = "=X.Y.Z" # must match the version ocrs depends on
```

Read the pinned API: `ls ~/.cargo/registry/src/*/ocrs-*/examples/` and the README there. Confirm `OcrEngine::new(OcrEngineParams { detection_model, recognition_model, .. })`, `ImageSource::from_bytes(&[u8], (u32, u32))`, `prepare_input`, `get_text`. If names differ in the pinned version, adapt Step 3's code to them, keeping the `Engine` interface.

Find the thread-count knob: `grep -rn "RTEN_NUM_THREADS\|fn thread_pool" ~/.cargo/registry/src/*/rten-*/src | head`.

- [ ] **Step 2: Make the test fixture**

```bash
mkdir -p src-tauri/resources/ocr-test
font=$(fc-match -f '%{file}' 'DejaVu Sans:bold')
nix run nixpkgs#imagemagick -- -size 1280x720 xc:white -fill black -font "$font" \
  -pointsize 56 -annotate +80+160 'Quarterly results' \
  -pointsize 40 -annotate +80+280 'Revenue grew 40 percent in Q3' \
  -annotate +80+360 'Driven by enterprise renewals' -quality 90 src-tauri/resources/ocr-test/slide.jpg
```

Open the image (Read tool) and check the three lines are rendered.

- [ ] **Step 3: Write the failing (ignored) test and the engine**

Create `src-tauri/src/screen/ocr.rs`:

```rust
//! Screen OCR for "reading off the slides": the text in each changed screen
//! frame (ocrs), so the report can compare speech to what was on screen. See
//! docs/superpowers/specs/2026-10-21-screen-reading-design.md.
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;
use std::path::Path;

pub const DETECTION: &str = "text-detection.rten";
pub const RECOGNITION: &str = "text-recognition.rten";
pub const MODEL_URL: &str = "https://ocrs-models.s3-accelerate.amazonaws.com/";

pub struct Engine(OcrEngine);

impl Engine {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let model = |f: &str| Model::load_file(dir.join(f)).map_err(|e| format!("{f}: {e}"));
        let params = OcrEngineParams {
            detection_model: Some(model(DETECTION)?),
            recognition_model: Some(model(RECOGNITION)?),
            ..Default::default()
        };
        OcrEngine::new(params).map(Engine).map_err(|e| e.to_string())
    }

    /// The text in a packed-RGB frame, lines joined with "\n".
    pub fn text(&self, w: usize, h: usize, rgb: &[u8]) -> Result<String, String> {
        let src = ImageSource::from_bytes(rgb, (w as u32, h as u32)).map_err(|e| e.to_string())?;
        let input = self.0.prepare_input(src).map_err(|e| e.to_string())?;
        self.0.get_text(&input).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models() -> std::path::PathBuf {
        // Where the app downloads them (Linux app data dir).
        dirs_home().join(".local/share/com.marsont.speech/models")
    }
    fn dirs_home() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("HOME").unwrap())
    }

    // Needs the models: turn Record screen on once in the app, then
    // `cargo test screen::ocr -- --ignored`.
    #[test]
    #[ignore]
    fn reads_a_rendered_slide() {
        let img = image::open(concat!(env!("CARGO_MANIFEST_DIR"), "/resources/ocr-test/slide.jpg")).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let t0 = std::time::Instant::now();
        let text = Engine::load(&models()).unwrap().text(w, h, img.as_raw()).unwrap().to_lowercase();
        eprintln!("[ocr] {} ms: {text:?}", t0.elapsed().as_millis());
        assert!(text.contains("revenue grew 40 percent in q3"), "{text}");
        assert!(text.contains("driven by enterprise renewals"), "{text}");
    }
}
```

Add `mod ocr;` to `src-tauri/src/screen/mod.rs` next to `mod frames;`.

- [ ] **Step 4: Generalise the downloader and add `ocr_models`**

In `src-tauri/src/models.rs` replace `download_model` and `fetch` with:

```rust
/// Downloads a model, emitting `model_download` progress events. Writes to a
/// `.part` file and renames on success, so a failed or partial download never
/// shows up as installed.
#[tauri::command]
pub async fn download_model(app: AppHandle, id: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let f = file(&id)?;
        download(&app, &format!("{BASE_URL}{f}"), &downloaded(&app, f)?, Some(&id))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// `url` to `dest` via a `.part` file. Progress events only for a settings-page
/// model (`progress` = its id).
fn download(app: &AppHandle, url: &str, dest: &PathBuf, progress: Option<&str>) -> Result<(), String> {
    std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
    let part = dest.with_extension("part");
    let result = fetch(app, url, &part, progress).and_then(|_| std::fs::rename(&part, dest).map_err(|e| e.to_string()));
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result
}

/// Where the screen-OCR models (screen/ocr.rs) live; they may not be there yet.
pub fn ocr_dir(app: &AppHandle) -> Result<PathBuf, String> {
    downloaded(app, crate::screen::ocr::DETECTION).map(|p| p.parent().unwrap().to_path_buf())
}

/// The screen-OCR models' directory, downloading any that are missing. Blocking.
pub fn ocr_models(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = ocr_dir(app)?;
    for f in [crate::screen::ocr::DETECTION, crate::screen::ocr::RECOGNITION] {
        let dest = dir.join(f);
        if !dest.exists() {
            download(app, &format!("{}{f}", crate::screen::ocr::MODEL_URL), &dest, None)?;
        }
    }
    Ok(dir)
}

fn fetch(app: &AppHandle, url: &str, part: &PathBuf, progress: Option<&str>) -> Result<(), String> {
    // ureq defaults to rustls; only native-tls is compiled in (see Cargo.toml).
    let tls = ureq::tls::TlsConfig::builder().provider(ureq::tls::TlsProvider::NativeTls).build();
    let agent: ureq::Agent = ureq::Agent::config_builder().tls_config(tls).build().into();
    let mut resp = agent.get(url).call().map_err(|e| e.to_string())?;
    let total: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok()?.parse().ok())
        .unwrap_or(0);
    let mut reader = resp.body_mut().as_reader();
    let mut out = std::fs::File::create(part).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 1 << 16];
    let (mut done, mut emitted) = (0u64, 0u64);
    let emit = |done, total| {
        if let Some(id) = progress {
            let _ = app.emit("model_download", Progress { id: id.into(), done, total });
        }
    };
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        done += n as u64;
        // ~1MB granularity keeps the event rate sane on a fast connection.
        if done - emitted >= 1 << 20 {
            emitted = done;
            emit(done, total);
        }
    }
    if total > 0 && done != total {
        return Err(format!("download of {url} incomplete ({done} of {total} bytes)"));
    }
    out.sync_all().map_err(|e| e.to_string())?;
    emit(done, done);
    Ok(())
}
```

`screen::ocr` must be reachable from `models.rs`: make it `pub(crate) mod ocr;` in `screen/mod.rs`.

- [ ] **Step 5: Download when Record screen turns on**

In `screen_prepare` (`src-tauri/src/screen/mod.rs`), after `app.state::<ScreenState>().release();`:

```rust
    {
        let app = app.clone(); // models for the OCR worker; sessions without them just skip OCR
        thread::spawn(move || {
            if let Err(e) = crate::models::ocr_models(&app) {
                eprintln!("[ocr] model download: {e}");
            }
        });
    }
```

- [ ] **Step 6: Cap OCR threads**

If Step 1 found that rten reads `RTEN_NUM_THREADS`, add at the top of `pub fn run()` in `src-tauri/src/lib.rs` (before the builder, while the process is single-threaded):

```rust
    // Screen OCR (rten) would take every core from whisper and the camera.
    if std::env::var_os("RTEN_NUM_THREADS").is_none() {
        std::env::set_var("RTEN_NUM_THREADS", "2");
    }
```

If the knob is an API instead (a thread-pool constructor), use it in `Engine::load` with 2 threads, and note which in a comment.

- [ ] **Step 7: Build, then run the ignored test**

Run: `nix develop -c bash -c 'cd src-tauri && CARGO_TARGET_DIR=$SCRATCH/target cargo build'`
Expected: builds.

Fetch the models once for the test (the same URLs the app uses):

```bash
d=~/.local/share/com.marsont.speech/models; mkdir -p $d
for f in text-detection.rten text-recognition.rten; do [ -f $d/$f ] || curl -fL -o $d/$f https://ocrs-models.s3-accelerate.amazonaws.com/$f; done
```

Run: `nix develop -c bash -c 'cd src-tauri && CARGO_TARGET_DIR=$SCRATCH/target cargo test --release screen::ocr -- --ignored --nocapture'`
Expected: PASS; note the printed ms (the spec's 0.5–2 s budget). If it fails on the text, print it and adjust only the assertion's expected casing or spacing, not the engine.

- [ ] **Step 8: Checkpoint**

Commit only if asked.

---

### Task 5: OCR worker per session and `screen_text`

**Files:**
- Modify: `src-tauri/src/screen/ocr.rs`: queue, worker, `start`/`feed`/`close`/`results`
- Modify: `src-tauri/src/screen/mod.rs`: `begin`, `end`, `screen_sample`, new command `screen_text`
- Modify: `src-tauri/src/lib.rs`: register `screen::screen_text`

**Interfaces:**
- Consumes: `Engine` (Task 4), `models::ocr_dir` (Task 4), `frames::resize` (existing)
- Produces:
  - `ocr::start(ts: i64, models: PathBuf)`, `ocr::feed(ms: u32, w: usize, h: usize, rgb: &[u8])`, `ocr::close()`, `ocr::results(ts: i64, wait: Duration) -> Vec<(u32, String)>`
  - Tauri command `screen_text(ts: i64) -> Vec<(u32, String)>` (JS: `invoke<[number, string][]>("screen_text", { ts })`)

- [ ] **Step 1: Write the failing queue tests**

Append to the `tests` module in `ocr.rs`:

```rust
    fn f(ms: u32) -> Frame {
        Frame { ms, w: 1, h: 1, rgb: vec![0; 3], at: std::time::Instant::now() }
    }
    fn ms(q: &VecDeque<Frame>) -> Vec<u32> {
        q.iter().map(|f| f.ms).collect()
    }

    // A frame replaced within TRANSIENT_MS is a transient: the newer one takes its place.
    #[test]
    fn push_replaces_a_transient_tail() {
        let mut q = VecDeque::new();
        for m in [0, 1500, 1900, 2300, 4000] {
            push(&mut q, f(m));
        }
        assert_eq!(ms(&q), [0, 2300, 4000]);
    }

    // Review focus 5: a busy screen can't grow the queue without bound, and the
    // newest frame (where the screen settles) is the one kept.
    #[test]
    fn push_caps_the_queue_keeping_the_newest() {
        let mut q = VecDeque::new();
        for m in (0..20).map(|i| i * 2000) {
            push(&mut q, f(m));
        }
        assert_eq!(q.len(), QUEUE_CAP);
        assert_eq!(q.back().unwrap().ms, 38000);
    }

    // Review focus 4: a new session's results never include the last one's.
    #[test]
    fn results_are_per_session() {
        let _g = TEST_LOCK.lock().unwrap();
        start_with(1, |_, _, _| Ok("slide one".into()));
        feed(0, 1, 1, &[0, 0, 0]);
        close();
        assert_eq!(results(1, Duration::from_secs(5)), [(0, "slide one".to_string())]);
        assert!(results(2, Duration::from_secs(1)).is_empty());
    }

    // The worker skips a frame that fails and keeps going.
    #[test]
    fn a_failed_frame_is_skipped() {
        let _g = TEST_LOCK.lock().unwrap();
        start_with(3, |w, _, _| if w == 2 { Err("bad".into()) } else { Ok("ok".into()) });
        feed(0, 2, 1, &[0; 6]);
        feed(5000, 1, 1, &[0; 3]);
        close();
        assert_eq!(results(3, Duration::from_secs(5)), [(5000, "ok".to_string())]);
    }

    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
```

Run: `nix develop -c bash -c 'cd src-tauri && CARGO_TARGET_DIR=$SCRATCH/target cargo test screen::ocr'`
Expected: FAIL to compile (`Frame`, `push`, `QUEUE_CAP`, `start_with`, `feed`, `close`, `results` not found).

- [ ] **Step 2: Implement the queue and worker**

Add to `ocr.rs` (below `Engine`), and extend the `use` lines with `std::collections::VecDeque`, `std::path::PathBuf`, `std::sync::{Arc, Condvar, Mutex}`, `std::thread::{self, JoinHandle}`, `std::time::{Duration, Instant}`:

```rust
/// ponytail: a frame replaced within this long is a transient (scroll,
/// animation, slide transition) and isn't read; tune from real sessions.
const TRANSIENT_MS: u32 = 1000;
/// ponytail: frames waiting for OCR at most (≤ ~6 MB each at OCR_MAX_W); a
/// screen busy for longer than OCR keeps up loses its oldest frames.
const QUEUE_CAP: usize = 8;
/// Frames are read at most this wide: slide text stays legible, memory bounded.
const OCR_MAX_W: usize = 1920;

pub struct Frame {
    ms: u32,
    w: usize,
    h: usize,
    rgb: Vec<u8>,
    at: Instant, // when it was queued (a transient is judged in wall time)
}

/// Queues `f`: it replaces a tail it follows within TRANSIENT_MS, and the
/// oldest frame goes when the queue is over QUEUE_CAP.
fn push(q: &mut VecDeque<Frame>, f: Frame) {
    if q.back().is_some_and(|b| f.ms.saturating_sub(b.ms) < TRANSIENT_MS) {
        q.pop_back();
    }
    q.push_back(f);
    if q.len() > QUEUE_CAP {
        q.pop_front();
    }
}

struct Queue {
    frames: VecDeque<Frame>,
    open: bool,
}

struct Job {
    ts: i64,
    queue: Arc<(Mutex<Queue>, Condvar)>,
    out: Arc<Mutex<Vec<(u32, String)>>>,
    thread: JoinHandle<()>,
}

/// The current (or last) session's OCR, past any poisoning like screen's SESSION.
static JOB: Mutex<Option<Job>> = Mutex::new(None);
fn job() -> std::sync::MutexGuard<'static, Option<Job>> {
    JOB.lock().unwrap_or_else(|e| e.into_inner())
}

/// Starts session `ts`'s worker with the models in `models` (loaded on the
/// worker, so the session start never waits). No models → no worker.
pub fn start(ts: i64, models: PathBuf) {
    if !models.join(DETECTION).exists() || !models.join(RECOGNITION).exists() {
        return;
    }
    start_with(ts, move |w, h, rgb| {
        thread_local!(static ENGINE: std::cell::OnceCell<Result<Engine, String>> = const { std::cell::OnceCell::new() });
        ENGINE.with(|e| match e.get_or_init(|| Engine::load(&models)) {
            Ok(e) => e.text(w, h, rgb),
            Err(err) => Err(err.clone()),
        })
    });
}

fn start_with(ts: i64, read: impl Fn(usize, usize, &[u8]) -> Result<String, String> + Send + 'static) {
    close(); // a previous session's worker finishes on its own
    let queue = Arc::new((Mutex::new(Queue { frames: VecDeque::new(), open: true }), Condvar::new()));
    let out = Arc::new(Mutex::new(Vec::new()));
    let thread = {
        let (queue, out) = (queue.clone(), out.clone());
        thread::spawn(move || {
            while let Some(f) = next(&queue) {
                match read(f.w, f.h, &f.rgb) {
                    Ok(text) => out.lock().unwrap().push((f.ms, text)),
                    Err(e) => eprintln!("[ocr] frame at {} ms: {e}", f.ms),
                }
            }
        })
    };
    *job() = Some(Job { ts, queue, out, thread });
}

/// The next frame to read: the oldest, once it can't be a transient any more
/// (a newer frame follows it, it's TRANSIENT_MS old, or the session ended).
/// None once the session ended and the queue is empty.
fn next(queue: &(Mutex<Queue>, Condvar)) -> Option<Frame> {
    let (lock, cv) = queue;
    let mut q = lock.lock().unwrap();
    loop {
        let wait = match q.frames.front() {
            None if !q.open => return None,
            None => None,
            Some(f) => {
                let left = Duration::from_millis(TRANSIENT_MS as u64).saturating_sub(f.at.elapsed());
                if q.frames.len() > 1 || !q.open || left.is_zero() {
                    return q.frames.pop_front();
                }
                Some(left)
            }
        };
        q = match wait {
            Some(t) => cv.wait_timeout(q, t).unwrap().0,
            None => cv.wait(q).unwrap(),
        };
    }
}

/// A changed screen frame of the running session (screen_sample). Cheap when
/// no worker runs.
pub fn feed(ms: u32, w: usize, h: usize, rgb: &[u8]) {
    let g = job();
    let Some(j) = g.as_ref() else { return };
    let (lock, cv) = &*j.queue;
    let mut q = lock.lock().unwrap();
    if !q.open {
        return;
    }
    let (nw, nh) = if w > OCR_MAX_W { (OCR_MAX_W, h * OCR_MAX_W / w) } else { (w, h) };
    let rgb = if nw == w { rgb.to_vec() } else { super::frames::resize(rgb, w, h, nw, nh) };
    push(&mut q.frames, Frame { ms, w: nw, h: nh, rgb, at: Instant::now() });
    cv.notify_one();
}

/// The session ended: the worker reads what's queued, then exits.
pub fn close() {
    if let Some(j) = job().as_ref() {
        j.queue.0.lock().unwrap().open = false;
        j.queue.1.notify_one();
    }
}

/// Session `ts`'s text by frame ms, in time order, waiting up to `wait` for
/// its worker to finish; whatever's done by then otherwise. Empty for any
/// other session.
pub fn results(ts: i64, wait: Duration) -> Vec<(u32, String)> {
    let deadline = Instant::now() + wait;
    loop {
        let g = job();
        let Some(j) = g.as_ref().filter(|j| j.ts == ts) else { return Vec::new() };
        if j.thread.is_finished() || Instant::now() >= deadline {
            let mut out = j.out.lock().unwrap().clone();
            out.sort_by_key(|r| r.0);
            return out;
        }
        drop(g);
        thread::sleep(Duration::from_millis(50));
    }
}
```

(The engine lives in a thread-local because each session's worker is a fresh thread: it loads once per session, off the audio path. Note this in a `ponytail:` comment if load time in Task 4's test was > 300 ms: "keep one engine across sessions if load time shows".)

- [ ] **Step 3: Run the queue tests**

Run: `nix develop -c bash -c 'cd src-tauri && CARGO_TARGET_DIR=$SCRATCH/target cargo test screen::'`
Expected: the 4 new tests pass, the existing `screen::` tests still pass, `reads_a_rendered_slide` ignored.

- [ ] **Step 4: Hook the worker into the session**

In `src-tauri/src/screen/mod.rs`:

`begin`: after `start_session` succeeds with `true`, start OCR:

```rust
pub fn begin(app: &AppHandle, ts: i64) {
    let live = app.state::<ScreenState>().live();
    match crate::history::detail_dir(app).and_then(|d| start_session(&d, ts, live)) {
        Ok(true) => {
            if let Ok(dir) = crate::models::ocr_dir(app) {
                ocr::start(ts, dir);
            }
        }
        Ok(false) => {}
        Err(e) => {
            let _ = app.emit("screen_ended", format!("couldn't save the video — {e}"));
        }
    }
}
```

`end`: first line `ocr::close();`.

`screen_sample`: feed frames the recorder saved:

```rust
fn screen_sample(w: usize, h: usize, rgb: &[u8]) {
    let mut g = session();
    if let Some(s) = g.as_mut() {
        let ms = s.ms();
        match s.rec.screen(ms, w, h, rgb) {
            Ok(true) => ocr::feed(ms, w, h, rgb),
            Ok(false) => {}
            Err(e) => {
                eprintln!("[screen] writing a frame failed, video stops here: {e}");
                *g = None; // keep recording audio; drop the replay's tail
            }
        }
    }
}
```

New command, after `session_has_frames`:

```rust
/// Session `ts`'s screen text by frame ms (screen/ocr.rs), once its OCR is
/// done or after 60 s; empty without OCR.
#[tauri::command]
pub async fn screen_text(ts: i64) -> Vec<(u32, String)> {
    tauri::async_runtime::spawn_blocking(move || ocr::results(ts, Duration::from_secs(60))).await.unwrap_or_default()
}
```

Register `screen::screen_text,` after `screen::session_has_frames,` in `lib.rs`'s `generate_handler!`.

- [ ] **Step 5: Build and test**

Run: `nix develop -c bash -c 'cd src-tauri && CARGO_TARGET_DIR=$SCRATCH/target cargo test screen:: && CARGO_TARGET_DIR=$SCRATCH/target cargo clippy --all-targets 2>&1 | tail -20'`
Expected: tests pass; no new clippy warnings in `screen/` or `models.rs`.

- [ ] **Step 6: Checkpoint**

Commit only if asked.

---

### Task 6: Screen text as a reading source

**Files:**
- Modify: `src/reading.ts`: `withoutSpoken`, `groupScreens`, `frameAt`
- Modify: `src/reading.check.ts`: new cases
- Modify: `src/main.ts`: `screenTexts`/`screenTextsReady` state, Stop handler, `corrections_done` handler, `clearLive`, `readingResult`, new `frameThumbs` and its calls

**Interfaces:**
- Consumes: `screen_text` (Task 5), `slidesPanelHtml`'s `img.slide-thumb[data-frame]` (Task 3), `runsOf`/`reading` (Task 1)
- Produces:
  - `withoutSpoken(text: string[], spoken: Set<string>): string[]`
  - `groupScreens(frames: { ms: number; text: string[] }[]): { ms: number; slide: number; text: string[] }[]` (`slide` −1 = blank)
  - `frameAt<T extends { ms: number }>(frames: T[], ms: number): T | undefined`

- [ ] **Step 1: Write the failing checks**

Append to `src/reading.check.ts` before the final `console.log` (and add `frameAt, groupScreens, runsOf, withoutSpoken` to its import):

```ts
// --- Screen OCR source -------------------------------------------------------

// The app's own transcript on the recorded display: runs already spoken are cut,
// and no new run spans a cut.
const spoken = new Set(runsOf(w("we grew a lot this year")));
assert.deepStrictEqual(withoutSpoken(w("Transcript we grew a lot this year Revenue grew 40 percent"), spoken), w("transcript ¦ revenue grew 40 percent"));

const slideA = w("Revenue grew 40 percent in Q3 driven by enterprise renewals");
const slideB = w("Hiring plan two new teams in Berlin and Lisbon next spring");
const g = groupScreens([
  { ms: 0, text: slideA },
  { ms: 3000, text: [...slideA, "12:01"] }, // clock ticked: same slide
  { ms: 6000, text: w("ok") }, // blank (too few words)
  { ms: 9000, text: slideB },
  { ms: 15000, text: slideA }, // back to the first slide
]);
assert.deepStrictEqual(g.map((x) => x.slide), [0, 0, -1, 1, 0]);
assert.deepStrictEqual(g[1].text, [...slideA, "12:01"]); // each frame keeps its own text

// Review focus 3: the same app chrome/transcript on every frame, once cut, doesn't merge slides.
const chrome = w("live transcript so the big story this quarter");
const said = new Set(runsOf(chrome));
const g2 = groupScreens([slideA, slideB].map((t, i) => ({ ms: i * 5000, text: withoutSpoken([...chrome, ...t], said) })));
assert.deepStrictEqual(g2.map((x) => x.slide), [0, 1]);

// The frame on screen when an utterance started: the last at or before it.
const frames = [{ ms: 0 }, { ms: 3000 }, { ms: 9000 }];
assert.equal(frameAt(frames, 2999), frames[0]);
assert.equal(frameAt(frames, 3000), frames[1]);
assert.equal(frameAt([{ ms: 500 }], 100), undefined);
```

Run: `bun src/reading.check.ts`
Expected: FAIL: `withoutSpoken` is not exported.

- [ ] **Step 2: Implement in `reading.ts`**

Append:

```ts
// ponytail: a frame sharing this much of its runs with an earlier slide is that
// slide again (a clock ticking, a cursor); tune from real decks.
const SAME_SLIDE = 0.5;

// `text` minus every 3-word run already in `spoken`: the app's own transcript,
// when it shares the recorded display. A "¦" marks each cut so no run spans it.
export function withoutSpoken(text: string[], spoken: Set<string>): string[] {
  const cut = text.map(() => false);
  for (let i = 0; i + 2 < text.length; i++)
    if (spoken.has(`${text[i]} ${text[i + 1]} ${text[i + 2]}`)) cut[i] = cut[i + 1] = cut[i + 2] = true;
  const out: string[] = [];
  text.forEach((w, i) => {
    if (!cut[i]) out.push(w);
    else if (out[out.length - 1] !== "¦") out.push("¦");
  });
  return out;
}

// Numbers recorded screens as slides by first appearance: a frame sharing
// ≥ SAME_SLIDE of its runs (of the smaller set) with an earlier slide is that
// slide again. Blank frames (no runs) get slide -1. Frames keep their own text.
export function groupScreens(frames: { ms: number; text: string[] }[]): { ms: number; slide: number; text: string[] }[] {
  const seen: Set<string>[] = [];
  return frames.map(({ ms, text }) => {
    const r = new Set(runsOf(text).filter((x) => !x.includes("¦")));
    if (!r.size) return { ms, slide: -1, text };
    let [best, share] = [-1, 0];
    seen.forEach((s, i) => {
      let n = 0;
      for (const x of r) if (s.has(x)) n++;
      const sh = n / Math.min(r.size, s.size);
      if (sh > share) [best, share] = [i, sh];
    });
    if (share >= SAME_SLIDE) return { ms, slide: best, text };
    seen.push(r);
    return { ms, slide: seen.length - 1, text };
  });
}

// The last of `frames` (in time order) at or before `ms`.
export function frameAt<T extends { ms: number }>(frames: T[], ms: number): T | undefined {
  let out: T | undefined;
  for (const f of frames) {
    if (f.ms > ms) break;
    out = f;
  }
  return out;
}
```

Run: `bun src/reading.check.ts`
Expected: `reading ok`.

- [ ] **Step 3: Fetch the screen text at Stop, before the report is built**

In `src/main.ts`, next to `slideThumbs`:

```ts
// The session's screen text by frame ms (screen/ocr.rs), fetched at Stop; the
// report waits for it (screen_text gives up after 60 s).
let screenTexts: [number, string][] = [];
let screenTextsReady: Promise<void> = Promise.resolve();
```

In `clearLive` add `screenTexts = [];`.

In the Stop branch, right after `await invoke("stop_recording");`:

```ts
      const ts = sessionStartTs;
      screenTextsReady = invoke<[number, string][]>("screen_text", { ts }).then(
        (r) => void (ts === sessionStartTs && (screenTexts = r)),
        () => {},
      );
```

In the `corrections_done` listener, wrap the existing `finishCorrecting(() => { … });` call so it runs once the text is in, and is dropped if a new session started meanwhile (that session's start already flushed this one's save):

```ts
    const ts = sessionStartTs;
    void screenTextsReady.then(() => {
      if (ts !== sessionStartTs) return;
      finishCorrecting(() => {
        // …existing body unchanged…
      });
    });
```

Keep the `for (const el of segmentEls.values()) el.classList.replace("draft", "final");` line before the wrapper, unchanged.

- [ ] **Step 4: Use the screen text when the deck didn't show**

Replace `readingResult()` with:

```ts
// Read off the slides (reading.ts): the in-app deck's text when it showed, else
// the recorded screen's OCR text (not with a read-along script: the script is
// on screen and graded already).
function readingResult(): Reading | null {
  const utts = [...tokensByIndex.keys()]
    .sort((a, b) => a - b)
    .map((index) => {
      const t = timingByIndex.get(index);
      return { index, ms: t?.start ?? 0, dur: t ? t.end - t.start : 0, said: tokensByIndex.get(index) ?? [] };
    });
  if (slideByIndex.size)
    return reading(utts.flatMap((u) => {
      const d = slideByIndex.get(u.index);
      return d ? [{ ...u, ...d }] : [];
    }));
  if (!screenTexts.length || scriptUsed) return null;
  // Each frame minus what had been said before it (the app's own transcript).
  const spoken = new Set<string>();
  let next = 0;
  const frames = groupScreens(
    screenTexts.map(([ms, text]) => {
      for (; next < utts.length && utts[next].ms < ms; next++) runsOf(utts[next].said).forEach((r) => spoken.add(r));
      return { ms, text: withoutSpoken(tokenizeWithOffsets(text).map((t) => t.norm), spoken) };
    }),
  );
  const r = reading(utts.flatMap((u) => {
    const f = frameAt(frames, u.ms);
    return f && f.slide >= 0 ? [{ ...u, slide: f.slide, shown: f.text }] : [];
  }));
  return r && { ...r, screen: true };
}
```

Add `frameAt, groupScreens, runsOf, withoutSpoken` to the reading import.

- [ ] **Step 5: Screen thumbnails in the panel**

Next to `replays` in `src/main.ts`:

```ts
// Fills the Slides panel's recorded-screen thumbnails (data-frame = session ms).
// A pruned .frames file leaves them blank.
const frameThumbs = (root: ParentNode, ts: number) =>
  root.querySelectorAll<HTMLImageElement>("img.slide-thumb[data-frame]").forEach((img) =>
    void invoke<ArrayBuffer>("session_frame", { ts, kind: 0, ms: Number(img.dataset.frame) })
      .then((buf) => {
        if (buf.byteLength) img.src = URL.createObjectURL(new Blob([buf], { type: "image/jpeg" }));
      })
      .catch(() => {}),
  );
```

Call it where reports render: in `renderReport` right after `body.innerHTML = reportBodyHtml(…);` add `frameThumbs(body, d.ts);`; at the two `replays(el, history[i].ts)` call sites add `frameThumbs(el, history[i].ts);` next to them (the second inside the same `if (i >= 0)`, as a block).

(`kind: 0` is `SCREEN` in `screen/frames.rs`; `replay.ts` uses the same value.)

- [ ] **Step 6: Typecheck and checks**

Run: `bunx tsc --noEmit -p . && bun src/reading.check.ts && bun src/crutch.check.ts && bun src/replay.check.ts`
Expected: no tsc output; `reading ok`, `crutch ok`, replay check passes.

- [ ] **Step 7: Checkpoint**

Commit only if asked.

---

### Task 7: End-to-end check in the app

**Files:** none (fix-ups go in the owning task's files)

- [ ] **Step 1: Ask the user to restart `tauri dev`** (Cargo.toml changed). If their build fails with "build-script-build … No such file" or links fail after our scratch builds, see memory: `cargo clean -p speech` and `touch src-tauri/src/lib.rs`.

- [ ] **Step 2: Models download.** User turns Record screen on. Expect `~/.local/share/com.marsont.speech/models/text-detection.rten` and `text-recognition.rten` within a minute (`ls -la` them).

- [ ] **Step 3: Other app, app off the recorded display.** User opens Google Slides in a browser on the recorded display (the speech app on another display or workspace), records ~2 minutes: reads slides 1 and 3 verbatim, talks around 2 and 4. Expect after Stop: report shows "Slides · read N%" with 4 rows (screen frame thumbnails), rows 1 and 3 marked read at ≥ 80%, rows 2 and 4 ≤ 15%; the tip names slides 1 and 3; clicking a row jumps the transcript and replay.

- [ ] **Step 4: App on the recorded display.** Same run with the speech app visible on the recorded display beside the slides. Expect the same rows (±1 slide from OCR noise), not one merged slide and not 100% everywhere.

- [ ] **Step 5: Script session.** Record with a read-along script open and Record screen on. Expect no Slides panel.

- [ ] **Step 6: Live latency.** During Step 3, watch the live transcript: interim text should keep up as without Record screen. If it lags, lower the thread cap to 1 (Task 4 Step 6) and re-check.

- [ ] **Step 7: Note tuning.** Record the observed per-slide numbers and OCR ms in the Task 4/6 `ponytail:` comments if thresholds needed changing; otherwise leave them.

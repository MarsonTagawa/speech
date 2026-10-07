# Video Presence Grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional camera mode that grades eye contact, facial expressiveness, stillness and hand gestures. The result is a **Presence** score in the live view, the report, the coach tips and history.

**Architecture:** MediaPipe Tasks Vision (face + pose + hand landmarkers) runs in the Tauri webview on frames downscaled to 640×360, at ~12 fps. `src/camera.ts` handles capture and inference, and reduces each frame to a small `Frame`. `src/presence.ts` is pure: it reduces raw results and turns a session's frames into metrics and sub-scores. `main.ts` stores frames only while recording and adds `presence` to the composite score, the same way it handles `articulation`. On the Rust side, the only change is two WebKitGTK workarounds in `lib.rs`.

**Tech Stack:** Tauri 2 / wry / webkit2gtk 2.0.2 (Rust), TypeScript + Vite, `@mediapipe/tasks-vision` 1.1.0, Bun for `.check.ts` files.

**Spec:** `docs/superpowers/specs/2026-10-07-video-presence-design.md` (read it first, especially "Feasibility").

## Before you start

- The working tree may contain the user's own uncommitted edits to `src/main.ts` and `src/styles.css`. **Don't commit them as part of a task.** If `git status` shows them modified before Task 1, stop and ask the user to commit or stash them first.
- Rust commands need the dev shell: prefix them with `nix develop -c`.
- Line numbers below are approximate (the file is ~4.7k lines). Find anchors by the quoted code or function name, not by number.

## Global Constraints

- With the camera off, the app behaves exactly as today: no camera, no MediaPipe load, identical scores.
- Everything runs locally. Never save video, frames, landmarks, or per-second presence timelines. Persist only the `Presence` summary numbers.
- MediaPipe assets live in git-ignored `public/mediapipe/` and are fetched via README steps, never committed.
- Inference runs on a 640×360 canvas, never on raw camera frames (1080p halves fps).
- Every threshold lives in `PRESENCE_TUNING` in `src/presence.ts`.
- `presence` is `null`/absent when the camera was off or the session has < 3 s of frames. Sessions saved before this feature must still load and render.
- The Rust permission handler and env var are Linux-only (`#[cfg(target_os = "linux")]`).
- Commit messages follow repo style (short, lowercase) and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Speaker out of frame for the whole session.** Expected: eye contact 0, expressiveness `null`, no `NaN` anywhere. Covered by a Task 3 test.
2. **Degenerate pose (shoulders at the same point, e.g. side-on).** Expected: `body` is `null`, no `Infinity`/`NaN` in motion. Covered by a Task 2 test.
3. **Frame gaps (window hidden → rAF stops; camera hiccup).** Expected: the gap doesn't count as a burst of motion. Covered by a Task 3 test.
4. **Camera unplugged, or toggled off, mid-recording.** Expected: the toggle turns off, the recording continues, and the frames so far are still scored. Covered by a Task 4 manual step.
5. **Old saved sessions without `presence`.** Expected: History, report, and speech overview render with no Presence tile and no errors. Covered by a Task 6 manual step.

## File map

| File | Responsibility |
|---|---|
| `src-tauri/src/lib.rs` (modify) | dma-buf env var; enable media stream + grant user-media permission requests |
| `src/presence.ts` (create) | Pure: `PRESENCE_TUNING`, `scoreLinear` (moved from main.ts), `headAngles`, `toFrame`, `summarize`, `gestureScore`, `awayMs`, `weakestCue`, types |
| `src/presence.check.ts` (create) | Assertions for presence.ts (`bun src/presence.check.ts`) |
| `src/camera.ts` (create) | getUserMedia, lazy MediaPipe load (GPU → CPU fallback), ~12 fps loop → `Frame` callback |
| `index.html` (modify) | camera button, self-view `<video>`, Presence dock card |
| `src/styles.css` (modify) | self-view, dock/tile grid variants for the extra card/tile, tile rows |
| `src/main.ts` (modify) | frame storage, toggle, live card, `Scores.presence`, `SessionSummary.presence`, report tile, tips, `DIMENSIONS` |
| `.gitignore`, `README.md`, `package.json` (modify) | asset ignore + fetch steps, dependency |

---

### Task 1: Let the webview open the camera (Rust)

**Files:**
- Modify: `src-tauri/src/lib.rs` (`run()` top, and `setup` closure before `app.manage(RecordingState::default());`)

**Interfaces:**
- Consumes: nothing.
- Produces: `navigator.mediaDevices.getUserMedia({video})` resolves in the main window on Linux without crashing the web process.

- [ ] **Step 1: Add the env var at the top of `run()`**

In `src-tauri/src/lib.rs`, make the first statement inside `pub fn run() {`:

```rust
    // The webview's camera feed (camera.ts) crashes WebKitWebProcess — SIGSEGV
    // in GStreamer's GL dma-buf upload (webkitgtk 2.54 / gst 1.28, AMD iGPU) —
    // unless WebKit's dma-buf video sink is off. Must be set before any webview
    // exists.
    #[cfg(target_os = "linux")]
    std::env::set_var("WEBKIT_GST_DMABUF_SINK_DISABLED", "1");
```

- [ ] **Step 2: Enable media stream and grant camera/mic permission requests**

In the `setup` closure, immediately before `app.manage(RecordingState::default());`, insert:

```rust
            // Camera for presence grading (camera.ts). wry leaves WebKitGTK's
            // permission requests unanswered, so getUserMedia would never
            // resolve. Grant camera/mic requests only; anything else keeps
            // WebKit's default (deny).
            #[cfg(target_os = "linux")]
            app.get_webview_window("main")
                .ok_or("no main window")?
                .with_webview(|wv| {
                    use gtk::glib::object::Cast;
                    use webkit2gtk::{PermissionRequestExt, SettingsExt, UserMediaPermissionRequest, WebViewExt};
                    let view = wv.inner();
                    if let Some(s) = WebViewExt::settings(&view) {
                        s.set_enable_media_stream(true);
                    }
                    view.connect_permission_request(|_, req| {
                        if req.downcast_ref::<UserMediaPermissionRequest>().is_none() {
                            return false;
                        }
                        req.allow();
                        true
                    });
                })?;
```

- [ ] **Step 3: Type-check**

Run: `nix develop -c cargo check --manifest-path src-tauri/Cargo.toml`
Expected: `Finished` with no errors. If `Cast` doesn't resolve, use `use gtk::prelude::Cast;` instead. Both re-export the same glib trait.

- [ ] **Step 4: Run existing Rust tests**

Run: `nix develop -c cargo test --manifest-path src-tauri/Cargo.toml`
Expected: all pass (17 at time of writing).

- [ ] **Step 5: Commit**

```bash
git add src-tauri/src/lib.rs
git commit -m "presence: let the webview open the camera

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Per-frame reduction — `presence.ts` part 1

**Files:**
- Create: `src/presence.ts`
- Create: `src/presence.check.ts`
- Modify: `src/main.ts`: delete the local `scoreLinear` (search `function scoreLinear(v: number, good: number, bad: number)`) and import it instead

**Interfaces:**
- Consumes: nothing.
- Produces (exact exports):
  - `scoreLinear(v: number, good: number, bad: number): number` (moved verbatim from main.ts)
  - `PRESENCE_TUNING` (const, `as const`)
  - `type XY = [number, number]`
  - `interface Pt { x: number; y: number }`
  - `interface Frame { t: number; face: boolean; looking: boolean; smile: number; brows: number; body: XY[] | null; wrists: XY[] }`. `body` is `[shoulderMid, nose]` and `wrists` is in shoulder widths. `wrists` is empty when `body` is `null`.
  - `interface RawResults { blend: Record<string, number> | null; matrix: number[] | null; pose: Pt[] | null; wrists: Pt[] }`
  - `headAngles(m: number[]): { yaw: number; pitch: number }` in degrees
  - `toFrame(r: RawResults, w: number, h: number, t: number): Frame`

- [ ] **Step 1: Write the failing check**

Create `src/presence.check.ts`:

```ts
// Run: bun src/presence.check.ts
import assert from "node:assert";
import { headAngles, toFrame, scoreLinear, type RawResults } from "./presence";

const near = (a: number, b: number, eps = 0.01) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const rad = (d: number) => (d * Math.PI) / 180;
// 4×4 flattened column-major (MediaPipe's layout) from a 3×3 rotation, with translation.
const mat = (R: number[][]): number[] => {
  const m = new Array(16).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m[c * 4 + r] = R[r][c];
  m[12] = 1; m[13] = 2; m[14] = -30; m[15] = 1;
  return m;
};
const yawM = (d: number) => mat([[Math.cos(rad(d)), 0, Math.sin(rad(d))], [0, 1, 0], [-Math.sin(rad(d)), 0, Math.cos(rad(d))]]);
const pitchM = (d: number) => mat([[1, 0, 0], [0, Math.cos(rad(d)), -Math.sin(rad(d))], [0, Math.sin(rad(d)), Math.cos(rad(d))]]);
const transpose = (m: number[]) => m.map((_, i) => m[(i % 4) * 4 + Math.floor(i / 4)]);

// scoreLinear (moved from main.ts): unchanged behaviour.
assert.equal(scoreLinear(1, 1, 12), 100);
assert.equal(scoreLinear(12, 1, 12), 0);
assert.equal(scoreLinear(6.5, 1, 12), 50);
assert.equal(scoreLinear(3, 5, 5), 100);

// headAngles: magnitudes are right in either flattening order.
near(headAngles(mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).yaw, 0);
near(Math.abs(headAngles(yawM(30)).yaw), 30);
near(Math.abs(headAngles(yawM(30)).pitch), 0);
near(Math.abs(headAngles(pitchM(20)).pitch), 20);
near(Math.abs(headAngles(transpose(pitchM(20))).pitch), 20);
near(Math.abs(headAngles(transpose(yawM(30))).yaw), 30);

// toFrame
const facing: Record<string, number> = { mouthSmileLeft: 0.4, mouthSmileRight: 0.6, browInnerUp: 0.2, browOuterUpLeft: 0.3, eyeLookDownLeft: 0.2 };
const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5 }));
pose[0] = { x: 0.5, y: 0.3 }; // nose
pose[11] = { x: 0.4, y: 0.5 }; // left shoulder
pose[12] = { x: 0.6, y: 0.5 }; // right shoulder
const raw = (o: Partial<RawResults> = {}): RawResults => ({ blend: facing, matrix: yawM(5), pose, wrists: [], ...o });

const none = toFrame(raw({ blend: null, matrix: null, pose: null }), 640, 360, 0);
assert.deepStrictEqual(none, { t: 0, face: false, looking: false, smile: 0, brows: 0, body: null, wrists: [] });

const f = toFrame(raw({ wrists: [{ x: 0.5, y: 0.8 }] }), 640, 360, 42);
assert.equal(f.t, 42);
assert.equal(f.face, true);
assert.equal(f.looking, true); // 5° yaw, eyes 0.2 < 0.5
near(f.smile, 0.5);
near(f.brows, 0.3);
// shoulders 128 px apart; midpoint (320,180) px → (2.5, 1.40625) shoulder widths
near(f.body![0][0], 2.5);
near(f.body![0][1], 1.40625);
near(f.body![1][1], 108 / 128); // nose y = 0.3·360
near(f.wrists[0][1], 288 / 128); // wrist y = 0.8·360

assert.equal(toFrame(raw({ matrix: yawM(30) }), 640, 360, 0).looking, false); // head turned
assert.equal(toFrame(raw({ blend: { ...facing, eyeLookOutLeft: 0.7 } }), 640, 360, 0).looking, false); // eyes away
assert.equal(toFrame(raw({ matrix: null }), 640, 360, 0).looking, false); // no matrix → can't tell → not looking

// Degenerate pose (shoulders coincide): no body, no wrists, nothing infinite.
const flat = pose.map((p) => ({ ...p }));
flat[12] = { ...flat[11] };
const d = toFrame(raw({ pose: flat, wrists: [{ x: 0.5, y: 0.8 }] }), 640, 360, 0);
assert.equal(d.body, null);
assert.deepStrictEqual(d.wrists, []);

console.log("presence part 1 ok");
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bun src/presence.check.ts`
Expected: FAIL. The module `./presence` doesn't exist yet.

- [ ] **Step 3: Write `src/presence.ts` (part 1)**

```ts
// Video presence: reduces MediaPipe results to small per-frame records and
// (summarize, below) turns a session's frames into metrics + a 0–100 score.
// Pure; capture and inference live in camera.ts.

export function scoreLinear(v: number, good: number, bad: number): number {
  if (good === bad) return 100;
  const t = (v - bad) / (good - bad);
  return Math.round(Math.max(0, Math.min(1, t)) * 100);
}

// ponytail: initial guesses from one spike session (2026-10-07); tune against
// real recordings before trusting the numbers.
export const PRESENCE_TUNING = {
  maxYaw: 15, // degrees off-axis still counted as eye contact
  maxPitch: 15,
  maxEyeLook: 0.5, // any eyeLook* blendshape above this = looking away
  eye: [0.8, 0.3], // eye-contact fraction: good, bad
  smileOn: 0.3, // smile blendshape counted as smiling
  smileFrac: [0.25, 0],
  exprRange: [0.08, 0.01], // mean 1 s std-dev of smile + brows
  motion: [0.15, 0.6], // shoulder widths travelled per second
  gestureStep: 0.04, // wrist travel per frame (shoulder widths) that counts as gesturing
  gestureBand: [0.2, 0.6], // gesturing fraction that scores 100
  maxGapMs: 500, // frames further apart than this aren't compared
  minMs: 3000, // shorter sessions aren't scored
  awayCueMs: 2000, // live "look at the camera" cue after this long looking away
} as const;

export type XY = [number, number];
export interface Pt {
  x: number;
  y: number;
}

export interface Frame {
  t: number; // ms, monotonic
  face: boolean;
  looking: boolean; // face toward the camera and eyes centred
  smile: number;
  brows: number;
  body: XY[] | null; // [shoulder midpoint, nose], in shoulder widths
  wrists: XY[]; // in shoulder widths; empty without a body to scale by
}

// One frame of MediaPipe output, already unpacked by camera.ts.
export interface RawResults {
  blend: Record<string, number> | null; // face blendshapes by name; null = no face
  matrix: number[] | null; // 4×4 facial transformation, flattened
  pose: Pt[] | null; // 33 normalised pose landmarks
  wrists: Pt[]; // hand landmark 0 per detected hand, normalised
}

// Head yaw/pitch in degrees from the facial transformation matrix. Works for
// either flattening order: for a single-axis turn the mirrored off-diagonal
// entries differ only in sign, and callers only use magnitudes.
export function headAngles(m: number[]): { yaw: number; pitch: number } {
  const deg = 180 / Math.PI;
  return { yaw: Math.asin(Math.max(-1, Math.min(1, m[2]))) * deg, pitch: Math.atan2(m[6], m[10]) * deg };
}

const EYE_LOOK = ["Up", "Down", "In", "Out"].flatMap((d) => [`eyeLook${d}Left`, `eyeLook${d}Right`]);

export function toFrame(r: RawResults, w: number, h: number, t: number): Frame {
  const T = PRESENCE_TUNING;
  const b = r.blend;
  let looking = false;
  if (b && r.matrix) {
    const { yaw, pitch } = headAngles(r.matrix);
    looking =
      Math.abs(yaw) <= T.maxYaw && Math.abs(pitch) <= T.maxPitch && Math.max(...EYE_LOOK.map((k) => b[k] ?? 0)) < T.maxEyeLook;
  }
  const smile = b ? ((b.mouthSmileLeft ?? 0) + (b.mouthSmileRight ?? 0)) / 2 : 0;
  const brows = b ? Math.max(b.browInnerUp ?? 0, b.browOuterUpLeft ?? 0, b.browOuterUpRight ?? 0) : 0;

  // Pixels so x and y share a unit, then shoulder widths so distance from the
  // camera doesn't change the numbers.
  let body: XY[] | null = null;
  let wrists: XY[] = [];
  const p = r.pose;
  if (p && p.length > 12) {
    const px = (q: Pt): XY => [q.x * w, q.y * h];
    const [l, rs, nose] = [px(p[11]), px(p[12]), px(p[0])];
    const sw = Math.hypot(l[0] - rs[0], l[1] - rs[1]);
    if (sw >= 1) {
      const n = (q: XY): XY => [q[0] / sw, q[1] / sw];
      body = [n([(l[0] + rs[0]) / 2, (l[1] + rs[1]) / 2]), n(nose)];
      wrists = r.wrists.map((q) => n(px(q)));
    }
  }
  return { t, face: !!b, looking, smile, brows, body, wrists };
}
```

- [ ] **Step 4: Run the check to verify it passes**

Run: `bun src/presence.check.ts`
Expected: `presence part 1 ok`

- [ ] **Step 5: Point main.ts at the moved `scoreLinear`**

In `src/main.ts`, delete the whole block:

```ts
function scoreLinear(v: number, good: number, bad: number): number {
  if (good === bad) return 100;
  const t = (v - bad) / (good - bad);
  return Math.round(Math.max(0, Math.min(1, t)) * 100);
}
```

and add after the line `import { chunkSpeech, type Chunk } from "./chunks";`:

```ts
import { scoreLinear } from "./presence";
```

- [ ] **Step 6: Build**

Run: `bun run build`
Expected: `tsc` and `vite build` succeed with no errors.

- [ ] **Step 7: Commit**

```bash
git add src/presence.ts src/presence.check.ts src/main.ts
git commit -m "presence: per-frame reduction of face/pose/hand results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(If `src/main.ts` also has the user's unrelated edits, see "Before you start". Never stage them.)

---

### Task 3: Session summary and scores — `presence.ts` part 2

**Files:**
- Modify: `src/presence.ts` (append)
- Modify: `src/presence.check.ts` (append, and extend the import)

**Interfaces:**
- Consumes: `Frame`, `PRESENCE_TUNING`, `scoreLinear` from Task 2.
- Produces (exact exports):
  - `interface PresenceScores { eye: number; expr: number | null; still: number | null; gesture: number | null }`
  - `interface Presence { eyeContact: number; smileFrac: number; exprRange: number; motion: number; gestureFrac: number; scores: PresenceScores; eyeScored: boolean; score: number }`
  - `gestureScore(frac: number): number`
  - `summarize(frames: Frame[], opts: { scriptOpen: boolean }): Presence | null`
  - `awayMs(frames: Frame[]): number`
  - `weakestCue(p: Presence): string` returns `"smile more" | "stay still" | "use your hands" | "looking good"`

Rules (from the spec, plus these refinements):
- A sub-score is `null` when its signal had no data. Expressiveness is `null` with no 1 s window of ≥ 3 face frames. Stillness and gestures are `null` with < 1 s of comparable body frames. Eye contact is always defined, and having no face counts as not looking.
- `score` = rounded mean of the non-null sub-scores, leaving out `eye` when `scriptOpen`. If nothing is left, `summarize` returns `null`.

- [ ] **Step 1: Write the failing checks**

Change the import line in `src/presence.check.ts` to:

```ts
import { headAngles, toFrame, scoreLinear, summarize, gestureScore, awayMs, weakestCue, type RawResults, type Frame } from "./presence";
```

Replace the final `console.log("presence part 1 ok");` with:

```ts
// --- summarize ------------------------------------------------------------
const STEP = 100; // 10 fps keeps the expected timings round
const F = (t: number, o: Partial<Frame> = {}): Frame => ({ t, face: true, looking: true, smile: 0, brows: 0, body: [[2.5, 1.4], [2.5, 0.8]], wrists: [], ...o });
const frames = (secs: number, make: (t: number, i: number) => Partial<Frame> = () => ({})): Frame[] =>
  Array.from({ length: (secs * 1000) / STEP + 1 }, (_, i) => F(i * STEP, make(i * STEP, i)));
const open = { scriptOpen: false };

// Too short to score.
assert.equal(summarize(frames(2), open), null);
assert.equal(summarize([], open), null);

// Steady, always looking, flat face, no hands.
const steady = summarize(frames(10), open)!;
assert.equal(steady.eyeContact, 1);
assert.equal(steady.scores.eye, 100);
assert.equal(steady.motion, 0);
assert.equal(steady.scores.still, 100);
assert.equal(steady.gestureFrac, 0);
assert.equal(steady.scores.gesture, 0);
assert.equal(steady.scores.expr, 0); // never smiles, no variation
assert.equal(steady.eyeScored, true);
assert.equal(steady.score, Math.round((100 + 0 + 100 + 0) / 4));

// Out of frame the whole session: eye 0, nothing else measurable, no NaN.
const gone = summarize(frames(10, () => ({ face: false, looking: false, body: null })), open)!;
assert.equal(gone.eyeContact, 0);
assert.deepStrictEqual(gone.scores, { eye: 0, expr: null, still: null, gesture: null });
assert.equal(gone.score, 0);
for (const v of [gone.smileFrac, gone.exprRange, gone.motion, gone.gestureFrac]) assert.ok(Number.isFinite(v));

// Script open: eye contact reported but not scored.
const away = frames(10, () => ({ looking: false }));
const withScript = summarize(away, { scriptOpen: true })!;
assert.equal(withScript.eyeScored, false);
assert.equal(withScript.eyeContact, 0);
assert.equal(withScript.score, Math.round((0 + 100 + 0) / 3)); // expr, still, gesture
assert.equal(summarize(away, open)!.score, Math.round((0 + 0 + 100 + 0) / 4));
// Script open and nothing else measurable → no score.
assert.equal(summarize(frames(10, () => ({ face: false, looking: false, body: null })), { scriptOpen: true }), null);

// Jitter scores worse than steady.
const jitter = summarize(frames(10, (_, i) => ({ body: [[2.5 + (i % 2) * 0.2, 1.4], [2.5 + (i % 2) * 0.2, 0.8]] })), open)!;
assert.ok(jitter.motion > 1, `motion ${jitter.motion}`);
assert.equal(jitter.scores.still, 0);

// A hidden-window gap isn't motion: two still halves 5 s apart, shifted.
const gap = [...frames(4), ...frames(4).map((f) => ({ ...f, t: f.t + 9000, body: [[4, 1.4], [4, 0.8]] as Frame["body"] }))];
assert.equal(summarize(gap, open)!.motion, 0);

// Expressive face: smile alternating 0 / 0.6.
const lively = summarize(frames(10, (_, i) => ({ smile: i % 2 ? 0.6 : 0 })), open)!;
assert.ok(lively.smileFrac > 0.45 && lively.smileFrac < 0.55);
assert.equal(lively.scores.expr, 100);

// Hands moving every frame → gesturing ~100% → band ceiling 70.
const busy = summarize(frames(10, (_, i) => ({ wrists: [[1 + (i % 2) * 0.1, 2]] })), open)!;
assert.ok(busy.gestureFrac > 0.95);
assert.equal(busy.scores.gesture, 70);
// A hand held still isn't gesturing.
assert.equal(summarize(frames(10, () => ({ wrists: [[1, 2]] })), open)!.gestureFrac, 0);

// --- gestureScore band ----------------------------------------------------
assert.equal(gestureScore(0), 0);
assert.equal(gestureScore(0.1), 50);
assert.equal(gestureScore(0.4), 100);
assert.equal(gestureScore(0.6), 100);
assert.equal(gestureScore(1), 70);

// --- live helpers -----------------------------------------------------------
assert.equal(awayMs([]), 0);
assert.equal(awayMs(frames(3)), 0);
assert.equal(awayMs(frames(3, (t) => ({ looking: t < 1000 }))), 3000 - 900); // last look at t=900
assert.equal(awayMs(frames(3, () => ({ looking: false }))), 3000);
assert.equal(weakestCue(steady), "smile more"); // expr 0 and gesture 0 tie → first listed (expr)
assert.equal(weakestCue(lively), "use your hands");
assert.equal(weakestCue(summarize(frames(10, (_, i) => ({ smile: i % 2 ? 0.6 : 0, wrists: [[1 + (i % 4 < 2 ? 0 : 0.1), 2]] })), open)!), "looking good");

console.log("presence ok");
```

- [ ] **Step 2: Run to verify it fails**

Run: `bun src/presence.check.ts`
Expected: FAIL. `summarize` is not exported.

- [ ] **Step 3: Implement (append to `src/presence.ts`)**

```ts
export interface PresenceScores {
  eye: number;
  expr: number | null; // null = no measurable face
  still: number | null; // null = no measurable body
  gesture: number | null;
}

export interface Presence {
  eyeContact: number; // fraction of frames looking at the camera
  smileFrac: number; // fraction of face frames smiling
  exprRange: number; // mean per-second std-dev of smile + brows
  motion: number; // shoulder widths per second
  gestureFrac: number; // fraction of body frames with a moving hand
  scores: PresenceScores;
  eyeScored: boolean; // false when a script was open (reading looks down)
  score: number;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const sd = (xs: number[]) => {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
};
const dist = (a: XY, b: XY) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// 100 inside the band; ramps up from 0 below it, eases down to 70 for constant motion.
export function gestureScore(frac: number): number {
  const [lo, hi] = PRESENCE_TUNING.gestureBand;
  if (frac < lo) return Math.round((frac / lo) * 100);
  if (frac <= hi) return 100;
  return Math.round(100 - 30 * Math.min(1, (frac - hi) / (1 - hi)));
}

export function summarize(frames: Frame[], opts: { scriptOpen: boolean }): Presence | null {
  const T = PRESENCE_TUNING;
  if (frames.length < 2 || frames[frames.length - 1].t - frames[0].t < T.minMs) return null;

  const eyeContact = frames.filter((f) => f.looking).length / frames.length;

  const faces = frames.filter((f) => f.face);
  const smileFrac = faces.length ? faces.filter((f) => f.smile > T.smileOn).length / faces.length : 0;
  const windows = new Map<number, number[]>();
  for (const f of faces) {
    const k = Math.floor((f.t - frames[0].t) / 1000);
    if (!windows.has(k)) windows.set(k, []);
    windows.get(k)!.push(f.smile + f.brows);
  }
  const spreads = [...windows.values()].filter((w) => w.length >= 3).map(sd);
  const exprRange = mean(spreads);

  let travelled = 0;
  let secs = 0;
  let pairs = 0;
  let gesturing = 0;
  for (let i = 1; i < frames.length; i++) {
    const a = frames[i - 1];
    const b = frames[i];
    if (!a.body || !b.body || b.t - a.t > T.maxGapMs) continue;
    pairs++;
    secs += (b.t - a.t) / 1000;
    travelled += mean(b.body.map((q, j) => dist(q, a.body![j])));
    if (a.wrists.length && b.wrists.some((w) => Math.min(...a.wrists.map((v) => dist(w, v))) > T.gestureStep)) gesturing++;
  }
  const motion = secs > 0 ? travelled / secs : 0;
  const gestureFrac = pairs ? gesturing / pairs : 0;

  const scores: PresenceScores = {
    eye: scoreLinear(eyeContact, ...T.eye),
    expr: spreads.length ? Math.round((scoreLinear(smileFrac, ...T.smileFrac) + scoreLinear(exprRange, ...T.exprRange)) / 2) : null,
    still: secs >= 1 ? scoreLinear(motion, ...T.motion) : null,
    gesture: secs >= 1 ? gestureScore(gestureFrac) : null,
  };
  const parts = [opts.scriptOpen ? null : scores.eye, scores.expr, scores.still, scores.gesture].filter(
    (s): s is number => s !== null,
  );
  if (!parts.length) return null;
  return { eyeContact, smileFrac, exprRange, motion, gestureFrac, scores, eyeScored: !opts.scriptOpen, score: Math.round(mean(parts)) };
}

// How long the speaker has been looking away, as of the latest frame.
export function awayMs(frames: Frame[]): number {
  if (!frames.length) return 0;
  const last = frames[frames.length - 1].t;
  for (let i = frames.length - 1; i >= 0; i--) if (frames[i].looking) return last - frames[i].t;
  return last - frames[0].t;
}

// The live card's nudge: the weakest non-eye signal under 70, if any.
export function weakestCue(p: Presence): string {
  const cues: Array<[number | null, string]> = [
    [p.scores.expr, "smile more"],
    [p.scores.still, "stay still"],
    [p.scores.gesture, "use your hands"],
  ];
  const worst = cues.filter((c): c is [number, string] => c[0] !== null).sort((a, b) => a[0] - b[0])[0];
  return worst && worst[0] < 70 ? worst[1] : "looking good";
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `bun src/presence.check.ts`
Expected: `presence ok`

- [ ] **Step 5: Build**

Run: `bun run build`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/presence.ts src/presence.check.ts
git commit -m "presence: session summary, sub-scores and live cues

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Camera capture and live UI

**Files:**
- Modify: `package.json` (via `bun add`), `.gitignore`, `README.md`
- Create: `public/mediapipe/*` (git-ignored, not committed)
- Create: `src/camera.ts`
- Modify: `index.html` (strip, scope-tr, dock)
- Modify: `src/styles.css` (append at end of file)
- Modify: `src/main.ts` (imports, session state near `let recording = false;`, `clearLive`, init near `setRibbon`)

**Interfaces:**
- Consumes: `toFrame`, `Frame`, `summarize`, `awayMs`, `weakestCue`, `PRESENCE_TUNING` from presence.ts.
- Produces:
  - `startCamera(video: HTMLVideoElement, onFrame: (f: Frame) => void, onEnded: () => void): Promise<void>` and `stopCamera(): void` in `src/camera.ts`
  - In main.ts: `let presenceFrames: Frame[]` (the session's recorded frames, used by Task 5), plus `setCamera(on: boolean)` and `renderPresenceCard()`
  - The CSS class `camera-on` on `<body>` while the camera is on

- [ ] **Step 1: Dependency and assets**

```bash
bun add @mediapipe/tasks-vision@1.1.0
mkdir -p public/mediapipe
cp node_modules/@mediapipe/tasks-vision/wasm/* public/mediapipe/
M=https://storage.googleapis.com/mediapipe-models
curl -L -o public/mediapipe/face_landmarker.task $M/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -L -o public/mediapipe/pose_landmarker_lite.task $M/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
curl -L -o public/mediapipe/hand_landmarker.task $M/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
ls -la public/mediapipe
```

Expected: 3 `.task` files (≈3.8 MB, 5.8 MB, 7.8 MB) and the `vision_wasm_*` js/wasm files.

Append to `.gitignore`, right after the `src-tauri/resources/*.onnx` line:

```
# MediaPipe wasm + models for camera presence grading — fetched on setup (see README)
public/mediapipe/
```

In `README.md`, at the end of the "## Models (required, not in git)" section (after the paragraph ending "freed when it's done."), add:

````markdown
The camera's presence grading (optional) needs MediaPipe's wasm and three
models (~30 MB) in `public/mediapipe/`:

```sh
mkdir -p public/mediapipe
cp node_modules/@mediapipe/tasks-vision/wasm/* public/mediapipe/
M=https://storage.googleapis.com/mediapipe-models
curl -L -o public/mediapipe/face_landmarker.task $M/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -L -o public/mediapipe/pose_landmarker_lite.task $M/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
curl -L -o public/mediapipe/hand_landmarker.task $M/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Without them the camera toggle reports "Camera models missing" and the app
works audio-only.
````

- [ ] **Step 2: Create `src/camera.ts`**

```ts
// Webcam capture + MediaPipe inference for presence grading. Pure scoring is in
// presence.ts; this file only turns camera frames into Frames. Assets are
// served from public/mediapipe/ (see README). Nothing here loads until the
// camera is first switched on.
import { FaceLandmarker, FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import { toFrame, type Frame } from "./presence";

const BASE = "/mediapipe";
// Inference size. The camera delivers 1080p whatever we ask; feeding that
// straight in halves the frame rate (spike, 2026-10-07).
const W = 640;
const H = 360;
const STEP_MS = 80; // ~12 fps

interface Models {
  face: FaceLandmarker;
  pose: PoseLandmarker;
  hand: HandLandmarker;
}

async function create(delegate: "GPU" | "CPU"): Promise<Models> {
  const files = await FilesetResolver.forVisionTasks(BASE);
  const base = (file: string) => ({ baseOptions: { modelAssetPath: `${BASE}/${file}`, delegate }, runningMode: "VIDEO" as const });
  // Sequential: the spike created them one after another; untested in parallel.
  const face = await FaceLandmarker.createFromOptions(files, {
    ...base("face_landmarker.task"),
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
  const pose = await PoseLandmarker.createFromOptions(files, base("pose_landmarker_lite.task"));
  const hand = await HandLandmarker.createFromOptions(files, { ...base("hand_landmarker.task"), numHands: 2 });
  return { face, pose, hand };
}

// Loaded once per app run; GPU first, CPU (~10 fps) if WebGL init fails.
let models: Promise<Models> | null = null;
function loadModels(): Promise<Models> {
  models ??= create("GPU")
    .catch(() => create("CPU"))
    .catch((e) => {
      models = null; // let a later toggle retry
      throw new Error(`Camera models missing — see README (${e})`);
    });
  return models;
}

let stream: MediaStream | null = null;
let raf = 0;
let gen = 0; // bumps on every start/stop so a slow start can't outlive a stop

export async function startCamera(video: HTMLVideoElement, onFrame: (f: Frame) => void, onEnded: () => void): Promise<void> {
  stopCamera();
  const my = ++gen;
  const m = await loadModels();
  const s = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: W }, height: { ideal: H }, frameRate: { ideal: 30 } },
  });
  if (my !== gen) {
    s.getTracks().forEach((t) => t.stop());
    return;
  }
  stream = s;
  s.getVideoTracks()[0]?.addEventListener("ended", () => {
    if (my === gen) {
      stopCamera();
      onEnded();
    }
  });
  video.srcObject = s;
  await video.play();

  const cv = document.createElement("canvas");
  cv.width = W;
  cv.height = H;
  const cx = cv.getContext("2d")!;
  let last = 0;
  const tick = (now: number) => {
    if (my !== gen) return;
    raf = requestAnimationFrame(tick);
    if (now - last < STEP_MS || video.readyState < 2) return;
    last = now;
    cx.drawImage(video, 0, 0, W, H);
    const f = m.face.detectForVideo(cv, now);
    const p = m.pose.detectForVideo(cv, now);
    const h = m.hand.detectForVideo(cv, now);
    const shapes = f.faceBlendshapes[0]?.categories;
    onFrame(
      toFrame(
        {
          blend: shapes ? Object.fromEntries(shapes.map((c) => [c.categoryName, c.score])) : null,
          matrix: f.facialTransformationMatrixes[0]?.data ?? null,
          pose: p.landmarks[0] ?? null,
          wrists: h.landmarks.map((l) => l[0]),
        },
        W,
        H,
        now,
      ),
    );
  };
  raf = requestAnimationFrame(tick);
}

export function stopCamera() {
  gen++;
  cancelAnimationFrame(raf);
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
}
```

- [ ] **Step 3: Markup in `index.html`**

(a) In the `.strip`, immediately before `<button id="ribbon-btn"`, insert:

```html
          <button id="camera-btn" class="tool-btn icon" type="button" aria-pressed="false" aria-label="Camera" data-tip="Camera — grade eye contact, expression, stillness and gestures">
            <svg viewBox="0 0 24 24"><path d="m16 13 5.2 3.5a.5.5 0 0 0 .8-.4V7.9a.5.5 0 0 0-.8-.4L16 11" /><rect x="2" y="6" width="14" height="12" rx="2" /></svg>
          </button>
```

(b) In `<div class="scope-tr">`, after the `<div id="rms-label" …>…</div>` line, insert:

```html
            <video id="self-view" class="self-view" autoplay playsinline muted hidden></video>
```

(c) In `<section class="dock">`, immediately before `<div class="card card-pace">`, insert:

```html
          <div id="presence-card" class="card" hidden>
            <div class="card-top"><span id="stat-eye" class="big">–</span><span class="unit" data-tip="Eye contact — share of this session spent looking at the camera">% eye contact</span></div>
            <div class="track"><div id="eye-bar" class="fill ink-bg"></div></div>
            <div id="presence-sub" class="card-sub">camera on</div>
          </div>
```

- [ ] **Step 4: Styles (append to the END of `src/styles.css`)**

These must come after the existing `@media` blocks so they win ties.

```css
/* --- Camera / presence ------------------------------------------------------
   The Presence card and tile only exist with the camera on (or on sessions
   recorded with it), adding one column to the dock/tiles grids. */
.self-view {
  display: block;
  width: 160px;
  height: 90px;
  margin: 8px 0 0 auto;
  object-fit: cover;
  border-radius: 8px;
  border: 0.5px solid var(--edge);
  background: #000;
  transform: scaleX(-1); /* mirror, like every video-call self view */
}
.self-view[hidden],
.dock > .card[hidden] {
  display: none;
}
.camera-on .dock {
  grid-template-columns: 44px repeat(6, minmax(0, 1fr)) minmax(0, 1.5fr);
}
.tiles.cam {
  grid-template-columns: repeat(6, minmax(0, 1fr));
}
.tile-rows {
  margin-top: 6px;
  display: grid;
  gap: 2px;
  font-size: 10px;
  color: var(--ink-2);
}

@media (max-width: 900px) {
  .camera-on .dock {
    grid-template-columns: 44px repeat(6, minmax(0, 1fr));
  }
  .tiles.cam {
    grid-template-columns: repeat(6, minmax(0, 1fr));
  }
  /* six cards → two rows of three */
  .camera-on .dock > :nth-child(n + 5),
  .tiles.cam > .tile:nth-child(n + 4) {
    grid-column: span 2;
  }
  .camera-on .dock > .card-pace {
    grid-column: 2 / -1;
  }
}

@media (max-width: 600px) {
  .camera-on .dock,
  .tiles.cam {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  /* words + presence pair up; pace keeps its full row */
  .camera-on .dock > .card:nth-child(n + 6):not(.card-pace),
  .tiles.cam > .tile:last-child {
    grid-column: auto;
  }
  .self-view {
    width: 112px;
    height: 63px;
  }
}
```

- [ ] **Step 5: Wire it up in `src/main.ts`**

(a) Imports, after `import { scoreLinear } from "./presence";` (added in Task 2). Replace that line with:

```ts
import { scoreLinear, summarize, awayMs, weakestCue, PRESENCE_TUNING, type Frame } from "./presence";
import { startCamera, stopCamera } from "./camera";
```

(b) Session state: immediately after `let recording = false;` (near the top of the file), add:

```ts
// Camera presence (camera.ts / presence.ts): this session's frames, kept only
// while recording; summarized into the score by computeSummary.
let presenceFrames: Frame[] = [];
let presenceDrawnAt = 0;
```

(c) Add these functions directly above `function clearLive() {` (they are module-level functions, hoisted):

```ts
const CAMERA_KEY = "speech.camera";

function onPresenceFrame(f: Frame) {
  if (recording) presenceFrames.push(f);
  if (f.t - presenceDrawnAt >= 500) {
    presenceDrawnAt = f.t;
    renderPresenceCard();
  }
}

// Live Presence card: eye contact so far, plus one nudge.
function renderPresenceCard() {
  const p = summarize(presenceFrames, { scriptOpen: scriptUsed });
  setText("stat-eye", p ? String(Math.round(p.eyeContact * 100)) : "–");
  setWidth("eye-bar", p ? p.eyeContact : 0);
  const away = recording && awayMs(presenceFrames) > PRESENCE_TUNING.awayCueMs;
  setText("presence-sub", !recording ? "camera on" : away ? "look at the camera" : p ? weakestCue(p) : "warming up…");
}

async function setCamera(on: boolean) {
  const video = $<HTMLVideoElement>("self-view");
  $("camera-btn")?.setAttribute("aria-pressed", String(on));
  document.body.classList.toggle("camera-on", on);
  $("presence-card")?.toggleAttribute("hidden", !on);
  video?.toggleAttribute("hidden", !on);
  save(CAMERA_KEY, on ? "1" : "0");
  if (!on || !video) {
    stopCamera();
    return;
  }
  try {
    await startCamera(video, onPresenceFrame, () => setCamera(false));
    renderPresenceCard();
  } catch (e) {
    await setCamera(false);
    appendError(`Camera unavailable — ${e instanceof Error ? e.message : e}`);
    if (!recording) setStatus("Error"); // don't clobber "Recording"
  }
}
```

(d) In `function clearLive() {`, add as the first line of the body:

```ts
  presenceFrames = [];
```

and as the last line of the body (after the `transcriptEl.innerHTML = …` statement):

```ts
  if (document.body.classList.contains("camera-on")) renderPresenceCard();
```

(e) Init: directly after the line
`$("ribbon-btn")?.addEventListener("click", () => setRibbon(document.body.classList.contains("no-ribbon")));`
add:

```ts
  $("camera-btn")?.addEventListener("click", () => setCamera(!document.body.classList.contains("camera-on")));
  if (load(CAMERA_KEY) === "1") void setCamera(true);
```

- [ ] **Step 6: Build and check**

Run: `bun run build && bun src/presence.check.ts`
Expected: clean build; `presence ok`.

- [ ] **Step 7: Manual check in the real app**

Make sure the user's own dev instance isn't using port 1420 (or ask them to close it), then run:
`nix develop -c bun run tauri dev`

Verify each item:
1. Camera off (default): no self-view, no Presence card, and the dock looks exactly as before.
2. Click the camera button. The button goes pressed, a mirrored self-view appears under the clock, and the Presence card shows `–` / "camera on".
3. Start recording and talk for ~15 s. After about 3 s the card shows an eye-contact %. Look away for more than 2 s and the sub-line reads "look at the camera". Look back and it returns to a cue.
4. Leave it running for 60 s and confirm no web-process crash: `coredumpctl list --since "-5min"` shows no new `WebKitWebProcess` entry.
5. Toggle the camera off **while recording**. The recording continues and the self-view and card disappear. Toggle it back on and it resumes.
6. Rename `public/mediapipe/face_landmarker.task` temporarily, reload, and toggle on. The toggle turns back off and the transcript shows "Camera unavailable — Camera models missing — see README (…)". Restore the file.
7. Resize to ~632 px wide (Hyprland half tile) and ~560 px. With the camera on, the dock shows rows of three and then pairs, with no overlap and no hole.
8. Quit and relaunch with the camera on. It reopens automatically.

- [ ] **Step 8: Commit**

```bash
git add package.json bun.lock .gitignore README.md src/camera.ts index.html src/styles.css src/main.ts
git commit -m "presence: camera toggle, self-view and live presence card

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Never stage the user's unrelated edits; see "Before you start". Confirm `public/mediapipe` isn't staged: `git status --short | grep mediapipe` prints nothing.)

---

### Task 5: Presence in the composite score

**Files:**
- Modify: `src/main.ts`: `interface Scores`, `interface SessionSummary`, `SCORE_WEIGHTS`, `computeSummary`

**Interfaces:**
- Consumes: `presenceFrames`, `scriptUsed` (existing global), `summarize`, `Presence`.
- Produces: `Scores.presence?: number | null`, `SessionSummary.presence?: Presence`, `SCORE_WEIGHTS.presence = 1.5`. Task 6 relies on these names.

- [ ] **Step 1: Types**

Add `type Presence` to the presence import at the top of `src/main.ts`:

```ts
import { scoreLinear, summarize, awayMs, weakestCue, PRESENCE_TUNING, type Frame, type Presence } from "./presence";
```

In `interface Scores`, after `articulation: number | null; // null when no script was used`, add:

```ts
  presence?: number | null; // null/absent when the camera was off
```

In `interface SessionSummary`, after `crutch?: Crutch; …` add:

```ts
  presence?: Presence; // camera metrics; absent when the camera was off
```

- [ ] **Step 2: Weight**

Change

```ts
const SCORE_WEIGHTS = { pace: 1, fillers: 1.5, pauses: 0.75, pitch: 1, volume: 0.75, articulation: 1.5 };
```

to

```ts
const SCORE_WEIGHTS = { pace: 1, fillers: 1.5, pauses: 0.75, pitch: 1, volume: 0.75, articulation: 1.5, presence: 1.5 };
```

- [ ] **Step 3: `computeSummary`**

After `const articulation = script ? script.accuracy : null;` add:

```ts
  const presence = summarize(presenceFrames, { scriptOpen: scriptUsed });
```

After `if (articulation !== null) parts.push({ s: articulation, w: SCORE_WEIGHTS.articulation });` add:

```ts
  if (presence) parts.push({ s: presence.score, w: SCORE_WEIGHTS.presence });
```

Change the `scores` line to:

```ts
  const scores: Scores = { pace, fillers, pauses: pausesScore, pitch, volume, articulation, presence: presence?.score ?? null, overall };
```

In the returned object, after `crutch: crutch(spokenWords()),` add:

```ts
    presence: presence ?? undefined,
```

- [ ] **Step 4: Build**

Run: `bun run build && bun src/presence.check.ts`
Expected: clean; `presence ok`.

- [ ] **Step 5: Manual check: camera off changes nothing**

In the app with the camera **off**, record a ~20 s session and note the live ring score. In the devtools console (or a temporary `console.log`, removed afterwards) confirm `computeSummary().presence === undefined` and `scores.presence === null`. Then turn the camera **on**, record again while looking away the whole time, and confirm the ring score is lower than the same speech would get without the camera. That's presence pulling `overall` down.

- [ ] **Step 6: Commit**

```bash
git add src/main.ts
git commit -m "presence: add presence to the composite score

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Report tile, coach tips, speech overview

**Files:**
- Modify: `src/main.ts`: `reportTilesHtml`, `generateTips`, `DIMENSIONS` + `speechOverview`'s `detail` map

**Interfaces:**
- Consumes: `SessionSummary.presence`, `Scores.presence`, `SCORE_WEIGHTS.presence` (Task 5).
- Produces: user-visible output only.

- [ ] **Step 1: Report tile**

In `reportTilesHtml`, change the opening `` `<div class="tiles">` + `` to:

```ts
    `<div class="tiles${s.presence ? " cam" : ""}">` +
```

and immediately before the closing `` `</div>` `` of the return, after the `words` tile line, add:

```ts
    presenceTileHtml(s, prev) +
```

Then add this function directly below `reportTilesHtml`:

```ts
// Camera sessions only: presence score + the four signals behind it.
function presenceTileHtml(s: SessionSummary, prev: SessionSummary | undefined): string {
  const p = s.presence;
  if (!p) return "";
  const pct = (f: number) => `${Math.round(f * 100)}%`;
  const old = prev?.presence?.score;
  const [delta, cls] = typeof old === "number" ? [`${signed(p.score - old)} vs last`, tone(p.score - old)] : ["first with camera", "flat"];
  const moves = p.motion < 0.15 ? "steady" : p.motion < 0.4 ? "ok" : "restless";
  return (
    `<div class="tile" data-tip="Presence — eye contact, expression, stillness and gestures from the camera">` +
    `<div class="card-top"><span class="big">${p.score}</span><span class="unit">presence</span></div>` +
    `<span class="delta ${cls}">${delta}</span>` +
    `<div class="tile-rows">` +
    `<span>eye contact ${pct(p.eyeContact)}${p.eyeScored ? "" : " (not scored: script)"}</span>` +
    `<span>smiling ${pct(p.smileFrac)}</span>` +
    `<span>movement ${moves}</span>` +
    `<span>gesturing ${pct(p.gestureFrac)}</span>` +
    `</div></div>`
  );
}
```

- [ ] **Step 2: Coach tips**

In `generateTips`, immediately before `tips.sort((a, b) => b.impact - a.impact);`, add:

```ts
  const p = s.presence;
  if (p) {
    const w = SCORE_WEIGHTS.presence;
    const pct = (f: number) => `${Math.round(f * 100)}%`;
    if (p.eyeScored && p.eyeContact < 0.6)
      push(p.scores.eye, w, `You looked at the camera ${pct(p.eyeContact)} of the time — aim for 70%+. Glance at your notes, then come back to the lens.`);
    if (p.scores.expr !== null && p.scores.expr < 60)
      push(p.scores.expr, w, `Your face stayed mostly neutral (smiling ${pct(p.smileFrac)} of the time). Smile on your opening and closing lines.`);
    if (p.scores.still !== null && p.motion > 0.4)
      push(p.scores.still, w, `You moved around a lot (${p.motion.toFixed(2)} shoulder-widths a second). Plant your feet and keep your head steady.`);
    if (p.scores.gesture !== null && p.gestureFrac < 0.1)
      push(p.scores.gesture, w, `Your hands were still or out of frame ${pct(1 - p.gestureFrac)} of the time. Use them to mark your key points.`);
    else if (p.scores.gesture !== null && p.gestureFrac > 0.8)
      push(p.scores.gesture, w, `You gestured almost constantly (${pct(p.gestureFrac)} of the time). Save gestures for the points that matter.`);
  }
```

- [ ] **Step 3: Speech overview**

In `DIMENSIONS`, after `["articulation", "Script accuracy"],` add:

```ts
  ["presence", "Presence"],
```

In `speechOverview`'s `detail` object, after the `articulation: () => { … },` entry, add:

```ts
    presence: () =>
      `${Math.round(avg((h) => h.presence?.eyeContact) * 100)}% eye contact on average. Look at the lens on your key lines and let your hands carry the emphasis.`,
```

- [ ] **Step 4: Build**

Run: `bun run build && bun src/presence.check.ts`
Expected: clean; `presence ok`.

- [ ] **Step 5: Manual check**

1. Record a ~30 s camera session: look away a lot, stay expressionless, keep your hands down. Stop, and let the report open. The tiles row shows a sixth **Presence** tile with four rows. The coach's tips include the eye-contact and hands tips with real percentages.
2. Record a second camera session. The Presence tile shows "±N vs last".
3. **Old sessions:** open History and click a session saved *before* this feature. It renders with five tiles, no Presence tile, and no console errors. Open a saved speech that has old and new attempts and check the overview renders.
4. A session recorded with the camera off shows five tiles and no presence tips.
5. Narrow window (~632 px, ~560 px): the six report tiles form rows of three, then pairs.

- [ ] **Step 6: Commit**

```bash
git add src/main.ts
git commit -m "presence: report tile, coach tips and speech overview

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: End-to-end verification and tuning notes

**Files:**
- Modify (only if tuning is needed): `src/presence.ts` (`PRESENCE_TUNING` values only)

- [ ] **Step 1: All automated checks**

```bash
for f in src/*.check.ts; do bun "$f" || echo "FAIL $f"; done
bun run build
nix develop -c cargo test --manifest-path src-tauri/Cargo.toml
```

Expected: every check prints its ok line, there are no `FAIL` lines, the build is clean, and the Rust tests pass.

- [ ] **Step 2: Calibration pass (real footage)**

With the camera on, record three 30 s sessions:
- (a) looking straight at the lens, smiling, gesturing naturally, standing still;
- (b) reading notes on screen the whole time;
- (c) fidgeting and swaying.

Read the Presence tile rows for each. Expected ordering: (a) eye contact > 70% and movement "steady"; (b) eye contact < 40%; (c) movement "restless".
If (a) fails on eye contact, temporarily log `headAngles` / max `eyeLook*` from `camera.ts` and adjust `maxPitch` or `maxEyeLook` in `PRESENCE_TUNING`. A laptop camera sits above the screen, so a natural "looking at the screen" pitch may sit around 10–20°. Change **only** `PRESENCE_TUNING` values, rerun `bun src/presence.check.ts` (update expectations only where a test pins a changed constant), then remove the temporary logging.

- [ ] **Step 3: Spec checklist**

Walk the spec's sections (Rust, Assets, presence.ts, Metrics, Integration, Report/coach/history, Error handling, Out of scope) and confirm each item is implemented or explicitly out of scope. In particular confirm: no frames or landmarks in the saved session JSON (`history.rs` stores `computeSummary()`; inspect one saved file in the app data dir); with the camera off, `camera.ts`'s wasm and models are never fetched (devtools Network tab after a fresh launch).

- [ ] **Step 4: Commit (only if tuning changed)**

```bash
git add src/presence.ts src/presence.check.ts
git commit -m "presence: tune thresholds against real sessions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

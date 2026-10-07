// Run: bun src/presence.check.ts
import assert from "node:assert";
import { headAngles, toFrame, scoreLinear, summarize, gestureScore, awayMs, weakestCue, type RawResults, type Frame } from "./presence";

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

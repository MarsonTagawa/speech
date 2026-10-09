// Run: bun src/presence.check.ts
import assert from "node:assert";
import { toFrame, scoreLinear, summarize, gestureScore, awayMs, weakestCue, liveCue, movingHands, coverMap, type RawResults, type Frame } from "./presence";

const near = (a: number, b: number, eps = 0.01) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

// scoreLinear (moved from main.ts): unchanged behaviour.
assert.equal(scoreLinear(1, 1, 12), 100);
assert.equal(scoreLinear(12, 1, 12), 0);
assert.equal(scoreLinear(6.5, 1, 12), 50);
assert.equal(scoreLinear(3, 5, 5), 100);


// toFrame
const facing: Record<string, number> = { mouthSmileLeft: 0.4, mouthSmileRight: 0.6, browInnerUp: 0.2, browOuterUpLeft: 0.3, eyeLookDownLeft: 0.2 };
const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5 }));
pose[0] = { x: 0.5, y: 0.3 }; // nose
pose[11] = { x: 0.4, y: 0.5 }; // left shoulder
pose[12] = { x: 0.6, y: 0.5 }; // right shoulder
const raw = (o: Partial<RawResults> = {}): RawResults => ({ blend: facing, head: { yaw: 5, pitch: 0 }, pose, wrists: [], ...o });

const none = toFrame(raw({ blend: null, head: null, pose: null }), 640, 360, 0);
assert.deepStrictEqual(none, { t: 0, face: false, looking: false, smile: 0, brows: 0, body: null, wrists: [], sw: 0 });

const f = toFrame(raw({ wrists: [{ x: 0.5, y: 0.8 }] }), 640, 360, 42);
assert.equal(f.t, 42);
assert.equal(f.face, true);
assert.equal(f.looking, true); // 5° yaw, eyes 0.2 < 0.5
near(f.smile, 0.5);
near(f.brows, 0.3);
// shoulders 128 px apart, midpoint (320,180) px; points stay in pixels
near(f.sw, 128);
near(f.body![0][0], 320);
near(f.body![0][1], 180);
near(f.body![1][1], 108); // nose y = 0.3·360
near(f.wrists[0][1], 288); // wrist y = 0.8·360

assert.equal(toFrame(raw({ head: { yaw: 30, pitch: 0 } }), 640, 360, 0).looking, false); // head turned
assert.equal(toFrame(raw({ blend: { ...facing, eyeLookOutLeft: 0.7 } }), 640, 360, 0).looking, false); // eyes away
assert.equal(toFrame(raw({ head: null }), 640, 360, 0).looking, false); // no head pose → can't tell → not looking

// Degenerate pose (shoulders coincide): no body, no wrists, nothing infinite.
const flat = pose.map((p) => ({ ...p }));
flat[12] = { ...flat[11] };
const d = toFrame(raw({ pose: flat, wrists: [{ x: 0.5, y: 0.8 }] }), 640, 360, 0);
assert.equal(d.body, null);
assert.deepStrictEqual(d.wrists, []);

// --- summarize ------------------------------------------------------------
const STEP = 100; // 10 fps keeps the expected timings round
const F = (t: number, o: Partial<Frame> = {}): Frame => ({ t, face: true, looking: true, smile: 0, brows: 0, body: [[2.5, 1.4], [2.5, 0.8]], wrists: [], sw: 1, ...o });
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

// --- review fix: shoulder-width jitter isn't motion -------------------------
// A perfectly still speaker whose detected shoulder width wobbles ±2% frame to
// frame (landmark noise / leaning) must not read as movement or gesturing.
const stillRaw = (i: number): RawResults => {
  const half = 0.1 * (1 + (i % 2 ? 0.02 : -0.02)); // shoulders ±2% around a fixed midpoint
  const p = pose.map((q) => ({ ...q }));
  p[11] = { x: 0.5 - half, y: 0.5 };
  p[12] = { x: 0.5 + half, y: 0.5 };
  return { blend: facing, head: { yaw: 0, pitch: 0 }, pose: p, wrists: [{ x: 0.7, y: 0.8 }] };
};
const stillFrames = Array.from({ length: 101 }, (_, i) => toFrame(stillRaw(i), 640, 360, i * STEP));
const stillP = summarize(stillFrames, open)!;
assert.ok(stillP.motion < 0.01, `still speaker motion ${stillP.motion}`);
assert.equal(stillP.gestureFrac, 0);

// Near side-on: shoulders 10 px apart (< 5% of the frame width) → too unreliable to scale by.
const sideOn = pose.map((q) => ({ ...q }));
sideOn[11] = { x: 0.5, y: 0.5 };
sideOn[12] = { x: 0.5 + 10 / 640, y: 0.5 };
assert.equal(toFrame(raw({ pose: sideOn }), 640, 360, 0).body, null);

// --- review fix: live cue -------------------------------------------------
// Looking away for 3 s: nag only when eye contact is actually scored.
const lookedAway = frames(10, (t) => ({ looking: t < 7000 }));
assert.equal(liveCue(lookedAway, { recording: true, scriptOpen: false }), "look at the camera");
assert.notEqual(liveCue(lookedAway, { recording: true, scriptOpen: true }), "look at the camera"); // reading a script looks down
assert.equal(liveCue(lookedAway, { recording: false, scriptOpen: false }), "camera on");
assert.equal(liveCue(frames(2), { recording: true, scriptOpen: false }), "warming up…");
assert.equal(liveCue(frames(10), { recording: true, scriptOpen: false }), "smile more");

// --- hand highlight ---------------------------------------------------------
// A hand counts as moving when its wrist travels more than gestureStep shoulder
// widths since the previous frame — the same rule the gesture score uses.
const hand = (x: number, y: number) => Array.from({ length: 21 }, () => ({ x, y }));
// 640×360 frame, shoulders 128 px → gestureStep 0.04 = 5.12 px
assert.deepStrictEqual(movingHands([hand(0.5, 0.5)], [hand(0.5 + 10 / 640, 0.5)], 640, 360, 128), [true]); // 10 px
assert.deepStrictEqual(movingHands([hand(0.5, 0.5)], [hand(0.5 + 3 / 640, 0.5)], 640, 360, 128), [false]); // 3 px: jitter
assert.deepStrictEqual(movingHands([], [hand(0.5, 0.5)], 640, 360, 128), [false]); // just appeared: no motion yet
// two hands, matched to the nearest previous wrist; only the right one moved
assert.deepStrictEqual(movingHands([hand(0.2, 0.5), hand(0.8, 0.5)], [hand(0.2, 0.5), hand(0.8, 0.6)], 640, 360, 128), [false, true]);
// no body to scale by → assume a typical shoulder width (20% of the frame): 10 px still counts
assert.deepStrictEqual(movingHands([hand(0.5, 0.5)], [hand(0.5 + 10 / 640, 0.5)], 640, 360, 0), [true]);

// coverMap: normalised camera coords → pixels in an object-fit: cover box.
const exact = coverMap(1920, 1080, 160, 90); // same aspect: no crop
assert.deepStrictEqual(exact({ x: 0.5, y: 0.5 }), [80, 45]);
assert.deepStrictEqual(exact({ x: 1, y: 1 }), [160, 90]);
const wide = coverMap(1920, 1080, 300, 100); // wider box: crops top and bottom
assert.deepStrictEqual(wide({ x: 0.5, y: 0.5 }), [150, 50]);
near(wide({ x: 0, y: 0 })[1], -34.375);
const tall = coverMap(1920, 1080, 100, 100); // square box: crops the sides
near(tall({ x: 0, y: 0.5 })[0], -38.888);
near(tall({ x: 0.5, y: 0 })[1], 0);

console.log("presence ok");

// Run: bun src/gestures.check.ts
import assert from "node:assert";
import { readFileSync } from "node:fs";
import { counting, detect, gestureSummary, holds, lists, poses, readHand, type GestureId, type Hand } from "./gestures";
import { toFrame, whichHands, type Frame, type Pt, type XY } from "./presence";

const W = 640;
const H = 360;

// --- readHand: the golden test image --------------------------------------
// MediaPipe's thumbs-up photo: four fingers curled, thumb out. x/y from
// MediaPipe (goldens.json); z is our model's output on it (vision/hands.rs),
// with handedness 0.99. The curled fingers sit nearer the camera than the
// knuckles, so the palm faces the camera — which is what fixes the sign
// convention: a score over 0.5 is the speaker's right hand.
const golden = JSON.parse(readFileSync(new URL("../src-tauri/resources/vision/golden/goldens.json", import.meta.url), "utf8"))["thumb_up.jpg"];
const gz = [0, -11, -14, -16, -16, -7, -16, -18, -18, -4, -13, -14, -14, -3, -13, -10, -7, -3, -8, -6, -3];
const thumbUp = readHand(golden.hands[0].map(([x, y]: number[], i: number) => ({ x, y, z: gz[i] / golden.w })), true, golden.w, golden.h);
assert.equal(thumbUp.shape, "1");
assert.equal(thumbUp.palm, "out");
// The same hand's metric 3D "world" landmarks (frame axes, from vision/hands.rs)
// agree; they're what the app reads shape and palm from.
const gworld = [[0.0677, 0.0305, 0.0544], [0.064, -0.0042, 0.0219], [0.0543, -0.0385, 0.0095], [0.0357, -0.0679, -0.0004], [0.021, -0.0853, 0.0011], [0.0042, -0.0269, -0.004], [-0.0026, -0.0244, -0.0334], [0.0078, -0.0195, -0.0326], [0.0249, -0.0155, -0.006], [-0.0046, -0.0035, 0.0024], [-0.0102, -0.0038, -0.0353], [0.0169, 0.0016, -0.0352], [0.0204, -0.0035, 0.0043], [-0.0056, 0.0163, 0.0033], [-0.0089, 0.0166, -0.0286], [0.0162, 0.0182, -0.0262], [0.0225, 0.0126, -0.0004], [0.0022, 0.0424, 0.0065], [-0.0092, 0.0376, -0.0158], [0.0074, 0.0344, -0.0286], [0.0143, 0.0372, -0.0138]];
const thumbUpW = readHand(golden.hands[0].map(([x, y]: number[]) => ({ x, y })), true, golden.w, golden.h, undefined, gworld);
assert.equal(thumbUpW.shape, "1");
assert.equal(thumbUpW.palm, "out");

// --- synthetic hands ----------------------------------------------------------
// A right hand, palm to the camera, fingers up, in pixels around the wrist
// (x right, y down, z away from the camera). Curled fingers fold toward the
// palm side (nearer the camera); a tucked thumb crosses toward the pinky knuckle.
type V3 = [number, number, number];
const MCP: V3[] = [[12, -60, 0], [0, -62, 0], [-11, -60, 0], [-22, -55, 0]];
function canon(thumb: boolean, fingers: boolean[]): V3[] {
  const p: V3[] = Array(21);
  p[0] = [0, 0, 0];
  p[1] = [15, -10, -5];
  p[2] = [25, -20, -8];
  [p[3], p[4]] = thumb ? [[35, -30, -10], [55, -38, -12]] : [[15, -42, -12], [-5, -50, -15]];
  fingers.forEach((up, k) => {
    const [x, y] = MCP[k];
    const b = 5 + 4 * k;
    p[b] = MCP[k];
    p[b + 1] = up ? [x, y - 25, 0] : [x, y - 22, -5];
    p[b + 2] = up ? [x, y - 42, 0] : [x, y - 12, -22];
    p[b + 3] = up ? [x, y - 55, 0] : [x, y + 5, -15];
  });
  return p;
}
// Turn the palm: about x (+90° = palm down, fingers toward the camera; −90° =
// palm up), or about y (±90° = upright palm facing image left/right; 180° = back).
const rx = (deg: number) => ([x, y, z]: V3): V3 => {
  const [s, c] = [Math.sin((deg * Math.PI) / 180), Math.cos((deg * Math.PI) / 180)];
  return [x, y * c - z * s, y * s + z * c];
};
const ry = (deg: number) => ([x, y, z]: V3): V3 => {
  const [s, c] = [Math.sin((deg * Math.PI) / 180), Math.cos((deg * Math.PI) / 180)];
  return [x * c + z * s, y, -x * s + z * c];
};
// …or about z (the camera axis: +90° = fingers pointing image right).
const rz = (deg: number) => ([x, y, z]: V3): V3 => {
  const [s, c] = [Math.sin((deg * Math.PI) / 180), Math.cos((deg * Math.PI) / 180)];
  return [x * c - y * s, x * s + y * c, z];
};
interface Mk {
  right?: boolean;
  thumb?: boolean;
  fingers?: boolean[];
  turn?: (v: V3) => V3;
  at?: XY; // where the wrist goes, px
  scale?: number;
  edit?: (p: V3[]) => void;
}
function pts({ right = true, thumb = true, fingers = [true, true, true, true], turn = (v) => v, at = [320, 250], scale = 1, edit }: Mk = {}): Pt[] {
  const p = canon(thumb, fingers);
  edit?.(p);
  return p.map((v) => {
    const [x, y, z] = turn(v).map((c) => c * scale);
    return { x: (at[0] + (right ? x : -x)) / W, y: (at[1] + y) / H, z: z / W }; // a left hand is the mirror image
  });
}
const hand = (o: Mk = {}) => readHand(pts(o), o.right ?? true, W, H);

const shape = (thumb: boolean, ...fingers: boolean[]) => hand({ thumb, fingers }).shape;
assert.equal(shape(false, false, false, false, false), "fist");
assert.equal(shape(false, true, false, false, false), "1");
assert.equal(shape(false, true, true, false, false), "2");
assert.equal(shape(true, true, true, false, false), "3"); // thumb-first counting
assert.equal(shape(false, true, true, true, false), "3"); // index-first
assert.equal(shape(false, true, true, true, true), "4");
assert.equal(shape(true, true, true, true, true), "open");
// 2D-only landmarks (no z) still count fingers
assert.equal(readHand(pts({ thumb: false, fingers: [true, true, false, false] }).map(({ x, y }) => ({ x, y })), true, W, H).shape, "2");

// palm direction, on both hands
for (const right of [true, false]) {
  assert.equal(hand({ right }).palm, "out");
  assert.equal(hand({ right, turn: rx(90) }).palm, "down");
  assert.equal(hand({ right, turn: rx(-90) }).palm, "up");
  assert.equal(hand({ right, turn: ry(180) }).palm, "in");
  assert.equal(hand({ right, turn: ry(90) }).palm, "side");
}
assert.ok(hand({ turn: ry(90) }).n[0] < 0 && hand({ turn: ry(-90) }).n[0] > 0); // which way an upright palm faces

// Palm down, seen from a little above (a webcam at eye level, hand at chest
// height). The image z is squashed ~7× (measured on the golden hand), which
// hides that the palm faces down; the metric world landmarks don't.
const tilted = pts({ turn: rx(70) });
const squashed = tilted.map((q) => ({ ...q, z: q.z! / 7 }));
const worldOf = (q: Pt[]) => q.map((v) => [v.x * W, v.y * H, v.z! * W].map((c) => c / 1000)); // any unit: only directions matter
assert.notEqual(readHand(squashed, true, W, H).palm, "down");
assert.equal(readHand(squashed, true, W, H, undefined, worldOf(tilted)).palm, "down");
assert.equal(readHand(squashed, false, W, H, undefined, worldOf(tilted)).palm, "up"); // same points called a left hand: mirror sign

// Thumb pressed straight along the side of the hand: out on a flat hand, but a
// fist with the thumb resting on its side is still a fist.
const alongside = (p: V3[]) => {
  [p[1], p[2], p[3], p[4]] = [[12, -12, -3], [18, -28, -5], [20, -42, -6], [21, -55, -6]];
};
assert.equal(hand({ edit: alongside }).shape, "open");
assert.equal(hand({ fingers: [false, false, false, false], edit: alongside }).shape, "fist");
assert.equal(hand({ thumb: false }).shape, "4"); // tucked across the palm is still in

// Palm direction steadies: a palm wobbling either side of halfway between
// facing the camera and facing up (40° / 50°) reads one way every frame…
const wobble = (prevHand?: Hand) => {
  const seen: string[] = [];
  let h = prevHand;
  for (let i = 0; i < 12; i++) {
    h = readHand(pts({ turn: rx(i % 2 ? -50 : -40) }), true, W, H, prevHand && h);
    seen.push(h.palm);
  }
  return new Set(seen).size;
};
assert.equal(wobble(), 2); // frame by frame, unsmoothed, it flips
assert.equal(wobble(hand()), 1);
// …but a real turn still comes through within a few frames.
let turning = hand();
for (let i = 0; i < 4; i++) turning = readHand(pts({ turn: rx(-90) }), true, W, H, turning);
assert.equal(turning.palm, "up");
assert.ok(hand().up && !hand({ turn: rx(90) }).up);

// A palm tilted between two directions faces both: up and out at 45°, and
// only one when nearly square to one; side never pairs.
const tilt = (deg: number) => { const h = hand({ turn: rx(deg) }); return [h.palm, h.tilt].filter(Boolean).sort().join("+"); };
assert.equal(tilt(-45), "out+up");
assert.equal(tilt(45), "down+out");
assert.equal(tilt(-15), "out");
assert.equal(hand({ turn: ry(90) }).tilt, undefined);

// --- poses: one frame ------------------------------------------------------
const U = 128; // shoulder width, px
const CHEST: XY = [320, 150];
const has = (hs: Hand[], g: GestureId) => poses(hs, U, CHEST).some((p) => p.g === g);

assert.ok(has([hand({ thumb: false, fingers: [false, false, false, false] })], "fist"));
assert.ok(has([hand({ thumb: false, fingers: [true, false, false, false] })], "point"));
const tiny = hand({
  thumb: true,
  fingers: [false, false, false, false],
  edit: (p) => {
    [p[6], p[7], p[8]] = [[14, -82, -5], [20, -92, -5], [25, -95, -5]]; // index bent forward…
    [p[3], p[4]] = [[35, -60, -5], [40, -85, -5]]; // …thumb tip just below it
  },
});
assert.ok(has([tiny], "tiny") && !has([tiny], "point") && !has([tiny], "fist"));
assert.ok(has([hand({ turn: rx(-90) })], "palmsup"));
assert.ok(has([hand({ turn: rx(90) })], "palmsdown"));
assert.ok(has([hand({ turn: rx(-45) })], "palmsup")); // tilted up toward the camera still counts
assert.ok(has([hand({ turn: ry(180), at: [320, 230] })], "me")); // back of the hand to the camera, on the chest
assert.ok(!has([hand({ turn: ry(180), at: [500, 230] })], "me")); // off to the side
const upIn = hand({ turn: rx(-125), at: [320, 230] }); // palm up, tilted toward the body: a reach, not a hand on the chest
assert.ok(upIn.palm === "up" && upIn.tilt === "in" && !has([upIn], "me"));
const fistIn = hand({ thumb: false, fingers: [false, false, false, false], turn: ry(180), at: [320, 230] }); // a fist turned over, at the chest
assert.ok(fistIn.palm === "in" && !has([fistIn], "me"));
assert.ok(!poses([hand({ turn: ry(180), at: [320, 230] })], U, null).length); // no body, no chest
const L = (o: Mk = {}) => hand({ right: false, ...o });
assert.ok(has([L({ at: [180, 250] }), hand({ at: [460, 250] })], "grand"));
assert.ok(!has([L({ at: [180, 250] }), hand({ at: [460, 250] })], "stop")); // two palms out = grand, not stop
// Stop: raised to the shoulders (wrist at 200 puts the palm at ~shoulder height) and square on.
assert.ok(has([hand({ at: [320, 200] })], "stop"));
assert.ok(!has([hand()], "stop")); // palm out but low: just talking
assert.ok(!has([hand({ at: [320, 200], turn: rx(-45) })], "stop")); // tilted up, not flashed at the camera
assert.ok(!has([hand({ turn: rx(90) })], "stop"));
assert.ok(!poses([hand({ at: [320, 200] })], U, null).some((p) => p.g === "stop")); // no body to say it's raised
// I'm ready: fingertips touching (a steeple), not just hands close together
const steepleR = hand({ at: [360, 250] });
const steepleL: Hand = { ...L({ at: [280, 250] }), tips: steepleR.tips.map(([x, y], k): XY => [x + (k % 2 ? 4 : -4), y + 3]) };
assert.ok(has([steepleL, steepleR], "ready"));
const upR = hand({ at: [360, 250], turn: rx(-90) });
assert.ok(!has([{ ...L({ at: [280, 250], turn: rx(-90) }), tips: upR.tips.map(([x, y]): XY => [x + 4, y + 3]) }, upR], "ready")); // palms up
assert.ok(!has([L({ at: [300, 250] }), hand({ at: [340, 250] })], "ready")); // side by side, tips apart
const ghost: Hand = { ...steepleR, right: false, c: [steepleR.c[0] + 10, steepleR.c[1] + 5], tips: steepleR.tips.map(([x, y]): XY => [x + 10, y + 5]) };
assert.ok(!has([ghost, steepleR], "ready")); // one hand seen twice: tips line up, but so does everything
assert.ok(!has([{ ...steepleL, tips: steepleR.tips.map(([x, y], k): XY => (k < 3 ? [x + 60, y] : [x, y])) }, steepleR], "ready")); // only 2 touching
// fingertips touching but pointing at each other, or down: not ready
for (const deg of [90, 180]) {
  const r = hand({ at: [360, 250], turn: rz(-deg) });
  assert.ok(!has([{ ...L({ at: [280, 250], turn: rz(-deg) }), tips: r.tips.map(([x, y], k): XY => [x + (k % 2 ? 4 : -4), y + 3]) }, r], "ready"), String(deg));
}
const fwdR = hand({ at: [360, 250], turn: rx(60) }); // fingers tipped toward the camera
assert.ok(has([{ ...L({ at: [280, 250], turn: rx(60) }), tips: fwdR.tips.map(([x, y], k): XY => [x + (k % 2 ? 4 : -4), y + 3]) }, fwdR], "ready"));
// It's just so: two upright palms facing each other. The speaker's right hand
// is on the image left; its palm faces image right (ry(-90)), the left one's faces left.
assert.ok(has([hand({ at: [250, 250], turn: ry(-90) }), L({ at: [400, 250], turn: ry(-90) })], "justso"));
assert.ok(!has([hand({ at: [250, 250], turn: ry(90) }), L({ at: [400, 250], turn: ry(90) })], "justso")); // facing away
// Just a part: thumb out and index bent toward it, a wider gap than a tiny bit.
const part = hand({
  thumb: true,
  fingers: [false, false, false, false],
  edit: (p) => {
    [p[6], p[7], p[8]] = [[14, -82, -5], [20, -92, -5], [25, -95, -5]];
    [p[3], p[4]] = [[35, -45, -8], [50, -55, -8]];
  },
});
assert.ok(part.pinch >= 0.6 && part.pinch < 1.1);
assert.ok(has([part], "part") && !has([part], "tiny") && !has([part], "point"));
assert.ok(!has([tiny], "part"));
assert.ok(!has([hand({ thumb: false, fingers: [true, false, false, false] })], "part")); // pointing, thumb tucked
const tucked = hand({ thumb: false, fingers: [true, false, false, false], edit: (p) => ([p[3], p[4]] = [[15, -55, -12], [5, -65, -15]]) });
assert.ok(!tucked.ext[0] && tucked.pinch >= 0.6 && tucked.pinch < 1.1 && !has([tucked], "part")); // gap in range, but the thumb's tucked
const gun = hand({ thumb: true, fingers: [true, false, false, false] }); // thumb out in an L: too wide
assert.ok(gun.pinch >= 1.1 && has([gun], "point") && !has([gun], "part"));
assert.ok(!has([hand({ turn: ry(90) })], "part")); // the old edge-on hand isn't it any more
// We: both open hands in view, wide apart (2.2 u = 282 px).
assert.ok(has([L({ turn: rx(-90), at: [80, 250] }), hand({ turn: rx(-90), at: [560, 250] })], "we"));
assert.ok(!has([L({ turn: rx(-90), at: [200, 250] }), hand({ turn: rx(-90), at: [440, 250] })], "we")); // not wide enough
assert.ok(!has([hand({ turn: rx(-90), at: [80, 250] })], "we")); // one hand
assert.ok(!has([L({ turn: rx(-90), at: [80, 250], fingers: [false, false, false, false] }), hand({ turn: rx(-90), at: [560, 250] })], "we")); // a fist

// --- detect: over time ---------------------------------------------------------
const STEP = 80; // 12.5 Hz, like the camera
const frame = (t: number, hands: Hand[]): Frame => ({
  t, face: true, looking: true, smile: 0, brows: 0, body: [CHEST, [320, 80]], wrists: [], sw: U, u: U, hands, poses: poses(hands, U, CHEST),
});
const run = (n: number, make: (i: number) => Hand[], t0 = 0, step = STEP) => Array.from({ length: n }, (_, i) => frame(t0 + i * step, make(i)));
const kinds = (fs: Frame[]) => detect(fs).map((e) => e.g);
const count = (fs: Frame[], g: GestureId) => kinds(fs).filter((k) => k === g).length;

// A held pose is one event; flickering never holds.
assert.equal(count(run(15, () => [hand({ turn: rx(-90) })]), "palmsup"), 1);
assert.equal(count(run(15, (i) => (i % 3 ? [hand({ turn: rx(-90) })] : [])), "palmsup"), 0);

// Everything: a flat palm-down hand sweeping sideways.
assert.equal(count(run(10, (i) => [hand({ turn: rx(90), at: [150 + i * 25, 250] })]), "everything"), 1);
assert.equal(count(run(10, (i) => [hand({ turn: rx(90), at: [150 + i * 5, 250] })]), "everything"), 0); // too small
// a fast sweep where the hand blurs out of a frame or two still counts
assert.equal(count(run(8, (i) => (i === 3 || i === 5 ? [] : [hand({ turn: rx(90), at: [150 + i * 40, 250] })])), "everything"), 1);

// Growth: a hand rising; sideways travel isn't growth.
assert.equal(count(run(10, (i) => [hand({ thumb: false, fingers: [true, false, false, false], at: [320, 320 - i * 15] })]), "growth"), 1);
assert.equal(count(run(10, (i) => [hand({ at: [150 + i * 25, 250] })]), "growth"), 0);

// You: an open palm coming toward the camera (getting bigger)…
assert.equal(count(run(14, (i) => [hand({ turn: rx(-90), scale: 1 + Math.min(i, 7) * 0.08 })]), "you"), 1); // then held: "you" once it doesn't end at the chest
assert.equal(count(run(8, () => [hand({ turn: rx(-90) })]), "you"), 0);
assert.equal(count(run(14, (i) => [hand({ turn: rx(-125), scale: 1 + Math.min(i, 7) * 0.08 })]), "you"), 0); // palm tilted in: toward me, not you
// not a hand tipping away and back (shrinks, then regrows to its size)
assert.equal(count(run(20, (i) => [hand({ turn: rx(-90), scale: 1 - 0.3 * Math.sin(Math.min(i, 10) * Math.PI / 10) })]), "you"), 0);
// nor one lost and found bigger, far off (reaching for the mouse)
assert.equal(count(run(16, (i) => (i < 6 ? [hand({ turn: rx(-90), at: [320, 300] })] : i < 8 ? [] : [hand({ turn: rx(-90), at: [60, 200], scale: 1.4 })])), "you"), 0);
// …or a flat palm-down hand near the camera (big); further back it's palms down.
assert.ok(has([hand({ turn: rx(90), scale: 1.6 })], "you") && !has([hand({ turn: rx(90), scale: 1.6 })], "palmsdown"));
assert.ok(!has([hand({ turn: rx(90) })], "you") && has([hand({ turn: rx(90) })], "palmsdown"));
assert.ok(!has([hand({ scale: 1.6 })], "you")); // palm to the camera, held near: not the palm-down pose
assert.ok(!has([hand({ turn: rx(-90), scale: 1.6 })], "you")); // palm up
assert.equal(count(run(8, () => [hand({ turn: rx(90), scale: 1.6 })]), "you"), 1);

// A tiny bit held still fires; a hand closing through the same gap doesn't.
assert.equal(count(run(10, () => [tiny]), "tiny"), 1);
const closing = run(10, (i) => [hand({
  thumb: true,
  fingers: [false, false, false, false],
  edit: (p) => {
    [p[6], p[7], p[8]] = [[14, -82, -5], [20, -92, -5], [25, -95, -5]];
    [p[3], p[4]] = [[35, -60, -5], [45 - i * 2, -60 - i * 5, -5]];
  },
})]);
assert.ok(count(closing, "tiny") === 0 && closing.filter((f) => f.poses.some((p) => p.g === "tiny")).length >= 4);

// Listen here: one hand chops down onto the other.
const base = L({ turn: rx(-90), at: [300, 300] });
assert.equal(count(run(6, (i) => [base, hand({ turn: ry(90), at: [330, 180 + i * 25] })]), "listen"), 1);

// Come together: hands from wide apart to touching.
assert.equal(count(run(10, (i) => [L({ at: [150 + i * 16, 250] }), hand({ at: [490 - i * 16, 250] })]), "together"), 1);
assert.equal(count(run(10, () => [L({ at: [300, 250] }), hand({ at: [340, 250] })]), "together"), 0); // already together

// Small, medium, large: a palm-down hand held at one height, then another.
const levels = run(16, (i) => [hand({ turn: rx(90), at: [320, i < 6 ? 150 : i < 9 ? 150 + (i - 5) * 25 : 250] })]);
assert.equal(count(levels, "levels"), 1);
assert.equal(count(run(16, () => [hand({ turn: rx(90) })]), "levels"), 0); // one height only

// This and that: one hand moves while the other holds, then the other.
const tt = run(8, (i) => [L({ at: [200, 250 - (i < 4 ? i * 20 : 60)] }), hand({ at: [440, 250 - (i < 4 ? 0 : (i - 3) * 20)] })]);
assert.equal(count(tt, "thisthat"), 1);
assert.equal(count(run(8, (i) => [L({ at: [200, 250 - i * 20] }), hand({ at: [440, 250 - i * 20] })]), "thisthat"), 0); // both at once

// Listing: 1 → 2 → 3 on the fingers, each held.
const fingersUp = (n: number) => [true, true, true, true].map((_, k) => k < n);
const listing = run(30, (i) => [hand({ thumb: false, fingers: fingersUp(1 + Math.floor(i / 10)) })]);
assert.deepStrictEqual(kinds(listing).filter((k) => k === "list"), ["list"]);
// passing through the counts (~0.3 s each) isn't holding them
assert.equal(count(run(12, (i) => [hand({ thumb: false, fingers: fingersUp(1 + Math.floor(i / 4)) })]), "list"), 0);

// Half-closed fingers aren't a count, so they can't make a list even when the
// fingers that are straight go 1 → 2 → 3. (Half-bent: 30°/45°/30° at the
// knuckle/middle/end joints, toward the camera; this read as out before.)
const halfBend = (k: number) => (p: V3[]) => {
  const [x, y] = MCP[k];
  [p[6 + 4 * k], p[7 + 4 * k], p[8 + 4 * k]] = [[x, y - 21.7, -12.5], [x, y - 26.1, -28.9], [x, y - 22.7, -41.5]];
};
const halfClosed = (n: number) => hand({ thumb: false, fingers: fingersUp(n), edit: (p) => [0, 1, 2, 3].filter((k) => k >= n).forEach((k) => halfBend(k)(p)) });
assert.deepStrictEqual(halfClosed(1).ext, [false, true, false, false, false]);
// folded fingers that aren't fully curled (0.65–0.8 straight) still let a hand count
// finger k bent toward the camera by these angles (°) at its three joints
const bend = (k: number, angles: number[]) => (p: V3[]) => {
  let [y, z, a] = [MCP[k][1], 0, 0];
  [25, 17, 13].forEach((len, j) => {
    a += (angles[j] * Math.PI) / 180;
    [y, z] = [y - len * Math.cos(a), z - len * Math.sin(a)];
    p[6 + 4 * k + j] = [MCP[k][0], y, z];
  });
};
const looseFold = (n: number) => hand({ thumb: false, fingers: fingersUp(n), edit: (p) => [0, 1, 2, 3].filter((k) => k >= n).forEach((k) => bend(k, [45, 60, 40])(p)) });
assert.ok(looseFold(2).straightness[2] > 0.65 && looseFold(2).straightness[2] < 0.8);
assert.equal(counting(looseFold(2))?.map(Number).join(""), "01100");
assert.equal(counting(looseFold(3))?.map(Number).join(""), "01110");

// A cupped hand: all four fingers bent alike, a little, thumb toward the index
// tip. None reads out, but none is curled either: not tiny, part or pointing.
const cupped: Mk = { thumb: true, fingers: fingersUp(4), edit: (p) => {
  [0, 1, 2, 3].forEach((k) => bend(k, [35, 45, 30])(p)); // ~0.86 straight each
  [p[3], p[4]] = [[30, -50, -15], [30, -70, -25]];
} };
const cup = hand(cupped);
// cupped toward the chest is "me" (facing in, or read as palm up); a flat palm up there isn't
assert.ok(has([hand({ ...cupped, turn: rx(-90), at: [320, 230] })], "me"));
assert.ok(!has([hand({ turn: rx(-90), at: [320, 230] })], "me"));
// cupped, fingers pointing sideways at the other hand: "me", coming closer or not
assert.ok(has([hand({ ...cupped, turn: rz(90), at: [240, 240] })], "me"));
{ const downCup = hand({ ...cupped, turn: (v) => rz(90)(ry(-60)(v)), at: [240, 240] }); assert.ok(downCup.palm === "down" && has([downCup], "me")); } // turned palm down
{ const flatUp = hand({ turn: (v) => ry(90)(rx(-90)(v)), at: [240, 240] }); assert.ok(flatUp.palm === "up" && flatUp.axis[0] > 0.9 && !has([flatUp], "me")); } // flat, palm up: palms up, not me
assert.ok(!has([hand({ ...cupped, turn: rz(-90), at: [240, 240] })], "me")); // pointing away
assert.equal(count(run(14, (i) => [hand({ turn: rz(90), at: [240, 240], scale: 1 + Math.min(i, 7) * 0.08 })]), "you"), 0);
assert.equal(count(run(14, (i) => [hand({ turn: rz(-90), at: [240, 240], scale: 1 + Math.min(i, 7) * 0.08 })]), "you"), 1); // pointing away: a reach
// a palm-up hand coming up and growing, then held to the chest: "me", never "you"
const toChest = run(14, (i) => [i < 8 ? hand({ turn: rx(-90), at: [320, 300 - i * 8], scale: 1 + i * 0.08 }) : hand({ turn: ry(180), at: [320, 230], scale: 1.6 })]);
assert.equal(count(toChest, "you"), 0);
assert.equal(count(toChest, "me"), 1);
assert.equal(count(toChest.slice(0, 8), "you"), 0); // not yet: it might still end at the chest
// cupped sideways: "me"
{ const sc = hand({ ...cupped, turn: ry(-90), at: [260, 240] }); assert.ok(sc.palm === "side" && sc.n[0] > 0 && has([sc], "me")); }
{ const sc = hand({ ...cupped, turn: ry(90), at: [260, 240] }); assert.ok(sc.palm === "side" && sc.n[0] < 0 && has([sc], "me")); } // turning, it reads facing away too
assert.ok(has([hand({ ...cupped, turn: rx(90), at: [320, 230] })], "me")); // cupped palm down
assert.ok(!has([hand({ ...cupped, at: [320, 230] })], "me")); // cupped toward the camera: offering, not me
assert.ok(!has([hand({ ...cupped, turn: ry(180), at: [440, 230] })], "me")); // beside the body
{ const sc = hand({ turn: ry(-90), at: [260, 240] }); assert.ok(sc.palm === "side" && !has([sc], "me")); } // flat: just so, not me
// near the camera, palm down, but fingers pointing at the other hand: not "you"
{ const h = hand({ turn: (v) => ry(-90)(rx(90)(v)), at: [200, 240], scale: 1.6 }); assert.ok(h.palm === "down" && h.axis[0] > 0.9 && h.size >= 0.6 * U && !has([h], "you")); }
assert.ok(has([hand({ ...cupped, turn: ry(180), at: [320, 230] })], "me"));
assert.ok(has([hand({ ...cupped, edit: (p) => { cupped.edit!(p); [0, 1, 2, 3].forEach((k) => bend(k, [20, 20, 10])(p)); }, turn: ry(180), at: [320, 230] })], "me")); // cupped more
assert.ok(cup.ext.slice(1).every((e) => !e));
assert.ok(!["tiny", "part", "point"].some((g) => has([cup], g as GestureId)));
assert.equal(count(run(30, (i) => [halfClosed(1 + Math.floor(i / 10))]), "list"), 0);
// a finger flickering to half-bent for a frame doesn't break the hold…
const flicker = run(30, (i) => [i % 5 === 2 ? halfClosed(1 + Math.floor(i / 10)) : hand({ thumb: false, fingers: fingersUp(1 + Math.floor(i / 10)) })]);
assert.equal(count(flicker, "list"), 1);
// …but a different count does
assert.equal(count(run(30, (i) => [hand({ thumb: false, fingers: fingersUp(i % 4 === 3 ? 4 : 1 + Math.floor(i / 10)) })]), "list"), 0);

// counting(): what a hand counts, or null.
const counted = (h: Hand) => counting(h)?.map(Number).join("") ?? null;
assert.equal(counted(hand({ thumb: false, fingers: fingersUp(2) })), "01100");
assert.equal(counted(hand({ thumb: true, fingers: fingersUp(2) })), "01100"); // thumb beside fewer fingers: not counted
assert.equal(counted(hand({ thumb: true, fingers: fingersUp(4) })), "11111"); // thumb as the fifth
assert.equal(counted(hand({ thumb: true, fingers: fingersUp(0) })), null); // thumbs-up isn't 1
assert.equal(counted(hand({ thumb: false, fingers: [true, false, true, false] })), null); // index+ring: not side by side
assert.equal(counted(hand({ thumb: false, fingers: [false, false, false, true] })), "00001"); // pinky-first is fine
assert.equal(counted(tiny), null); // the tiny-bit pinch isn't 2
// "1" with the thumb holding the middle finger down, tip close to the index (pinch < 0.6) but by its knuckle
const holdingDown = hand({ fingers: fingersUp(1), edit: (p) => ([p[3], p[4]] = [[15, -70, -12], [10, -82, -12]]) });
assert.ok(holdingDown.pinch < 0.6);
assert.equal(counted(holdingDown), "01000");
assert.equal(counted(halfClosed(1)), null);

// Hands come at 25 Hz: the same gestures, at the same speeds, read the same.
const FAST = 40;
assert.equal(count(run(30, () => [hand({ turn: rx(-90) })], 0, FAST), "palmsup"), 1);
assert.equal(count(run(5, () => [hand({ turn: rx(-90) })], 0, FAST), "palmsup"), 0); // 160 ms: not held yet
assert.equal(count(run(60, (i) => [hand({ thumb: false, fingers: fingersUp(1 + Math.floor(i / 20)) })], 0, FAST), "list"), 1);
assert.equal(count(run(24, (i) => [hand({ thumb: false, fingers: fingersUp(1 + Math.floor(i / 8)) })], 0, FAST), "list"), 0); // 0.3 s a count
const levelsAt = (i: number) => (i < 12 ? 150 : i < 18 ? 150 + (i - 11) * 12.5 : 250); // half the step, twice the frames
assert.equal(count(run(32, (i) => [hand({ turn: rx(90), at: [320, levelsAt(i)] })], 0, FAST), "levels"), 1);
const tt40 = run(16, (i) => [L({ at: [200, 250 - (i < 8 ? i * 10 : 70)] }), hand({ at: [440, 250 - (i < 8 ? 0 : (i - 7) * 10)] })], 0, FAST);
assert.equal(count(tt40, "thisthat"), 1);

// lists(), directly: a lower count or a long wait starts over.
// fingers as thumb..pinky, e.g. "01100" = index and middle; "R"/"L" hand
const C = (...xs: Array<[number, string, string?]>) =>
  xs.map(([t, f, side = "R"]) => ({ t, right: side === "R", ext: [...f].map((c) => c === "1"), at: [0, 0] as XY }));
assert.equal(lists(C([0, "01000"], [1000, "01100"], [2000, "01110"])).length, 1);
assert.equal(lists(C([0, "01100"], [1000, "01110"])).length, 1); // missed the first item
assert.equal(lists(C([0, "01000"], [9000, "01100"])).length, 0); // too slow: two separate points
assert.equal(lists(C([0, "01000"], [1000, "01100"], [2000, "01000"], [3000, "01100"])).length, 2);
assert.equal(lists(C([0, "00000"], [1000, "01100"], [2000, "00000"])).length, 0);
assert.equal(lists(C([0, "01000"], [1000, "00110"])).length, 0); // more fingers, but different ones
assert.equal(lists(C([0, "01000"], [1000, "01110"])).length, 0); // jumped two
assert.equal(lists(C([0, "01000", "L"], [500, "01100", "R"])).length, 0); // 1 on one hand, 2 on the other
assert.equal(lists(C([0, "01000"], [500, "11111", "L"], [1000, "01100"])).length, 1); // other hand resting open
assert.equal(lists(C([0, "01111"], [1000, "11111"])).length, 1); // the thumb last

// holds(): once per run, broken by a frame gap.
const hf = (n: number, t0 = 0) => run(n, () => [hand()], t0);
assert.equal(holds(hf(10), (f) => f.hands.map((h) => h.shape)).length, 1);
assert.equal(holds([...hf(3), ...hf(3, 2000)], (f) => f.hands.map((h) => h.shape)).length, 0);

// --- toFrame: which hand is which ------------------------------------------
const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5 }));
pose[11] = { x: 0.6, y: 0.4 }; // speaker's left shoulder (image right)
pose[12] = { x: 0.4, y: 0.4 };
pose[15] = { x: 0.7, y: 0.7 }; // speaker's left wrist
pose[16] = { x: 0.3, y: 0.7 };
const onRight = pts({ right: true, at: [0.3 * W, 0.7 * H] }); // a right hand where the right wrist is
const raw = { blend: null, head: null, hands: [onRight] };
assert.equal(toFrame({ ...raw, pose, handed: [0] }, W, H, 0).hands[0].right, true); // pose overrides the model
assert.equal(toFrame({ ...raw, pose: null, handed: [0.9] }, W, H, 0).hands[0].right, true); // no pose: the model's score
assert.equal(toFrame({ ...raw, pose: null, handed: [0.1] }, W, H, 0).hands[0].right, false);
assert.equal(toFrame({ ...raw, pose, handed: [0] }, W, H, 0).hands[0].palm, "out");

// Two hands are always one of each, even when both sit nearer one pose wrist
// or the model calls both the same.
const PW: XY[] = [[450, 250], [190, 250]]; // speaker's left wrist (image right), right wrist
assert.deepStrictEqual(whichHands([[180, 250], [460, 250]], PW, []), [true, false]);
assert.deepStrictEqual(whichHands([[150, 250], [230, 250]], PW, []), [true, false]); // both near the right wrist
assert.deepStrictEqual(whichHands([[230, 250], [150, 250]], PW, []), [false, true]);
assert.deepStrictEqual(whichHands([[300, 250], [400, 250]], null, [0.9, 0.8]), [true, false]); // model: both "right"
assert.deepStrictEqual(whichHands([[300, 250]], null, [0.2]), [false]);

// --- summary line ---------------------------------------------------------
assert.equal(gestureSummary(undefined), "");
assert.equal(gestureSummary({ palmsup: 5, list: 2, growth: 1, fist: 1 }), "4 gestures: palms up 5×, listing 2×, growth 1×");

console.log("gestures ok");

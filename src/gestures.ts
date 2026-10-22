// Speaking gestures, after Vanessa Van Edwards' "20 Hand Gestures You Should
// Be Using" (Science of People), recognised from the hand and pose landmarks.
// Per frame, readHand describes each hand (fingers out, palm direction, where
// it is) and poses() the gestures held in it; detect() turns a run of frames
// into events — a pose held ~0.3 s, or a movement (a sweep, a rise, hands
// meeting). Pure; checked by gestures.check.ts.
import type { Frame, Pt, XY } from "./presence";

export type GestureId =
  | "list" | "tiny" | "listen" | "fist" | "everything" | "levels" | "point" | "part" | "grand" | "growth"
  | "you" | "me" | "thisthat" | "together" | "ready" | "justso" | "palmsup" | "palmsdown" | "stop" | "we";

// In the article's order. `short` labels the camera view; `caution` marks the
// ones the article says to use sparingly.
export const GESTURES: Array<{ id: GestureId; name: string; short: string; how: string; caution?: boolean }> = [
  { id: "list", name: "Listing", short: "listing", how: "Count your points off on your fingers: 1, 2, 3" },
  { id: "tiny", name: "A tiny bit", short: "tiny bit", how: "Thumb and index finger almost touching, other fingers curled" },
  { id: "listen", name: "Listen here", short: "listen here", how: "Chop one hand down onto the other" },
  { id: "fist", name: "Clenched hand", short: "fist", how: "A firm fist for your most important point", caution: true },
  { id: "everything", name: "Everything", short: "everything", how: "Sweep a flat hand sideways, palm down" },
  { id: "levels", name: "Small, medium, large", short: "levels", how: "Hold a flat, palm-down hand at one height, then another" },
  { id: "point", name: "Let me tell you", short: "pointing", how: "Point your index finger", caution: true },
  { id: "part", name: "Just a part", short: "just a part", how: "Thumb and index apart, the other fingers curled: a gap wider than a tiny bit" },
  { id: "grand", name: "The grand gesture", short: "grand", how: "Both hands wide apart, palms to the audience" },
  { id: "growth", name: "Growth", short: "growth", how: "Raise a hand upward" },
  { id: "you", name: "You", short: "you", how: "Reach an open palm toward the audience, or hold a flat hand palm down close to the camera" },
  { id: "me", name: "Me", short: "me", how: "An open hand at your chest, palm toward you; a cupped one any way but toward the camera; or fingers toward the other hand" },
  { id: "thisthat", name: "This and that", short: "this & that", how: "Gesture with one hand, then the other" },
  { id: "together", name: "Come together", short: "together", how: "Bring your hands together from apart" },
  { id: "ready", name: "I'm ready", short: "ready", how: "The fingertips of both hands touching, pointing up or toward the camera" },
  { id: "justso", name: "It's just so", short: "just so", how: "Both palms upright, facing each other" },
  { id: "palmsup", name: "I'm not hiding anything", short: "palms up", how: "Open hands, palms up" },
  { id: "palmsdown", name: "You listen to me", short: "palms down", how: "Flat hands, palms down — commanding", caution: true },
  { id: "stop", name: "Stop", short: "stop", how: "Raise one open palm to shoulder height, square to the audience" },
  { id: "we", name: "We", short: "we", how: "Open both arms wide, as if hugging the audience — hands still in view" },
];
export const gestureInfo = (id: GestureId) => GESTURES.find((g) => g.id === id)!;

// Distances in hand sizes (wrist → middle knuckle) or u (shoulder widths).
// ponytail: reasoned from hand proportions and one test image, not from
// recordings of each gesture; tune against real sessions.
export const GESTURE_TUNING = {
  fingerOut: 1.1, // the thumb sticks out when its tip is this much further from the pinky knuckle than its joint (golden thumbs-up: 1.18)
  fingerStraight: 0.9, // knuckle→tip over the finger's length: a finger is out at least this straight (straight 1, half-bent ~0.85, golden curled 0.24–0.38)
  fingerCurled: 0.65, // …and curled at most this; between, it's half-bent (◐ on the readout)
  countGap: 0.15, // counting, and point / tiny bit / just a part: the folded fingers are at least this much less straight than the straight ones (the index for those)
  thumbStraight: 0.92, // …or, for the thumb, when it's this straight (base → tip over the joint path; tucked ~0.75)
  palmBlend: 0.5, // weight of the previous frame's palm direction, smoothing landmark jitter
  palmStick: 0.2, // a new palm direction must win by this much (of 1) to replace the last
  palmAlso: [0.4, 0.5], // a palm also faces the other axis (up/down with out/in) once it leans this far along it (cos 60°), until under the first
  // Times and speeds, not frame counts: hands are tracked at 25 Hz, the face and body at 12.5.
  holdMs: 240, // a pose held this long counts
  listHoldMs: 320, // each count of a list is held, not passed through…
  listMissMs: 200, // …through this long of the hand not reading as counting
  stillSpeed: 0.6, // shoulder widths a second: a hand slower than this is holding still…
  moveSpeed: 1.25, // …and faster than this, moving ("this and that", "levels")
  segMs: 80, // "this and that": each hand's move lasts at least this
  maxGapMs: 500, // frames further apart than this aren't compared
  refractoryMs: 1000, // one movement or pose = one event
  listGapMs: 4000, // max time between counted fingers in a list
  tinyGap: [0.12, 0.6], // "a tiny bit": thumb–index tip gap, hand sizes (under = touching)
  tinyReach: 1.2, // …with the index tip at least this far from the wrist (not a fist)
  partGap: 1.1, // "just a part": the same, thumb out, gap from tinyGap's top to this (wider is an L, pointing)
  touch: 0.35, // "ready": fingertips this close (hand sizes) are touching…
  readyApart: 0.8, // …with the palms at least this far apart (hand sizes): two copies of one hand sit on top of each other
  readyPoint: 0.5, // …and each hand's fingers pointing at least this much up or toward the camera (cos 60°)
  pinchSteady: 0.1, // "tiny bit" / "just a part": the gap holds within this (hand sizes), not a hand closing through it
  moveMs: 1000, // movements are judged over the last second
  sweep: 1.2, // "everything": sideways travel, u
  rise: 0.8, // "growth": upward travel, u
  reach: 1.3, // "you": hand grows this much (comes toward the camera)…
  reachOver: 1.15, // …to this much bigger than it was any time in the second before the growth…
  reachDrift: 0.6, // …staying within this (u) of where it was: a hand lost and found elsewhere isn't growing…
  chestWide: 0.8, // "me": the hand this near the body's midline (u)
  inward: 0.7, // "me": fingers pointing at least this much (of 1) sideways at the midline; not "you" then
  youWaitMs: 400, // a reach is "you" only if the hand isn't at the chest as "me" within this long after
  near: 0.6, // …or is held palm down this big (wrist → middle knuckle, shoulder widths; ~0.3 at the body, ~0.4 in front of the chest)
  chop: 0.5, // "listen here": drop in ≤ 400 ms, u, landing within 0.7 u of the other hand
  apart: 1.2, // "come together": from this far apart (u)…
  met: 0.5, // …to this close
  levelStep: 0.4, // "levels": height change between two still palm-down holds, u
  levelsMs: 4000,
  wide: 2.2, // "we": both hands in view, open, this many shoulder widths apart
  stopFacing: 0.8, // "stop": how squarely the palm faces the camera (1 = straight on)
  stopRaise: 0.2, // …with the palm centre no lower than this far (u) below the shoulders
} as const;

export type Shape = "fist" | "1" | "2" | "3" | "4" | "open";
export type Palm = "up" | "down" | "out" | "in" | "side";
export interface Hand {
  right: boolean; // the speaker's own right hand
  ext: boolean[]; // fingers out: thumb, index, middle, ring, pinky
  straightness: number[]; // index..pinky: knuckle→tip over the finger's length (1 = straight)
  shape: Shape; // fingers out, counted (thumb-first and index-first counting agree)
  pinch: number; // thumb–index tip gap, hand sizes
  pinched: boolean; // the thumb tip is nearer the index tip than the index knuckle
  reach: number; // index tip from the wrist, hand sizes
  tips: XY[]; // fingertips thumb..pinky, image px
  palm: Palm; // which way the palm faces; out = toward the camera
  tilt?: Palm; // a second way the palm faces, when tilted between up/down and out/in
  n: [number, number, number]; // palm normal, unit, smoothed over frames; n[0] + = toward image right
  up: boolean; // fingers point up
  axis: V3; // wrist → middle knuckle, unit (frame axes: y down, z − = toward the camera)
  c: XY; // palm centre, px
  size: number; // wrist → middle knuckle, px
}
export interface Pose {
  g: GestureId;
  at: XY; // px, where to label it
}
export interface GestureEvent extends Pose {
  t: number;
}

type V3 = [number, number, number];
const d3 = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const dist = (a: XY, b: XY) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const midpoint = (a: XY, b: XY): XY => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
const flat = (h: Hand) => h.ext.slice(1).every(Boolean); // four fingers out, thumb either way

const unit = (v: number[]): V3 => {
  const len = Math.hypot(...v) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
};

// Which way a palm normal faces. Each direction scores how far the normal
// points along it; `prev` keeps its place unless another beats it by palmStick,
// so a palm halfway between two directions doesn't flip every frame.
export function palmOf(n: V3, prev?: Palm): Palm {
  const score: Record<Palm, number> = { out: -n[2], in: n[2], up: -n[1], down: n[1], side: Math.abs(n[0]) };
  const best = (Object.keys(score) as Palm[]).reduce((a, b) => (score[b] > score[a] ? b : a));
  return prev && score[prev] >= score[best] - GESTURE_TUNING.palmStick ? prev : best;
}

// A palm tilted between up/down and out/in faces both: the other axis's
// direction, when the normal leans far enough along it. Not with "side".
export function tiltOf(n: V3, palm: Palm, prev?: Palm): Palm | undefined {
  const vert = palm === "up" || palm === "down";
  if (!vert && palm !== "out" && palm !== "in") return undefined;
  const k = vert ? 2 : 1;
  const t: Palm = vert ? (n[2] < 0 ? "out" : "in") : n[1] < 0 ? "up" : "down";
  return Math.abs(n[k]) >= GESTURE_TUNING.palmAlso[prev === t ? 0 : 1] ? t : undefined;
}
const faces = (h: Hand, d: Palm) => h.palm === d || h.tilt === d;
// Fingers pointing sideways at the body's midline: hands held to the chest,
// fingers toward each other.
const inward = (h: Hand, chest: XY) => Math.sign(chest[0] - h.c[0]) * h.axis[0] >= GESTURE_TUNING.inward;

// One hand from its 21 landmarks (0 wrist, 1–4 thumb base → tip, 5/9/13/17
// knuckles, 6/10/14/18 middle joints, 8/12/16/20 tips; z on x's scale,
// negative = nearer). `world` is the same 21 in metric 3D (frame axes), when
// the model gave them: the image z is squashed ~7×, which hid a horizontal
// palm, so shape and palm direction come from `world`; position and size stay
// in image pixels. `prev` is the same hand last frame, for smoothing.
export function readHand(l: Pt[], right: boolean, w: number, h: number, prev?: Hand, world?: number[][]): Hand {
  const T = GESTURE_TUNING;
  const px = l.map((q): V3 => [q.x * w, q.y * h, (q.z ?? 0) * w]);
  const p = world?.length === 21 ? world.map((v): V3 => [v[0], v[1], v[2]]) : px;
  // How straight each finger is: knuckle → tip over the length along its joints.
  // (Tip-vs-middle-joint distance from the wrist called a half-bent finger out,
  // so a half-closed hand flickered between counts and looked like a list.)
  const straightness = [5, 9, 13, 17].map((b) => d3(p[b], p[b + 3]) / (d3(p[b], p[b + 1]) + d3(p[b + 1], p[b + 2]) + d3(p[b + 2], p[b + 3]) || 1));
  const fingers = straightness.map((r) => r >= T.fingerStraight);
  // The thumb is out when it sticks away from the hand (its tip further from the
  // pinky knuckle than its joint — a thumbs-up), or, with any finger out, when
  // it's straight: pressed along the side of a flat hand still counts. A thumb
  // tucked across the palm is bent; one resting on the side of a fist isn't out.
  const away = d3(p[4], p[17]) > T.fingerOut * d3(p[3], p[17]);
  const straight = d3(p[1], p[4]) > T.thumbStraight * (d3(p[1], p[2]) + d3(p[2], p[3]) + d3(p[3], p[4]));
  const ext = [away || (straight && fingers.some(Boolean)), ...fingers];
  const n = ext.filter(Boolean).length;
  // Palm normal: the fan of (knuckle − wrist) × (next knuckle − wrist) across
  // all four knuckles (averages out a jittery one), flipped for the left hand
  // (a mirror image) so it points out of the palm on both; then blended with
  // last frame's.
  const arm = [5, 9, 13, 17].map((k) => p[k].map((v, i) => v - p[0][i]));
  const fan = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const [a, b] = [arm[k], arm[k + 1]];
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]].forEach((v, i) => (fan[i] += (right ? 1 : -1) * v));
  }
  let nrm = unit(fan);
  if (prev) nrm = unit(nrm.map((v, i) => (1 - T.palmBlend) * v + T.palmBlend * prev.n[i]));
  const palm = palmOf(nrm, prev?.palm);
  const c = [0, 5, 9, 13, 17].reduce<XY>((m, k) => [m[0] + px[k][0] / 5, m[1] + px[k][1] / 5], [0, 0]);
  const size = d3(px[0], px[9]) || 1;
  const span = d3(p[0], p[9]) || 1; // hand size in p's units
  return {
    right,
    ext,
    straightness,
    shape: n === 0 ? "fist" : n === 5 ? "open" : (String(n) as Shape),
    pinch: d3(p[4], p[8]) / span,
    pinched: d3(p[4], p[8]) < d3(p[4], p[5]),
    reach: d3(p[8], p[0]) / span,
    tips: [4, 8, 12, 16, 20].map((k): XY => [px[k][0], px[k][1]]),
    palm,
    tilt: tiltOf(nrm, palm, prev?.tilt),
    n: nrm,
    up: px[0][1] - px[9][1] > 0.5 * size,
    axis: unit(p[9].map((v, i) => v - p[0][i])),
    c,
    size,
  };
}

// The gestures held in one frame. u = shoulder width (px); chest = shoulder
// midpoint, null without a body.
export function poses(hands: Hand[], u: number, chest: XY | null): Pose[] {
  const T = GESTURE_TUNING;
  const out: Pose[] = [];
  const add = (g: GestureId, at: XY) => out.push({ g, at });
  for (const h of hands) {
    const [, index, middle, ring, pinky] = h.ext;
    // Middle, ring and pinky folded, and clearly more than the index: in a cupped
    // hand all four are bent alike (and a little, so none reads out).
    const curled = !middle && !ring && !pinky && Math.max(...h.straightness.slice(1)) <= h.straightness[0] - T.countGap;
    if (h.shape === "fist") add("fist", h.c);
    const part = curled && h.ext[0] && h.reach > T.tinyReach && h.pinch >= T.tinyGap[1] && h.pinch < T.partGap;
    if (curled && index && h.pinch >= T.tinyGap[1] && !part) add("point", h.c);
    if (curled && h.reach > T.tinyReach && h.pinch > T.tinyGap[0] && h.pinch < T.tinyGap[1]) add("tiny", h.c);
    if (part) add("part", h.c);
    if (flat(h) && faces(h, "up")) add("palmsup", h.c);
    if (flat(h) && faces(h, "down")) add(chest && h.size >= T.near * u && !inward(h, chest) ? "you" : "palmsdown", h.c);
    // Me: an open hand at the chest, palm in; or a cupped one (one finger
    // curled at most: not a fist, which turned over faces in too) any way but
    // toward the camera, as hands turning to the chest read up, side or down;
    // or fingers pointing at the other hand, unless flat palms up / down.
    const cupped = !flat(h) && h.straightness.filter((r) => r <= T.fingerCurled).length <= 1;
    const level = flat(h) && (faces(h, "up") || faces(h, "down"));
    const atChest = chest && Math.abs(h.c[0] - chest[0]) < T.chestWide * u && h.c[1] > chest[1] && h.c[1] < chest[1] + 1.3 * u;
    if (atChest && (flat(h) || cupped) && (h.palm === "in" || (cupped && !faces(h, "out")) || (inward(h, chest) && !level))) add("me", h.c);
  }
  const outs = hands.filter((h) => flat(h) && faces(h, "out"));
  const sides = hands.filter((h) => flat(h) && faces(h, "side"));
  if (hands.length === 2) {
    const [a, b] = hands;
    const gap = dist(a.c, b.c);
    const m = midpoint(a.c, b.c);
    if (outs.length === 2 && gap >= 1.5 * u) add("grand", m);
    const touching = a.tips.filter((t, k) => dist(t, b.tips[k]) < T.touch * (a.size + b.size) / 2).length;
    // palms up and touching is offering, not ready
    // and fingers pointing up or at the camera, not flat at each other
    const pointing = (h: Hand) => Math.max(-h.axis[1], -h.axis[2]) >= T.readyPoint;
    if (touching >= 3 && gap >= T.readyApart * (a.size + b.size) / 2 && !faces(a, "up") && !faces(b, "up") && pointing(a) && pointing(b)) add("ready", m);
    const [l, r] = a.c[0] < b.c[0] ? [a, b] : [b, a]; // in the image
    if (sides.length === 2 && l.n[0] > 0 && r.n[0] < 0 && gap >= 0.4 * u && gap <= 2 * u) add("justso", m);
  }
  // Stop: a palm raised and flashed square at the camera — not just any hand
  // that happens to face it while talking.
  const s = outs[0];
  if (chest && outs.length === 1 && s.up && -s.n[2] >= T.stopFacing && s.c[1] <= chest[1] + T.stopRaise * u) add("stop", s.c);
  // We: both open hands in view, wide apart. (The pose model's wrists can't be
  // used: off-frame it guesses them, often far apart.)
  if (hands.length === 2 && hands.every(flat) && !outs.length && dist(hands[0].c, hands[1].c) >= T.wide * u) add("we", midpoint(hands[0].c, hands[1].c));
  return out;
}

// Keys held for holdMs in every frame (once per run), by frame index.
export function holds<K>(frames: Frame[], keys: (f: Frame) => K[]): Array<{ i: number; key: K }> {
  const out: Array<{ i: number; key: K }> = [];
  let runs = new Map<K, { from: number; done: boolean }>();
  frames.forEach((f, i) => {
    if (i && f.t - frames[i - 1].t > GESTURE_TUNING.maxGapMs) runs = new Map();
    const prev = runs;
    runs = new Map([...new Set(keys(f))].map((k) => [k, prev.get(k) ?? { from: f.t, done: false }]));
    for (const [k, r] of runs)
      if (!r.done && f.t - r.from >= GESTURE_TUNING.holdMs) {
        r.done = true;
        out.push({ i, key: k });
      }
  });
  return out;
}

// What a hand counts on its fingers (thumb..pinky), or null if it isn't
// counting: the fingers out sit side by side (index+middle, not index+ring),
// and the thumb only counts as the fifth — a thumb out beside fewer fingers is
// just resting. The folded fingers are clearly more bent than the straight ones
// (a half-closed hand's fingers are all about as bent, flickering in and out;
// folded fingers aren't always fully curled, so they can't just be ruled
// half-bent: that lost every count but 4). Not an index pinched to the thumb (the
// tiny-bit gesture; holding a curled finger down for "1" puts the thumb down by
// the knuckles, not at the tip).
export function counting(h: Hand): boolean[] | null {
  const ext = [h.ext[0] && h.ext.slice(1).every(Boolean), ...h.ext.slice(1)];
  const out = h.straightness.filter((_, k) => h.ext[k + 1]);
  const folded = h.straightness.filter((_, k) => !h.ext[k + 1]);
  if (folded.length && Math.max(...folded) > Math.min(...out, 1) - GESTURE_TUNING.countGap) return null;
  if ((h.ext[1] && h.pinched) || !/^0*1+0*$/.test(ext.map(Number).join(""))) return null;
  return ext;
}

// Listing: on one hand, the same fingers held out plus one more each time
// (1 → 2 → 3…), each within listGapMs; one event per list, at its second count.
// Any other count (different fingers, a jump, a long gap) starts over. Per
// hand, so two hands showing 1 and 2 aren't a list.
export function lists(counts: Array<{ t: number; right: boolean; ext: boolean[]; at: XY }>): GestureEvent[] {
  const out: GestureEvent[] = [];
  const state = new Map<boolean, { ext: boolean[]; t: number; len: number }>();
  for (const c of counts) {
    const s = state.get(c.right);
    const grew = s && c.t - s.t <= GESTURE_TUNING.listGapMs && s.ext.every((e, k) => !e || c.ext[k]) && c.ext.filter(Boolean).length === s.ext.filter(Boolean).length + 1;
    const len = grew ? s.len + 1 : 1;
    if (len === 2) out.push({ t: c.t, g: "list", at: c.at });
    state.set(c.right, { ext: c.ext, t: c.t, len });
  }
  return out;
}

// Each count held long enough to be one, per hand, in time order. A moment of
// the hand not reading as counting (a finger flickering to half-bent, the hand
// blurred out) doesn't end the hold; a different count does.
export function counts(frames: Frame[]): Array<{ t: number; right: boolean; ext: boolean[]; at: XY }> {
  const T = GESTURE_TUNING;
  const out: Array<{ t: number; right: boolean; ext: boolean[]; at: XY }> = [];
  const run = new Map<boolean, { key: string; from: number; seen: number; done: boolean }>();
  frames.forEach((f, i) => {
    if (i && f.t - frames[i - 1].t > T.maxGapMs) run.clear();
    for (const right of [true, false]) {
      const h = f.hands.find((h) => h.right === right);
      const ext = h && counting(h);
      const r = run.get(right);
      if (!ext) {
        if (r && f.t - r.seen > T.listMissMs) run.delete(right);
        continue;
      }
      const key = ext.map(Number).join("");
      const cur = r?.key === key ? { ...r, seen: f.t } : { key, from: f.t, seen: f.t, done: false };
      if (!cur.done && f.t - cur.from >= T.listHoldMs) {
        cur.done = true;
        out.push({ t: f.t, right, ext, at: h.c });
      }
      run.set(right, cur);
    }
  });
  return out;
}

const sideOf = (f: Frame, right: boolean) => f.hands.find((h) => h.right === right);
const isMe = (f: Frame, right: boolean) => {
  const h = sideOf(f, right);
  return !!h && f.poses.some((p) => p.g === "me" && p.at[0] === h.c[0] && p.at[1] === h.c[1]);
};

// Every gesture in a run of frames, in time order.
export function detect(frames: Frame[]): GestureEvent[] {
  const T = GESTURE_TUNING;
  const ev: GestureEvent[] = [];
  const last = new Map<string, number>();
  const fire = (g: GestureId, t: number, at: XY, key = "") => {
    if (t - (last.get(g + key) ?? -Infinity) < T.refractoryMs) return;
    last.set(g + key, t);
    ev.push({ t, g, at });
  };

  // A hand closing passes through a tiny gap: those two only count while the
  // gap holds still (by the hand nearest the pose, over the hold).
  const steady = (i: number, at: XY) => {
    const gaps = frames.slice(0, i + 1).filter((f) => f.t >= frames[i].t - T.holdMs).map((f) => f.hands.reduce<Hand | undefined>((m, h) => (!m || dist(h.c, at) < dist(m.c, at) ? h : m), undefined)?.pinch ?? NaN);
    return Math.max(...gaps) - Math.min(...gaps) <= T.pinchSteady;
  };
  for (const { i, key } of holds(frames, (f) => f.poses.map((p) => p.g))) {
    const at = frames[i].poses.find((p) => p.g === key)!.at;
    if ((key !== "tiny" && key !== "part") || steady(i, at)) fire(key, frames[i].t, at);
  }
  ev.push(...lists(counts(frames)));

  // Each hand's biggest size over the moveMs up to each frame: a hand coming
  // back to its size (it had tipped away) isn't reaching ("you").
  const biggest = new Map([false, true].map((right) => [right, new Map(frames.map((f, i) => {
    const sizes = frames.slice(0, i + 1).filter((g) => f.t - g.t <= T.moveMs).map((g) => sideOf(g, right)?.size ?? 0);
    return [f, Math.max(...sizes)];
  }))]));
  const stillSince = new Map<boolean, number | null>();
  const level = new Map<boolean, { t: number; y: number }>();
  let seg = { side: "", end: 0, start: 0 };
  let done: typeof seg | null = null;
  for (let i = 1; i < frames.length; i++) {
    const f = frames[i];
    const u = f.u;
    const prev = f.t - frames[i - 1].t > T.maxGapMs ? null : frames[i - 1];
    // Back over the last second while the frames are continuous. A frame
    // where the hand dropped out (a fast hand blurs) is skipped, not the end.
    const back = (each: (g: Frame, dt: number) => boolean) => {
      for (let j = i - 1; j >= 0 && f.t - frames[j].t <= T.moveMs && frames[j + 1].t - frames[j].t <= T.maxGapMs; j--)
        if (!each(frames[j], f.t - frames[j].t)) return;
    };

    for (const right of [false, true]) {
      const h = sideOf(f, right);
      if (!h) {
        stillSince.set(right, null);
        continue;
      }
      const other = sideOf(f, !right);
      // Bringing hands up to the chest grows them too (a palm-up hand is
      // foreshortened): wait out youWaitMs, and no reach that ends in "me".
      const ahead = frames.filter((g) => g.t >= f.t && g.t - f.t <= T.youWaitMs);
      const reached = frames[frames.length - 1].t - f.t >= T.youWaitMs && !ahead.some((g) => isMe(g, right));
      back((g, dt) => {
        const p = sideOf(g, right);
        if (!p) return true;
        const [dx, dy] = [h.c[0] - p.c[0], h.c[1] - p.c[1]];
        if (Math.abs(dx) >= T.sweep * u && Math.abs(dx) >= 2 * Math.abs(dy) && flat(h) && (faces(h, "down") || faces(h, "side")))
          fire("everything", f.t, h.c, String(right));
        if (-dy >= T.rise * u && -dy >= 2 * Math.abs(dx)) fire("growth", f.t, h.c, String(right));
        if (h.size >= T.reach * p.size && h.size >= T.reachOver * biggest.get(right)!.get(g)! && dist(h.c, p.c) <= T.reachDrift * u && flat(h) && (faces(h, "up") || faces(h, "out")) && !faces(h, "in") && !(f.body && inward(h, f.body[0])) && reached) fire("you", f.t, h.c, String(right));
        if (dt <= 400 && dy >= T.chop * u && other && dist(h.c, other.c) <= 0.7 * u) fire("listen", f.t, h.c, String(right));
        return true;
      });
      // Levels: a flat palm-down hand held still at one height, then another.
      const p = prev && sideOf(prev, right);
      const still = p && flat(h) && faces(h, "down") && dist(h.c, p.c) < T.stillSpeed * u * ((f.t - prev.t) / 1000);
      const since = still ? (stillSince.get(right) ?? prev.t) : null;
      const wasHeld = stillSince.get(right) != null && frames[i - 1].t - stillSince.get(right)! >= T.holdMs;
      stillSince.set(right, since);
      if (since != null && f.t - since >= T.holdMs && !wasHeld) {
        const lv = level.get(right);
        if (lv && f.t - lv.t <= T.levelsMs && Math.abs(h.c[1] - lv.y) >= T.levelStep * u) fire("levels", f.t, h.c, String(right));
        level.set(right, { t: f.t, y: h.c[1] });
      }
    }

    const [l, r] = [sideOf(f, false), sideOf(f, true)];
    if (!l || !r) continue;
    const gap = dist(l.c, r.c);
    const m = midpoint(l.c, r.c);
    if (gap <= T.met * u)
      back((g) => {
        const [pl, pr] = [sideOf(g, false), sideOf(g, true)];
        if (!pl || !pr) return true;
        if (dist(pl.c, pr.c) < T.apart * u) return true;
        fire("together", f.t, m);
        return false;
      });
    // This and that: one hand moves while the other holds still, then the other.
    const [pl, pr] = prev ? [sideOf(prev, false), sideOf(prev, true)] : [];
    if (!pl || !pr) continue;
    const step = u * ((f.t - prev!.t) / 1000);
    const [ml, mr] = [dist(l.c, pl.c), dist(r.c, pr.c)];
    const [go, hold] = [T.moveSpeed * step, T.stillSpeed * step];
    const mover = ml > go && mr < hold ? "L" : mr > go && ml < hold ? "R" : "";
    if (!mover) continue;
    if (mover === seg.side) seg.end = f.t;
    else {
      if (seg.end - seg.start >= T.segMs) done = seg;
      seg = { side: mover, start: f.t, end: f.t };
    }
    if (seg.end - seg.start >= T.segMs && done && done.side !== seg.side && seg.start - done.end <= 1500 && gap >= u) {
      fire("thisthat", f.t, m);
      done = null;
    }
  }
  return ev.sort((a, b) => a.t - b.t);
}

// "4 gestures: palms up 5×, listing 2×, growth 1×" — what a session used, or "".
export function gestureSummary(used: Partial<Record<GestureId, number>> | undefined): string {
  const xs = Object.entries(used ?? {}).sort((a, b) => b[1] - a[1]) as Array<[GestureId, number]>;
  if (!xs.length) return "";
  return `${xs.length} gesture${xs.length === 1 ? "" : "s"}: ${xs.slice(0, 3).map(([g, n]) => `${gestureInfo(g).short} ${n}×`).join(", ")}`;
}

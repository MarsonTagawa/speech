// Looping demos of each gesture for the Gestures page: a stick figure whose
// hands are drawn as the camera overlay draws tracked hands (the same 21
// landmarks), posed by keyframes and shown mirrored, so you copy it as you
// would a mirror.
import { HAND_BONES } from "./camera";
import type { GestureId } from "./gestures";

type V3 = [number, number, number];
// A hand at one moment. World axes are the speaker's: x to their right, y up,
// z toward the camera; units are hand lengths (wrist → middle knuckle), with
// the shoulders 4 apart.
interface HandKey {
  p: V3; // wrist
  d: V3; // where the fingers point
  n: V3; // where the palm faces
  f: number[]; // curl, thumb..pinky: 0 straight, 1 folded
  pinch?: number; // thumb tip this far below the index tip
}
interface Key {
  t: number; // 0..1 of the loop
  R: HandKey;
  L: HandKey;
}
export interface Figure {
  ms: number; // loop length
  keys: Key[];
}

const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => mul(a, 1 / (Math.hypot(...a) || 1));
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
const lerp3 = (a: V3, b: V3, k: number): V3 => [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)];
const ease = (k: number) => k * k * (3 - 2 * k);

const UP: V3 = [0, 1, 0];
const DOWN: V3 = [0, -1, 0];
const FWD: V3 = [0, 0, 1];
const BACK: V3 = [0, 0, -1];
const IN: V3 = [-1, 0, 0]; // for the right hand: toward the body's midline

const OPEN = [0, 0, 0, 0, 0];
const FIST = [0.75, 1, 1, 1, 1];
const RELAXED = [0.3, 0.35, 0.4, 0.45, 0.5];
const POINT = [0.8, 0, 1, 1, 1];

// The right hand's key, mirrored onto the left.
const mir = (k: HandKey): HandKey => ({
  ...k,
  p: [-k.p[0], k.p[1], k.p[2]],
  d: [-k.d[0], k.d[1], k.d[2]],
  n: [-k.n[0], k.n[1], k.n[2]],
});
const hk = (p: V3, d: V3, n: V3, f = OPEN, pinch?: number): HandKey => ({ p, d, n, f, pinch });
const REST = hk([1.5, -4.3, 1.2], [-0.3, -0.25, 1], IN, RELAXED);
const both = (t: number, R: HandKey, L = mir(R)): Key => ({ t, R, L });
const right = (t: number, R: HandKey): Key => ({ t, R, L: mir(REST) });
// One pose held through the middle of the loop, from and back to rest.
const hold = (R: HandKey, L = mir(REST), ms = 2600): Figure => ({
  ms,
  keys: [both(0, REST, mir(REST)), both(0.2, R, L), both(0.75, R, L), both(1, REST, mir(REST))],
});

const count = (n: number) => [1, ...[0, 1, 2, 3].map((k) => (k < n ? 0 : 1))];
const listAt = (f: number[]) => hk([1.6, -2.2, 1.6], UP, FWD, f);
const chopHi = hk([0.7, -1.2, 1.8], FWD, IN);
const chopLo = hk([0.5, -2.9, 1.8], FWD, IN);
const shelf = mir(hk([0.6, -3.4, 1.7], [-0.9, 0, 0.45], UP));
const level = (y: number) => hk([1.8, y, 1.7], FWD, DOWN);
const sweep = (x: number) => hk([x, -2.4, 2.2], [0.1, 0, 1], DOWN);
const point = (z: number) => hk([1.5, -1.8, z], [-0.5, 0.7, 0.5], [-0.6, -0.3, -0.6], POINT);
const grandR = hk([4.6, -0.9, 1.2], [0.35, 1, 0], FWD);
const lift = (y: number) => hk([1.8, y, 1.8], FWD, UP);
const reach = (z: number, y: number) => hk([1.8, y, z], [0.1, 0.1, 1], UP);
const chest = hk([0.8, -1.7, 1.0], [-1, 0.35, 0], BACK);
const offer = (x: number) => hk([x, -2.7, 2.0], [0.4, 0, 1], UP);
const meet = (x: number) => hk([x, -2.4, 2.0], [-0.2, 0.25, 1], IN, [0.2, 0.05, 0.05, 0.05, 0.05]);
const steeple = hk([1.15, -2.6, 1.8], [-0.32, 1, 0.15], [-0.7, 0, 0.35], [0.2, 0.12, 0.12, 0.12, 0.12]);
const upright = (y: number) => hk([0.95, y, 1.9], [0, 1, 0.35], IN);
const palmsUp = (x: number) => hk([x, -3.0, 2.0], [0.3, 0, 1], UP);
const palmsDown = (y: number) => hk([1.9, y, 2.0], FWD, DOWN);
const wide = hk([4.4, -1.4, 1.8], [0.55, 0.25, 0.8], [-0.7, 0.7, 0]);

export const FIGURES: Record<GestureId, Figure> = {
  list: {
    ms: 3600,
    keys: [
      right(0, REST), right(0.12, listAt(count(1))), right(0.3, listAt(count(1))),
      right(0.36, listAt(count(2))), right(0.55, listAt(count(2))),
      right(0.61, listAt(count(3))), right(0.85, listAt(count(3))), right(1, REST),
    ],
  },
  tiny: hold(hk([1.5, -1.6, 1.8], UP, [-0.35, 0, 1], [0.35, 0.3, 1, 1, 1], 0.25)),
  listen: {
    ms: 2400,
    keys: [
      both(0, REST, mir(REST)), both(0.15, chopHi, shelf), both(0.3, chopLo, shelf),
      both(0.5, chopHi, shelf), both(0.65, chopLo, shelf), both(0.8, chopLo, shelf), both(1, REST, mir(REST)),
    ],
  },
  fist: {
    ms: 2400,
    keys: [
      right(0, REST), right(0.2, hk([1.4, -1.7, 1.6], UP, [-0.8, 0, 0.4], FIST)),
      right(0.35, hk([1.4, -2.3, 1.7], UP, [-0.8, 0, 0.4], FIST)), right(0.5, hk([1.4, -1.7, 1.6], UP, [-0.8, 0, 0.4], FIST)),
      right(0.65, hk([1.4, -2.3, 1.7], UP, [-0.8, 0, 0.4], FIST)), right(0.8, hk([1.4, -2.3, 1.7], UP, [-0.8, 0, 0.4], FIST)), right(1, REST),
    ],
  },
  everything: {
    ms: 2600,
    keys: [right(0, REST), right(0.2, sweep(-0.8)), right(0.55, sweep(4)), right(0.7, sweep(4)), right(1, REST)],
  },
  levels: {
    ms: 3600,
    keys: [
      right(0, REST), right(0.1, level(-4)), right(0.28, level(-4)), right(0.36, level(-2.9)),
      right(0.55, level(-2.9)), right(0.63, level(-1.7)), right(0.85, level(-1.7)), right(1, REST),
    ],
  },
  point: {
    ms: 2400,
    keys: [right(0, REST), right(0.2, point(1.6)), right(0.35, point(2.6)), right(0.5, point(1.6)), right(0.65, point(2.6)), right(0.8, point(2.6)), right(1, REST)],
  },
  part: hold(hk([1.5, -1.6, 1.8], UP, [-0.35, 0, 1], [0.05, 0.3, 1, 1, 1], 0.85)),
  grand: hold(grandR, mir(grandR)),
  growth: {
    ms: 2600,
    keys: [right(0, REST), right(0.15, lift(-4)), right(0.6, lift(1.2)), right(0.75, lift(1.2)), right(1, REST)],
  },
  you: {
    ms: 2600,
    keys: [right(0, REST), right(0.15, reach(0.9, -3)), right(0.55, reach(4.4, -2)), right(0.75, reach(4.4, -2)), right(1, REST)],
  },
  me: hold(chest),
  thisthat: {
    ms: 3200,
    keys: [
      both(0, REST, mir(REST)), both(0.1, offer(1.5), mir(offer(1.5))), both(0.25, offer(3.4), mir(offer(1.5))),
      both(0.4, offer(1.5), mir(offer(1.5))), both(0.55, offer(1.5), mir(offer(3.4))), both(0.7, offer(1.5), mir(offer(1.5))),
      both(0.85, offer(1.5), mir(offer(1.5))), both(1, REST, mir(REST)),
    ],
  },
  together: {
    ms: 2600,
    keys: [both(0, REST), both(0.15, meet(3.6)), both(0.3, meet(3.6)), both(0.6, meet(0.8)), both(0.8, meet(0.8)), both(1, REST)],
  },
  ready: hold(steeple, mir(steeple)),
  justso: {
    ms: 2600,
    keys: [both(0, REST), both(0.2, upright(-2.4)), both(0.35, upright(-2.9)), both(0.5, upright(-2.4)), both(0.65, upright(-2.9)), both(0.8, upright(-2.9)), both(1, REST)],
  },
  palmsup: hold(palmsUp(2.2), mir(palmsUp(2.2))),
  palmsdown: {
    ms: 2600,
    keys: [both(0, REST), both(0.2, palmsDown(-2.6)), both(0.4, palmsDown(-3.2)), both(0.55, palmsDown(-2.8)), both(0.75, palmsDown(-3.2)), both(1, REST)],
  },
  stop: hold(hk([2.0, -0.3, 2.3], UP, FWD)),
  we: hold(wide, mir(wide), 3000),
};

// --- Posing ------------------------------------------------------------------

function keyAt(fig: Figure, t: number): Key {
  const k = (((t % fig.ms) + fig.ms) % fig.ms) / fig.ms;
  const i = Math.max(0, fig.keys.findIndex((x) => x.t > k) - 1);
  const [a, b] = [fig.keys[i], fig.keys[i + 1] ?? fig.keys[i]];
  const m = b.t > a.t ? ease((k - a.t) / (b.t - a.t)) : 0;
  const mix = (x: HandKey, y: HandKey): HandKey => ({
    p: lerp3(x.p, y.p, m),
    d: lerp3(x.d, y.d, m),
    n: lerp3(x.n, y.n, m),
    f: x.f.map((c, j) => lerp(c, y.f[j], m)),
    pinch: x.pinch == null && y.pinch == null ? undefined : lerp(x.pinch ?? 1, y.pinch ?? 1, m),
  });
  return { t: k, R: mix(a.R, b.R), L: mix(a.L, b.L) };
}

// Knuckles (side, along the hand), finger bone lengths and splay, index..pinky.
const KNUCKLES: Array<[number, number]> = [[0.32, 0.95], [0.1, 1], [-0.12, 0.95], [-0.32, 0.85]];
const BONES = [[0.45, 0.27, 0.22], [0.5, 0.3, 0.23], [0.46, 0.28, 0.22], [0.36, 0.22, 0.2]];
const SPLAY = [0.08, 0, -0.06, -0.14];
const BEND = [1.45, 1.75, 1.3]; // radians per joint at curl 1

const HAND_SCALE = 1.35; // drawn larger than life, so the fingers read at card size

// The 21 landmarks (MediaPipe's order) of a hand posed by `k`.
export function handPoints(k: HandKey, isRight: boolean): V3[] {
  const D = norm(k.d);
  const N = norm(sub(k.n, mul(D, dot(k.n, D))));
  const S = isRight ? cross(N, D) : cross(D, N); // toward the thumb
  const at = (s: number, a: number, b: number) => add(k.p, add(mul(S, s), add(mul(D, a), mul(N, b))));
  const local = (v: V3) => add(mul(S, v[0]), add(mul(D, v[1]), mul(N, v[2])));
  const pts: V3[] = [k.p];

  // Thumb: from its base by the wrist, swinging across the palm as it folds.
  const c = k.f[0];
  const open: V3[] = [[0.8, 0.6, 0.25], [0.55, 0.85, 0.15], [0.45, 0.9, 0.1]];
  const shut: V3[] = [[0.35, 0.6, 0.7], [-0.5, 0.4, 0.75], [-0.9, 0, 0.4]];
  let p = at(0.22, 0.2, 0.08);
  pts.push(p);
  [0.42, 0.32, 0.27].forEach((len, j) => {
    p = add(p, mul(norm(local(lerp3(open[j], shut[j], Math.min(1, c * (0.6 + 0.2 * j))))), len));
    pts.push(p);
  });

  KNUCKLES.forEach(([s, a], i) => {
    const base = norm(add(D, mul(S, SPLAY[i])));
    let q = at(s, a, 0);
    let phi = 0;
    pts.push(q);
    BONES[i].forEach((len, j) => {
      phi += k.f[i + 1] * BEND[j];
      q = add(q, mul(add(mul(base, Math.cos(phi)), mul(N, Math.sin(phi))), len));
      pts.push(q);
    });
  });

  // A pinch puts the thumb tip just under the index tip.
  if (k.pinch != null) {
    const tip = add(pts[8], mul(D, -k.pinch));
    pts[4] = tip;
    pts[3] = add(lerp3(pts[2], tip, 0.5), mul(S, 0.12));
  }
  return pts.map((v) => add(k.p, mul(sub(v, k.p), HAND_SCALE)));
}

// Elbow by two-bone IK, bending down and out.
function elbow(sh: V3, w: V3, side: number): V3 {
  const [l1, l2] = [2.9, 2.6];
  const v = sub(w, sh);
  const d = Math.min(Math.hypot(...v), (l1 + l2) * 0.999);
  const ax = norm(v);
  const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
  const pole: V3 = [side * 0.6, -1, -0.4];
  const perp = norm(sub(pole, mul(ax, dot(pole, ax))));
  return add(sh, add(mul(ax, a), mul(perp, h)));
}

// --- Drawing -------------------------------------------------------------------

export interface FigureColors {
  body: string;
  hand: string;
}

// Draws gesture `g` at time `t` (ms) into a w×h box at the context's origin.
export function drawFigure(cx: CanvasRenderingContext2D, g: GestureId, t: number, w: number, h: number, col: FigureColors) {
  const pose = keyAt(FIGURES[g], t);
  const k = Math.min(w / 15, h / 9.6); // x ±7.5, y −6 to 3.6
  const F = 16; // camera distance: a hand reaching forward grows
  const proj = (v: V3): [number, number] => {
    const s = (F / (F - v[2])) * k;
    return [w / 2 + v[0] * s, h / 2 + (-1.2 - v[1]) * s]; // mirrored: their right is on your right
  };
  const line = (pts: V3[]) => {
    cx.beginPath();
    pts.forEach((v, i) => (i ? cx.lineTo(...proj(v)) : cx.moveTo(...proj(v))));
    cx.stroke();
  };
  cx.lineCap = cx.lineJoin = "round";

  // Body: head, shoulders, torso.
  cx.strokeStyle = col.body;
  cx.lineWidth = Math.max(1, k * 0.14);
  cx.globalAlpha = 0.45;
  const [hx, hy] = proj([0, 2.3, 0]);
  cx.beginPath();
  cx.arc(hx, hy, 1.05 * k, 0, Math.PI * 2);
  cx.stroke();
  line([[0, 1.25, 0], [0, 0.4, 0]]);
  line([[-1.7, -7, 0], [-2, -0.3, 0], [-1.6, 0.25, 0], [1.6, 0.25, 0], [2, -0.3, 0], [1.7, -7, 0]]);

  // Arms, then hands over them.
  cx.globalAlpha = 0.8;
  cx.lineWidth = Math.max(1.2, k * 0.2);
  const hands = (["R", "L"] as const).map((s) => {
    const side = s === "R" ? 1 : -1;
    const pts = handPoints(pose[s], s === "R");
    const sh: V3 = [side * 1.9, 0, 0];
    line([sh, elbow(sh, pts[0], side), pts[0]]);
    return pts;
  });
  cx.globalAlpha = 1;
  cx.strokeStyle = cx.fillStyle = col.hand;
  for (const pts of hands) {
    const xy = pts.map(proj);
    cx.globalAlpha = 0.18;
    cx.beginPath();
    [0, 1, 5, 9, 13, 17].forEach((i, j) => (j ? cx.lineTo(...xy[i]) : cx.moveTo(...xy[i])));
    cx.fill();
    cx.globalAlpha = 1;
    cx.lineWidth = Math.max(1.2, k * 0.12);
    cx.beginPath();
    for (const { start, end } of HAND_BONES) {
      cx.moveTo(...xy[start]);
      cx.lineTo(...xy[end]);
    }
    cx.stroke();
    for (const [x, y] of xy) {
      cx.beginPath();
      cx.arc(x, y, Math.max(1.2, k * 0.1), 0, Math.PI * 2);
      cx.fill();
    }
  }
}

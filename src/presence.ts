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

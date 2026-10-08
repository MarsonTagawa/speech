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
  minShoulder: 0.05, // shoulder width below this fraction of the frame (side-on) is too unreliable to scale by
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
  body: XY[] | null; // [shoulder midpoint, nose], in pixels
  wrists: XY[]; // in pixels; empty without a body to scale by
  sw: number; // shoulder width in pixels; 0 without a body
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

  // Pixels so x and y share a unit. summarize scales movement by the session's
  // median shoulder width, not each frame's: per-frame widths wobble, and
  // dividing absolute positions by them turns that wobble into fake motion.
  let body: XY[] | null = null;
  let wrists: XY[] = [];
  let sw = 0;
  const p = r.pose;
  if (p && p.length > 12) {
    const px = (q: Pt): XY => [q.x * w, q.y * h];
    const [l, rs, nose] = [px(p[11]), px(p[12]), px(p[0])];
    const width = Math.hypot(l[0] - rs[0], l[1] - rs[1]);
    if (width >= T.minShoulder * w) {
      sw = width;
      body = [[(l[0] + rs[0]) / 2, (l[1] + rs[1]) / 2], nose];
      wrists = r.wrists.map(px);
    }
  }
  return { t, face: !!b, looking, smile, brows, body, wrists, sw };
}

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

  // Movement in shoulder widths, using the session's median width so distance
  // from the camera doesn't matter but frame-to-frame width noise does not count.
  const widths = frames.filter((f) => f.body).map((f) => f.sw).sort((x, y) => x - y);
  const scale = widths.length ? widths[widths.length >> 1] : 1;
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
    travelled += mean(b.body.map((q, j) => dist(q, a.body![j]))) / scale;
    if (a.wrists.length && b.wrists.some((w) => Math.min(...a.wrists.map((v) => dist(w, v))) / scale > T.gestureStep)) gesturing++;
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

// Which of the current hands moved since the previous frame, by the gesture
// score's own rule (wrist travel > gestureStep shoulder widths), so what the
// live highlight shows is what gets counted. Hands are 21 normalised landmarks
// (0 = wrist); each is matched to the nearest previous wrist. Without a body
// to scale by, assume a typical shoulder width of 20% of the frame.
export function movingHands(prev: Pt[][], cur: Pt[][], w: number, h: number, sw: number): boolean[] {
  const step = PRESENCE_TUNING.gestureStep * (sw || 0.2 * w);
  const px = (q: Pt): XY => [q.x * w, q.y * h];
  const before = prev.map((l) => px(l[0]));
  return cur.map((l) => before.length > 0 && Math.min(...before.map((b) => dist(px(l[0]), b))) > step);
}

// Normalised camera coords → pixels in a box showing the video with
// `object-fit: cover` (scaled to fill, centred, overflow cropped).
export function coverMap(srcW: number, srcH: number, boxW: number, boxH: number): (p: Pt) => XY {
  const k = Math.max(boxW / srcW, boxH / srcH);
  const dx = (boxW - srcW * k) / 2;
  const dy = (boxH - srcH * k) / 2;
  return (p) => [p.x * srcW * k + dx, p.y * srcH * k + dy];
}

// The live card's sub-line. No "look at the camera" while a script is open:
// reading looks down, and eye contact isn't scored then anyway.
export function liveCue(frames: Frame[], opts: { recording: boolean; scriptOpen: boolean }): string {
  if (!opts.recording) return "camera on";
  if (!opts.scriptOpen && awayMs(frames) > PRESENCE_TUNING.awayCueMs) return "look at the camera";
  const p = summarize(frames, { scriptOpen: opts.scriptOpen });
  return p ? weakestCue(p) : "warming up…";
}

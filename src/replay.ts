// Screen replay in the report: the session's saved screen frames, with the
// camera as a bubble, drawn for the moment the session audio is playing
// (screen/ in Rust records them; session_frame returns the last frame at or
// before a time). Without playback the canvas holds the last frame drawn.
import { invoke } from "@tauri-apps/api/core";

const SCREEN = 0;
const CAMERA = 1;
const SCREEN_EVERY = 250; // ms of session time between fetches; frames are ≥ 250 ms apart anyway
const CAMERA_EVERY = 100; // ~10 fps, as recorded

interface Rect { x: number; y: number; w: number; h: number }

// An iw×ih image letterboxed into cw×ch.
export function fitRect(iw: number, ih: number, cw: number, ch: number): Rect {
  const s = Math.min(cw / iw, ch / ih);
  const w = Math.round(iw * s);
  const h = Math.round(ih * s);
  return { x: Math.round((cw - w) / 2), y: Math.round((ch - h) / 2), w, h };
}

// The camera bubble: 22% of the width, 16:9, 2% in from the bottom-right.
export function bubbleRect(cw: number, ch: number): Rect {
  const w = Math.round(cw * 0.22);
  const h = Math.round((w * 9) / 16);
  const m = Math.round(cw * 0.02);
  return { x: cw - w - m, y: ch - h - m, w, h };
}

// Whether a frame fetched at session time `last` is stale at `ms` (or never fetched).
export const due = (last: number, ms: number, period: number) => last < 0 || ms < last || ms - last >= period;

interface View {
  ts: number;
  endsMs: number | null;
  screen: ImageBitmap | null;
  cam: ImageBitmap | null;
  screenAt: number;
  camAt: number;
  busy: boolean;
}
const views = new WeakMap<HTMLCanvasElement, View>();

async function frame(ts: number, kind: number, ms: number): Promise<ImageBitmap | null> {
  const buf = await invoke<ArrayBuffer>("session_frame", { ts, kind, ms: Math.max(0, Math.round(ms)) });
  return buf.byteLength ? createImageBitmap(new Blob([buf], { type: "image/jpeg" })) : null;
}

function paint(canvas: HTMLCanvasElement, v: View, ended: boolean) {
  const cx = canvas.getContext("2d");
  if (!cx) return;
  const { width: cw, height: ch } = canvas;
  cx.fillStyle = "#000";
  cx.fillRect(0, 0, cw, ch);
  if (v.screen) {
    const r = fitRect(v.screen.width, v.screen.height, cw, ch);
    cx.drawImage(v.screen, r.x, r.y, r.w, r.h);
  }
  if (v.cam && !ended) {
    const b = bubbleRect(cw, ch);
    cx.save();
    cx.beginPath();
    cx.roundRect(b.x, b.y, b.w, b.h, b.h * 0.18);
    cx.clip();
    cx.drawImage(v.cam, b.x, b.y, b.w, b.h);
    cx.restore();
  }
  if (ended) {
    cx.fillStyle = "rgba(0,0,0,0.55)";
    cx.fillRect(0, ch - 28, cw, 28);
    cx.fillStyle = "#fff";
    cx.font = "12px 'IBM Plex Mono', monospace";
    cx.fillText("Video ended here", 10, ch - 10);
  }
}

// Brings `canvas` to session time `ms`, fetching only what's stale. A fetch
// already in flight makes this a no-op (the next tick catches up).
async function drawAt(canvas: HTMLCanvasElement, ms: number) {
  const v = views.get(canvas);
  if (!v || v.busy) return;
  const ended = v.endsMs !== null && ms >= v.endsMs;
  v.busy = true;
  try {
    // Past the cut the screen keeps its last frame (the record there is empty).
    if (!ended && due(v.screenAt, ms, SCREEN_EVERY)) {
      v.screen = (await frame(v.ts, SCREEN, ms)) ?? v.screen;
      v.screenAt = ms;
    }
    if (due(v.camAt, ms, CAMERA_EVERY)) {
      v.cam = await frame(v.ts, CAMERA, ms); // empty = camera off → no bubble
      v.camAt = ms;
    }
  } catch {
    canvas.hidden = true; // pruned since it was shown
  } finally {
    v.busy = false;
  }
  paint(canvas, v, ended);
}

// Shows `canvas` with session `ts`'s first frame if the session has a replay.
export async function showReplay(canvas: HTMLCanvasElement, ts: number) {
  const f = await invoke<{ screen: boolean; camera: boolean; endsMs: number | null } | null>("session_has_frames", { ts }).catch(() => null);
  if (!f?.screen) return;
  canvas.width = 1280;
  canvas.height = 720;
  canvas.dataset.ts = String(ts);
  views.set(canvas, { ts, endsMs: f.endsMs, screen: null, cam: null, screenAt: -1, camAt: -1, busy: false });
  canvas.hidden = false;
  await drawAt(canvas, 0);
}

// Follows audio playback of session `ts` from `startMs` on `ctx`'s clock,
// drawing every shown replay canvas of that session. Returns a token for
// stopFollow; a newer follow replaces an older one.
let following = 0;
export function followClip(ts: number, startMs: number, ctx: AudioContext): number {
  const token = ++following;
  const t0 = ctx.currentTime;
  const tick = () => {
    if (token !== following) return;
    const ms = startMs + (ctx.currentTime - t0) * 1000;
    document.querySelectorAll<HTMLCanvasElement>(`canvas.rep-video[data-ts="${ts}"]:not([hidden])`).forEach((c) => void drawAt(c, ms));
    requestAnimationFrame(tick);
  };
  tick();
  return token;
}

export function stopFollow(token: number) {
  if (token === following) following++;
}

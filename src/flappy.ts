// Flappy-bird on the "Correcting…" overlay, to fill the wait (Space swaps it in
// for the coach, who becomes the bird). Space or a click flaps through the
// gaps; a crash pauses briefly, then another press restarts.
// The overlay can end it at any moment, so there's nothing to lose.

type Pipe = { x: number; gapY: number; scored: boolean };
export type FlappyState = { mode: "ready" | "play" | "dead"; y: number; vy: number; pipes: Pipe[]; score: number; best: number; t: number };

const R = 11; // bird hitbox radius, px (inside the coach's body, arms excluded)
export const BIRD = 34; // px box the coach is drawn in
const PIPE_W = 28;
const birdX = (w: number) => w * 0.25;
const gap = (h: number) => Math.max(h * 0.4, R * 7);
const G = 2.4; // gravity, board heights per s²
const FLAP = 0.75; // flap speed, board heights per s
const SPEED = 0.32; // scroll, board widths per s
const SPACING = 0.5; // between pipes, board widths
const RETRY = 0.5; // s after a crash before a click restarts (so a mid-flap click doesn't skip the crash)

export const newState = (h: number, best = 0): FlappyState => ({ mode: "ready", y: h / 2, vy: 0, pipes: [], score: 0, best, t: 0 });

export function flap(s: FlappyState, h: number) {
  if (s.mode === "dead") {
    if (s.t < RETRY) return;
    Object.assign(s, newState(h, s.best));
  }
  s.mode = "play";
  s.vy = -FLAP * h;
}

// Advance dt seconds.
export function step(s: FlappyState, dt: number, w: number, h: number) {
  s.t += dt;
  if (s.mode === "ready") return;
  s.vy += G * h * dt;
  s.y = Math.min(h - R, s.y + s.vy * dt);
  if (s.mode === "dead") return; // just falls to the floor

  for (const p of s.pipes) p.x -= SPEED * w * dt;
  s.pipes = s.pipes.filter((p) => p.x > -PIPE_W);
  const last = s.pipes[s.pipes.length - 1];
  if (!last || last.x < w - SPACING * w) {
    const g = gap(h);
    s.pipes.push({ x: w, gapY: g / 2 + 8 + Math.random() * (h - g - 16), scored: false });
  }

  const bx = birdX(w), g = gap(h);
  const crashed =
    s.y - R < 0 ||
    s.y + R >= h ||
    s.pipes.some((p) => bx + R > p.x && bx - R < p.x + PIPE_W && Math.abs(s.y - p.gapY) > g / 2 - R);
  for (const p of s.pipes) {
    if (!p.scored && p.x + PIPE_W < bx - R) (p.scored = true), s.score++;
  }
  s.best = Math.max(s.best, s.score);
  if (crashed) (s.mode = "dead"), (s.t = 0);
}

export class Flappy {
  private ctx: CanvasRenderingContext2D | null;
  private s: FlappyState | null = null;
  private best = 0; // across passes this session
  private raf = 0;
  private last = 0;
  private ink = "#888";
  private mode: FlappyState["mode"] | "" = "";

  // `bird` is an element over the canvas (the coach) moved to the bird each
  // frame; `react` hears mode changes, to animate him.
  constructor(
    private canvas: HTMLCanvasElement,
    private bird: HTMLElement,
    private react: (mode: FlappyState["mode"]) => void,
  ) {
    this.ctx = canvas.getContext("2d");
    canvas.addEventListener("pointerdown", (e) => {
      e.preventDefault(); // keep focus where it was
      this.flap();
    });
  }

  // Ignored while frozen.
  flap() {
    if (this.raf && this.s) flap(this.s, this.canvas.clientHeight);
  }

  start() {
    this.stop();
    const css = getComputedStyle(this.canvas);
    this.ink = css.getPropertyValue("--ink-3").trim() || this.ink;
    this.s = null; // fresh game, sized on the first visible frame
    this.mode = "";
    this.last = performance.now();
    const loop = (now: number) => {
      this.frame(Math.min(0.05, (now - this.last) / 1000));
      this.last = now;
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  // Freezes the board where it is.
  stop() {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  private frame(dt: number) {
    const cv = this.canvas, ctx = this.ctx;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!ctx || w < 60 || h < 40) return; // hidden or squashed: skip
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const s = (this.s ??= newState(h, this.best));
    step(s, dt, w, h);
    this.best = s.best;
    if (s.mode !== this.mode) this.react((this.mode = s.mode));

    const g = gap(h);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = this.ink;
    ctx.globalAlpha = 0.5;
    for (const p of s.pipes) {
      ctx.fillRect(p.x, 0, PIPE_W, p.gapY - g / 2);
      ctx.fillRect(p.x, p.gapY + g / 2, PIPE_W, h - p.gapY - g / 2);
    }
    ctx.globalAlpha = 1;
    ctx.font = '500 13px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = "center";
    ctx.fillText(String(s.score), w / 2, 18);
    if (s.mode !== "play") {
      ctx.globalAlpha = 0.8;
      const msg = s.mode === "ready" ? "space to flap" : s.t < RETRY ? "" : `space to retry${s.best ? ` · best ${s.best}` : ""}`;
      ctx.fillText(msg, w / 2, h / 2 + g / 2 - 10);
      ctx.globalAlpha = 1;
    }
    // the coach: bobs while waiting, tilts with his speed
    const y = s.mode === "ready" ? s.y + Math.sin(s.t * 4) * 4 : s.y;
    const tilt = Math.max(-0.5, Math.min(1.2, s.vy / h));
    this.bird.style.transform = `translate(${birdX(w) - BIRD / 2}px, ${y - BIRD / 2}px) rotate(${tilt}rad)`;
  }
}

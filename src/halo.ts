// Coach halo: the Ribbon's look wrapped around the avatar's silhouette. The
// outline is read from the avatar's svg paths (head + arm nodes) whenever they
// change, so it follows him as his animations turn him. The prism orbits the
// outline (chromatic aberration that travels round him) and the ribbon's
// filaments become wisps circling just outside it. Drawn on a canvas behind
// the avatar svg; setMode() eases between the ribbon's modes.
import { RADIUS } from "@bible-strong/avatar-core";
import { MODES, PROPS, hexToRgb, hueShift, lerp, prism, rgba, type RibbonMode } from "./ribbon";

const deep = hexToRgb(PROPS.blueColor);
const cool = hexToRgb(PROPS.greenColor);
const warm = hexToRgb(PROPS.redColor);
const core = hexToRgb(PROPS.coreColor);
const SPECTRUM = prism(deep, cool, warm);
const TINTS = [deep, cool, core, hueShift(warm, 34, 1.4), warm];
// Body radius as a fraction of the avatar box: the 120-unit sphere in a 300-unit viewBox.
const BODY = RADIUS / 300;
const TAU = Math.PI * 2;
const S = 360; // silhouette samples round the circle

// Outline radius per angle (in body radii) from svg path data in viewBox units:
// the furthest point of any path along each angle, then the body/arm notches
// filled in with a triangle blur so the halo hugs rather than kinks.
// ponytail: an arc path is taken as a circle round the origin (avatar-core's
// head), other paths by their points (fine for its dense bezier nodes).
export function silhouette(ds: string[]): Float32Array {
  const raw = new Float32Array(S).fill(1), out = new Float32Array(S), soft = 7;
  for (const d of ds) {
    const n = (d.match(/-?\d*\.?\d+/g) ?? []).map(Number);
    if (d.includes("A")) {
      for (let i = 0; i < S; i++) raw[i] = Math.max(raw[i], n[2] / RADIUS);
      continue;
    }
    // Walk the point sequence (on-curve and control points: a polyline hugging
    // the curve) in ~1-unit steps, so every angle it crosses gets a radius even
    // where the path's points are sparse (a foreshortened arm).
    for (let k = 0; k + 1 < n.length; k += 2) {
      const x0 = n[k], y0 = n[k + 1], x1 = n[(k + 2) % (n.length & ~1)], y1 = n[(k + 3) % (n.length & ~1)];
      const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0)));
      for (let t = 0; t < steps; t++) {
        const x = x0 + ((x1 - x0) * t) / steps, y = y0 + ((y1 - y0) * t) / steps;
        const i = Math.round((((Math.atan2(y, x) + TAU) % TAU) / TAU) * S) % S;
        raw[i] = Math.max(raw[i], Math.hypot(x, y) / RADIUS);
      }
    }
  }
  for (let i = 0; i < S; i++) {
    let sum = 0, w = 0;
    for (let k = -soft; k <= soft; k++) {
      const wk = soft + 1 - Math.abs(k);
      sum += raw[(i + k + S) % S] * wk;
      w += wk;
    }
    out[i] = Math.max(raw[i], sum / w);
  }
  return out;
}

function edge(sh: Float32Array, a: number) {
  const f = ((((a / TAU) % 1) + 1) % 1) * S, i = Math.floor(f), t = f - i;
  return sh[i % S] * (1 - t) + sh[(i + 1) % S] * t;
}

function outline(ctx: CanvasRenderingContext2D, sh: Float32Array, cx: number, cy: number, R: number, pad = 0) {
  ctx.beginPath();
  for (let i = 0; i <= 120; i++) {
    const a = (i / 120) * TAU, r = R * (edge(sh, a) + pad);
    const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
    if (i) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
  }
  ctx.closePath();
}

// Strokes the outline blurred by `blur` px. WebKitGTK ignores ctx.filter (an
// unblurred stroke shows as a thick band), so the stroke is drawn far off-canvas
// and only its shadow, which does blur, lands. Shadow offset/blur are in device
// pixels, and shadowBlur is twice the gaussian's sigma.
const OFF = 10000;
function blurOutline(ctx: CanvasRenderingContext2D, sh: Float32Array, dpr: number, color: string, blur: number, cx: number, cy: number, R: number, pad = 0) {
  ctx.shadowColor = color;
  ctx.shadowBlur = blur * 2 * dpr;
  ctx.shadowOffsetX = OFF * dpr;
  outline(ctx, sh, cx - OFF, cy, R, pad);
  ctx.stroke();
}

export const still = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

export class Halo {
  mode: RibbonMode = "idle";
  private p = { ...MODES.idle };
  private phase = 0; // integrated like Ribbon's, so speed changes never snap
  private boost = 0; // pulse(): 1 → 0, swells and spins up the wisps and glow on top of the mode
  // pulse() sparks: angle, distance from centre and speed in body radii, life 1 → 0
  private sparks: { a: number; d: number; v: number; life: number; c: number[] }[] = [];
  private last = performance.now();
  private raf = 0;
  private ctx: CanvasRenderingContext2D | null;
  private shape = new Float32Array(S).fill(1); // round until the avatar svg appears
  private shapeKey = "";
  private watch: MutationObserver;
  // Knobs for small halos (the flappy coach): extra glow, a lower resolution
  // cap, and the costly bloom + aberration re-rendered only every Nth frame
  // (blitted from a cache between).
  gain = 1;
  maxDpr = 2;
  cacheEvery = 1;
  paused = false;
  private cache: HTMLCanvasElement | null = null;
  private frameN = 0;

  // follow = false: keep the round body outline instead of tracking the
  // avatar svg beside the canvas (fast-flapping arms trailed as ghost arms).
  constructor(private canvas: HTMLCanvasElement, private follow = true) {
    this.ctx = canvas.getContext("2d");
    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - this.last) / 1000);
      this.last = now;
      if (!this.paused) this.step(dt);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
    // The avatar redraws in its own rAF, which can run after ours in the same
    // frame; redraw (without advancing time) the moment its paths change, so
    // the outline never trails his arms by a frame.
    this.watch = new MutationObserver(() => this.readShape() && this.step(0));
    if (follow && canvas.parentElement) this.watch.observe(canvas.parentElement, { subtree: true, childList: true, attributeFilter: ["d"] });
  }

  // Re-reads the outline from the avatar's svg (beside the canvas); true if it changed.
  private readShape() {
    const ds = [...(this.canvas.parentElement?.querySelectorAll("svg > path") ?? [])].map((p) => p.getAttribute("d") ?? "");
    const key = ds.join();
    if (key === this.shapeKey) return false;
    this.shapeKey = key;
    this.shape = silhouette(ds);
    return true;
  }

  setMode(mode: RibbonMode) {
    this.mode = mode;
  }

  // A brief flare (decays over ~0.8s) and a burst of sparks off his outline, e.g. when he's clicked.
  // Smaller/tinted for minor beats (a corrected line).
  pulse(sparks = 26, strength = 1, tint?: number[]) {
    this.boost = Math.max(this.boost, strength);
    if (still) return;
    for (let i = 0; i < sparks; i++) {
      const c = tint ?? TINTS[i % TINTS.length];
      const a = Math.random() * TAU;
      this.sparks.push({ a, d: edge(this.shape, a), v: 1.5 + Math.random() * 2.5, life: 0.7 + Math.random() * 0.3, c });
    }
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.watch.disconnect();
  }

  private step(dt: number) {
    const cv = this.canvas, ctx = this.ctx;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!ctx || w < 8 || h < 8) return; // hidden view: skip the draw
    const dpr = Math.min(window.devicePixelRatio || 1, this.maxDpr);
    if (cv.width !== Math.round(w * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const reshaped = this.follow && this.readShape();
    const target = MODES[this.mode];
    const k = 1 - Math.pow(0.001, dt);
    for (const key of ["amp", "speed", "ab", "glow"] as const) this.p[key] = lerp(this.p[key], target[key], k);
    this.boost = Math.max(0, this.boost - dt / 0.8);
    const b = this.boost * this.boost; // eased out
    if (!still) this.phase += this.p.speed * (1 + 6 * b) * dt; // pulse spins the wisps up

    const P = { amp: this.p.amp * (1 + 2.5 * b), ab: this.p.ab * (1 + b), glow: (this.p.glow + b) * this.gain };
    const cx = w / 2, cy = h / 2;
    const R = (cv.parentElement?.clientWidth ?? w) * BODY;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const heavy = (ctx: CanvasRenderingContext2D) => {
      ctx.globalCompositeOperation = "lighter";
      // bloom hugging the outline: wide blurred strokes along the silhouette
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#000"; // only the shadow shows; its alpha comes from shadowColor
      ctx.lineWidth = R * 0.5;
      blurOutline(ctx, this.shape, dpr, rgba(core, 0.3 * P.glow + 0.08), R * 0.3, cx, cy, R, 0.05);
      ctx.lineWidth = R * 0.9;
      blurOutline(ctx, this.shape, dpr, rgba(cool, 0.04 * P.glow), R * 0.3, cx, cy, R, 0.3);

      // aberration: the prism as outlines, each nudged along a direction that
      // orbits the body, so the colour split travels round him
      const th = this.phase * TAU * 0.8;
      const ab = P.ab * PROPS.aberration * R * 0.03;
      ctx.lineWidth = R * 0.09;
      for (const f of SPECTRUM) {
        blurOutline(ctx, this.shape, dpr, rgba(f.c, f.a), R * 0.035, cx + Math.cos(th) * f.o * ab, cy + Math.sin(th) * f.o * ab, R);
        ctx.fill(); // a ring nudged outward would lift off his edge; fill the gap (the rest hides behind him)
      }
      ctx.shadowColor = "transparent";
      ctx.shadowOffsetX = 0;
      ctx.globalCompositeOperation = "source-over";
    };
    if (this.cacheEvery > 1) {
      const cc = (this.cache ??= document.createElement("canvas"));
      const fresh = cc.width !== cv.width || cc.height !== cv.height;
      if (fresh) (cc.width = cv.width), (cc.height = cv.height);
      if (fresh || reshaped || this.frameN++ % this.cacheEvery === 0) {
        const cctx = cc.getContext("2d")!;
        cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        cctx.clearRect(0, 0, w, h);
        heavy(cctx);
      }
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(cc, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    } else heavy(ctx);
    ctx.globalCompositeOperation = "lighter";

    // wisps: the ribbon's filaments bent into arcs, alternate ones counter-
    // rotating, radius wobbling with the mode's amplitude, faded at both ends
    const N = 48;
    for (let j = 0; j < PROPS.strands; j++) {
      const dir = j % 2 ? -1 : 1;
      const a0 = dir * this.phase * TAU * (0.55 + j * 0.12) + j * 1.9;
      const len = Math.PI * (0.55 + 0.25 * Math.sin(this.phase * 3 + j));
      const c = TINTS[j % TINTS.length];
      ctx.lineWidth = Math.max(0.7, R * 0.035 * (0.6 + 0.4 * ((j + 1) % 2)) * (1 + 1.5 * b));
      let px = 0, py = 0;
      for (let s = 0; s <= N; s++) {
        const u = s / N, a = a0 + u * len;
        const r = R * (1.12 + j * 0.05 + (0.1 + P.amp * 0.6) * Math.sin(a * 3 + this.phase * TAU * 1.3 + j * 0.7));
        const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
        if (s) {
          ctx.strokeStyle = rgba(c, (0.75 + 0.25 * b) * Math.sin(Math.PI * u));
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
        px = x;
        py = y;
      }
    }
    // sparks: fly out from the outline, slowing and shrinking as they fade
    for (const sp of this.sparks) {
      sp.d += sp.v * dt;
      sp.v *= Math.pow(0.04, dt); // drag
      sp.life -= dt / 0.9;
      if (sp.life <= 0) continue;
      ctx.fillStyle = rgba(sp.c, sp.life);
      ctx.beginPath();
      ctx.arc(cx + Math.cos(sp.a) * sp.d * R, cy + Math.sin(sp.a) * sp.d * R, Math.max(0.6, R * 0.07 * sp.life), 0, TAU);
      ctx.fill();
    }
    this.sparks = this.sparks.filter((sp) => sp.life > 0);
    ctx.globalCompositeOperation = "source-over";
  }
}

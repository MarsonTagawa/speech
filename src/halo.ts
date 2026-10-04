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
  // Knobs for small halos (the flappy coach): extra glow and a lower resolution cap.
  gain = 1;
  maxDpr = 2;
  paused = false;
  // The bloom and aberration are the same blurred outline every frame; only
  // their alpha and (aberration) offset move. Shadow blurs are the costly part,
  // so they're baked at alpha 1 once per outline/size and blitted: bloom ×2,
  // the aberration's white mask, then the mask tinted per SPECTRUM colour.
  private layers = Array.from({ length: SPECTRUM.length + 3 }, () => document.createElement("canvas"));
  private bakedKey = "";

  // follow = false: keep the round body outline instead of tracking the
  // avatar svg beside the canvas (fast-flapping arms trailed as ghost arms).
  constructor(private canvas: HTMLCanvasElement, follow = true) {
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
    if (follow && canvas.parentElement) {
      this.watch.observe(canvas.parentElement, { subtree: true, childList: true, attributeFilter: ["d"] });
      this.readShape(); // the observer catches every later change
    }
  }

  // Re-reads the outline from the avatar's svg (beside the canvas); true if it changed.
  private readShape() {
    const ds = [...(this.canvas.parentElement?.querySelectorAll("svg > path") ?? [])].map((p) => p.getAttribute("d") ?? "");
    const key = ds.join();
    if (key === this.shapeKey) return false;
    this.shapeKey = key;
    this.shape = silhouette(ds);
    this.bakedKey = "";
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

  private bake(w: number, h: number, dpr: number, R: number) {
    const cx = w / 2, cy = h / 2;
    const [b1, b2, mask, ...tints] = this.layers.map((c) => {
      if (c.width !== this.canvas.width || c.height !== this.canvas.height) (c.width = this.canvas.width), (c.height = this.canvas.height);
      const ctx = c.getContext("2d")!;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalCompositeOperation = "source-over";
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.lineJoin = "round";
      return ctx;
    });
    // bloom hugging the outline: wide blurred strokes along the silhouette
    b1.lineWidth = R * 0.5;
    blurOutline(b1, this.shape, dpr, rgba(core, 1), R * 0.3, cx, cy, R, 0.05);
    b2.lineWidth = R * 0.9;
    blurOutline(b2, this.shape, dpr, rgba(cool, 1), R * 0.3, cx, cy, R, 0.3);
    mask.lineWidth = R * 0.09;
    blurOutline(mask, this.shape, dpr, "#fff", R * 0.035, cx, cy, R);
    mask.fill(); // a ring nudged outward would lift off his edge; fill the gap (the rest hides behind him)
    tints.forEach((t, i) => {
      t.setTransform(1, 0, 0, 1, 0, 0);
      t.drawImage(mask.canvas, 0, 0);
      t.globalCompositeOperation = "source-in";
      t.fillStyle = rgba(SPECTRUM[i].c, 1);
      t.fillRect(0, 0, t.canvas.width, t.canvas.height);
    });
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

    const key = cv.width + "," + cv.height + "," + R;
    if (this.bakedKey !== key) {
      this.bake(w, h, dpr, R);
      this.bakedKey = key;
    }
    ctx.globalCompositeOperation = "lighter";
    const blit = (i: number, alpha: number, dx = 0, dy = 0) => {
      ctx.globalAlpha = alpha;
      ctx.drawImage(this.layers[i], dx, dy, w, h);
    };
    blit(0, 0.3 * P.glow + 0.08);
    blit(1, 0.04 * P.glow);
    // aberration: the prism as outlines, each nudged along a direction that
    // orbits the body, so the colour split travels round him
    const th = this.phase * TAU * 0.8;
    const ab = P.ab * PROPS.aberration * R * 0.03;
    SPECTRUM.forEach((f, i) => blit(i + 3, f.a, Math.cos(th) * f.o * ab, Math.sin(th) * f.o * ab));
    ctx.globalAlpha = 1;

    // wisps: the ribbon's filaments bent into arcs, alternate ones counter-
    // rotating, radius wobbling with the mode's amplitude, faded at both ends
    // (a conic gradient round the centre, so each is one stroke)
    const N = 48, G = 8;
    for (let j = 0; j < PROPS.strands; j++) {
      const dir = j % 2 ? -1 : 1;
      const a0 = dir * this.phase * TAU * (0.55 + j * 0.12) + j * 1.9;
      const len = Math.PI * (0.55 + 0.25 * Math.sin(this.phase * 3 + j));
      const c = TINTS[j % TINTS.length];
      const fade = ctx.createConicGradient(a0, cx, cy);
      for (let g = 0; g <= G; g++) fade.addColorStop(((g / G) * len) / TAU, rgba(c, (0.75 + 0.25 * b) * Math.sin((Math.PI * g) / G)));
      ctx.strokeStyle = fade;
      ctx.lineWidth = Math.max(0.7, R * 0.035 * (0.6 + 0.4 * ((j + 1) % 2)) * (1 + 1.5 * b));
      ctx.beginPath();
      for (let s = 0; s <= N; s++) {
        const a = a0 + (s / N) * len;
        const r = R * (1.12 + j * 0.05 + (0.1 + P.amp * 0.6) * Math.sin(a * 3 + this.phase * TAU * 1.3 + j * 0.7));
        const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
        if (s) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      }
      ctx.stroke();
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

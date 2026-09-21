// Siri-style ribbon canvas animation, lifted verbatim from "Siri Ribbon.html"
// (a bundled DCLogic/React component) and stripped of that framework so it runs
// as a plain canvas driver. Call setMode() to cross-fade between states.

export type RibbonMode = "idle" | "listening" | "thinking" | "speaking";

const MODES: Record<RibbonMode, any> = {
  idle: { label: "Idle", amp: 0.10, speed: 0.09, ab: 1.0, glow: 0.30, spread: 0.32, width: 0.075, split: 0.00 },
  listening: { label: "Listening", amp: 0.24, speed: 0.30, ab: 1.3, glow: 0.55, spread: 0.40, width: 0.125, split: 0.18 },
  thinking: { label: "Thinking", amp: 0.13, speed: 0.62, ab: 1.7, glow: 0.36, spread: 0.28, width: 0.070, split: 0.08 },
  speaking: { label: "Speaking", amp: 0.26, speed: 0.48, ab: 1.8, glow: 0.52, spread: 0.40, width: 0.105, split: 0.16 },
};

const PROPS = {
  strands: 5,
  blueColor: "#1a37ff",
  greenColor: "#2bff8a",
  redColor: "#ff2f45",
  coreColor: "#ffffff",
  aberration: 1.4,
};

function hexToRgb(h: string): number[] {
  const s = String(h || "#ffffff").replace("#", "");
  const f = s.length === 3 ? s.split("").map((c) => c + c).join("") : s;
  const n = parseInt(f, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const rgba = (c: number[], a: number) => "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")";

// rotate a colour's hue (deg) and optionally boost saturation — used to spread one
// anchor colour into a small prism of neighbouring wavelengths
function hueShift(c: number[], deg: number, sat?: number): number[] {
  const r = c[0] / 255, g = c[1] / 255, b = c[2] / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d) {
    if (mx === r) h = ((g - b) / d) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
  }
  h = (h * 60 + deg + 360) % 360;
  const l = (mx + mn) / 2;
  let s = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
  s = Math.min(1, s * (sat == null ? 1 : sat));
  const cc = (1 - Math.abs(2 * l - 1)) * s, hp = h / 60, x = cc * (1 - Math.abs((hp % 2) - 1));
  let rp = [cc, x, 0];
  if (hp >= 1 && hp < 2) rp = [x, cc, 0];
  else if (hp >= 2 && hp < 3) rp = [0, cc, x];
  else if (hp >= 3 && hp < 4) rp = [0, x, cc];
  else if (hp >= 4 && hp < 5) rp = [x, 0, cc];
  else if (hp >= 5) rp = [cc, 0, x];
  const m = l - cc / 2;
  return [Math.round((rp[0] + m) * 255), Math.round((rp[1] + m) * 255), Math.round((rp[2] + m) * 255)];
}
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export class Ribbon {
  props = PROPS;
  canvas: HTMLCanvasElement;
  mode: RibbonMode;
  base: any; // smoothly-lerped mode params
  p: any;    // per-frame render params = base folded with the live mic level
  level = 0; // smoothed mic level, 0..1
  private targetLevel = 0;
  // Integrated wave phase. Motion advances by speed·dt each frame instead of
  // (elapsed·speed): that way changing speed — on a mode switch or with the mic
  // level — only bends future motion, and never snaps the phase by an amount
  // proportional to how long the app has been open.
  private phase = 0;
  // Per-sheet phase. Each sheet's travel direction (dir) depends on `split`,
  // which changes between modes; integrating dir into the velocity keeps the
  // motion continuous instead of snapping dir·phase when split animates.
  private sphase = [0, 0, 0];

  private raf = 0;
  private t0 = 0;
  private last = 0;
  private ctx: CanvasRenderingContext2D | null = null;
  private w = 0;
  private h = 0;
  private dpr = 1;
  private sx?: Float32Array;
  private sxW = 0;
  private cSp: Float32Array[] = [];
  private cBd: Float32Array[] = [];
  private off?: HTMLCanvasElement;
  private octx: CanvasRenderingContext2D | null = null;
  private small?: HTMLCanvasElement;
  private refl?: HTMLCanvasElement;
  private sctx: CanvasRenderingContext2D | null = null;
  private rctx: CanvasRenderingContext2D | null = null;

  constructor(canvas: HTMLCanvasElement, mode: RibbonMode = "idle") {
    this.canvas = canvas;
    this.mode = mode;
    this.base = Object.assign({}, MODES[mode]);
    this.p = Object.assign({}, this.base);
    this.t0 = performance.now();
    this.last = this.t0;
    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - this.last) / 1000);
      this.last = now;
      this.step(dt, (now - this.t0) / 1000);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  setMode(mode: RibbonMode) {
    this.mode = mode;
  }

  // live mic loudness, 0..1; drives amplitude/glow/width/speed
  setLevel(v: number) {
    this.targetLevel = Math.max(0, Math.min(1, v));
  }

  destroy() {
    cancelAnimationFrame(this.raf);
  }

  // keeps the backing store in sync with layout, every frame
  fit(): boolean {
    const cv = this.canvas;
    if (!cv) return false;
    if (!this.ctx || this.ctx.canvas !== cv) this.ctx = cv.getContext("2d");
    const r = cv.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width));
    const h = Math.max(1, Math.round(r.height));
    if (w < 8 || h < 8) return false;
    // The source component pinned dpr to 1 (fine in its editor preview); on a
    // HiDPI display that renders at 1x and the browser upscales it → pixelated.
    // Match the device so the strands/meniscus render crisp.
    const dpr = this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (this.w !== w || this.h !== h || cv.width !== Math.round(w * dpr)) {
      this.w = w;
      this.h = h;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    return true;
  }

  step(dt: number, t: number) {
    if (!this.fit()) return;
    const target = MODES[this.mode] || MODES.idle;
    const k = 1 - Math.pow(0.001, dt);
    for (const key in target) {
      if (typeof target[key] === "number") this.base[key] = lerp(this.base[key], target[key], k);
    }
    // Envelope-follow the mic level: fast attack so each syllable punches the
    // ribbon (the "jump"), slower release so it eases back between words.
    const rate = this.targetLevel > this.level
      ? 1 - Math.pow(0.0000005, dt) // attack ~65ms
      : 1 - Math.pow(0.04, dt); // release ~300ms
    this.level = lerp(this.level, this.targetLevel, rate);
    const d = this.level;
    this.p = Object.assign({}, this.base);
    this.p.amp = this.base.amp * (1 + d * 2.8);
    this.p.glow = this.base.glow * (1 + d * 2.2);
    this.p.width = this.base.width * (1 + d * 1.6);
    this.p.speed = this.base.speed * (1 + d * 2.8); // go faster when loud
    this.phase += this.p.speed * dt;
    const sp = this.p.split == null ? 1 : this.p.split;
    for (let i = 0; i < 3; i++) {
      const dir = 1 - 2 * (i % 2) * sp;
      this.sphase[i] += dir * this.p.speed * dt;
    }
    this.draw(t);
  }

  // vertical position of a ribbon's spine at normalized x. Reads the sheet's own
  // integrated phase (this.sphase[i]) so direction can shift with `split` without
  // snapping.
  spine(x: number, i: number): number {
    const P = this.p;
    const h = this.h;
    const env = Math.exp(-Math.pow((x - 0.5) / P.spread, 2));
    const sp = P.split == null ? 1 : P.split;
    const ph = i * 2.1 * sp;
    const f = 1 + i * 0.18 * sp;
    const phase = this.sphase[i];
    const a =
      Math.sin((x * 0.95 * f - phase * 1.4) * Math.PI * 2 + ph) * 1.0 +
      Math.sin((x * 1.9 * f + phase * 0.9) * Math.PI * 2 + ph * 1.6) * 0.26 +
      // loud-only ripple: a faster, tighter wave that makes it jump around when
      // you speak; scaled by level so it vanishes in silence. Uses the same
      // integrated phase, so it can't snap.
      Math.sin((x * 3.3 * f - phase * 2.7) * Math.PI * 2 + ph * 2.3) * (0.85 * this.level);
    return h * 0.5 + a * env * P.amp * h * 0.30;
  }

  band(x: number, i: number): number {
    const env = Math.pow(Math.max(0, 1 - Math.pow((x - 0.5) / 0.40, 2)), 1.7);
    const pulse = 0.65 + 0.35 * Math.sin((x * 2.2 + this.phase * 1.8) * Math.PI * 2 + i);
    return this.p.width * this.h * env * pulse;
  }

  path(ctx: CanvasRenderingContext2D, i: number, dy: number, scale?: number, frac?: number) {
    const N = 96, k = (scale == null ? 1 : scale) * 0.5, fr = frac || 0;
    const xs = this.sx!, sp = this.cSp[i], bd = this.cBd[i];
    ctx.beginPath();
    for (let j = 0; j <= N; j++) {
      const b = bd[j];
      const y = sp[j] + dy + b * fr - b * k;
      j === 0 ? ctx.moveTo(xs[j], y) : ctx.lineTo(xs[j], y);
    }
    for (let j = N; j >= 0; j--) {
      const b = bd[j];
      ctx.lineTo(xs[j], sp[j] + dy + b * fr + b * k);
    }
    ctx.closePath();
  }

  grad(ctx: CanvasRenderingContext2D, alpha: number, color: number[]): CanvasGradient {
    const g = ctx.createLinearGradient(0, 0, this.w, 0);
    g.addColorStop(0.0, rgba(color, 0));
    g.addColorStop(0.10, rgba(color, 0));
    g.addColorStop(0.30, rgba(color, alpha * 0.45));
    g.addColorStop(0.5, rgba(color, alpha));
    g.addColorStop(0.70, rgba(color, alpha * 0.45));
    g.addColorStop(0.90, rgba(color, 0));
    g.addColorStop(1.0, rgba(color, 0));
    return g;
  }

  draw(t: number) {
    const ctx = this.ctx!, P = this.p, w = this.w, h = this.h;
    if (!ctx) return;
    const cool = hexToRgb(this.props.greenColor || "#2bff8a");
    const warm = hexToRgb(this.props.redColor || "#ff2f45");
    const core = hexToRgb(this.props.coreColor || "#ffffff");
    const deep = hexToRgb(this.props.blueColor || "#1a37ff");
    const abScale = this.props.aberration == null ? 1 : this.props.aberration;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    const dpr = this.dpr || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalCompositeOperation = "lighter";

    // ambient bloom hugging the ribbon spine
    const sy = this.spine(0.5, 1);
    const bl = ctx.createRadialGradient(w * 0.5, sy, 0, w * 0.5, sy, w * 0.14);
    bl.addColorStop(0, rgba(core, 0.045 * P.glow));
    bl.addColorStop(0.45, rgba(cool, 0.02 * P.glow));
    bl.addColorStop(1, rgba(core, 0));
    ctx.filter = "none";
    ctx.fillStyle = bl;
    ctx.fillRect(0, 0, w, h);

    // sample each sheet's spine/thickness once per frame; every pass below reads these arrays
    const N = 96;
    if (!this.sx || this.sxW !== w) {
      this.sx = new Float32Array(N + 1);
      for (let k = 0; k <= N; k++) this.sx[k] = (k / N) * w;
      this.sxW = w;
      this.cSp = [0, 1, 2].map(() => new Float32Array(N + 1));
      this.cBd = [0, 1, 2].map(() => new Float32Array(N + 1));
    }
    // Every sheet's vertical deviation from the centerline (its own wave + the
    // inter-sheet offset) is windowed by sin(pi*x), which is exactly 0 at both
    // edges and 1 in the middle. So all three sheets collapse onto the same
    // centerline at x=0 and x=1 (joined edges) and bloom apart in the center.
    const half = h * 0.5;
    const sepAmt = (P.split == null ? 1 : P.split) * h * 0.017;
    for (let i = 0; i < 3; i++) {
      const sp = this.cSp[i], bd = this.cBd[i];
      const sep = (i - 1) * sepAmt;
      for (let k = 0; k <= N; k++) {
        const x = k / N;
        const edge = Math.sin(Math.PI * x);
        sp[k] = half + (this.spine(x, i) - half + sep) * edge;
        bd[k] = this.band(x, i);
      }
    }

    // ribbon stack: drawn crisp into a half-res offscreen, then composited with two blurred blits
    const S = 0.5;
    if (!this.off) this.off = document.createElement("canvas");
    if (this.off.width !== Math.round(w * S) || this.off.height !== Math.round(h * S)) {
      this.off.width = Math.max(1, Math.round(w * S));
      this.off.height = Math.max(1, Math.round(h * S));
      this.octx = this.off.getContext("2d");
    }
    const o = this.octx!;
    o.setTransform(1, 0, 0, 1, 0, 0);
    o.clearRect(0, 0, this.off.width, this.off.height);
    o.setTransform(S, 0, 0, S, 0, 0);
    o.globalCompositeOperation = "lighter";
    o.filter = "none";

    // frosted-glass bridge across the split: as the sheets pull apart, wash a
    // soft milky tint into the gap between them so the space reads as blurred
    // frosted glass rather than empty black. The colour is the cool/warm blend
    // pulled most of the way to white, and the fill itself is blurred. Fades to
    // nothing at the joined edges and vanishes entirely at split=0.
    const gap = P.split == null ? 1 : P.split;
    const fade = Math.min(1, gap * 4); // split values run ~0..0.18; map to 0..1
    if (fade > 0.001) {
      // full vertical envelope across all three sheets (spine ± half thickness),
      // so every gap between the weaving sheets gets filled, not just the 0↔2 band
      const yTop = new Float32Array(N + 1), yBot = new Float32Array(N + 1);
      for (let j = 0; j <= N; j++) {
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < 3; i++) {
          lo = Math.min(lo, this.cSp[i][j] - this.cBd[i][j] * 0.5);
          hi = Math.max(hi, this.cSp[i][j] + this.cBd[i][j] * 0.5);
        }
        yTop[j] = lo; yBot[j] = hi;
      }
      o.beginPath();
      for (let j = 0; j <= N; j++) (j === 0 ? o.moveTo : o.lineTo).call(o, this.sx![j], yTop[j]);
      for (let j = N; j >= 0; j--) o.lineTo(this.sx![j], yBot[j]);
      o.closePath();
      const tint = cool.map((c, k) => (c + warm[k]) / 2 * 0.35 + core[k] * 0.65); // mostly white
      const gw = o.createLinearGradient(0, 0, w, 0);
      gw.addColorStop(0.0, rgba(tint, 0));
      gw.addColorStop(0.5, rgba(tint, 0.25 * fade));
      gw.addColorStop(1.0, rgba(tint, 0));
      o.globalCompositeOperation = "source-over";
      o.filter = "blur(" + Math.max(1, h * 0.012).toFixed(1) + "px)";
      o.fillStyle = gw;
      o.fill();
      o.filter = "none";
      o.globalCompositeOperation = "lighter";
    }

    const ab = P.ab * abScale * 0.34; // in units of local sheet thickness
    // full prism: violet→cyan on the upper edge, yellow→red on the lower one
    const spectrum = [
      { o: -1.80, c: hueShift(deep, -10, 1.5), a: 0.30 },
      { o: -1.55, c: deep, a: 0.38 },
      { o: -1.18, c: hueShift(cool, -34, 1.6), a: 0.52 },
      { o: -0.86, c: cool, a: 0.82 },
      { o: -0.50, c: hueShift(cool, 30, 1.5), a: 0.46 },
      { o: 0.50, c: hueShift(warm, 34, 1.5), a: 0.46 },
      { o: 0.85, c: warm, a: 0.76 },
      { o: 1.15, c: hueShift(warm, 8, 1.6), a: 0.38 },
    ];
    for (let i = 0; i < 3; i++) {
      const sp = P.split == null ? 1 : P.split;
      const depth = (1 - i * 0.28) * (i === 0 ? 1 : 0.30 + 0.70 * sp);
      const dy = 0; // separation is now baked into cSp (windowed), see sampling loop above
      const flip = i === 1 ? 1 - 2 * sp : 1; // middle sheet inverts only as the stack separates

      for (let k = 0; k < spectrum.length; k++) {
        const f = spectrum[k];
        this.path(o, i, dy, 1.0, f.o * ab * flip);
        o.fillStyle = this.grad(o, f.a * depth, f.c);
        o.fill();
      }

      this.path(o, i, dy, 0.42);
      o.fillStyle = this.grad(o, 0.045 * depth, core);
      o.fill();

      // hollow out the interior so the sheet reads as edge-lit glass rather than a solid glow
      o.globalCompositeOperation = "destination-out";
      this.path(o, i, dy, 0.30);
      const hol = o.createLinearGradient(0, 0, w, 0);
      hol.addColorStop(0, "rgba(0,0,0,0)");
      hol.addColorStop(0.5, "rgba(0,0,0," + (0.42 * depth).toFixed(3) + ")");
      hol.addColorStop(1, "rgba(0,0,0,0)");
      o.fillStyle = hol;
      o.fill();
      o.globalCompositeOperation = "lighter";

      // meniscus: a crisp specular line riding the upper surface + a softer wet line below
      for (let pass = 0; pass < 2; pass++) {
        const off = pass === 0 ? -0.30 : 0.34;
        o.lineWidth = Math.max(1, h * (pass === 0 ? 0.0022 : 0.0014) * depth);
        o.strokeStyle = this.grad(o, (pass === 0 ? 0.42 : 0.16) * depth, core);
        o.beginPath();
        for (let j = 0; j <= N; j++) {
          const y = this.cSp[i][j] + dy + this.cBd[i][j] * off;
          j === 0 ? o.moveTo(this.sx![j], y) : o.lineTo(this.sx![j], y);
        }
        o.stroke();
      }

      // travelling caustic glint, as light focused through moving water
      const gx = 0.5 + 0.34 * Math.sin((t * (0.22 + i * 0.07) + i * 0.6) * Math.PI * 2);
      const gi = Math.max(0, Math.min(N, Math.round(gx * N)));
      const gy = this.cSp[i][gi] + dy - this.cBd[i][gi] * 0.28;
      const gr = h * 0.10 * depth;
      const gg = o.createRadialGradient(gx * w, gy, 0, gx * w, gy, gr);
      gg.addColorStop(0, rgba(core, 0.18 * depth));
      gg.addColorStop(0.45, rgba(core, 0.05 * depth));
      gg.addColorStop(1, rgba(core, 0));
      o.fillStyle = gg;
      o.beginPath();
      o.ellipse(gx * w, gy, gr * 5.0, gr * 0.34, 0, 0, Math.PI * 2);
      o.fill();
    }

    // halo + reflection are built from a quarter-res downscale (cheap blur)
    const qw = Math.max(1, Math.round(w * 0.22)), qh = Math.max(1, Math.round(h * 0.22));
    if (!this.small) { this.small = document.createElement("canvas"); this.refl = document.createElement("canvas"); }
    if (this.small.width !== qw || this.small.height !== qh) {
      this.small.width = this.refl!.width = qw;
      this.small.height = this.refl!.height = qh;
      this.sctx = this.small.getContext("2d");
      this.rctx = this.refl!.getContext("2d");
    }
    const sc = this.sctx!, rc = this.rctx!;
    sc.setTransform(1, 0, 0, 1, 0, 0);
    sc.globalCompositeOperation = "source-over";
    sc.globalAlpha = 1;
    sc.filter = "none";
    sc.clearRect(0, 0, qw, qh);
    sc.drawImage(this.off, 0, 0, qw, qh);

    // reflection: mirrored copy, masked to a soft vertical band so no straight edge survives
    rc.setTransform(1, 0, 0, 1, 0, 0);
    rc.globalCompositeOperation = "source-over";
    rc.globalAlpha = 1;
    rc.clearRect(0, 0, qw, qh);
    rc.save();
    rc.translate(0, qh);
    rc.scale(1, -1);
    rc.drawImage(this.small, 0, -qh * 0.05, qw, qh);
    rc.restore();
    rc.globalCompositeOperation = "destination-in";
    const mask = rc.createLinearGradient(0, 0, 0, qh);
    mask.addColorStop(0, "rgba(0,0,0,0)");
    mask.addColorStop(0.40, "rgba(0,0,0,0)");
    mask.addColorStop(0.58, "rgba(0,0,0,0.8)");
    mask.addColorStop(0.78, "rgba(0,0,0,0.25)");
    mask.addColorStop(1, "rgba(0,0,0,0)");
    rc.fillStyle = mask;
    rc.fillRect(0, 0, qw, qh);

    ctx.imageSmoothingEnabled = true;
    ctx.filter = "blur(" + Math.max(2, h * 0.012).toFixed(1) + "px)";
    ctx.globalAlpha = 0.5 * (0.6 + P.glow);
    ctx.drawImage(this.small, 0, 0, w, h);
    ctx.globalAlpha = 0.20;
    ctx.drawImage(this.refl!, 0, 0, w, h);

    // crisp pass
    ctx.globalAlpha = 1;
    ctx.filter = "blur(" + Math.max(0.6, h * 0.003).toFixed(1) + "px)";
    ctx.drawImage(this.off, 0, 0, w, h);

    ctx.filter = "none";
    // individual filaments: thin plucked strings weaving over the sheet
    const nS = Math.round(this.props.strands == null ? 5 : this.props.strands);
    if (nS > 0) {
      const tints = [deep, cool, core, hueShift(warm, 34, 1.4), warm];
      ctx.filter = "none";
      for (let j = 0; j < nS; j++) {
        const u = nS === 1 ? 0.5 : j / (nS - 1);
        const sgn = u < 0.5 ? -1 : 1;
        const spreadOut = 0.35 + 1.5 * Math.abs(u - 0.5) * 2;
        const ph = j * 1.9 + 0.7;
        const sw = 0.55 + 0.9 * Math.abs(0.5 - u);
        ctx.lineWidth = Math.max(0.8, h * 0.0022 * sw);
        ctx.strokeStyle = this.grad(ctx, 0.55 + 0.30 * (1 - Math.abs(u - 0.5) * 2), tints[j % tints.length]);
        ctx.beginPath();
        for (let k = 0; k <= N; k++) {
          const x = k / N;
          const env = Math.pow(Math.max(0, 1 - Math.pow((x - 0.5) / 0.44, 2)), 1.3);
          const wob =
            Math.sin((x * (1.5 + j * 0.45) - this.phase * (1.1 + j * 0.22)) * Math.PI * 2 + ph) * 1.0 +
            Math.sin((x * (3.4 + j * 0.3) + this.phase * 0.8) * Math.PI * 2 + ph * 1.4) * 0.22;
          const y = this.cSp[0][k]
            + sgn * this.cBd[0][k] * spreadOut * 0.55
            + wob * env * h * 0.012 * (0.5 + P.amp * 2.2);
          k === 0 ? ctx.moveTo(this.sx![k], y) : ctx.lineTo(this.sx![k], y);
        }
        ctx.stroke();
      }
    }

    ctx.filter = "none";
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }
}

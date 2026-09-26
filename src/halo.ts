// Coach halo: the Ribbon's look wrapped around a circle. The prism orbits the
// avatar's outline (chromatic aberration that travels round him) and the
// ribbon's filaments become wisps circling just outside it. Drawn on a canvas
// behind the avatar svg; setMode() eases between the ribbon's modes.
import { MODES, PROPS, hexToRgb, hueShift, lerp, prism, rgba, type RibbonMode } from "./ribbon";

const deep = hexToRgb(PROPS.blueColor);
const cool = hexToRgb(PROPS.greenColor);
const warm = hexToRgb(PROPS.redColor);
const core = hexToRgb(PROPS.coreColor);
const SPECTRUM = prism(deep, cool, warm);
const TINTS = [deep, cool, core, hueShift(warm, 34, 1.4), warm];
// ponytail: assumes a round body filling 80% of the avatar box (Strobi's
// 240-unit sphere in a 300-unit viewBox); read it from the definition if other avatars land.
const BODY = 0.4;
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

export class Halo {
  mode: RibbonMode = "idle";
  private p = { ...MODES.idle };
  private phase = 0; // integrated like Ribbon's, so speed changes never snap
  private last = performance.now();
  private raf = 0;
  private ctx: CanvasRenderingContext2D | null;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d");
    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - this.last) / 1000);
      this.last = now;
      this.step(dt);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  setMode(mode: RibbonMode) {
    this.mode = mode;
  }

  destroy() {
    cancelAnimationFrame(this.raf);
  }

  private step(dt: number) {
    const cv = this.canvas, ctx = this.ctx;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!ctx || w < 8 || h < 8) return; // hidden view: skip the draw
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (cv.width !== Math.round(w * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const target = MODES[this.mode];
    const k = 1 - Math.pow(0.001, dt);
    for (const key of ["amp", "speed", "ab", "glow"] as const) this.p[key] = lerp(this.p[key], target[key], k);
    if (!still) this.phase += this.p.speed * dt;

    const P = this.p, cx = w / 2, cy = h / 2;
    const R = (cv.parentElement?.clientWidth ?? w) * BODY;
    const TAU = Math.PI * 2;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.globalCompositeOperation = "lighter";

    // bloom hugging the outline
    const bl = ctx.createRadialGradient(cx, cy, R * 0.9, cx, cy, R * 1.7);
    bl.addColorStop(0, rgba(core, 0.35 * P.glow + 0.1));
    bl.addColorStop(0.4, rgba(cool, 0.05 * P.glow));
    bl.addColorStop(1, rgba(core, 0));
    ctx.fillStyle = bl;
    ctx.fillRect(0, 0, w, h);

    // aberration: the prism as rings, each nudged along a direction that
    // orbits the body, so the colour split travels round him
    const th = this.phase * TAU * 0.8;
    const ab = P.ab * PROPS.aberration * R * 0.03;
    ctx.filter = `blur(${(R * 0.035).toFixed(1)}px)`;
    ctx.lineWidth = R * 0.09;
    for (const f of SPECTRUM) {
      ctx.strokeStyle = rgba(f.c, f.a);
      ctx.beginPath();
      ctx.arc(cx + Math.cos(th) * f.o * ab, cy + Math.sin(th) * f.o * ab, R, 0, TAU);
      ctx.stroke();
    }
    ctx.filter = "none";

    // wisps: the ribbon's filaments bent into arcs, alternate ones counter-
    // rotating, radius wobbling with the mode's amplitude, faded at both ends
    const N = 36;
    for (let j = 0; j < PROPS.strands; j++) {
      const dir = j % 2 ? -1 : 1;
      const a0 = dir * this.phase * TAU * (0.55 + j * 0.12) + j * 1.9;
      const len = Math.PI * (0.55 + 0.25 * Math.sin(this.phase * 3 + j));
      const c = TINTS[j % TINTS.length];
      ctx.lineWidth = Math.max(0.7, R * 0.035 * (0.6 + 0.4 * ((j + 1) % 2)));
      let px = 0, py = 0;
      for (let s = 0; s <= N; s++) {
        const u = s / N, a = a0 + u * len;
        const r = R * (1.12 + j * 0.05 + (0.1 + P.amp * 0.6) * Math.sin(a * 3 + this.phase * TAU * 1.3 + j * 0.7));
        const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
        if (s) {
          ctx.strokeStyle = rgba(c, 0.75 * Math.sin(Math.PI * u));
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
        px = x;
        py = y;
      }
    }
    ctx.globalCompositeOperation = "source-over";
  }
}

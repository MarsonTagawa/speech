// Frost-sparkle hover trail, lifted from "Ice Shard Hover.dc.html" (a DCLogic
// component) and stripped to a plain canvas driver. Moving the pointer over
// `host` spawns wobbly rings of ice-shard pixels that ripple out and fade.

const CELL = 14; // px per sparkle cell
const DECAY = 0.9; // per-frame persistence of lit cells
const SPEED = 420; // ring growth, px/s (time-based so it holds when many rings drop the frame rate)
const STEP = 18; // px of pointer travel between ring spawns

type Ring = { x: number; y: number; r: number; g: number; w: number[] };

export class Shards {
  private ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private cols = 0;
  private rows = 0;
  private f = new Float32Array(0);
  private hash = new Float32Array(0);
  private rings: Ring[] = [];
  private groups: Record<number, { t: number }> = {};
  private gid = 0;
  private gdir: { x: number; y: number } | null = null;
  private lastRing: { x: number; y: number } | null = null;
  private lastMove = 0;
  private lastTick = performance.now();

  constructor(private canvas: HTMLCanvasElement, host: HTMLElement) {
    this.ctx = canvas.getContext("2d")!;
    new ResizeObserver(() => this.setup()).observe(canvas);
    host.addEventListener("mousemove", (e) => this.move(e));
    host.addEventListener("mouseleave", () => { this.lastRing = null; this.gdir = null; });
    const loop = () => { this.tick(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  private setup() {
    const r = this.canvas.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = r.width * this.dpr;
    this.canvas.height = r.height * this.dpr;
    this.cols = Math.ceil(r.width / CELL);
    this.rows = Math.ceil(r.height / CELL);
    const n = this.cols * this.rows;
    this.f = new Float32Array(n);
    this.hash = new Float32Array(n);
    for (let i = 0; i < n; i++) { const s = Math.sin(i * 12.9898) * 43758.5453; this.hash[i] = s - Math.floor(s); }
  }

  private spawn(x: number, y: number) {
    if (this.rings.length > 160) return;
    if (this.rings.some((r) => r.g === this.gid && Math.hypot(x - r.x, y - r.y) < r.r - 2)) return;
    const R = Math.random, P = () => R() * 6.283;
    this.rings.push({ x, y, r: 0, g: this.gid, w: [.12 + R() * .1, P(), .08 + R() * .08, P(), .05 + R() * .05, P(), .03 + R() * .03, P()] });
  }

  private newGroup() {
    this.gid++;
    const live = new Set(this.rings.map((r) => r.g));
    for (const id in this.groups) if (!live.has(+id)) delete this.groups[id];
    this.groups[this.gid] = { t: performance.now() };
  }

  private move(e: MouseEvent) {
    const b = this.canvas.getBoundingClientRect();
    const zx = e.clientX - b.left, zy = e.clientY - b.top;
    const now = performance.now(), lr = this.lastRing;
    if (!lr || now - this.lastMove > 250) {
      this.newGroup(); this.gdir = null;
      this.lastRing = { x: zx, y: zy };
      this.spawn(zx, zy);
    } else {
      const dx = zx - lr.x, dy = zy - lr.y, len = Math.hypot(dx, dy);
      if (len >= STEP) {
        const ux = dx / len, uy = dy / len;
        // sharp reversal starts a fresh group so it isn't swallowed by its own rings
        if (this.gdir && ux * this.gdir.x + uy * this.gdir.y < -.3) this.newGroup();
        this.gdir = { x: ux, y: uy };
        const n = Math.floor(len / STEP);
        for (let i = 1; i <= n; i++) {
          const px = lr.x + ux * STEP * i, py = lr.y + uy * STEP * i;
          const gs = this.groups[this.gid];
          if (!gs || now - gs.t > 300) {
            const swallowed = this.rings.some((r) => r.g === this.gid && r.r > 40 && Math.hypot(px - r.x, py - r.y) < r.r * .85);
            if (swallowed) { this.newGroup(); this.groups[this.gid].t = now; }
          }
          this.spawn(px, py);
        }
        this.lastRing = { x: lr.x + ux * STEP * n, y: lr.y + uy * STEP * n };
      }
    }
    this.lastMove = now;
  }

  private tick() {
    const { cols, rows, hash, f, ctx } = this;
    if (!f.length) return;
    const cs = CELL, maxR = Math.hypot(cols * cs, rows * cs), band = cs * 2.2;
    const alive = new Set<number>();
    for (const r of this.rings) if (r.r < maxR + 40) alive.add(r.g);
    this.rings = this.rings.filter((r) => alive.has(r.g));
    const now = performance.now(), dt = Math.min(0.05, (now - this.lastTick) / 1000);
    this.lastTick = now;
    for (const rg of this.rings) rg.r += SPEED * dt;

    for (let k = 0; k < f.length; k++) {
      let h = f[k] * (f[k] > .9 ? DECAY * .9 : DECAY);
      if (this.rings.length) {
        const x = (k % cols) * cs + cs / 2, y = ((k / cols) | 0) * cs + cs / 2;
        const G: Record<number, { sd: number; amp: number }> = {};
        for (const rg of this.rings) {
          const dx = x - rg.x, dy = y - rg.y, dist = Math.hypot(dx, dy);
          if (dist > rg.r * 1.55 + cs * 2) continue;
          if (dist < rg.r * .45 - band * 2 - cs * 2) { G[rg.g] = { sd: -1e9, amp: 0 }; continue; }
          const th = Math.atan2(dy, dx), w = rg.w;
          const wob = 1 + w[0] * Math.sin(2 * th + w[1]) + w[2] * Math.sin(3 * th + w[3]) + w[4] * Math.sin(5 * th + w[5]) + w[6] * Math.sin(9 * th + w[7]);
          const d = dist - rg.r * wob, g = G[rg.g];
          if (!g || d < g.sd) G[rg.g] = { sd: d, amp: 1 - rg.r / maxR * .5 };
        }
        for (const id in G) {
          const sd = G[id].sd + (hash[k] - .5) * cs * .6, fw = cs * 1.6;
          if (sd < fw && sd > -band * 2) {
            const t = sd >= 0 ? 1 - Math.pow(sd / fw, 2) : Math.pow(1 + sd / (band * 2), 3);
            const a = G[id].amp * 1.6 * t; if (a > h) h = a;
          }
        }
      }
      f[k] = h;
    }

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = "#ffffff30";
    for (let k = 0; k < f.length; k++) {
      const h = f[k]; if (h < .002) continue;
      const v = Math.sqrt(h); if (v < .9 && hash[k] > v * 2.2) continue;
      const x = (k % cols) * cs + cs / 2, y = ((k / cols) | 0) * cs + cs / 2;
      ctx.globalAlpha = Math.min(1, v * 1.3);
      if (h > .7 && hash[k] < .35) { ctx.fillRect(x - 2.5, y - .5, 5, 1); ctx.fillRect(x - .5, y - 2.5, 1, 5); }
      else { const s = 1 + Math.min(1, v) * 1.4; ctx.fillRect(x - s / 2, y - s / 2, s, s); }
    }
    ctx.globalAlpha = 1;
  }
}

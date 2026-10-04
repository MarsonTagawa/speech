// Flappy-bird on the "Correcting…" overlay, to fill the wait (Space swaps it in
// for the coach, who becomes the bird). Space or a click flaps through the
// gaps; a crash pauses briefly, then another press restarts.
// The overlay can end it at any moment, so there's nothing to lose.
// Board from the "Flappy Coach" design (2a): the coach faces right, flaps his
// arms on every press and keeps his halo; glass pipes, and (unless plain) a
// scope grid, prism streaks and motes on a black board behind him.
import { expressionFromDefinition, renderAvatarExpression, type AvatarDefinition } from "@bible-strong/avatar-core";
import { Halo, still } from "./halo";
import { PROPS, hexToRgb, hueShift, rgba, type RibbonMode } from "./ribbon";
import strobi from "./strobi.avatar.json";

type Pipe = { x: number; gapY: number; scored: boolean };
export type FlappyState = { mode: "ready" | "play" | "dead"; y: number; vy: number; pipes: Pipe[]; score: number; best: number; t: number };

const R = 8; // physics hitbox radius, px; the board tests his drawn body too (Flappy.frame)
const BIRD = 30; // px box the coach is drawn in
const PIPE_W = 20;
const PIPE_R = 5; // corner radius
const birdX = (w: number) => w * 0.25;
// opening height: 30% of the board, but never under ~2 coach heights
export const gap = (h: number) => Math.max(h * 0.3, BIRD * 1.9);
const G = 2.4; // gravity, board heights per s²
const FLAP = 0.75; // flap speed, board heights per s
const SPEED = 0.32; // scroll, board widths per s
const SPACING = 0.5; // between pipes, board widths
const RETRY = 0.5; // s after a crash before a click restarts (so a mid-flap click doesn't skip the crash)

export const newState = (h: number, best = 0): FlappyState => ({ mode: "ready", y: h / 2, vy: 0, pipes: [], score: 0, best, t: 0 });

// false when ignored (too soon after a crash).
export function flap(s: FlappyState, h: number) {
  if (s.mode === "dead") {
    if (s.t < RETRY) return false;
    Object.assign(s, newState(h, s.best));
  }
  s.mode = "play";
  s.vy = -FLAP * h;
  return true;
}

// Advance dt seconds.
export function step(s: FlappyState, dt: number, w: number, h: number) {
  s.t += dt;
  if (s.mode === "ready") return;
  s.vy += G * h * dt;
  s.y = Math.min(h - R, s.y + s.vy * dt);
  if (s.mode === "dead") return; // just falls to the floor

  for (const p of s.pipes) p.x -= SPEED * w * dt;
  s.pipes = s.pipes.filter((p) => p.x > -PIPE_W - 8); // -8: the glass rim overhangs
  const last = s.pipes[s.pipes.length - 1];
  if (!last || last.x < w - SPACING * w) {
    // Opening centre, anywhere it fits except within 30% of the range of the last
    // pipe's, so short boards (the loading screen) still vary.
    const g = gap(h), lo = g / 2 + 8, span = Math.max(0, h - g - 16);
    let u = Math.random() * span;
    if (last) {
      const r = Math.min(span, Math.max(0, last.gapY - lo)), d = span * 0.3;
      const below = Math.max(0, r - d), above = Math.max(0, span - r - d);
      u = Math.random() * (below + above);
      if (u >= below) u += 2 * d; // skip the band round the last opening
    }
    s.pipes.push({ x: w, gapY: lo + u, scored: false });
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

const TAU = Math.PI * 2;
const deep = hexToRgb(PROPS.blueColor), cool = hexToRgb(PROPS.greenColor), warm = hexToRgb(PROPS.redColor), core = hexToRgb(PROPS.coreColor);
const TINTS = [deep, cool, core, hueShift(warm, 34, 1.4), warm];
const GREEN_RGB = hexToRgb("#30d158");
const RED_RGB = hexToRgb("#ff453a");
const HALO_MODE: Record<FlappyState["mode"], RibbonMode> = { ready: "idle", play: "speaking", dead: "thinking" };
const EYE_SHAPE = ["widthLeft", "widthRight", "heightLeft", "heightRight", "spacing", "positionXLeft", "positionXRight", "positionYLeft", "positionYRight", "leftAngle", "rightAngle"] as const;
const SVGNS = "http://www.w3.org/2000/svg";

const def0 = strobi as unknown as AvatarDefinition;
const NEUTRAL = expressionFromDefinition("neutral", def0.expressions.neutral);
const JOYFUL = expressionFromDefinition("joyful-wide", def0.expressions["joyful-wide"]);
const SCARED = expressionFromDefinition("surprised-wide-left", def0.expressions["surprised-wide-left"]);
type Vec = [number, number, number];
const baseNodes = def0.body.nodes as unknown as { position: Vec; rotation: Vec }[];
// rotate about z, degrees
function rot(p: Vec, deg: number): Vec {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]];
}

function rrect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, style: Partial<CSSStyleDeclaration>, parent: HTMLElement) {
  const e = document.createElement(tag);
  Object.assign(e.style, style);
  parent.append(e);
  return e;
}

type Streak = { x: number; y: number; thick: boolean; wd: number; len: number; v: number; a: number; c: number[] };
// ribbon-coloured light lines rushing past; ~1 in 4 a thick ribbon (wider, slower, fainter: further back)
function newStreak(x = 1 + Math.random() * 0.6): Streak {
  const thick = Math.random() < 0.25;
  return {
    x,
    y: 0.06 + Math.random() * 0.88,
    thick,
    wd: thick ? 5 + Math.random() * 7 : 1.4,
    len: (thick ? 0.25 : 0.08) + Math.random() * (thick ? 0.3 : 0.22),
    v: (thick ? 0.35 : 0.55) + Math.random() * (thick ? 0.25 : 0.6),
    a: thick ? 0.1 + Math.random() * 0.1 : 0.12 + Math.random() * 0.22,
    c: TINTS[(Math.random() * TINTS.length) | 0],
  };
}

export class Flappy {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private bird: HTMLDivElement;
  private fx: HTMLDivElement;
  private halo: Halo;
  private back: SVGPathElement[] = [];
  private front: SVGPathElement[] = [];
  private head: SVGPathElement;
  private eyeL: SVGPathElement;
  private eyeR: SVGPathElement;
  // a copy of the coach whose arm nodes are re-posed each frame
  private def = { ...def0, body: { ...def0.body, nodes: baseNodes.map((n) => ({ ...n, position: [...n.position], rotation: [...n.rotation] })) } };
  private s: FlappyState | null = null;
  best = 0; // seeds each new game; raised as games beat it
  private mode: FlappyState["mode"] | "" = "";
  private raf = 0;
  private go = false; // flap as soon as the board is sized (skips "ready")
  private last = 0;
  private scroll = 0; // grid drift, px
  private shake = 0; // 1 → 0 after a crash
  private offX = 0; // knock-back out of a pipe face, px
  private side = false; // crashed into a pipe's face: bounce back out of it
  private pipeHit = false; // crashed into a pipe (vs the floor/ceiling)
  private fallY: number | null = null; // hit a pipe's top/bottom: falls through it and out the ground
  private joy = 0;
  private joyUntil = 0;
  private scare = 0;
  private blinkAt = 1.5;
  private wing = 0; // arm angle, degrees (+ = raised)
  private wingV = 0;
  private poseKey = "";
  private streaks = Array.from({ length: 16 }, () => newStreak(Math.random() * 1.6));
  // background motes: three depths, drifting left with the board (slower = further)
  private motes = Array.from({ length: 70 }, () => ({ x: Math.random(), y: Math.random(), z: 0.15 + Math.random() * 0.85, ph: Math.random() * TAU }));

  private fg = [242, 242, 247]; // pipes and text: light on the black board, the theme's ink when plain

  // onBest: a game just beat `best` (to persist it). plain: no backdrop, pipes
  // and text in the root's CSS `color`, to blend into whatever is behind.
  constructor(
    private root: HTMLElement,
    private onBest?: (best: number) => void,
    private plain = false,
  ) {
    this.canvas = el("canvas", { position: "relative", display: "block", width: "100%", height: "100%" }, root);
    this.ctx = this.canvas.getContext("2d")!;
    this.bird = el("div", { position: "absolute", top: "0", left: "0", width: BIRD + "px", height: BIRD + "px", pointerEvents: "none" }, root);
    this.fx = el("div", { position: "absolute", inset: "0" }, this.bird);
    const haloCv = el("canvas", { position: "absolute", inset: "-75%", width: "250%", height: "250%", pointerEvents: "none" }, this.fx);
    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("viewBox", "-150 -150 300 300");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Coach as the flappy bird");
    Object.assign(svg.style, { position: "relative", display: "block", width: "100%", height: "100%", overflow: "visible" });
    this.fx.append(svg);
    const colors = (strobi as { colors: { body: string; eyes: string } }).colors;
    const mk = (fill: string) => {
      const p = document.createElementNS(SVGNS, "path");
      p.setAttribute("fill", fill);
      return p;
    };
    for (let i = 0; i < baseNodes.length; i++) this.back.push(mk(colors.body));
    this.head = mk(colors.body);
    for (let i = 0; i < baseNodes.length; i++) this.front.push(mk(colors.body));
    svg.append(...this.back, this.head, ...this.front);
    const eyes = document.createElementNS(SVGNS, "g");
    this.eyeL = mk(colors.eyes);
    this.eyeR = mk(colors.eyes);
    eyes.append(this.eyeL, this.eyeR);
    svg.append(eyes);
    // halo hugs the round body only: one that followed the arms trailed the fast flaps as ghost arms
    this.halo = new Halo(haloCv, false);
    this.halo.gain = 1.3;
    this.halo.maxDpr = 1;
    this.halo.paused = true;
    this.canvas.addEventListener("pointerdown", (e) => {
      e.preventDefault(); // keep focus where it was
      this.flap();
    });
  }

  // Ignored while frozen.
  flap() {
    const s = this.s;
    if (!this.raf || !s) return;
    const was = s.mode;
    if (!flap(s, this.canvas.clientHeight)) return;
    if (was === "dead") (this.offX = 0), (this.side = false), (this.fallY = null);
    this.wingV += 1100; // kick the arms up
    this.halo.pulse(was === "play" ? 2 : 10, 0.3);
  }

  start(go = false) {
    this.stop();
    this.go = go;
    this.s = null; // fresh game, sized on the first visible frame
    this.mode = "";
    this.offX = 0;
    this.shake = 0;
    this.halo.paused = false;
    if (this.plain) this.fg = (getComputedStyle(this.root).color.match(/[\d.]+/g) ?? []).slice(0, 3).map(Number); // theme may have changed
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
    this.halo.paused = true;
    this.canvas.style.transform = "";
  }

  private frame(dt: number) {
    const cv = this.canvas, ctx = this.ctx;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (w < 60 || h < 40) return; // hidden or squashed: skip
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    if (!this.s) {
      this.s = newState(h, this.best);
      if (this.go) this.flap();
    }
    const s = this.s;
    const prev = s.score, prevY = s.y;
    step(s, dt, w, h);

    // hit test against his drawn body (the physics radius is smaller), so he never sinks into a pipe
    const HR = BIRD * 0.38, HX = BIRD * 0.42, hx = birdX(w), g = gap(h);
    if (s.mode === "play") {
      const hit = s.pipes.find((p) => hx + HX > p.x && hx - HX < p.x + PIPE_W && Math.abs(s.y - p.gapY) > g / 2 - HR);
      if (hit || s.y - HR < 0 || s.y + HR >= h) {
        s.mode = "dead";
        s.t = 0;
        this.side = !!hit && hx < hit.x; // centre still left of the pipe: hit its face
        this.pipeHit = !!hit; // floor/ceiling crashes rest on the ground instead of falling through
      }
    }
    if (s.mode === "dead" && this.side) {
      // bounce back out of the pipe face he hit
      let want = 0;
      for (const p of s.pipes) {
        const inGap = Math.abs(s.y - p.gapY) <= g / 2 - HR;
        if (!inGap && hx + HX > p.x && hx - HX < p.x + PIPE_W) want = Math.min(want, p.x - (hx + HX) - 4);
      }
      this.offX += (want - this.offX) * Math.min(1, dt * 18);
    }
    this.fallY = s.mode === "dead" && this.pipeHit && !this.side ? (this.fallY ?? prevY) + s.vy * dt : null;
    if (s.best > this.best) this.onBest?.((this.best = s.best));
    if (s.mode !== this.mode) {
      this.mode = s.mode;
      this.halo.setMode(HALO_MODE[s.mode]);
      if (s.mode === "dead") {
        if (!still) this.shake = 1;
        this.halo.pulse(26, 1, RED_RGB);
      }
    }
    if (s.score > prev) {
      this.halo.pulse(9, 0.55, GREEN_RGB);
      this.joyUntil = performance.now() + 500;
      this.boop();
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.plain) this.drawBackdrop(s, w, h, dt);
    this.drawPipes(s, h, g);

    ctx.fillStyle = rgba(this.fg, 0.55);
    ctx.font = '500 13px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = "center";
    ctx.fillText(String(s.score), w / 2, 18);
    if (s.mode !== "play") {
      ctx.globalAlpha = 0.8;
      const msg = s.mode === "ready" ? "space to flap" : s.t < RETRY ? "" : `space to retry${s.best ? ` · best ${s.best}` : ""}`;
      ctx.fillText(msg, w / 2, h / 2 + g / 2 - 10);
      ctx.globalAlpha = 1;
    }

    const y = s.mode === "ready" ? s.y + Math.sin(s.t * 4) * 4 : s.y;
    this.shake = Math.max(0, this.shake - dt / 0.4);
    const k = this.shake * this.shake;
    const kx = k ? (Math.random() * 2 - 1) * 7 * k : 0, ky = k ? (Math.random() * 2 - 1) * 5 * k : 0;
    cv.style.transform = k ? `translate(${kx}px, ${ky}px)` : "";
    const by = this.fallY != null ? Math.min(this.fallY, h + BIRD) : Math.min(y, h - HR);
    this.bird.style.transform = `translate(${hx + this.offX - BIRD / 2 + kx}px, ${by - BIRD / 2 + ky}px)`;
    this.drawCoach(s, dt);
  }

  // The design's dark-board backdrop: scope grid, prism streaks and motes.
  private drawBackdrop(s: FlappyState, w: number, h: number, dt: number) {
    const ctx = this.ctx;
    this.scroll += (s.mode === "play" ? 1 : 0.25) * SPEED * w * dt;
    // the scope grid's hairlines (88 × 56px), drifting with the board
    ctx.fillStyle = "rgba(255,255,255,0.055)";
    for (let x = -(this.scroll % 88); x < w; x += 88) ctx.fillRect(Math.round(x), 0, 1, h);
    for (let y = 0; y < h; y += 56) ctx.fillRect(0, y, w, 1);

    // streaks: prism fringes above/below a core line, fading toward the tail
    const pace = s.mode === "play" ? 1 : 0.3;
    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round";
    for (const st of this.streaks) {
      st.x -= st.v * pace * dt;
      if (st.x + st.len < -0.05) Object.assign(st, newStreak());
      const x0 = st.x * w, x1 = (st.x + st.len) * w, y = st.y * h;
      const sp = st.thick ? st.wd * 0.55 : 1.6;
      for (const f of [{ c: deep, o: -sp, a: 0.5 }, { c: st.c, o: 0, a: 1 }, { c: warm, o: sp, a: 0.45 }]) {
        const grad = ctx.createLinearGradient(x0, 0, x1, 0);
        grad.addColorStop(0, rgba(f.c, st.a * f.a * 0.5));
        grad.addColorStop(1, rgba(f.c, 0));
        ctx.strokeStyle = grad;
        ctx.lineWidth = st.thick ? (f.o ? st.wd * 0.35 : st.wd) : f.o ? 0.8 : 1.4;
        ctx.beginPath();
        ctx.moveTo(x0, y + f.o);
        ctx.lineTo(x1, y + f.o);
        ctx.stroke();
      }
    }
    ctx.globalCompositeOperation = "source-over";
    const drift = SPEED * pace * dt;
    for (const m of this.motes) {
      m.x -= drift * m.z * 0.6;
      if (m.x < -0.02) (m.x = 1.02), (m.y = Math.random());
      m.ph += dt * (0.6 + m.z);
      const r = 0.5 + m.z * 1.1, a = (0.08 + 0.22 * m.z) * (0.7 + 0.3 * Math.sin(m.ph));
      ctx.fillStyle = `rgba(242,242,247,${a})`;
      ctx.beginPath();
      ctx.arc(m.x * w, m.y * h + Math.sin(m.ph * 0.7) * 3 * m.z, r, 0, TAU);
      ctx.fill();
    }
  }

  // Glass pipes: a dark rounded pane, a faint sheen, a prism split at the
  // edges and a rim lit from the top-left.
  private drawPipes(s: FlappyState, h: number, g: number) {
    const ctx = this.ctx;
    for (const p of s.pipes) {
      const top = p.gapY - g / 2, bot = p.gapY + g / 2;
      for (const [y, ph] of [[-8, top + 8], [bot, h - bot + 8]]) {
        if (ph <= 8) continue;
        const x = p.x;
        rrect(ctx, x, y, PIPE_W, ph, PIPE_R);
        const tint = ctx.createLinearGradient(x, 0, x + PIPE_W, 0);
        tint.addColorStop(0, rgba(this.fg, 0.12));
        tint.addColorStop(0.5, rgba(this.fg, 0.03));
        tint.addColorStop(1, rgba(this.fg, 0.08));
        ctx.fillStyle = tint;
        ctx.fill();
        ctx.fillStyle = rgba(cool, 0.14);
        ctx.fillRect(x + 1.5, y + PIPE_R, 1, ph - 2 * PIPE_R);
        ctx.fillStyle = rgba(warm, 0.14);
        ctx.fillRect(x + PIPE_W - 2.5, y + PIPE_R, 1, ph - 2 * PIPE_R);
        const rim = ctx.createLinearGradient(x, y, x + PIPE_W, y + Math.min(ph, 120));
        rim.addColorStop(0, rgba(this.fg, 0.55));
        rim.addColorStop(0.5, rgba(this.fg, 0.12));
        rim.addColorStop(1, rgba(this.fg, 0.3));
        rrect(ctx, x + 0.5, y + 0.5, PIPE_W - 1, ph - 1, PIPE_R - 0.5);
        ctx.strokeStyle = rim;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  // A swell on scoring.
  private boop() {
    if (still) return;
    const grow = (k: number, offset: number, easing = "ease-in-out") => ({ transform: `scale(${k})`, offset, easing });
    this.fx.animate([grow(1, 0, "cubic-bezier(.2, 1.4, .5, 1)"), grow(1.22, 0.28), grow(0.95, 0.55), grow(1.03, 0.78), grow(1, 1)], { duration: 600 });
  }

  // The coach, turned to face right: arms on a damped spring (each flap kicks
  // them up, they swing past rest and settle), joyful eyes on a point, scared
  // ones on a crash, and a blink now and then.
  private drawCoach(s: FlappyState, dt: number) {
    const now = performance.now();
    this.joy += ((now < this.joyUntil ? 1 : 0) - this.joy) * Math.min(1, dt * 14);
    this.scare += ((s.mode === "dead" ? 1 : 0) - this.scare) * Math.min(1, dt * 10);
    // rest: a slow hover while waiting, limp after a crash, a flutter while falling
    const rest = s.mode === "ready" ? 24 + 30 * Math.sin(s.t * 5) : s.mode === "dead" ? -38 : 4 + 10 * Math.sin(s.t * 8);
    const n = 4, h = dt / n;
    for (let i = 0; i < n; i++) {
      this.wingV += (170 * (rest - this.wing) - 9 * this.wingV) * h;
      this.wing += this.wingV * h;
    }
    this.wing = Math.max(-45, Math.min(95, this.wing));
    const ang = still ? 10 : this.wing;
    this.def.body.nodes.forEach((node, i) => {
      const b = baseNodes[i], side = b.position[0] < 0 ? 1 : -1; // left arm swings +z, right -z to raise
      const far = side < 0; // the right arm is on the far side as he turns right: tucked behind, only its tip peeks out
      const p0: Vec = [b.position[0] * (far ? 1.14 : 1.04), b.position[1] - (far ? 10 : 0), b.position[2] + (far ? 35 : 0)];
      node.position = rot(p0, side * ang * 0.6);
      node.rotation = [b.rotation[0], b.rotation[1], b.rotation[2] + side * ang];
    });
    const e = { ...NEUTRAL, headY: 32 - 6 * this.scare, headX: -6 + (s.mode === "play" ? Math.max(-10, Math.min(10, -s.vy / 40)) : 0), headZ: 0 };
    for (const k of EYE_SHAPE) e[k] += (JOYFUL[k] - NEUTRAL[k]) * this.joy + (SCARED[k] - NEUTRAL[k]) * this.scare * (1 - this.joy);
    this.blinkAt -= dt;
    if (this.blinkAt < 0.12 && this.blinkAt > 0) {
      const k = 1 - Math.sin((this.blinkAt / 0.12) * Math.PI) * 0.9;
      e.heightLeft *= k;
      e.heightRight *= k;
    }
    if (this.blinkAt <= 0) this.blinkAt = 2 + Math.random() * 3;
    const key = [ang.toFixed(1), this.joy.toFixed(2), this.scare.toFixed(2), e.heightLeft.toFixed(1), e.headX.toFixed(1)].join();
    if (key === this.poseKey) return;
    this.poseKey = key;
    const geo = renderAvatarExpression(this.def as AvatarDefinition, e).geometry;
    this.back.forEach((p, i) => p.setAttribute("d", geo.backPaths[i] ?? ""));
    this.head.setAttribute("d", geo.headPath);
    this.front.forEach((p, i) => p.setAttribute("d", geo.frontPaths[i] ?? ""));
    this.eyeL.setAttribute("d", geo.leftPath);
    this.eyeL.setAttribute("opacity", geo.leftVisible ? "1" : "0");
    this.eyeR.setAttribute("d", geo.rightPath);
    this.eyeR.setAttribute("opacity", geo.rightVisible ? "1" : "0");
  }
}

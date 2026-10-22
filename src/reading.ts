// Reading off the slides: how much of what was said while a slide showed is
// that slide's own text, word for word, per slide and overall. Pure; checked
// by reading.check.ts.
//
// ponytail: verbatim 3-word runs only, so a close paraphrase ("grew forty
// percent" vs "grew 40%") isn't caught, and image slides (no text) never count.
// Thresholds are guesses; tune against real decks.

import { isHesitation } from "./crutch";
import { scoreLinear } from "./presence";

export const READ_FRAC = 0.4; // a slide with at least this share of its runs on it was read
const MIN_RUNS = 10; // fewer runs said over text slides isn't enough to judge
const MIN_SLIDE_RUNS = 3; // nor on one slide

export interface Said {
  index: number; // utterance (transcript line)
  slide: number; // 0-based slide shown when it started
  ms: number; // its start, session ms
  dur: number; // its length, ms
  said: string[]; // normalized spoken tokens
  shown: string[]; // that slide's normalized text tokens
}

export interface SlideRead {
  slide: number; // 1-based
  frac: number; // share of its spoken runs found on it; 0 when too few to judge
  runs: number; // spoken runs counted on it
  index: number; // first utterance on it
  ms: number; // that utterance's start
  spokeMs: number; // summed utterance lengths on it
  shownMs?: number; // time it was up (slideSpans); absent on older sessions
  score?: number | null; // delivery on it (slideScore); null when too little was said
}

export interface Reading {
  frac: number | null; // share of runs said over text slides that are on the slide; null: too few to judge
  slides: SlideRead[]; // every slide spoken on, in slide order
  screen?: true; // slides are recorded screens (OCR), not the in-app deck
  spans?: SlideSpan[]; // when each slide was up, for the pace chart
}

export interface SlideSpan {
  slide: number; // 0-based
  start: number; // session ms
  end: number;
}

// Slide changes ([session ms, 0-based slide], in time order; -1 = no slide up)
// as the spans each slide was up, until `endMs`.
export function slideSpans(changes: [number, number][], endMs: number): SlideSpan[] {
  const out: SlideSpan[] = [];
  let open: SlideSpan | undefined;
  for (const [ms, slide] of changes) {
    if (open?.slide === slide) continue;
    if (open) open.end = ms;
    open = slide >= 0 ? { slide, start: ms, end: endMs } : undefined;
    if (open) out.push(open);
  }
  return out;
}

// ponytail: the session's weights (main.ts SCORE_WEIGHTS) plus reading at 1;
// keep in step if those change.
const SLIDE_WEIGHTS = { pace: 1, fillers: 1.5, pauses: 0.75, reading: 1 };

// A slide's delivery, 0–100, scored like the session's: `pace` already scored
// (main.ts scorePace), fillers and pauses per minute on it, and the share of it
// read word for word (null: not judged, left out).
export function slideScore(p: { pace: number; fillersPerMin: number; pausesPerMin: number; read: number | null }): number {
  const parts = [
    { s: p.pace, w: SLIDE_WEIGHTS.pace },
    { s: scoreLinear(p.fillersPerMin, 1, 12), w: SLIDE_WEIGHTS.fillers },
    { s: scoreLinear(p.pausesPerMin, 2, 14), w: SLIDE_WEIGHTS.pauses },
  ];
  if (p.read !== null) parts.push({ s: Math.round((1 - p.read) * 100), w: SLIDE_WEIGHTS.reading });
  return Math.round(parts.reduce((a, x) => a + x.s * x.w, 0) / parts.reduce((a, x) => a + x.w, 0));
}

export const isRead = (r: SlideRead) => r.runs >= MIN_SLIDE_RUNS && r.frac >= READ_FRAC;

// The rows of `r`; none for summaries saved before rows existed (slide numbers).
export const slideRows = (r: Reading) => r.slides.filter((x): x is SlideRead => typeof x === "object");

export function runsOf(ws: string[]): string[] {
  const t = ws.filter((w) => !isHesitation(w));
  return t.slice(2).map((w, i) => `${t[i]} ${t[i + 1]} ${w}`);
}

export function reading(utts: Said[]): Reading | null {
  const per = new Map<number, { hits: number; runs: number; index: number; ms: number; spokeMs: number }>();
  for (const u of utts) {
    const p = per.get(u.slide) ?? { hits: 0, runs: 0, index: u.index, ms: u.ms, spokeMs: 0 };
    per.set(u.slide, p);
    if (u.ms < p.ms) [p.index, p.ms] = [u.index, u.ms];
    p.spokeMs += u.dur;
    if (u.shown.length < 3) continue;
    const onSlide = new Set(runsOf(u.shown));
    const said = runsOf(u.said);
    p.hits += said.filter((r) => onSlide.has(r)).length;
    p.runs += said.length;
  }
  if (!per.size) return null;
  const all = [...per.values()];
  const total = all.reduce((a, p) => a + p.runs, 0);
  return {
    frac: total >= MIN_RUNS ? all.reduce((a, p) => a + p.hits, 0) / total : null,
    slides: [...per]
      .sort((a, b) => a[0] - b[0])
      .map(([n, p]) => ({
        slide: n + 1,
        frac: p.runs >= MIN_SLIDE_RUNS ? p.hits / p.runs : 0,
        runs: p.runs,
        index: p.index,
        ms: p.ms,
        spokeMs: p.spokeMs,
      })),
  };
}

// ponytail: a frame sharing this much of its runs with an earlier slide is that
// slide again (a clock ticking, a cursor); tune from real decks.
const SAME_SLIDE = 0.5;

// `text` minus every 3-word run already in `spoken`: the app's own transcript,
// when it shares the recorded display. A "¦" marks each cut so no run spans it.
export function withoutSpoken(text: string[], spoken: Set<string>): string[] {
  const cut = text.map(() => false);
  for (let i = 0; i + 2 < text.length; i++)
    if (spoken.has(`${text[i]} ${text[i + 1]} ${text[i + 2]}`)) cut[i] = cut[i + 1] = cut[i + 2] = true;
  const out: string[] = [];
  text.forEach((w, i) => {
    if (!cut[i]) out.push(w);
    else if (out[out.length - 1] !== "¦") out.push("¦");
  });
  return out;
}

// ponytail: a run on at least this share of the frames (3+ frames) is screen
// chrome (a browser's menus, a deck footer), not a slide's own text; tune from
// real sessions.
const CHROME = 0.8;

// Numbers recorded screens as slides by first appearance. Each frame has its
// text with what had been said cut out (`text`, see withoutSpoken) and as read
// (`raw`). Frames are told apart by their cut runs (or raw ones, when it was
// all said already: a slide read aloud, then come back to), less chrome unless
// that's all there is. A frame sharing ≥ SAME_SLIDE of those (of the smaller
// set) with an earlier slide is that slide again, scored against the slide's
// first frame text. Blank frames (no runs) get slide -1.
export function groupScreens(
  frames: { ms: number; text: string[]; raw: string[] }[],
): { ms: number; slide: number; text: string[] }[] {
  const runSet = (t: string[]) => new Set(runsOf(t).filter((x) => !x.includes("¦")));
  const raws = frames.map((f) => runSet(f.raw));
  const df = new Map<string, number>();
  for (const r of raws) for (const x of r) df.set(x, (df.get(x) ?? 0) + 1);
  const chrome = (x: string) => frames.length >= 3 && (df.get(x) ?? 0) >= CHROME * frames.length;
  const seen: { key: Set<string>; text: string[] }[] = [];
  return frames.map(({ ms, text }, i) => {
    const cut = runSet(text);
    const base = cut.size ? cut : raws[i];
    if (!base.size) return { ms, slide: -1, text };
    const own = new Set([...base].filter((x) => !chrome(x)));
    const key = own.size ? own : base;
    let [best, share] = [-1, 0];
    seen.forEach((s, n) => {
      let hits = 0;
      for (const x of key) if (s.key.has(x)) hits++;
      const sh = hits / Math.min(key.size, s.key.size);
      if (sh > share) [best, share] = [n, sh];
    });
    if (share >= SAME_SLIDE) return { ms, slide: best, text: seen[best].text };
    seen.push({ key, text });
    return { ms, slide: seen.length - 1, text };
  });
}

// The last of `frames` (in time order) at or before `ms`.
export function frameAt<T extends { ms: number }>(frames: T[], ms: number): T | undefined {
  let out: T | undefined;
  for (const f of frames) {
    if (f.ms > ms) break;
    out = f;
  }
  return out;
}

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/400-italic.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "@fontsource/ibm-plex-mono/700.css";
import { Ribbon } from "./ribbon";
import { Shards } from "./shards";
import { troubleSpots } from "./passages";

let ribbon: Ribbon | null = null;

interface TranscriptSegment {
  index: number;
  text: string;
  start_ms: number;
  end_ms: number;
  is_final: boolean;
  // True only for the deferred accurate-model (medium) correction of an already
  // committed line. Lets it overwrite the fast final and re-tally that line.
  refined: boolean;
  // Preview chunk an interim covers; see withPreviewPrefix. 0 for finals.
  chunk: number;
}

// A silent gap within an utterance; offsets are ms relative to the utterance start.
interface Pause {
  start_ms: number;
  end_ms: number;
}

// Per-utterance acoustic analysis (waveform + hesitation pauses), emitted once
// when an utterance commits on the `utterance_analysis` event and correlated to
// the transcript line by index. Pauses come from the VAD trace; the envelope is
// the amplitude for the waveform sparkline. See UtteranceAnalysis in audio.rs.
interface UtteranceAnalysis {
  index: number;
  duration_ms: number;
  envelope: number[];
  pauses: Pause[];
  pause_count: number;
  total_pause_ms: number;
  // Absolute loudness (0-1), comparable across utterances (unlike `envelope`,
  // which is self-normalized). Median voiced pitch in Hz (0 = unvoiced), the
  // within-utterance inflection range in semitones, and an uptalk flag. See
  // UtteranceAnalysis / pitch.rs in the backend.
  rms_level: number;
  f0_median: number;
  f0_range_semitones: number;
  f0_terminal_rising: boolean;
}

// Gaps between separate utterances this long or longer are counted as pauses
// too (matches the backend's MIN_SILENCE_MS that ends an utterance), so the
// pause metric covers both within- and between-utterance hesitation.
let INTER_PAUSE_MS = 600;

let recording = false;
let recordBtn: HTMLButtonElement | null;
let statusEl: HTMLElement | null;
let transcriptEl: HTMLElement | null;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T | null;
function setText(id: string, text: string) {
  const el = $(id);
  if (el) el.textContent = text;
}

// One <p> per utterance index, so interim decodes update a line in place and
// the final decode commits it. `finalized` guards against a slow interim
// result landing after the final and clobbering it.
const segmentEls = new Map<number, HTMLElement>();
const finalized = new Set<number>();
// Indices whose line has been replaced by the accurate-model correction. The
// correction can win the shared-GPU race and emit *before* the fast final (both
// decode the same utterance), so a fast `draft`/interim segment may arrive after
// the `corrected` one. Without this guard that late fast segment would clobber
// the correction — reverting the text and re-tallying stats back to the fast
// result. Once refined, later non-refined segments for the index are ignored.
const refinedIndices = new Set<number>();

// Running session metrics, accumulated from committed (final) utterances only.
// Tracked per utterance index (not just a "counted" flag) so the deferred
// accurate-model correction can re-tally its line by delta — replacing the fast
// final's word/filler counts with the corrected ones — instead of either
// double-counting or being ignored. `speakingMs` is the summed duration of
// spoken utterances (not wall-clock), so WPM reflects speaking pace and ignores
// the gaps between them; a correction reuses the same timing, so it's added
// once per index (tracked by `timedIndices`).
let totalWords = 0;
let totalFillers = 0;
let speakingMs = 0;
const wordsByIndex = new Map<number, number>();
const fillersByIndex = new Map<number, number>();
// The flagged filler words themselves (lowercased), for the report's breakdown.
const fillerWordsByIndex = new Map<number, string[]>();
const timedIndices = new Set<number>();

// Per-utterance acoustic analysis keyed by index, for the waveform under each
// line and the pause tally. `timingByIndex` keeps each utterance's start/end so
// the gaps *between* utterances can be counted as pauses too. Both are cleared
// per session (indices restart at 1).
const analysisByIndex = new Map<number, UtteranceAnalysis>();
const timingByIndex = new Map<number, { start: number; end: number }>();

// One session's per-line data: the live session's maps above, or a saved
// session's (History). The report helpers take one, defaulting to live.
interface Lines {
  timing: Map<number, { start: number; end: number }>;
  words: Map<number, number>;
  fillers: Map<number, number>;
  fillerWords: Map<number, string[]>;
  analysis: Map<number, UtteranceAnalysis>;
}
const live: Lines = { timing: timingByIndex, words: wordsByIndex, fillers: fillersByIndex, fillerWords: fillerWordsByIndex, analysis: analysisByIndex };

// Bumped on every new recording. Captured when a contextual review is scheduled
// so a review that resolves after the user has restarted is discarded instead
// of writing into the new session's transcript or stats.
let sessionId = 0;

// Last script-alignment result (null until a script is used), so the end-of-
// session report can score articulation. Updated every re-align in renderScriptMatch.
let lastScriptResult:
  | { total: number; hits: number; misses: number; subs: number; accuracy: number; missed: number[] }
  | null = null;
// Whether the read-along pane was open for this session. A script stays loaded
// after the pane is closed, so the pane — not the loaded text — decides whether
// the report scores against it. Frozen at Stop (jumpToLine closes the pane).
let scriptUsed = false;

// Set at Stop; the report opens once the correction pass finishes (corrections_done).
let reportPending = false;

// The report re-renders live as deferred corrections trickle in after stop, so
// this tracks whether it's on screen (see renderStats).
let reportVisible = false;

// Filler detection runs in two tiers:
//
//  Tier 1 — hesitation markers ("um", "uh", ...). These are never legitimate
//  content words, so a plain string match is safe. Flagged and counted
//  immediately as each line is rendered.
//
//  Tier 2 — discourse markers ("so", "like", "you know", ...). These are filler
//  only in context ("so, anyway...") and ordinary words otherwise ("so many
//  clouds"; "do you know him?"), so they can't be matched blindly. Each
//  committed line is first shown unflagged, then reviewed with a local
//  part-of-speech tagger (compromise, lazy-loaded, offline) that decides per
//  occurrence whether the usage is filler. Confirmed ones are flagged and
//  counted after the text is on screen.
// Unambiguous hesitation/backchannel sounds, flagged verbatim (no POS needed).
// Covers the spellings Whisper actually emits — "um/umm", "uh/uhh", "uhm", "er/
// erm", "eh/ehm", "ah", "hm/hmm", "mm", "mhm/mmhmm", "uh-huh" — since a narrow
// list silently drops mid-sentence fillers it spells differently.
const HESITATION_PATTERN =
  /\b(uh-?huh|u[hm]+|e+r+m*|e+h+m*|a+h+|h+m+|m+h+m+|m+m+)\b/gi;
const AMBIGUOUS_WORDS = new Set([
  "so",
  "like",
  "well",
  "actually",
  "basically",
  "literally",
  "and",
  "right",
  "okay",
  "ok",
]);
// Multi-word discourse markers. compromise splits these into separate terms, so
// they're matched as consecutive-term sequences (lowercased). Listed longest
// first isn't required here since none is a prefix of another, but each is tried
// before the single-word check so "you"/"know" aren't judged in isolation.
const AMBIGUOUS_PHRASES: string[][] = [
  ["you", "know"],
  ["i", "mean"],
  ["kind", "of"],
  ["sort", "of"],
];

interface Range {
  start: number;
  end: number;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Wraps the given character ranges in highlight spans, escaping everything.
// Ranges are sorted and overlaps dropped so the output stays well-formed.
function renderHighlighted(text: string, ranges: Range[]): string {
  if (ranges.length === 0) return escapeHtml(text);
  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  let html = "";
  let pos = 0;
  for (const r of sorted) {
    if (r.start < pos) continue;
    html += escapeHtml(text.slice(pos, r.start));
    html += `<span class="filler">${escapeHtml(text.slice(r.start, r.end))}</span>`;
    pos = r.end;
  }
  return html + escapeHtml(text.slice(pos));
}

function hesitationRanges(text: string): Range[] {
  const ranges: Range[] = [];
  for (const m of text.matchAll(HESITATION_PATTERN)) {
    const start = m.index ?? 0;
    ranges.push({ start, end: start + m[0].length });
  }
  return ranges;
}

function countWords(text: string): number {
  return text
    .trim()
    .split(/\s+/)
    .filter((w) => /[a-z0-9]/i.test(w)).length;
}

// Lazily-imported POS tagger. Kept out of the initial bundle since it's only
// needed for the deferred tier-2 review, never for rendering text.
let nlpPromise: Promise<typeof import("compromise").default> | null = null;
function loadNlp() {
  if (!nlpPromise) nlpPromise = import("compromise").then((m) => m.default);
  return nlpPromise;
}

function nonSpaceBefore(text: string, i: number): string {
  let j = i - 1;
  while (j >= 0 && /\s/.test(text[j])) j--;
  return j >= 0 ? text[j] : "";
}

function nonSpaceAfter(text: string, i: number): string {
  let j = i;
  while (j < text.length && /\s/.test(text[j])) j++;
  return j < text.length ? text[j] : "";
}

// Decides whether one occurrence of an ambiguous word is filler, given its POS
// tags and its punctuation neighbourhood. Two signals, both low-false-positive:
//  - compromise tags discourse/interjection uses as "Expression" (reliably
//    separates filler "so" from the "so many" adverb), and
//  - genuine filler inserts are set off by commas — a mid-clause ", like," or a
//    clause-initial "So, ..." — which the tagger alone doesn't always catch.
function isFillerUsage(
  tags: string[],
  text: string,
  start: number,
  end: number,
): boolean {
  if (tags.includes("Expression")) return true;
  const before = nonSpaceBefore(text, start);
  const after = nonSpaceAfter(text, end);
  const clauseInitial = before === "" || ".!?".includes(before);
  if (before === "," && after === ",") return true;
  if (clauseInitial && after === ",") return true;
  return false;
}

// POS-tags the text and returns the ranges of ambiguous words and phrases used
// as filler. Multi-word phrases are matched first so their component words
// aren't also judged individually.
async function ambiguousFillerRanges(text: string): Promise<Range[]> {
  const nlp = await loadNlp();
  const sentences = nlp(text).json({ offset: true }) as Array<{
    terms: Array<{ text: string; tags: string[]; offset?: { start: number; length: number } }>;
  }>;
  const ranges: Range[] = [];
  for (const sentence of sentences) {
    // Normalize to a flat list with resolved offsets so phrases can span terms.
    const terms = sentence.terms
      .filter((t) => t.offset)
      .map((t) => ({
        word: t.text.toLowerCase(),
        tags: t.tags ?? [],
        start: t.offset!.start,
        end: t.offset!.start + t.offset!.length,
      }));

    for (let i = 0; i < terms.length;) {
      const phrase = AMBIGUOUS_PHRASES.find(
        (p) =>
          i + p.length <= terms.length &&
          p.every((w, k) => terms[i + k].word === w),
      );
      if (phrase) {
        const span = terms.slice(i, i + phrase.length);
        const start = span[0].start;
        const end = span[span.length - 1].end;
        const tags = span.flatMap((t) => t.tags);
        if (isFillerUsage(tags, text, start, end)) ranges.push({ start, end });
        i += phrase.length;
        continue;
      }
      const t = terms[i];
      if (AMBIGUOUS_WORDS.has(t.word) && isFillerUsage(t.tags, text, t.start, t.end)) {
        ranges.push({ start: t.start, end: t.end });
      }
      i++;
    }
  }
  return ranges;
}

// Local (offline) filler detection: unambiguous hesitation markers plus the
// POS-reviewed discourse markers. The two sets are disjoint.
async function localFillerRanges(text: string): Promise<Range[]> {
  const ranges = hesitationRanges(text);
  try {
    return ranges.concat(await ambiguousFillerRanges(text));
  } catch {
    return ranges; // tagger failed to load; still return the hesitation markers
  }
}

// Tier-2 pass: runs after the line is already on screen. Detects fillers locally
// (POS-tagged discourse markers + hesitation sounds), then re-renders the line
// with them highlighted and adds them to the count.
async function reviewFillers(
  entry: HTMLElement,
  index: number,
  startMs: number,
  text: string,
  reviewSession: number,
) {
  const ranges = await localFillerRanges(text);

  // Discard if the session ended or the line was removed while we were working.
  if (reviewSession !== sessionId || !entry.isConnected) return;

  if (ranges.length > 0) {
    entry.innerHTML = segmentHtml(startMs, renderHighlighted(text, ranges));
    refreshWaveform(index); // innerHTML rewrite wiped it
  }
  // Adjust the running total by the delta for this line, so a re-review of a
  // corrected line replaces its earlier filler count rather than stacking on it.
  totalFillers += ranges.length - (fillersByIndex.get(index) ?? 0);
  fillersByIndex.set(index, ranges.length);
  fillerWordsByIndex.set(index, ranges.map((r) => text.slice(r.start, r.end).toLowerCase()));
  renderStats();
}

function resetStats() {
  totalWords = 0;
  totalFillers = 0;
  speakingMs = 0;
  wordsByIndex.clear();
  fillersByIndex.clear();
  fillerWordsByIndex.clear();
  timedIndices.clear();
  analysisByIndex.clear();
  timingByIndex.clear();
  sessionId++;
  renderStats();
}

// Total hesitation pauses across the session: the within-utterance gaps found
// by the VAD trace, plus the gaps *between* consecutive utterances that are long
// enough to count (INTER_PAUSE_MS). Recomputed from scratch each render so a
// re-tally is always consistent regardless of event ordering.
function countPauses(): number {
  let count = 0;
  for (const a of analysisByIndex.values()) count += a.pause_count;
  const idx = [...timingByIndex.keys()].sort((a, b) => a - b);
  for (let i = 1; i < idx.length; i++) {
    const gap = timingByIndex.get(idx[i])!.start - timingByIndex.get(idx[i - 1])!.end;
    if (gap >= INTER_PAUSE_MS) count++;
  }
  return count;
}

// Live metrics dock (2a). Recomputed from the summary on every change; cheap
// enough (a pass over this session's utterances) to run per segment.
function renderStats() {
  const s = computeSummary();
  const preset = PRESETS[currentPreset];

  // WPM track: the preset's target band sits in the middle 40% of the scale.
  const span = preset.wpmHigh - preset.wpmLow;
  const lo = preset.wpmLow - span * 0.75;
  const hi = preset.wpmHigh + span * 0.75;
  const at = (v: number) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
  setText("stat-wpm", String(s.wpm));
  const band = $("wpm-band");
  if (band) {
    band.style.left = `${at(preset.wpmLow)}%`;
    band.style.width = `${at(preset.wpmHigh) - at(preset.wpmLow)}%`;
  }
  const mark = $("wpm-mark");
  if (mark) {
    mark.style.left = `${at(s.wpm)}%`;
    const inBand = s.wpm >= preset.wpmLow && s.wpm <= preset.wpmHigh;
    mark.className = `wpm-mark${s.wpm === 0 ? "" : inBand ? " in" : " out"}`;
  }
  setText("wpm-sub", `target ${preset.wpmLow}–${preset.wpmHigh}`);

  setText("stat-fpm", s.fillersPerMin.toFixed(1));
  setWidth("fpm-bar", s.fillersPerMin / 12); // 12/min scores 0 (computeSummary)
  setText("fpm-sub", `${s.fillers} total · aim <3`);

  // Pauses: six time bins across the session, lit where a pause landed.
  setText("stat-pauses", String(s.pauses));
  const pips = $("pause-pips");
  if (pips) {
    if (pips.childElementCount === 0) pips.innerHTML = "<div></div>".repeat(6);
    const end = sessionEndMs();
    const lit = new Set(pauseTimesMs().map((ms) => Math.min(5, Math.floor((ms / Math.max(1, end)) * 6))));
    [...pips.children].forEach((d, i) => d.classList.toggle("on", lit.has(i)));
  }
  setText("pause-sub", `${s.pausesPerMin.toFixed(1)} / min`);

  setText("stat-pitch", s.pitchRange.toFixed(1));
  setWidth("pitch-bar", s.pitchRange / 5); // 5 st scores 100
  setText("pitch-sub", s.pitchRange === 0 ? "inflection" : s.pitchRange < 3 ? "inflection · flat" : "inflection · varied");

  // Words bar: share of the session spent actually speaking.
  setText("stat-words", String(s.words));
  setWidth("words-bar", speakingMs / Math.max(1, sessionEndMs()));
  setText("words-sub", `${formatTimestamp(speakingMs)} speaking`);

  const series = perSecondPace();
  drawSpark(series);
  setText("peak-label", `peak ${Math.max(0, ...series.map((p) => p.wpm))}`);
  setLiveRing(s.words > 0 ? s.scores.overall : null);

  // Keep the report in sync as deferred corrections/analyses trickle in after stop.
  if (reportVisible) renderReport();
}

function setWidth(id: string, frac: number) {
  const el = $(id);
  if (el) el.style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
}

// Latest end time across committed utterances: the session's length so far.
function sessionEndMs(L = live): number {
  let end = 0;
  for (const t of L.timing.values()) end = Math.max(end, t.end);
  return end;
}

// Session-relative start of every counted pause (the same set countPauses
// tallies), for the pip bins and the report's pause timeline.
function pauseTimesMs(L = live): number[] {
  const out: number[] = [];
  for (const [i, a] of L.analysis) {
    const t = L.timing.get(i);
    if (t) for (const p of a.pauses) out.push(t.start + p.start_ms);
  }
  const idx = [...L.timing.keys()].sort((a, b) => a - b);
  for (let k = 1; k < idx.length; k++) {
    const prev = L.timing.get(idx[k - 1])!;
    if (L.timing.get(idx[k])!.start - prev.end >= INTER_PAUSE_MS) out.push(prev.end);
  }
  return out.sort((a, b) => a - b);
}

// Shared WPM axis for both pace charts: 80 at the floor, 200 (or the next 40
// above the peak) at the top.
function wpmAxis(peak: number): [number, number] {
  return [80, Math.max(200, Math.ceil(peak / 40) * 40)];
}

// Contents of the dock's pace sparkline: target band, WPM line, a dot at "now"
// and an amber × at each filler onset. Drawn in CSS pixels (W×H = the element's
// size), not a stretched viewBox, so the dot and × keep their shape at any width.
function sparkSvg(series: SecondPace[], preset: Preset, W: number, H: number): string {
  if (series.length < 2) return "";
  const [lo, hi] = wpmAxis(Math.max(...series.map((p) => p.wpm)));
  const x = (sec: number) => (sec / (series.length - 1)) * W;
  const y = (v: number) => H - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * H;
  const pts = series.map((p, sec) => `${x(sec).toFixed(1)},${y(p.wpm).toFixed(1)}`).join(" ");
  const last = series.length - 1;
  const crosses = series
    .map((p, sec) => (p.fillerHere ? `M${(x(sec) - 2.5).toFixed(1)},${(y(p.wpm) - 2.5).toFixed(1)} l5,5 m0,-5 l-5,5` : ""))
    .join(" ");
  return (
    `<rect x="0" y="${y(preset.wpmHigh).toFixed(1)}" width="${W}" height="${(y(preset.wpmLow) - y(preset.wpmHigh)).toFixed(1)}" fill="rgba(48,209,88,0.12)" />` +
    `<polyline points="${pts}" fill="none" style="stroke:var(--blue)" stroke-width="1.5" vector-effect="non-scaling-stroke" />` +
    `<circle cx="${x(last).toFixed(1)}" cy="${y(series[last].wpm).toFixed(1)}" r="2.5" style="fill:var(--blue)" />` +
    (crosses.trim() ? `<path d="${crosses}" stroke="#ffd60a" stroke-width="1.5" stroke-linecap="round" fill="none" vector-effect="non-scaling-stroke" />` : "")
  );
}

// Also run by a ResizeObserver: while the live view is hidden the spark has no
// size, so it's skipped and redrawn when the view (or window) resizes.
function drawSpark(series = perSecondPace()) {
  const spark = $("pace-spark");
  if (!spark) return;
  const { width, height } = spark.getBoundingClientRect();
  if (!width || !height) return;
  spark.setAttribute("viewBox", `0 0 ${width} ${height}`);
  spark.innerHTML = sparkSvg(series, PRESETS[currentPreset], width, height);
}

const GRADE_COLOR: Record<string, string> = { A: "#30d158", B: "#30d158", C: "#ffd60a", D: "#ff9f0a", E: "#ff453a" };

function setLiveRing(score: number | null) {
  const ring = $("live-ring");
  if (!ring) return;
  const g = score === null ? "" : grade(score);
  ring.style.setProperty("--p", String(score ?? 0));
  ring.style.setProperty("--c", g ? GRADE_COLOR[g] : "#30d158");
  setText("live-score", score === null ? "–" : String(score));
  setText("live-grade", g);
  ring.dataset.tip = score === null ? "Delivery score — appears once you start speaking" : `Delivery score — ${score} of 100 so far, grade ${g}`;
}

// --- Per-utterance waveform --------------------------------------------------
// A small amplitude sparkline drawn under each committed line, with the VAD-
// detected hesitation pauses shaded on top. Built as an inline SVG string so it
// needs no dependency and re-renders cheaply. Because the line's innerHTML is
// rewritten in several places (draft, correction, filler review) and the
// analysis event can arrive before or after the text, `refreshWaveform` is the
// single point that (re)attaches it — idempotent, so any of those can call it.

function buildWaveformSvg(a: UtteranceAnalysis): string {
  const W = 300;
  const H = 32;
  const n = a.envelope.length;
  const barW = (W / n) * 0.8;
  let bars = "";
  for (let i = 0; i < n; i++) {
    const h = Math.max(1, (a.envelope[i] / 255) * H);
    const x = (i / n) * W;
    const y = (H - h) / 2;
    bars += `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${barW.toFixed(2)}" height="${h.toFixed(2)}" rx="0.5" />`;
  }
  let pauses = "";
  if (a.duration_ms > 0) {
    for (const p of a.pauses) {
      const x = (p.start_ms / a.duration_ms) * W;
      const w = Math.min(W - x, ((p.end_ms - p.start_ms) / a.duration_ms) * W);
      pauses += `<rect class="waveform-pause" x="${x.toFixed(2)}" y="0" width="${Math.max(1, w).toFixed(2)}" height="${H}" />`;
    }
  }
  const title = a.pause_count > 0 ? `${a.pause_count} pause${a.pause_count > 1 ? "s" : ""}` : "";
  return `<svg class="waveform" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true"><title>${title}</title>${pauses}<g class="waveform-bars">${bars}</g></svg>`;
}

function refreshWaveform(index: number) {
  const entry = segmentEls.get(index);
  if (!entry) return; // text line not on screen yet; drawn when it lands
  entry.querySelector(".waveform")?.remove();
  const analysis = analysisByIndex.get(index);
  if (!analysis || analysis.envelope.length === 0) return;
  entry.querySelector(".seg-body")?.insertAdjacentHTML("beforeend", buildWaveformSvg(analysis));
}

// --- Pace-over-time data -----------------------------------------------------
// One WPM sample per elapsed second of the session (index 0 = 0:00). Each
// utterance's words are spread evenly across its duration, then a second's WPM
// is the words falling in a short trailing window scaled to per-minute — so the
// line is a smooth pace curve, not a per-word sawtooth, and dips through pauses.
// Fillers are attributed to the second at their utterance's midpoint; `fillersCum`
// is the running total up to that second and `fillerHere` flags a filler onset.
interface SecondPace {
  wpm: number;
  fillersCum: number;
  fillerHere: boolean;
}

// ponytail: 5s trailing window — widen to smooth more, narrow to react faster.
const PACE_WINDOW_MS = 5000;

function perSecondPace(L = live): SecondPace[] {
  const idx = [...L.timing.keys()].sort((a, b) => a - b);
  if (idx.length === 0) return [];
  const utts = idx.map((i) => {
    const t = L.timing.get(i)!;
    return {
      start: t.start,
      end: t.end,
      wordsPerMs: (L.words.get(i) ?? 0) / Math.max(1, t.end - t.start),
      fillers: L.fillers.get(i) ?? 0,
      mid: (t.start + t.end) / 2,
    };
  });
  const totalSec = Math.max(1, Math.ceil(Math.max(...utts.map((u) => u.end)) / 1000));
  const fillerBySec = new Map<number, number>();
  for (const u of utts)
    if (u.fillers > 0) {
      const sec = Math.min(totalSec, Math.floor(u.mid / 1000));
      fillerBySec.set(sec, (fillerBySec.get(sec) ?? 0) + u.fillers);
    }
  const out: SecondPace[] = [];
  let cum = 0;
  for (let sec = 0; sec <= totalSec; sec++) {
    const tEnd = sec * 1000;
    const tStart = Math.max(0, tEnd - PACE_WINDOW_MS);
    let words = 0;
    for (const u of utts) {
      const a = Math.max(tStart, u.start);
      const b = Math.min(tEnd, u.end);
      if (b > a) words += u.wordsPerMs * (b - a);
    }
    const minutes = (tEnd - tStart) / 60000;
    cum += fillerBySec.get(sec) ?? 0;
    out.push({
      wpm: minutes > 0 ? Math.round(words / minutes) : 0,
      fillersCum: cum,
      fillerHere: (fillerBySec.get(sec) ?? 0) > 0,
    });
  }
  return out;
}


// Inner markup of one transcript line; the pass-state label comes from the
// line's class via CSS, so re-renders (filler review) don't need to know it.
function segmentHtml(startMs: number, textHtml: string): string {
  return `<div class="seg-body"><div class="seg-meta"><span class="seg-ts">${formatTimestamp(startMs)}</span><span class="seg-state"></span><span class="seg-countdown"><span class="bar"><i></i></span><span class="secs"></span></span></div><p class="seg-text">${textHtml}</p></div>`;
}

// Hover text for a line's state label (a CSS ::before, so it can't carry its own
// data-tip). Filled in on mouseover from the line's class, so it survives the
// innerHTML rewrites and applies to the report's transcript copy too.
const STATE_TIPS: Record<string, string> = {
  partial: "Live — rough preview from the live model while you're still speaking",
  draft: "Draft — the live model's text. The correction model re-checks it after you stop",
  corrected: "Corrected — re-transcribed by the correction model after you stopped",
  final: "Final — the correction model had nothing to replace it with (it heard only noise, failed, or correction is off), so the live model's text stands",
};
document.addEventListener("mouseover", (e) => {
  const label = (e.target as Element).closest?.<HTMLElement>(".seg-state");
  const line = label?.closest(".segment");
  if (!label || !line) return;
  const state = Object.keys(STATE_TIPS).find((k) => line.classList.contains(k));
  if (state) label.dataset.tip = STATE_TIPS[state];
});

// --- Correction countdown ----------------------------------------------------
// After Stop, the medium model corrects drafts one at a time, in order
// (audio.rs). Each waiting draft line gets a bar that fills toward its projected
// correction time — the audio queued ahead of it × `correctionRtf` — resynced on
// every `correction_started`.
let correctionRtf = 0.15; // medium decode time ÷ audio length; learned as lines land
let countdownStart = 0;
let countdownRaf = 0;
let lastCorrection: { index: number; at: number } | null = null;
const etaByIndex = new Map<number, number>(); // projected correction time (performance.now)
let countdownEnd = 0; // projected end of the whole pass, for the overall bar
let correctShown = 0; // overall bar fill; only moves forward as the ETA is re-learned

// Overall "Correcting…" bar over the transcript, from Stop until corrections_done.
function showCorrectBar(on: boolean) {
  correctShown = 0;
  countdownEnd = 0;
  $("correct-bar")?.toggleAttribute("hidden", !on);
  $("correct-bar")?.style.setProperty("--p", "0");
  setText("correct-label", "Correcting…");
}

function utteranceMs(index: number): number {
  const t = timingByIndex.get(index);
  return t ? t.end - t.start : 0;
}

function onCorrectionStarted(index: number) {
  const now = performance.now();
  if (lastCorrection) {
    // The gap between starts is the previous line's decode time.
    const ms = utteranceMs(lastCorrection.index);
    if (ms > 0) correctionRtf = (correctionRtf + (now - lastCorrection.at) / ms) / 2;
  } else {
    countdownStart = now;
  }
  lastCorrection = { index, at: now };
  etaByIndex.clear();
  let eta = now;
  for (const [i, el] of segmentEls) {
    const waiting = i >= index && el.classList.contains("draft");
    el.classList.toggle("counting", waiting);
    if (!waiting) continue;
    eta += utteranceMs(i) * correctionRtf;
    etaByIndex.set(i, eta);
  }
  countdownEnd = eta;
  if (!countdownRaf) countdownRaf = requestAnimationFrame(tickCountdown);
}

function tickCountdown() {
  const now = performance.now();
  for (const [i, eta] of etaByIndex) {
    const el = segmentEls.get(i);
    // Corrected (or removed): its countdown is over.
    if (!el?.classList.contains("draft")) {
      etaByIndex.delete(i);
      el?.classList.remove("counting");
      continue;
    }
    const frac = Math.min(1, (now - countdownStart) / Math.max(1, eta - countdownStart));
    el.querySelector<HTMLElement>(".seg-countdown i")?.style.setProperty("width", `${frac * 100}%`);
    const secs = el.querySelector(".seg-countdown .secs");
    if (secs) secs.textContent = `~${Math.max(0, Math.ceil((eta - now) / 1000))}s`;
  }
  if (countdownEnd) {
    correctShown = Math.max(correctShown, Math.min(1, (now - countdownStart) / Math.max(1, countdownEnd - countdownStart)));
    $("correct-bar")?.style.setProperty("--p", String(correctShown));
    setText("correct-label", `Correcting… ~${Math.max(0, Math.ceil((countdownEnd - now) / 1000))}s`);
  }
  countdownRaf = etaByIndex.size ? requestAnimationFrame(tickCountdown) : 0;
}

function resetCountdown() {
  showCorrectBar(false);
  etaByIndex.clear();
  lastCorrection = null;
  for (const el of segmentEls.values()) el.classList.remove("counting");
}

function formatTimestamp(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

// --- Script read-along -------------------------------------------------------
// Optional teleprompter: paste/upload a target script and have spoken words
// matched against it in real time. Pure client-side text alignment over the
// same transcript segments, so it works in every transcription/filler mode.

interface ScriptToken {
  norm: string;
  start: number;
  end: number;
}

const SCRIPT_KEY = "speech.script";
let scriptText = "";
let scriptTokens: ScriptToken[] = [];
// One <span> per script token, built once when the script is set. Re-aligning
// only flips class names on these (cheap) instead of rebuilding the whole
// display's innerHTML on every interim decode. `scriptSpanState` mirrors each
// span's current state class so we can skip unchanged spans.
let scriptSpans: HTMLElement[] = [];
let scriptSpanState: string[] = [];
// Cached normalized word tokens per utterance index (interim or final). Only the
// changed segment is re-tokenized per event, so building the spoken sequence
// stays cheap instead of re-parsing the whole transcript each time. Interim-to-
// final replacement is automatic since both share an index.
const tokensByIndex = new Map<number, string[]>();

let scriptInput: HTMLTextAreaElement | null;
let scriptFile: HTMLInputElement | null;
let scriptDisplay: HTMLElement | null;
let scriptProgress: HTMLElement | null;

// Word tokens: letters/digits with internal apostrophes or hyphens.
const WORD_RE = /[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*/g;

function tokenizeWithOffsets(text: string): ScriptToken[] {
  const tokens: ScriptToken[] = [];
  WORD_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = WORD_RE.exec(text)) !== null) {
    tokens.push({
      norm: m[0].toLowerCase().replace(/’/g, "'"),
      start: m.index,
      end: m.index + m[0].length,
    });
  }
  return tokens;
}

// The full spoken sequence so far, from the cached per-segment tokens.
function spokenWords(): string[] {
  const indices = [...tokensByIndex.keys()].sort((a, b) => a - b);
  const out: string[] = [];
  for (const i of indices) {
    const words = tokensByIndex.get(i);
    if (words) out.push(...words);
  }
  return out;
}

// Bounded edit-distance ≤ 1 check, to tolerate minor ASR slips (plurals, a
// dropped letter) without a full Levenshtein matrix.
function editDistLE1(a: string, b: string): boolean {
  const la = a.length;
  const lb = b.length;
  if (Math.abs(la - lb) > 1) return false;
  let i = 0;
  let j = 0;
  let edits = 0;
  while (i < la && j < lb) {
    if (a[i] === b[j]) {
      i++;
      j++;
      continue;
    }
    if (++edits > 1) return false;
    if (la > lb) i++;
    else if (lb > la) j++;
    else {
      i++;
      j++;
    }
  }
  if (i < la || j < lb) edits++;
  return edits <= 1;
}

// Homophones (and near-homophones ASR routinely confuses) that should count as
// a match. This matters most for short words, which otherwise require an exact
// match — and for a *speech* grader tolerating true homophones is correct, since
// they're indistinguishable in audio. Words are normalized like script tokens
// (lowercase, straight apostrophes); contractions stay single tokens.
const HOMOPHONE_GROUPS: string[][] = [
  ["their", "there", "they're"],
  ["your", "you're"],
  ["its", "it's"],
  ["whose", "who's"],
  ["to", "too", "two"],
  ["for", "four", "fore"],
  ["by", "buy", "bye"],
  ["hear", "here"],
  ["know", "no"],
  ["knew", "new"],
  ["right", "write", "rite"],
  ["one", "won"],
  ["be", "bee"],
  ["see", "sea"],
  ["so", "sew"],
  ["some", "sum"],
  ["son", "sun"],
  ["our", "hour"],
  ["would", "wood"],
  ["weak", "week"],
  ["threw", "through"],
  ["peace", "piece"],
  ["plain", "plane"],
  ["break", "brake"],
  ["cell", "sell"],
  ["cent", "scent", "sent"],
  ["fair", "fare"],
  ["great", "grate"],
  ["heal", "heel"],
  ["hole", "whole"],
  ["male", "mail"],
  ["meat", "meet"],
  ["pair", "pear", "pare"],
  ["rain", "reign", "rein"],
  ["role", "roll"],
  ["sail", "sale"],
  ["steal", "steel"],
  ["tail", "tale"],
  ["wait", "weight"],
  ["ware", "wear", "where"],
  ["way", "weigh"],
  ["weather", "whether"],
  ["which", "witch"],
  ["aloud", "allowed"],
  ["board", "bored"],
  ["flour", "flower"],
];
const HOMOPHONES = new Map<string, number>();
HOMOPHONE_GROUPS.forEach((group, id) => {
  for (const w of group) HOMOPHONES.set(w, id);
});

// Fuzzy only for longer words; short words must match exactly so common ones
// ("a"/"I"/"the") don't collapse together — except known homophones, which are
// allowed to match at any length.
function wordsMatch(a: string, b: string): boolean {
  if (a === b) return true;
  const ga = HOMOPHONES.get(a);
  if (ga !== undefined && ga === HOMOPHONES.get(b)) return true;
  return Math.min(a.length, b.length) >= 4 && editDistLE1(a, b);
}

// Greedy forward alignment: walk the spoken words, advancing a cursor through
// the script. Each spoken word matches the next occurrence within a small
// look-ahead window (tolerating skipped script words); words not found in the
// window are insertions (fillers, ASR noise, misreads) and are ignored.
//
// Substitution leniency: when a spoken word has no match but the *next* spoken
// word re-syncs exactly one script word ahead, the script word at the cursor is
// treated as a substitution (a misread or ASR slip like "and" heard for "in")
// rather than a hard miss. This tells "you said a different/misheard word here"
// apart from "you skipped this word entirely" — only the latter stays a miss.
function alignToScript(spoken: string[]): {
  matched: boolean[];
  substituted: boolean[];
  cursor: number;
} {
  const matched = new Array<boolean>(scriptTokens.length).fill(false);
  const substituted = new Array<boolean>(scriptTokens.length).fill(false);
  const WINDOW = 8;
  let cursor = 0;
  for (let s = 0; s < spoken.length; s++) {
    const sw = spoken[s];
    const limit = Math.min(scriptTokens.length, cursor + WINDOW);
    let found = -1;
    for (let i = cursor; i < limit; i++) {
      if (wordsMatch(scriptTokens[i].norm, sw)) {
        found = i;
        break;
      }
    }
    if (found >= 0) {
      matched[found] = true;
      cursor = found + 1;
      continue;
    }
    // `sw` didn't match. If the next spoken word matches the script word right
    // after the cursor, `sw` stood in for the cursor word: a substitution.
    if (cursor < scriptTokens.length && s + 1 < spoken.length) {
      const next = spoken[s + 1];
      const lim2 = Math.min(scriptTokens.length, cursor + 1 + WINDOW);
      for (let i = cursor + 1; i < lim2; i++) {
        if (wordsMatch(scriptTokens[i].norm, next)) {
          if (i === cursor + 1) {
            substituted[cursor] = true;
            cursor += 1;
          }
          break;
        }
      }
    }
    // Otherwise `sw` is a pure insertion (filler / ASR noise) — ignored.
  }
  return { matched, substituted, cursor };
}

// Builds the read-along display once for the current script: a <span> per word
// token plus the surrounding text/punctuation, all spans starting "pending".
// Later re-alignment only mutates span class names.
function buildScriptDisplay() {
  if (!scriptDisplay) return;
  if (scriptTokens.length === 0) {
    scriptDisplay.innerHTML = "";
    scriptSpans = [];
    scriptSpanState = [];
    return;
  }
  let html = "";
  let pos = 0;
  for (const t of scriptTokens) {
    html += escapeHtml(scriptText.slice(pos, t.start));
    html += `<span class="script-tok script-pending">${escapeHtml(scriptText.slice(t.start, t.end))}</span>`;
    pos = t.end;
  }
  html += escapeHtml(scriptText.slice(pos));
  scriptDisplay.innerHTML = html;
  scriptSpans = Array.from(scriptDisplay.querySelectorAll<HTMLElement>(".script-tok"));
  scriptSpanState = scriptSpans.map(() => "script-pending");
}

// Re-aligns the spoken words and updates only the spans whose state changed:
// matched words solid, current word highlighted, skipped/misread red, upcoming
// dimmed. No innerHTML rebuild, so this stays cheap on every interim decode.
function renderScriptMatch() {
  if (!scriptDisplay || !scriptProgress) return;
  if (scriptTokens.length === 0) {
    scriptDisplay.setAttribute("hidden", "");
    scriptProgress.setAttribute("hidden", "");
    return;
  }
  scriptDisplay.removeAttribute("hidden");
  scriptProgress.removeAttribute("hidden");

  const { matched, substituted, cursor } = alignToScript(spokenWords());
  let hits = 0;
  let misses = 0;
  let subs = 0;
  const missed: number[] = [];
  let currentSpan: HTMLElement | null = null;
  for (let i = 0; i < scriptSpans.length; i++) {
    let state: string;
    if (matched[i] || substituted[i]) {
      // Substitutions (a different/misheard word) count the same as exact hits.
      state = "script-hit";
      hits++;
      if (substituted[i]) subs++;
    } else if (i === cursor) {
      state = "script-current";
    } else if (i < cursor) {
      // The cursor advanced past this word without matching it: skipped.
      state = "script-miss";
      misses++;
      missed.push(i);
    } else {
      state = "script-pending";
    }
    if (state !== scriptSpanState[i]) {
      scriptSpans[i].className = `script-tok ${state}`;
      scriptSpanState[i] = state;
    }
    if (state === "script-current") currentSpan = scriptSpans[i];
  }

  const pct = Math.round((hits / scriptTokens.length) * 100);
  const missText = misses > 0 ? ` · ${misses} missed` : "";
  scriptProgress.textContent = `${hits} / ${scriptTokens.length} words · ${pct}%${missText}`;

  // Stash the alignment result so the end-of-session report can score
  // articulation against the script (accuracy / omissions / substitutions).
  lastScriptResult = {
    total: scriptTokens.length,
    hits,
    misses,
    subs,
    accuracy: pct,
    missed,
  };

  // Keep the current word in view within the panel.
  if (currentSpan) {
    scriptDisplay.scrollTop = currentSpan.offsetTop - scriptDisplay.clientHeight / 2;
  }
}

// Saved speeches: named scripts you come back to. Sessions recorded against one
// carry its id (SessionSummary.speech), so the report and profile can chart
// progress on that speech across attempts.
interface Speech {
  id: string;
  title: string;
  text: string;
  color?: string; // one of SPEECH_COLORS; older speeches have none
}
// Tag colors, fixed across themes so a speech keeps its color.
const SPEECH_COLORS: Array<[string, string]> = [
  ["#0a84ff", "Blue"],
  ["#30d158", "Green"],
  ["#ffd60a", "Yellow"],
  ["#ff9f0a", "Orange"],
  ["#ff453a", "Red"],
  ["#bf5af2", "Purple"],
  ["#64d2ff", "Teal"],
  ["#ff375f", "Pink"],
];
const speechColor = (sp: Speech) => sp.color ?? SPEECH_COLORS[0][0];
const SPEECHES_KEY = "speech.speeches";
const SPEECH_KEY = "speech.currentSpeech";
let speeches: Speech[] = [];
let currentSpeech = ""; // id of the saved speech in the script box; "" = unsaved

const speechTitle = (id: string | undefined) => speeches.find((s) => s.id === id)?.title;

function saveSpeeches() {
  save(SPEECHES_KEY, JSON.stringify(speeches));
  save(SPEECH_KEY, currentSpeech);
}

function renderSpeechPicker() {
  const sel = $<HTMLSelectElement>("speech-sel");
  if (sel) {
    sel.innerHTML =
      `<option value="">New script</option>` +
      speeches.map((s) => `<option value="${s.id}">${escapeHtml(s.title)}</option>`).join("");
    sel.value = currentSpeech;
    syncSelect(sel);
  }
  const btn = $("speech-save");
  if (btn) {
    btn.textContent = currentSpeech ? "Clear" : "Save";
    btn.dataset.tip = currentSpeech ? "Clear the box to start a new script — the speech stays saved" : "Save this script as a speech";
  }
}

// Two-click confirm: the first click relabels the button "Sure?" for 3 s and
// returns false; a second click within that window returns true.
function confirmClick(btn: HTMLElement): boolean {
  if (btn.dataset.armed) return true;
  const label = btn.textContent;
  btn.dataset.armed = "1";
  btn.textContent = "Sure?";
  setTimeout(() => {
    if (!btn.dataset.armed) return;
    delete btn.dataset.armed;
    btn.textContent = label;
  }, 3000);
  return false;
}

// Past sessions keep the id, so re-saving the text starts a fresh progress line.
function deleteSpeech(id: string) {
  speeches = speeches.filter((s) => s.id !== id);
  if (currentSpeech === id) currentSpeech = "";
  saveSpeeches();
  renderSpeechPicker();
}

function loadScript(text: string) {
  if (scriptInput) scriptInput.value = text;
  setScript(text);
}

// Puts a saved speech (or, for "", a blank script) in the read-along box.
function openSpeech(id: string) {
  currentSpeech = id;
  saveSpeeches();
  renderSpeechPicker();
  loadScript(speeches.find((s) => s.id === id)?.text ?? "");
}

function setScript(text: string) {
  scriptText = text;
  scriptTokens = tokenizeWithOffsets(text);
  try {
    localStorage.setItem(SCRIPT_KEY, text);
  } catch {
    // storage unavailable; script just won't persist
  }
  buildScriptDisplay();
  renderScriptMatch();
}

// Feeds a segment into the read-along state and re-aligns. Called for every
// segment; only the changed segment is re-tokenized. A blank final drops it.
function updateScriptFromSegment(segment: TranscriptSegment) {
  // An empty correction keeps the fast result (see upsertSegment); don't let it
  // clear the tokens the fast final already contributed.
  if (segment.refined && segment.text === "") return;
  // Correction already applied; ignore a late fast segment (see refinedIndices).
  if (!segment.refined && refinedIndices.has(segment.index)) return;
  if (segment.is_final && segment.text === "") {
    tokensByIndex.delete(segment.index);
  } else {
    tokensByIndex.set(
      segment.index,
      tokenizeWithOffsets(segment.text).map((t) => t.norm),
    );
  }
  if (scriptTokens.length > 0) renderScriptMatch();
}

// Long utterances are previewed in chunks (audio.rs PARTIAL_WINDOW_SAMPLES): each
// interim covers only its chunk, and earlier chunks are sealed. Keep every
// chunk's latest text so the preview shows the whole utterance so far; the
// final replaces it with a full-utterance decode.
const previewChunks = new Map<number, string[]>();
function withPreviewPrefix(segment: TranscriptSegment): TranscriptSegment {
  if (segment.is_final) {
    previewChunks.delete(segment.index);
    return segment;
  }
  if (finalized.has(segment.index)) return segment; // late interim; upsert drops it
  const chunks = previewChunks.get(segment.index) ?? [];
  chunks[segment.chunk] = segment.text;
  previewChunks.set(segment.index, chunks);
  return { ...segment, text: chunks.filter(Boolean).join(" ") };
}

function upsertSegment(segment: TranscriptSegment) {
  if (!transcriptEl) return;

  // A late interim result for an already-committed utterance must not overwrite it.
  if (!segment.is_final && finalized.has(segment.index)) return;

  // The correction already landed for this index (it can emit before the fast
  // final; see refinedIndices): drop any later fast final/interim so it can't
  // revert the corrected text.
  if (!segment.refined && refinedIndices.has(segment.index)) return;

  // An empty correction: keep the fast result rather than blanking the line.
  if (segment.refined && segment.text === "") return;

  // A fast final with no intelligible speech: drop any interim line we may have
  // shown. Corrections never blank a line (guarded above), so this is fast-only.
  if (segment.is_final && !segment.refined && segment.text === "") {
    segmentEls.get(segment.index)?.remove();
    segmentEls.delete(segment.index);
    finalized.add(segment.index);
    return;
  }

  transcriptEl.querySelector(".placeholder")?.remove();

  let entry = segmentEls.get(segment.index);
  if (!entry) {
    entry = document.createElement("div");
    entry.className = "segment";
    entry.dataset.index = String(segment.index);
    transcriptEl.appendChild(entry);
    segmentEls.set(segment.index, entry);
  }
  // Visual state of the line, so the two-pass pipeline is legible:
  //   partial   — live interim from the fast (tiny) model, still being spoken
  //   draft     — fast model's committed text, correction pending
  //   corrected — replaced by the accurate (medium) model; flashes once on change
  // A refined segment is also is_final, so check it first.
  entry.classList.remove("partial", "draft", "corrected", "final");
  if (!segment.is_final) {
    entry.classList.add("partial");
  } else if (segment.refined) {
    entry.classList.add("corrected");
  } else {
    entry.classList.add("draft");
  }
  // Tier-1 render: hesitation sounds ("um", "uh") are highlighted right away —
  // including on live partials — since they're a plain string match. Discourse
  // markers need the POS tagger, so they're flagged (and all fillers counted)
  // in the deferred review below, which runs on committed lines only.
  entry.innerHTML = segmentHtml(
    segment.start_ms,
    renderHighlighted(segment.text, hesitationRanges(segment.text)),
  );
  // Re-attach the waveform: innerHTML above wiped it, and this also covers the
  // correction rewriting a committed line. No-op until the analysis has arrived.
  refreshWaveform(segment.index);

  // Tally word/time metrics once the utterance is committed. Interim decodes are
  // skipped so partial re-reads of the same speech don't inflate the counts.
  // This runs for both the fast final and the later correction: word count is
  // applied by delta per index, so the correction replaces (not stacks on) the
  // fast final's contribution. Speaking time is added only the first time we see
  // an index, since the correction reuses the same timing. Filler counting
  // happens in the deferred review (it needs the detector) and is likewise
  // re-tallied by delta there.
  if (segment.is_final) {
    const words = countWords(segment.text);
    totalWords += words - (wordsByIndex.get(segment.index) ?? 0);
    wordsByIndex.set(segment.index, words);
    if (!timedIndices.has(segment.index)) {
      timedIndices.add(segment.index);
      speakingMs += Math.max(0, segment.end_ms - segment.start_ms);
    }
    // Track timing so gaps between utterances can be counted as pauses.
    timingByIndex.set(segment.index, { start: segment.start_ms, end: segment.end_ms });
    renderStats();

    // Tier-2 filler review, deferred so the text renders first. Captures the
    // element, index, timestamp and session; adds the detected fillers afterward.
    const reviewEntry = entry;
    const index = segment.index;
    const startMs = segment.start_ms;
    const text = segment.text;
    const reviewSession = sessionId;
    setTimeout(() => void reviewFillers(reviewEntry, index, startMs, text, reviewSession), 0);
  }

  if (segment.is_final) finalized.add(segment.index);
  if (segment.refined) refinedIndices.add(segment.index);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function appendError(message: string) {
  if (!transcriptEl) return;
  const entry = document.createElement("p");
  entry.className = "error";
  entry.textContent = message;
  transcriptEl.appendChild(entry);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function setStatus(text: string) {
  if (statusEl) statusEl.textContent = text;
  const pill = $("status-pill");
  pill?.classList.toggle("recording", text === "Recording");
  pill?.classList.toggle("error", text === "Error");
}

// Start/stop hotkey, as a layout-independent KeyboardEvent.code.
const RECORD_KEY = "speech.recordKey";
let recordKey = "Space";

// "KeyR" → "R", "Digit5" → "5", "ArrowUp" → "Up"; others ("Space", "F2") as-is.
function keyName(code: string): string {
  return code.replace(/^(Key|Digit|Arrow)/, "");
}

function setRecordingUi(on: boolean) {
  if (!recordBtn) return;
  recordBtn.classList.toggle("recording", on);
  const k = keyName(recordKey);
  const label = on ? `Stop recording — ends the session and finalizes the transcript (${k})` : `Start recording (${k})`;
  recordBtn.setAttribute("aria-label", on ? "Stop recording" : "Start recording");
  recordBtn.dataset.tip = label;
  $("rec-label")?.classList.toggle("on", on);
  $("live-btn")?.classList.toggle("recording", on);
}

// Wall-clock session timer on the scope; also counts the timer (drill) down
// and stops the recording when it runs out. drillMs 0 = no timer.
let clockTimer = 0;
let recordStartedAt = 0;
let drillEndsAt = 0;

function tickClock() {
  const now = performance.now();
  setText("clock", formatTimestamp(recordStartedAt ? now - recordStartedAt : 0));
  const left = drillEndsAt - now;
  if (drillEndsAt && left <= 0 && recording) {
    drillEndsAt = 0;
    void toggleRecording(); // stopClock re-ticks the display
    return;
  }
  const running = left > 0;
  const label = drillMs ? formatTimestamp(running ? Math.ceil(left / 1000) * 1000 : drillMs) : "Off";
  setText("drill-label", label);
  setText("countdown", label);
  document.body.classList.toggle("has-timer", drillMs > 0);
  $("drill-btn")?.classList.toggle("drilling", running);
}

function startClock() {
  recordStartedAt = performance.now();
  drillEndsAt = drillMs ? recordStartedAt + drillMs : 0;
  clearInterval(clockTimer);
  clockTimer = window.setInterval(tickClock, 250);
  tickClock();
}

function stopClock() {
  clearInterval(clockTimer);
  drillEndsAt = 0;
  tickClock();
}

// Input level meter in the dock: live level plus a slowly falling peak.
let peakLevel = 0;
function setLevelMeter(level: number) {
  peakLevel = Math.max(level, peakLevel * 0.97);
  const live = $("lvl-live");
  const peak = $("lvl-peak");
  if (live) live.style.height = `${level * 100}%`;
  if (peak) peak.style.height = `${peakLevel * 100}%`;
  const micLive = $("mic-live");
  const micPeak = $("mic-peak");
  if (micLive) micLive.style.width = `${level * 100}%`;
  if (micPeak) micPeak.style.width = `${peakLevel * 100}%`;
}

// Settings mic test: live meter + speech light while on, then plays back the
// last ~10 s the backend kept (16 kHz mono f32 — exactly what the models hear).
const MIC_HINT = "Talk for a few seconds, then stop to hear it back";
let micTesting = false;
let micAudio: AudioContext | null = null;
async function startMicTest() {
  const btn = $<HTMLButtonElement>("mic-test-btn");
  try {
    await invoke("start_mic_test");
  } catch (e) {
    setText("mic-status", String(e));
    return;
  }
  micTesting = true;
  if (btn) btn.textContent = "Stop";
  setText("mic-status", "Listening — speak normally");
}
async function stopMicTest(playBack: boolean) {
  if (!micTesting) return;
  micTesting = false;
  const btn = $<HTMLButtonElement>("mic-test-btn");
  if (btn) btn.textContent = "Test";
  let buf: ArrayBuffer;
  try {
    buf = await invoke<ArrayBuffer>("stop_mic_test");
  } catch (e) {
    setText("mic-status", String(e));
    return;
  } finally {
    peakLevel = 0;
    setLevelMeter(0);
    ribbon?.setLevel(0);
    $("mic-speech")?.classList.remove("on");
  }
  const samples = new Float32Array(buf);
  if (!playBack || !samples.length) {
    setText("mic-status", MIC_HINT);
    return;
  }
  micAudio ??= new AudioContext();
  const clip = micAudio.createBuffer(1, samples.length, 16000);
  clip.copyToChannel(samples, 0);
  const src = micAudio.createBufferSource();
  src.buffer = clip;
  src.connect(micAudio.destination);
  src.onended = () => setText("mic-status", MIC_HINT);
  setText("mic-status", `Playing back ${(samples.length / 16000).toFixed(1)} s…`);
  src.start();
}

async function toggleRecording() {
  if (!recordBtn) return;

  recordBtn.disabled = true;
  try {
    if (!recording) {
      flushSave();
      resetCountdown();
      // The backend restarts utterance indices each session; drop stale
      // line references so a new session's index 1 starts a fresh line.
      segmentEls.clear();
      previewChunks.clear();
      finalized.clear();
      refinedIndices.clear();
      tokensByIndex.clear();
      renderScriptMatch(); // reset read-along highlights to the start
      resetStats();
      reportVisible = false;
      reportPending = false;
      scriptUsed = !$("script-pane")?.hidden;
      const statsBtn = $<HTMLButtonElement>("stats-btn");
      if (statsBtn) statsBtn.disabled = true;
      await stopMicTest(false);
      await invoke("start_recording", { correct: accurateCorrection, correctionModel });
      recording = true;
      sessionStartTs = Date.now();
      currentSaved = false;
      renderSessionLabel();
      startClock();
      showView("live");
      setRecordingUi(true);
      ribbon?.setMode("listening");
      setStatus("Recording");
    } else {
      await invoke("stop_recording");
      recording = false;
      stopClock();
      setRecordingUi(false);
      ribbon?.setMode("idle");
      ribbon?.setLevel(0); // no more level events once stopped; settle to rest
      peakLevel = 0;
      setLevelMeter(0);
      setStatus("Correcting…"); // back to Idle on corrections_done
      // The report opens on corrections_done, so it shows corrected text.
      reportPending = true;
      savePending = totalWords > 0;
      showCorrectBar(accurateCorrection && totalWords > 0);
    }
  } catch (e) {
    setStatus("Error");
    appendError(String(e));
  } finally {
    recordBtn.disabled = false;
  }
}

// Wipe the current session from the screen — transcript, live stats, and report
// — so the next attempt starts clean. Ignored mid-recording.
function resetSession() {
  if (recording) return;
  flushSave();
  reportPending = false;
  resetCountdown();
  segmentEls.clear();
  previewChunks.clear();
  finalized.clear();
  refinedIndices.clear();
  tokensByIndex.clear();
  renderScriptMatch();
  resetStats();
  renderReport(); // no words → disables STATS and returns to live
  recordStartedAt = 0;
  tickClock();
  $("practice-prompt")?.setAttribute("hidden", "");
  if (transcriptEl)
    transcriptEl.innerHTML =
      '<p class="placeholder">Your transcript will appear here as you speak.</p>';
  setStatus("Idle");
}

function toggleScript(open?: boolean) {
  const pane = $("script-pane");
  if (!pane) return;
  const show = open ?? pane.hidden;
  pane.hidden = !show;
  $("script-btn")?.setAttribute("aria-pressed", String(show));
  if (recording) scriptUsed = show;
}

// Styled hover tips for anything with data-tip (the design's tooltip).
function initTooltips() {
  const tip = $("tip");
  if (!tip) return;
  document.addEventListener("mousemove", (e) => {
    const el = (e.target as Element | null)?.closest?.("[data-tip]");
    const text = el?.getAttribute("data-tip");
    if (!text) {
      tip.style.opacity = "0";
      return;
    }
    if (tip.textContent !== text) tip.textContent = text;
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    const right = e.clientX + 14 + w < window.innerWidth;
    const below = e.clientY + 18 + h < window.innerHeight;
    tip.style.left = `${right ? e.clientX + 14 : e.clientX - 14 - w}px`;
    tip.style.top = `${below ? e.clientY + 18 : e.clientY - 12 - h}px`;
    tip.style.opacity = "1";
  });
  document.addEventListener("mouseleave", () => (tip.style.opacity = "0"));
}

// --- Scoring, report, practice -----------------------------------------------
// Everything below turns the per-utterance metrics accumulated above into an
// end-of-session score + report, persists the session, and drives the practice
// UI. All local; the report re-renders live as deferred corrections land.

interface Preset {
  name: string;
  wpmLow: number;
  wpmHigh: number;
}
// Target pace bands per speaking context; other targets (fillers, pitch, volume)
// are universal. Changing the context in the report re-scores the last session.
const PRESETS: Record<string, Preset> = {
  conversation: { name: "Conversation", wpmLow: 120, wpmHigh: 160 },
  presentation: { name: "Presentation", wpmLow: 100, wpmHigh: 140 },
  interview: { name: "Interview", wpmLow: 120, wpmHigh: 155 },
};
const PRESET_KEY = "speech.preset";
let currentPreset = "conversation";

interface Scores {
  pace: number;
  fillers: number;
  pauses: number;
  pitch: number;
  volume: number;
  articulation: number | null; // null when no script was used
  overall: number;
}

interface SessionSummary {
  ts: number;
  durationMs: number;
  words: number;
  wpm: number;
  fillers: number;
  fillersPerMin: number;
  pauses: number;
  pausesPerMin: number;
  pitchRange: number; // mean within-utterance inflection, semitones
  uptalk: number;
  loudnessCV: number; // coefficient of variation of loudness across utterances
  trailingOff: number; // 1 = steady, <1 = fades at sentence ends
  peakMinuteWpm: number;
  peakMinute: number;
  // `missed`: script token indices skipped (absent on sessions saved before it was tracked).
  script: { total: number; hits: number; misses: number; subs: number; accuracy: number; missed?: number[] } | null;
  preset: string;
  speech?: string; // saved speech id, when read against one
  saved?: boolean; // starred in Stats/History
  pace?: PaceData; // absent on sessions saved before it was kept
  scores: Scores;
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

// End-of-sentence volume decay: mean of the last ~20% of the envelope vs the
// whole. <1 means the speaker faded out (trailing off). Envelope is self-
// normalized per utterance, but this is a within-utterance ratio so that's fine.
function trailingOffRatio(env: number[]): number | null {
  if (env.length < 5) return null;
  const tailN = Math.max(1, Math.round(env.length * 0.2));
  const tailMean = mean(env.slice(env.length - tailN));
  const allMean = mean(env);
  return allMean > 0 ? tailMean / allMean : null;
}

// WPM per wall-clock minute, binning each utterance's words + speaking time by
// its start. Lets tips cite a specific stretch ("fastest around minute 3").
function perMinuteWpm(): { minute: number; wpm: number }[] {
  const wordsBin = new Map<number, number>();
  const msBin = new Map<number, number>();
  for (const idx of timingByIndex.keys()) {
    const t = timingByIndex.get(idx)!;
    const m = Math.floor(t.start / 60000);
    wordsBin.set(m, (wordsBin.get(m) ?? 0) + (wordsByIndex.get(idx) ?? 0));
    msBin.set(m, (msBin.get(m) ?? 0) + Math.max(0, t.end - t.start));
  }
  const out: { minute: number; wpm: number }[] = [];
  for (const m of [...wordsBin.keys()].sort((a, b) => a - b)) {
    const minutes = (msBin.get(m) ?? 0) / 60000;
    if (minutes > 0) out.push({ minute: m, wpm: Math.round((wordsBin.get(m) ?? 0) / minutes) });
  }
  return out;
}

// Linear score in [0,100]: `good` value → 100, `bad` value → 0 (either direction).
function scoreLinear(v: number, good: number, bad: number): number {
  if (good === bad) return 100;
  const t = (v - bad) / (good - bad);
  return Math.round(Math.max(0, Math.min(1, t)) * 100);
}

// Pace scores 100 inside the target band, falling off outside (0 at ~50 WPM out).
function scorePace(wpm: number, low: number, high: number): number {
  if (wpm <= 0) return 0;
  if (wpm >= low && wpm <= high) return 100;
  const dist = wpm < low ? low - wpm : wpm - high;
  return Math.round(Math.max(0, 1 - dist / 50) * 100);
}

// Relative weights of each dimension in the composite. Fillers and articulation
// (a graded read-along) weigh most; pauses/volume are secondary.
const SCORE_WEIGHTS = { pace: 1, fillers: 1.5, pauses: 0.75, pitch: 1, volume: 0.75, articulation: 1.5 };

function computeSummary(): SessionSummary {
  const minutes = speakingMs / 60000;
  const wpm = minutes > 0 ? Math.round(totalWords / minutes) : 0;
  const fillersPerMin = minutes > 0 ? totalFillers / minutes : 0;
  const pauses = countPauses();
  const pausesPerMin = minutes > 0 ? pauses / minutes : 0;

  const analyses = [...analysisByIndex.values()];
  const voiced = analyses.filter((a) => a.f0_median > 0);
  const pitchRange = mean(voiced.map((a) => a.f0_range_semitones));
  const uptalk = voiced.filter((a) => a.f0_terminal_rising).length;

  const levels = analyses.map((a) => a.rms_level).filter((x) => x > 0);
  const lMean = mean(levels);
  const lSd = levels.length > 1 ? Math.sqrt(mean(levels.map((x) => (x - lMean) ** 2))) : 0;
  const loudnessCV = lMean > 0 ? lSd / lMean : 0;

  const tails = analyses
    .map((a) => trailingOffRatio(a.envelope))
    .filter((x): x is number => x !== null);
  const trailingOff = tails.length ? mean(tails) : 1;

  let peak = { minute: 0, wpm: 0 };
  for (const p of perMinuteWpm()) if (p.wpm > peak.wpm) peak = p;

  const script = scriptUsed && lastScriptResult && scriptTokens.length > 0 ? { ...lastScriptResult } : null;
  const preset = PRESETS[currentPreset];

  const pace = scorePace(wpm, preset.wpmLow, preset.wpmHigh);
  const fillers = scoreLinear(fillersPerMin, 1, 12);
  const pausesScore = scoreLinear(pausesPerMin, 2, 14);
  const pitch = scoreLinear(pitchRange, 5, 1);
  const consistency = scoreLinear(loudnessCV, 0.25, 0.9);
  const finish = scoreLinear(trailingOff, 0.9, 0.4);
  const volume = Math.round((consistency + finish) / 2);
  const articulation = script ? script.accuracy : null;

  const parts: Array<{ s: number; w: number }> = [
    { s: pace, w: SCORE_WEIGHTS.pace },
    { s: fillers, w: SCORE_WEIGHTS.fillers },
    { s: pausesScore, w: SCORE_WEIGHTS.pauses },
    { s: pitch, w: SCORE_WEIGHTS.pitch },
    { s: volume, w: SCORE_WEIGHTS.volume },
  ];
  if (articulation !== null) parts.push({ s: articulation, w: SCORE_WEIGHTS.articulation });
  const wsum = parts.reduce((a, p) => a + p.w, 0);
  const overall = Math.round(parts.reduce((a, p) => a + p.s * p.w, 0) / wsum);

  const scores: Scores = { pace, fillers, pauses: pausesScore, pitch, volume, articulation, overall };
  return {
    ts: sessionStartTs,
    durationMs: speakingMs,
    words: totalWords,
    wpm,
    fillers: totalFillers,
    fillersPerMin,
    pauses,
    pausesPerMin,
    pitchRange,
    uptalk,
    loudnessCV,
    trailingOff,
    peakMinuteWpm: peak.wpm,
    peakMinute: peak.minute,
    script,
    preset: currentPreset,
    speech: script && currentSpeech ? currentSpeech : undefined,
    saved: currentSaved || undefined,
    scores,
  };
}

// Impact-ranked, specific-number tips. Each candidate's impact = its dimension
// weight × how far below 100 it scored, so the biggest weighted weakness leads.
function generateTips(s: SessionSummary): string[] {
  const preset = PRESETS[s.preset];
  const tips: Array<{ impact: number; text: string }> = [];
  const push = (score: number, weight: number, text: string) =>
    tips.push({ impact: weight * (100 - score), text });

  if (s.wpm > preset.wpmHigh)
    push(s.scores.pace, SCORE_WEIGHTS.pace, `You averaged ${s.wpm} WPM — ${s.wpm - preset.wpmHigh} above the ${preset.name.toLowerCase()} range (${preset.wpmLow}–${preset.wpmHigh}). Slow down, especially through longer sentences.`);
  else if (s.wpm > 0 && s.wpm < preset.wpmLow)
    push(s.scores.pace, SCORE_WEIGHTS.pace, `You averaged ${s.wpm} WPM — ${preset.wpmLow - s.wpm} below the ${preset.name.toLowerCase()} range (${preset.wpmLow}–${preset.wpmHigh}). Pick up the pace to keep energy up.`);
  if (s.peakMinuteWpm > preset.wpmHigh + 10)
    push(55, 0.5, `Your fastest stretch hit ${s.peakMinuteWpm} WPM around minute ${s.peakMinute + 1} — watch for rushing there.`);

  if (s.fillersPerMin >= 3)
    push(s.scores.fillers, SCORE_WEIGHTS.fillers, `You used ${s.fillers} filler words (${s.fillersPerMin.toFixed(1)}/min). Aim under 3/min — swap "um"/"like" for a brief silent pause.`);

  if (s.pitchRange < 3 && s.pitchRange > 0)
    push(s.scores.pitch, SCORE_WEIGHTS.pitch, `Your pitch varied only ${s.pitchRange.toFixed(1)} semitones — that reads as monotone. Stretch your intonation to hold attention.`);
  if (s.uptalk >= 3)
    push(50, 0.75, `${s.uptalk} statements rose in pitch at the end (uptalk), which can sound uncertain. Land statements on a falling tone.`);

  if (s.trailingOff < 0.7)
    push(s.scores.volume, SCORE_WEIGHTS.volume, `You trailed off at sentence ends (volume fell to ${Math.round(s.trailingOff * 100)}% of your average). Carry energy through the last word.`);
  else if (s.loudnessCV > 0.6)
    push(s.scores.volume, SCORE_WEIGHTS.volume, `Your volume was uneven across sentences (±${Math.round(s.loudnessCV * 100)}%). Keep a steadier level.`);

  if (s.pausesPerMin > 10)
    push(s.scores.pauses, SCORE_WEIGHTS.pauses, `You paused often (${s.pausesPerMin.toFixed(1)}/min). Some pausing lands well, but frequent hesitation gaps break flow.`);

  if (s.script)
    push(s.script.accuracy, SCORE_WEIGHTS.articulation, `You matched ${s.script.accuracy}% of the script${s.script.misses ? ` — ${s.script.misses} skipped` : ""}${s.script.subs ? `, ${s.script.subs} misread` : ""}.`);

  tips.sort((a, b) => b.impact - a.impact);
  const top = tips.filter((t) => t.impact > 0).slice(0, 4).map((t) => t.text);
  if (top.length === 0) top.push("Strong session — no standout weaknesses. Keep it up.");
  return top;
}

function grade(n: number): string {
  if (n >= 90) return "A";
  if (n >= 80) return "B";
  if (n >= 70) return "C";
  if (n >= 60) return "D";
  return "E";
}

// --- Session report (3a) -----------------------------------------------------

// Saved sessions (history.rs), oldest first. Loaded at startup and replaced by
// save_session's reply. The report compares against those saved before
// `sessionStartTs`, so this session's own saved copy is never its baseline.
let history: SessionSummary[] = [];
let sessionStartTs = Date.now();

function parseHistory(json: string): SessionSummary[] {
  try {
    const arr = JSON.parse(json);
    return Array.isArray(arr)
      ? arr.filter((h) => typeof h?.ts === "number" && typeof h?.scores?.overall === "number")
      : [];
  } catch {
    return [];
  }
}

const priorSessions = () => history.filter((h) => h.ts < sessionStartTs);

function fmtDate(ts: number): string {
  return new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function renderSessionLabel() {
  setText("session-label", `Session ${priorSessions().length + 1} · ${fmtDate(sessionStartTs)}`);
}

// The save waits for the post-stop correction pass (corrections_done) so the
// saved record scores corrected text; flushed early if the user moves on first.
let savePending = false;
function flushSave() {
  if (!savePending) return;
  savePending = false;
  void saveSession();
}

async function saveSession() {
  try {
    const session = JSON.stringify({ ...computeSummary(), pace: paceData() });
    history = parseHistory(await invoke<string>("save_session", { session, detail: sessionDetail() }));
    if (document.body.dataset.view === "speeches") renderSpeeches();
    if (document.body.dataset.view === "history") renderHistory();
  } catch (e) {
    appendError(`Couldn't save session: ${e}`);
  }
}

// The live session's per-line data, so History can redraw its full report.
// history.rs keeps it only for the newest sessions and starred ones.
function sessionDetail(): string {
  return JSON.stringify({
    timing: [...timingByIndex],
    words: [...wordsByIndex],
    fillers: [...fillersByIndex],
    fillerWords: [...fillerWordsByIndex],
    // The envelope only draws the waveform, which is already in `lines`.
    analysis: [...analysisByIndex].map(([i, a]) => [i, { ...a, envelope: [] }]),
    lines: liveLinesHtml(),
  });
}

interface Detail {
  ts: number;
  L: Lines;
  lines: Array<[number, string]>;
}
function parseDetail(ts: number, json: string): Detail | null {
  try {
    const d = JSON.parse(json);
    const L: Lines = {
      timing: new Map(d.timing),
      words: new Map(d.words),
      fillers: new Map(d.fillers),
      fillerWords: new Map(d.fillerWords),
      analysis: new Map(d.analysis),
    };
    return { ts, L, lines: d.lines ?? [] };
  } catch {
    return null;
  }
}

// Star on the session being recorded/reported, until its record is written
// (then the record's `saved` is the truth).
let currentSaved = false;
function isSaved(ts: number): boolean {
  const h = history.find((x) => x.ts === ts);
  return h ? !!h.saved : ts === sessionStartTs && currentSaved;
}

async function toggleSaved(ts: number) {
  const saved = !isSaved(ts);
  if (ts === sessionStartTs) currentSaved = saved; // carried by a save still pending
  try {
    if (history.some((h) => h.ts === ts))
      history = parseHistory(await invoke<string>("set_session_saved", { ts, saved }));
  } catch (e) {
    appendError(`Couldn't save session: ${e}`);
  }
  if (document.body.dataset.view === "report") renderReport();
  if (document.body.dataset.view === "history") renderHistory();
}

function starHtml(ts: number): string {
  const on = isSaved(ts);
  return `<button type="button" class="star-btn" data-star="${ts}" aria-pressed="${on}" aria-label="${on ? "Unsave" : "Save"} session" data-tip="${on ? "Saved — click to unsave" : "Save this session"}">${on ? "★" : "☆"}</button>`;
}

// "+4", "−1.2", "±0" — typographic minus to match the design.
function signed(n: number, digits = 0): string {
  const v = Number(n.toFixed(digits));
  if (v === 0) return "±0";
  return (v > 0 ? "+" : "−") + Math.abs(v).toFixed(digits);
}

function ringHtml(score: number, tip: string): string {
  const g = grade(score);
  return `<div class="ring lg" style="--p:${score};--c:${GRADE_COLOR[g]}" data-tip="${escapeHtml(tip)}"><div class="ring-in"><span class="ring-num">${score}</span><span class="ring-grade">${g}</span></div></div>`;
}

interface Moment {
  ms: number;
  index: number;
  color: string;
  title: string;
  note: string;
}

// Up to three jump-to points: the densest filler line, the fastest stretch,
// and the longest filler-free run.
function keyMoments(preset: Preset, L = live): Moment[] {
  const idx = [...L.timing.keys()].sort((a, b) => a - b);
  if (idx.length === 0) return [];
  const start = (i: number) => L.timing.get(i)!.start;
  const out: Moment[] = [];

  let worst = -1;
  for (const i of idx) {
    const n = L.fillers.get(i) ?? 0;
    if (n >= 2 && (worst < 0 || n > L.fillers.get(worst)!)) worst = i;
  }
  if (worst >= 0)
    out.push({
      ms: start(worst),
      index: worst,
      color: "#ffd60a",
      title: `${L.fillers.get(worst)} fillers in one sentence`,
      note: (L.fillerWords.get(worst) ?? []).join(" → "),
    });

  const series = perSecondPace(L);
  let ps = 0;
  series.forEach((p, sec) => {
    if (p.wpm > series[ps].wpm) ps = sec;
  });
  const peak = series[ps]?.wpm ?? 0;
  if (peak > 0) {
    // The peak second closes a trailing window; point at its middle.
    const ms = Math.max(0, ps * 1000 - PACE_WINDOW_MS / 2);
    const at = idx.filter((i) => start(i) <= ms).pop() ?? idx[0];
    out.push({
      ms,
      index: at,
      color: "var(--blue)",
      title: `Peak pace, ${peak} wpm`,
      note: peak > preset.wpmHigh ? `${peak - preset.wpmHigh} over target — breathe between phrases` : "Still inside your target range",
    });
  }

  let best = { from: -1, ms: 0 };
  let runStart = -1;
  for (let k = 0; k < idx.length; k++) {
    if ((L.fillers.get(idx[k]) ?? 0) > 0) {
      runStart = -1;
      continue;
    }
    if (runStart < 0) runStart = k;
    const ms = L.timing.get(idx[k])!.end - start(idx[runStart]);
    if (ms > best.ms) best = { from: runStart, ms };
  }
  if (best.ms >= 5000 && [...L.fillers.values()].some((n) => n > 0))
    out.push({
      ms: start(idx[best.from]),
      index: idx[best.from],
      color: "#30d158",
      title: "Cleanest stretch",
      note: `${Math.round(best.ms / 1000)} s with no fillers`,
    });

  return out.sort((a, b) => a.ms - b.ms);
}

const tone = (v: number) => (v > 0 ? "up" : v < 0 ? "down" : "flat");

// Report header: score ring, title + save star, delta vs the previous
// session, the top tip. Shared by the live report and History's session page.
function reportHeroHtml(s: SessionSummary, prev: SessionSummary | undefined, num: number, lengthMs: number): string {
  const c = s.scores;
  const title = speechTitle(s.speech);
  const d = prev ? c.overall - prev.scores.overall : null;
  return (
    `<div class="rep-hero">` +
    ringHtml(c.overall, `Delivery score — ${c.overall} of 100, grade ${grade(c.overall)}`) +
    `<div class="rep-title"><h2>Session ${num} report ${starHtml(s.ts)}</h2><div class="sub">${fmtDate(s.ts)} · ${formatTimestamp(lengthMs)} · ${PRESETS[s.preset]?.name ?? s.preset}${title ? ` · ${escapeHtml(title)}` : ""} · medium.en</div></div>` +
    (d === null
      ? ""
      : `<div class="rep-delta" data-tip="Score change since session ${num - 1}"><div class="num ${tone(d)}">${signed(d)}</div><div class="sub">vs session ${num - 1}</div></div>`) +
    `<div class="rep-vr"></div>` +
    `<div class="rep-tip" data-tip="The single change that would lift the score most">${escapeHtml(generateTips(s)[0])}</div>` +
    `</div>`
  );
}

// Headline numbers, each with its change vs the previous session, colored by
// whether that dimension's sub-score went up or down.
function reportTilesHtml(s: SessionSummary, prev: SessionSummary | undefined): string {
  const c = s.scores;
  const preset = PRESETS[s.preset];
  const vs = (cur: number, old: number | undefined, digits: number, key: keyof Scores): [string, string] => {
    if (!prev || typeof old !== "number") return ["first session", "flat"];
    return [`${signed(cur - old, digits)} vs last`, tone((c[key] ?? 0) - (prev.scores[key] ?? 0))];
  };
  const tile = (value: string, unit: string, [delta, cls]: [string, string], tip: string) =>
    `<div class="tile" data-tip="${escapeHtml(tip)}"><div class="card-top"><span class="big">${value}</span><span class="unit">${unit}</span></div><span class="delta ${cls}">${delta}</span></div>`;
  return (
    `<div class="tiles">` +
    tile(String(s.wpm), "wpm", vs(s.wpm, prev?.wpm, 0, "pace"), `Average pace — target ${preset.wpmLow}–${preset.wpmHigh}`) +
    tile(s.fillersPerMin.toFixed(1), "fillers/min", vs(s.fillersPerMin, prev?.fillersPerMin, 1, "fillers"), "Fillers per minute — aim under 3") +
    tile(s.pausesPerMin.toFixed(1), "pauses/min", vs(s.pausesPerMin, prev?.pausesPerMin, 1, "pauses"), "Hesitation pauses per minute") +
    tile(s.pitchRange.toFixed(1), "semitones", vs(s.pitchRange, prev?.pitchRange, 1, "pitch"), "Pitch range per sentence — under 3 reads as flat") +
    tile(String(s.words), "words", [`${formatTimestamp(s.durationMs)} speaking`, "flat"], `Words spoken across ${formatTimestamp(s.durationMs)} of speech`) +
    `</div>`
  );
}

function renderReport() {
  const body = $("report-body");
  const statsBtn = $<HTMLButtonElement>("stats-btn");
  if (!body) return;
  if (totalWords === 0) {
    reportVisible = false;
    if (statsBtn) statsBtn.disabled = true;
    if (document.body.dataset.view === "report") showView("live");
    return;
  }
  reportVisible = true;
  if (statsBtn) statsBtn.disabled = false;
  body.innerHTML = reportBodyHtml(computeSummary(), live, paceData());
  const rt = $("report-transcript");
  if (rt) rt.innerHTML = transcriptCopyHtml(liveLinesHtml());
}

// Every live transcript line's markup (fillers, waveform), in order.
function liveLinesHtml(): Array<[number, string]> {
  return [...segmentEls].sort((a, b) => a[0] - b[0]).map(([i, el]) => [i, el.outerHTML]);
}

// Full transcript below the fold: copies of the session's lines.
function transcriptCopyHtml(lines: Array<[number, string]>): string {
  return `<span class="label">Transcript</span>` + lines.map(([, html]) => html).join("");
}

// A session's pace, second by second — all the pace chart draws. Saved with
// every session, so older sessions (whose per-line data is pruned) keep it.
interface PaceData {
  wpm: number[];
  fillerSecs: number[];
  pauseMs: number[];
  lengthMs: number;
}
function paceData(L = live): PaceData {
  const series = perSecondPace(L);
  return {
    wpm: series.map((p) => p.wpm),
    fillerSecs: series.flatMap((p, sec) => (p.fillerHere ? [sec] : [])),
    pauseMs: pauseTimesMs(L).map(Math.round),
    lengthMs: sessionEndMs(L),
  };
}

// Pace across the session, with filler dots and pause ticks on shared time.
function paceChartHtml(p: PaceData, preset: Preset): string {
  const totalSec = Math.max(1, p.wpm.length - 1);
  const [lo, hi] = wpmAxis(Math.max(0, ...p.wpm));
  const W = 960;
  const H = 160;
  const y = (v: number) => H - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * H;
  const pts = p.wpm.map((v, sec) => `${((sec / totalSec) * W).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const pct = (sec: number) => `${Math.min(100, (sec / totalSec) * 100).toFixed(1)}%`;
  const dots = p.fillerSecs.map((sec) => `<div class="dot" style="left:${pct(sec)}"></div>`).join("");
  const ticks = p.pauseMs.map((ms) => `<div class="tick" style="left:${pct(ms / 1000)}"></div>`).join("");
  const yLabels = [0, 1, 2, 3].map((k) => `<span>${Math.round(hi - ((hi - lo) * k) / 3)}</span>`).join("");
  const xLabels = [0, 0.25, 0.5, 0.75, 1].map((f) => `<span>${formatTimestamp(f * p.lengthMs)}</span>`).join("");
  return (
    `<div class="panel pace-panel">` +
    `<div class="panel-head"><span class="label">Pace across session</span><div class="legend"><span style="color:#4aa8ff">— wpm</span><span style="color:#30d158">▮ target ${preset.wpmLow}–${preset.wpmHigh}</span><span style="color:#ffd60a">● filler</span><span>| pause</span></div></div>` +
    `<div class="pace-grid"><div class="y-labels">${yLabels}</div><div class="pace-plot" data-tip="">` +
    `<div class="pace-cursor"></div>` +
    `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">` +
    `<line x1="0" y1="0.5" x2="${W}" y2="0.5" stroke="currentColor" stroke-opacity="0.06" />` +
    `<line x1="0" y1="${H - 0.5}" x2="${W}" y2="${H - 0.5}" stroke="currentColor" stroke-opacity="0.1" />` +
    `<rect x="0" y="${y(preset.wpmHigh).toFixed(1)}" width="${W}" height="${(y(preset.wpmLow) - y(preset.wpmHigh)).toFixed(1)}" fill="rgba(48,209,88,0.12)" />` +
    `<polygon points="0,${H} ${pts} ${W},${H}" fill="rgba(10,132,255,0.12)" />` +
    `<polyline points="${pts}" fill="none" style="stroke:var(--blue)" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round" />` +
    `</svg>` +
    `<div class="marks">${dots}</div>` +
    `<div class="marks">${ticks}</div>` +
    `<div class="x-labels">${xLabels}</div>` +
    `</div></div></div>`
  );
}

// The Stats report for summary `s`, drawn from its per-line data `L`: the
// live session's, or a saved session's detail in History.
function reportBodyHtml(s: SessionSummary, L: Lines, pace: PaceData): string {
  const c = s.scores;
  const preset = PRESETS[s.preset];
  const prior = history.filter((h) => h.ts < s.ts);
  const prev = prior[prior.length - 1];
  const num = prior.length + 1;
  const title = speechTitle(s.speech);
  const hero = reportHeroHtml(s, prev, num, pace.lengthMs);
  const chart = paceChartHtml(pace, preset);
  const tiles = reportTilesHtml(s, prev);

  // Filler breakdown by word.
  const counts = new Map<string, number>();
  for (const ws of L.fillerWords.values()) for (const w of ws) counts.set(w, (counts.get(w) ?? 0) + 1);
  const rows = [...counts].sort((a, b) => b[1] - a[1]);
  const topN = rows[0]?.[1] ?? 1;
  const fillerRows = rows.length
    ? rows
        .map(([w, n]) => `<div class="fw-row" data-tip="“${escapeHtml(w)}” — ${n} of ${s.fillers} fillers this session"><span>${escapeHtml(w)}</span><div class="fw-bar"><div style="width:${(n / topN) * 100}%"></div></div><span class="n">${n}</span></div>`)
        .join("")
    : `<div class="empty">No fillers — clean session.</div>`;

  // Score trend: up to nine saved sessions plus this one — or, when reading a
  // saved speech, its last nine attempts.
  const pool = title ? prior.filter((h) => h.speech === s.speech) : prior;
  const shownPrior = pool.slice(-9);
  const bars = [
    ...shownPrior.map((h, i) => ({ n: pool.length - shownPrior.length + i + 1, score: h.scores.overall, now: false })),
    { n: pool.length + 1, score: c.overall, now: true },
  ];
  const what = title ? "Attempt" : "Session";
  const trend = bars.length > 1 ? `<span class="${tone(c.overall - bars[0].score)}">${signed(c.overall - bars[0].score)} since ${bars[0].n}</span>` : "";
  const hist = bars
    .map((b) => `<div class="hist-col${b.now ? " now" : ""}" data-tip="${what} ${b.n} — score ${b.score}"><span>${b.score}</span><div class="bar" style="height:${Math.min(96, Math.max(2, (b.score - 40) * 1.6))}%"></div><span>${b.n}</span></div>`)
    .join("");

  const moments = keyMoments(preset, L);
  const momentsHtml = moments.length
    ? moments
        .map((m) => `<button type="button" class="moment" data-index="${m.index}" data-ms="${m.ms}" data-tip="Jump to ${formatTimestamp(m.ms)} in the transcript"><span class="t">${formatTimestamp(m.ms)}</span><span class="bar" style="background:${m.color}"></span><span><b>${escapeHtml(m.title)}</b><span class="note">${escapeHtml(m.note)}</span></span></button>`)
        .join("")
    : `<div class="empty">Speak a little longer for highlights.</div>`;

  return (
    hero +
    chart +
    tiles +
    `<div class="rep-bottom">` +
    `<div class="panel"><span class="label">Fillers · ${s.fillers}</span><div class="fw-list">${fillerRows}</div></div>` +
    `<div class="panel"><div class="panel-head"><span class="label">${title ? `${escapeHtml(title)} · ${bars.length} attempt${bars.length === 1 ? "" : "s"}` : `Score · last ${bars.length}`}</span>${trend}</div><div class="hist">${hist}</div></div>` +
    `<div class="panel"><span class="label">Key moments</span><div class="moments">${momentsHtml}</div></div>` +
    `</div>`
  );
}

// The utterance (transcript line) at session time `ms`: the one spanning it,
// else the last one started before it (so a gap maps to the line it follows).
function lineAt(ms: number, L = live): number | undefined {
  let at: number | undefined;
  for (const [i, t] of [...L.timing].sort((a, b) => a[0] - b[0])) if (t.start <= ms) at = i;
  return at;
}

// How far across the report's pace plot the pointer is (0–1), and the session
// time there.
function plotFrac(plot: HTMLElement, clientX: number): number {
  const r = plot.getBoundingClientRect();
  return Math.max(0, Math.min(1, (clientX - r.left) / Math.max(1, r.width)));
}
const fracMs = (frac: number, pace: PaceData) => frac * Math.max(1, pace.wpm.length - 1) * 1000;

// Hover readout for the pace plot: pace at that second, plus the loudness,
// inflection, fillers and pauses of the line being spoken there — when the
// session's per-line data `L` is around.
function paceTip(ms: number, pace: PaceData, L?: Lines): string {
  const sec = Math.round(ms / 1000);
  const lines = [`${formatTimestamp(ms)} · ${pace.wpm[sec] ?? 0} wpm`];
  if (!L) return lines[0];
  const i = lineAt(ms, L);
  const t = i === undefined ? undefined : L.timing.get(i);
  const a = i === undefined ? undefined : L.analysis.get(i);
  const levels = [...L.analysis.values()].map((x) => x.rms_level).filter((x) => x > 0);
  if (a && a.rms_level > 0 && levels.length) lines.push(`Volume ${Math.round((a.rms_level / mean(levels)) * 100)}% of your average`);
  if (a) lines.push(a.f0_median > 0 ? `Pitch range ${a.f0_range_semitones.toFixed(1)} semitones${a.f0_terminal_rising ? " · rising end" : ""}` : "Pitch —");
  const fw = i === undefined ? [] : L.fillerWords.get(i) ?? [];
  lines.push(fw.length ? `Filler: ${fw.join(", ")}` : "No filler");
  // In a pause: inside a VAD gap of this line, or in a long gap after it.
  const inside = t && a?.pauses.find((p) => ms >= t.start + p.start_ms && ms <= t.start + p.end_ms);
  const next = i === undefined ? undefined : [...L.timing].filter(([k]) => k > i).sort((x, y) => x[0] - y[0])[0]?.[1];
  const gap = t && next && ms > t.end && next.start - t.end >= INTER_PAUSE_MS ? next.start - t.end : 0;
  const pauseMs = inside ? inside.end_ms - inside.start_ms : gap;
  lines.push(pauseMs ? `Pause ${(pauseMs / 1000).toFixed(1)} s` : a?.pause_count ? `${a.pause_count} pause${a.pause_count === 1 ? "" : "s"} in this line` : "No pause");
  return lines.join("\n");
}

// Pace plot / key moment → the matching line in that report's transcript copy.
function jumpInCopy(root: Element, index: number, ms: number, L: Lines) {
  const el = root.querySelector<HTMLElement>(`.rep-transcript [data-index="${index}"]`);
  if (el) flashLine(el, index, ms, L);
}

// Pace plot hover: moves the cursor and fills the tip. Runs before the
// document-level tooltip handler, so the tip reads fresh text.
function hoverPace(e: MouseEvent, pace: PaceData, L?: Lines) {
  const plot = (e.target as Element).closest<HTMLElement>(".pace-plot");
  if (!plot) return;
  const frac = plotFrac(plot, e.clientX);
  plot.dataset.tip = paceTip(fracMs(frac, pace), pace, L);
  const cursor = plot.querySelector<HTMLElement>(".pace-cursor");
  if (cursor) cursor.style.left = `${frac * 100}%`;
}

// Pace plot click → the line spoken there, in the report's transcript copy.
function clickPace(e: MouseEvent, root: Element, pace: PaceData, L: Lines) {
  const plot = (e.target as Element).closest<HTMLElement>(".pace-plot");
  if (!plot) return;
  const ms = fracMs(plotFrac(plot, e.clientX), pace);
  const i = lineAt(ms, L);
  if (i !== undefined) jumpInCopy(root, i, ms, L);
}

// Scrolls a transcript line into view, flashes it, and marks the word spoken at `ms`.
function flashLine(el: HTMLElement, index: number, ms: number, L = live) {
  el.scrollIntoView({ block: "center", behavior: "smooth" });
  el.classList.remove("flash");
  void el.offsetWidth; // restart the animation
  el.classList.add("flash");
  highlightWord(el, index, ms, L);
}

// Which of a line's `n` words was being spoken at session time `ms`. No word
// timestamps come from the decoder, so words are spread evenly across the
// line's voiced time (its duration minus the VAD pauses inside it).
// ponytail: even-spread estimate; enable whisper token timestamps if it drifts.
function wordIndexAt(index: number, ms: number, n: number, L = live): number {
  const t = L.timing.get(index);
  if (!t) return 0;
  let off = ms - t.start;
  let voiced = t.end - t.start;
  for (const p of L.analysis.get(index)?.pauses ?? []) {
    voiced -= p.end_ms - p.start_ms;
    off -= Math.max(0, Math.min(off, p.end_ms) - p.start_ms);
  }
  return Math.max(0, Math.min(n - 1, Math.floor((off / Math.max(1, voiced)) * n)));
}

// Wraps that word in a fading <mark>. Words can straddle text nodes (a filler
// span followed by punctuation), so a match glued to the previous node's last
// word continues it rather than starting a new one.
function highlightWord(el: HTMLElement, index: number, ms: number, L = live) {
  const p = el.querySelector(".seg-text");
  const n = p?.textContent?.match(/\S+/g)?.length ?? 0;
  if (!p || n === 0) return;
  const target = wordIndexAt(index, ms, n, L);
  const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
  let w = -1;
  let glued = false;
  for (let node = walker.nextNode() as Text | null; node; node = walker.nextNode() as Text | null) {
    for (const m of node.data.matchAll(/\S+/g)) {
      if (!(m.index === 0 && glued)) w++;
      if (w !== target) continue;
      const range = document.createRange();
      range.setStart(node, m.index);
      range.setEnd(node, m.index + m[0].length);
      const mark = document.createElement("mark");
      mark.className = "word-hit";
      range.surroundContents(mark);
      // Cancelled too (view hidden mid-fade), so no mark lingers into the report's copy.
      const unwrap = () => mark.isConnected && mark.replaceWith(...mark.childNodes);
      mark.addEventListener("animationend", unwrap);
      mark.addEventListener("animationcancel", unwrap);
      return;
    }
    glued = /\S$/.test(node.data);
  }
}

// Pages Tab cycles through, with their nav rail buttons.
const TAB_VIEWS: Array<[string, string]> = [
  ["live", "live-btn"],
  ["report", "stats-btn"],
  ["speeches", "speeches-btn"],
  ["history", "history-btn"],
];

function showView(view: "live" | "report" | "profile" | "speeches" | "history" | "settings") {
  document.body.dataset.view = view;
  if (view !== "settings") void stopMicTest(false);
  if (view === "profile") renderProfile();
  if (view === "speeches") renderSpeeches();
  if (view === "history") renderHistory();
}

// Speeches view: the list of saved speeches, or one speech's own page.
let speechPage = ""; // id of the open speech page; "" = the list. Persisted.
const SPEECH_PAGE_KEY = "speech.speechPage";
let speechMetric = "score";
const SPEECH_METRICS = ["score", "accuracy", "wpm", "fillers"];

function renderSpeeches() {
  const el = $("speeches");
  if (!el) return;
  const sp = speeches.find((x) => x.id === speechPage);
  if (!sp) speechPage = ""; // deleted since
  save(SPEECH_PAGE_KEY, speechPage);
  el.innerHTML = sp ? speechPageHtml(sp) : speechListHtml();
  const sort = $<HTMLSelectElement>("speech-sort");
  if (sort) enhanceSelect(sort);
}

// List controls. The color filter resets itself once no speech uses that color.
let speechQuery = "";
let speechFilter = ""; // color hex; "" = all
let speechSort = "recent";
const SPEECH_SORTS: Record<string, string> = {
  recent: "Last practiced",
  name: "Name",
  best: "Best score",
  attempts: "Most attempts",
  added: "Newest",
};

function speechListHtml(): string {
  const head =
    `<div class="panel-head"><span class="label">Your speeches</span>` +
    `<button type="button" class="ctl" data-new data-tip="Add a speech to practise">New speech</button></div>`;
  if (!speeches.length)
    return `<div class="panel">${head}<div class="empty">No saved speeches yet. Press New speech to add one.</div></div>`;
  const used = SPEECH_COLORS.filter(([c]) => speeches.some((sp) => speechColor(sp) === c));
  if (!used.some(([c]) => c === speechFilter)) speechFilter = "";
  const swatches =
    used.length > 1
      ? `<div class="swatches" role="group" aria-label="Filter by color">` +
        `<button type="button" class="ctl" data-cfilter="" aria-pressed="${!speechFilter}">All</button>` +
        used
          .map(([c, n]) => `<button type="button" class="swatch" style="--c:${c}" data-cfilter="${c}" aria-pressed="${speechFilter === c}" aria-label="${n}" data-tip="Only ${n.toLowerCase()} speeches"></button>`)
          .join("") +
        `</div>`
      : "";
  const sorts = Object.entries(SPEECH_SORTS)
    .map(([k, v]) => `<option value="${k}"${k === speechSort ? " selected" : ""}>${v}</option>`)
    .join("");
  return (
    `<div class="panel">${head}<div class="speech-filters">` +
    `<input id="speech-q" class="ctl" type="search" spellcheck="false" placeholder="Search names and scripts" aria-label="Search speeches" value="${escapeHtml(speechQuery)}" />` +
    `${swatches}<select id="speech-sort" aria-label="Sort by">${sorts}</select></div>` +
    `<table id="speech-table" class="prof-table">${speechRowsHtml()}</table></div>`
  );
}

// One row per speech that passes the search and color filter: attempts, best
// and latest score, and a sparkline of every attempt's score. A row opens
// that speech's page.
function speechRowsHtml(): string {
  const q = speechQuery.trim().toLowerCase();
  const rows = speeches
    .filter((sp) => !speechFilter || speechColor(sp) === speechFilter)
    .filter((sp) => !q || sp.title.toLowerCase().includes(q) || sp.text.toLowerCase().includes(q))
    .map((sp) => {
      const tries = history.filter((h) => h.speech === sp.id);
      const scores = tries.map((h) => h.scores.overall);
      const lastTs = tries.length ? tries[tries.length - 1].ts : Number(sp.id);
      return { sp, scores, best: scores.length ? Math.max(...scores) : -1, lastTs };
    });
  type Row = (typeof rows)[number];
  const orders: Record<string, (a: Row, b: Row) => number> = {
    recent: (a, b) => b.lastTs - a.lastTs,
    name: (a, b) => a.sp.title.localeCompare(b.sp.title),
    best: (a, b) => b.best - a.best,
    attempts: (a, b) => b.scores.length - a.scores.length,
    added: (a, b) => Number(b.sp.id) - Number(a.sp.id),
  };
  rows.sort(orders[speechSort] ?? orders.recent);
  const body = rows
    .map(({ sp, scores, best }) => {
      const color = speechColor(sp);
      const last = scores[scores.length - 1];
      const x = (i: number) => (scores.length > 1 ? (i / (scores.length - 1)) * 80 : 80);
      const y = (v: number) => 20 - (v / 100) * 20;
      const spark = scores.length
        ? `<svg class="speech-spark" viewBox="0 0 80 20" aria-hidden="true">` +
          `<polyline points="${scores.map((v, i) => `${x(i)},${y(v)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round" />` +
          `<circle cx="${x(scores.length - 1)}" cy="${y(last)}" r="2.5" fill="${color}" /></svg>`
        : `<span class="sub">no attempts</span>`;
      const lastCell = scores.length
        ? `${last} ${grade(last)}${scores.length > 1 ? ` <span class="sub">${signed(last - scores[0])}</span>` : ""}`
        : "—";
      return (
        `<tr data-open="${sp.id}" data-tip="Open ${escapeHtml(sp.title)}"><td><span class="speech-dot" style="--c:${color}"></span>${escapeHtml(sp.title)}</td>` +
        `<td>${scores.length}</td><td>${scores.length ? `${best} ${grade(best)}` : "—"}</td>` +
        `<td>${lastCell}</td><td>${spark}</td><td><div class="speech-actions">` +
        `<button type="button" class="ctl" data-practice="${sp.id}" data-tip="Load it into the read-along on the Live tab">Practice</button></div></td></tr>`
      );
    })
    .join("");
  return (
    `<tr><th>Speech</th><th>Attempts</th><th>Best</th><th>Last</th><th>Progress</th><th></th></tr>` +
    (body || `<tr><td colspan="6" class="empty">No speeches match.</td></tr>`)
  );
}

// Rule-based overview across a speech's attempts: the weakest areas on
// average (with the numbers behind them), what's improving or slipping from
// early to recent attempts, and the top fix from the latest attempt.
const DIMENSIONS: Array<[keyof Scores, string]> = [
  ["pace", "Pace"],
  ["fillers", "Filler words"],
  ["pauses", "Pauses"],
  ["pitch", "Vocal variety"],
  ["volume", "Volume"],
  ["articulation", "Script accuracy"],
];

// How often each word of the speech's current text was skipped. Only
// attempts that tracked skips against the same text (same word count) count,
// since indices from an older version of the script would point elsewhere.
interface SkipCounts {
  tokens: ScriptToken[];
  counts: number[];
  runs: number;
}
function skipCounts(sp: Speech, tries: SessionSummary[]): SkipCounts {
  const tokens = tokenizeWithOffsets(sp.text);
  const counts = new Array<number>(tokens.length).fill(0);
  let runs = 0;
  for (const h of tries) {
    if (!h.script?.missed || h.script.total !== tokens.length) continue;
    runs++;
    for (const i of h.script.missed) if (i < counts.length) counts[i]++;
  }
  return { tokens, counts, runs };
}

// The text from `from` to `to`, escaped, with skipped words shaded by how
// often they were skipped.
function shadeSkips(text: string, from: number, to: number, { tokens, counts, runs }: SkipCounts): string {
  let html = "";
  let pos = from;
  tokens.forEach((t, i) => {
    if (!counts[i] || t.start < from || t.end > to) return;
    html += escapeHtml(text.slice(pos, t.start));
    html += `<span class="skipped" style="--a:${(0.2 + 0.5 * (counts[i] / runs)).toFixed(2)}" data-tip="Skipped in ${counts[i]} of ${runs} attempt${runs === 1 ? "" : "s"}">${escapeHtml(text.slice(t.start, t.end))}</span>`;
    pos = t.end;
  });
  return html + escapeHtml(text.slice(pos, to));
}

function speechOverview(tries: SessionSummary[], sp: Speech, skips: SkipCounts): string[] {
  const n = tries.length;
  const latest = tries[n - 1];
  const avg = (f: (h: SessionSummary) => number | null | undefined, hs = tries) =>
    mean(hs.map(f).filter((v): v is number => typeof v === "number"));
  const preset = PRESETS[latest.preset] ?? PRESETS.conversation;
  const detail: Record<string, () => string> = {
    pace: () => {
      const w = Math.round(avg((h) => h.wpm));
      const dir = w > preset.wpmHigh ? "slow down" : w < preset.wpmLow ? "pick up the pace" : "hold it steady";
      return `you average ${w} wpm against a ${preset.wpmLow}–${preset.wpmHigh} target — ${dir}.`;
    },
    fillers: () => `${avg((h) => h.fillersPerMin).toFixed(1)} fillers a minute on average. Aim under 3 — pause silently where an "um" would go.`,
    pauses: () => `${avg((h) => h.pausesPerMin).toFixed(1)} hesitation pauses a minute. Rehearse the transitions between sections so the gaps close.`,
    pitch: () => `${avg((h) => h.pitchRange).toFixed(1)} semitones of inflection per sentence — under 3 sounds flat. Stress the key word in each line.`,
    volume: () => `sentence endings fade to ${Math.round(avg((h) => h.trailingOff) * 100)}% of your level. Carry energy through the last word.`,
    articulation: () => {
      const misses = Math.round(avg((h) => h.script?.misses));
      return `you match ${Math.round(avg((h) => h.script?.accuracy))}% of the script on average${misses ? `, skipping about ${misses} word${misses === 1 ? "" : "s"} a run` : ""}. Slow down on the passages you lose.`;
    },
  };

  const out: string[] = [];
  const dims = DIMENSIONS.filter(([k]) => tries.some((h) => typeof h.scores[k] === "number"));
  const weak = dims
    .map(([k, label]) => ({ k, label, v: avg((h) => h.scores[k]) }))
    .filter((d) => d.v < 80)
    .sort((a, b) => a.v - b.v)
    .slice(0, 2);
  for (const d of weak) out.push(`<b>${d.label}</b> <span class="sub">avg ${Math.round(d.v)}/100</span> — ${detail[d.k]()}`);
  if (!weak.length) out.push(`<b>No weak spots</b> — every area averages 80 or better across your attempts.`);

  for (const spot of troubleSpots(sp.text, skips.tokens, skips.counts))
    out.push(
      `<b>Trouble spot</b> <span class="sub">${spot.skips} skipped word${spot.skips === 1 ? "" : "s"} over ${skips.runs} attempt${skips.runs === 1 ? "" : "s"}</span> — ` +
        `“${shadeSkips(sp.text, spot.start, spot.end, skips)}” Read this passage aloud slowly a few times before your next run.`,
    );

  // Early vs recent: up to three attempts from each end, never overlapping.
  if (n >= 2) {
    const k = Math.min(3, Math.floor(n / 2));
    const early = tries.slice(0, k);
    const recent = tries.slice(-k);
    const d0 = Math.round(avg((h) => h.scores.overall, recent) - avg((h) => h.scores.overall, early));
    const moves = dims
      .map(([key, label]) => ({ label, d: avg((h) => h.scores[key], recent) - avg((h) => h.scores[key], early) }))
      .sort((a, b) => b.d - a.d);
    const up = moves[0];
    const down = moves[moves.length - 1];
    let trend = `<b>Trend</b> — your score is ${d0 > 0 ? `up ${d0}` : d0 < 0 ? `down ${-d0}` : "flat"} from your first ${k === 1 ? "attempt" : `${k} attempts`} to your latest.`;
    if (up && up.d >= 5) trend += ` Biggest gain: ${up.label.toLowerCase()} (${signed(up.d)}).`;
    if (down && down.d <= -5) trend += ` Slipping: ${down.label.toLowerCase()} (${signed(down.d)}).`;
    out.push(trend);
  }

  out.push(`<b>Last attempt</b> — ${escapeHtml(generateTips(latest)[0])}`);
  return out;
}

// A speech's own page: name (editable), headline numbers, a progress chart,
// every attempt, and the script itself.
function speechPageHtml(sp: Speech): string {
  const tries = history.filter((h) => h.speech === sp.id);
  const n = tries.length;
  const head =
    `<div class="panel prof-head"><div class="panel-head">` +
    `<button type="button" class="ctl" data-back>← Speeches</button><div class="speech-actions">` +
    `<button type="button" class="ctl" data-practice="${sp.id}" data-tip="Load it into the read-along on the Live tab">Practice</button>` +
    `<button type="button" class="ctl" data-delete="${sp.id}" data-tip="Delete this speech — its sessions stay in your history">Delete</button></div></div>` +
    `<button type="button" class="prof-name speech-name" data-rename data-tip="Rename or change color"><span class="speech-dot" style="--c:${speechColor(sp)}"></span>${escapeHtml(sp.title)}</button>` +
    `<span class="sub">${countWords(sp.text)} words · ${n} attempt${n === 1 ? "" : "s"} · added ${fmtDate(Number(sp.id))}</span></div>`;
  const skips = skipCounts(sp, tries);
  const script =
    `<div class="panel"><div class="panel-head"><span class="label">Script</span>` +
    (skips.runs ? `<span class="sub">shaded words: skipped in ${skips.runs} tracked attempt${skips.runs === 1 ? "" : "s"}</span>` : "") +
    `</div><p class="speech-text">${shadeSkips(sp.text, 0, sp.text.length, skips)}</p></div>`;
  if (!n) return head + `<div class="panel"><span class="sub">No attempts yet — press Practice to read it aloud.</span></div>` + script;

  const scores = tries.map((h) => h.scores.overall);
  const best = tries.reduce((a, h) => (h.scores.overall > a.scores.overall ? h : a));
  const last = scores[n - 1];
  const acc = tries.map((h) => h.script?.accuracy ?? 0);
  const tiles =
    `<div class="tiles">` +
    statTile(String(n), "attempts", `since ${fmtDate(tries[0].ts)}`, "Recordings read against this speech") +
    statTile(`${best.scores.overall} ${grade(best.scores.overall)}`, "best score", fmtDate(best.ts), "Highest delivery score on this speech") +
    statTile(`${last} ${grade(last)}`, "latest", n > 1 ? `${signed(last - scores[0])} since first` : "first attempt", "Score on your most recent attempt") +
    statTile(`${Math.max(...acc)}%`, "best accuracy", `latest ${acc[n - 1]}%`, "Share of the script's words you matched") +
    statTile(String(Math.round(mean(tries.map((h) => h.wpm)))), "avg wpm", `${mean(tries.map((h) => h.fillersPerMin)).toFixed(1)} fillers/min`, "Average pace across attempts") +
    `</div>`;

  const m = METRICS[speechMetric];
  const pts = m.scriptOnly ? tries.filter((h) => h.script) : tries;
  const segs = SPEECH_METRICS.map(
    (k) => `<button type="button" role="radio" data-smetric="${k}" aria-checked="${k === speechMetric}">${METRICS[k].name}</button>`,
  ).join("");
  const chart =
    `<div class="panel"><div class="panel-head"><span class="label">Progress · ${m.name}</span>` +
    `<div class="seg" role="radiogroup" aria-label="Metric">${segs}</div></div>${plotHtml(pts, m, pts[0]?.ts ?? Date.now(), Date.now())}</div>`;

  const rows = tries
    .map(
      (h, i) =>
        `<tr><td>${i + 1}</td><td>${fmtDate(h.ts)}</td><td>${formatTimestamp(h.durationMs)}</td><td>${Math.round(h.wpm)}</td>` +
        `<td>${h.fillersPerMin.toFixed(1)}</td><td>${h.script ? `${h.script.accuracy}%` : "—"}</td><td>${h.scores.overall} ${grade(h.scores.overall)}</td></tr>`,
    )
    .reverse()
    .join("");
  const table =
    `<div class="panel"><span class="label">Attempts</span><table class="prof-table">` +
    `<tr><th>#</th><th>Date</th><th>Speaking</th><th>WPM</th><th>Fillers/min</th><th>Accuracy</th><th>Score</th></tr>${rows}</table></div>`;
  const overview =
    `<div class="panel"><span class="label">What to work on</span><ul class="overview">` +
    speechOverview(tries, sp, skips).map((t) => `<li>${t}</li>`).join("") +
    `</ul></div>`;
  return head + tiles + overview + chart + table + script;
}

let ndColor = SPEECH_COLORS[0][0];
let ndFromLive = false;
let ndEditing = ""; // id of the speech being renamed; "" = creating one

function setNewSpeechColor(c: string) {
  ndColor = c;
  const box = $("nd-colors");
  if (box)
    box.innerHTML = SPEECH_COLORS.map(
      ([hex, name]) => `<button type="button" class="swatch" role="radio" style="--c:${hex}" data-ndcolor="${hex}" aria-checked="${hex === c}" aria-label="${name}" data-tip="${name}"></button>`,
    ).join("");
}

// The speech popup: creates a speech, or (with `edit`) renames and recolors
// one, in which case the script field is hidden.
function openSpeechDialog(text: string, fromLive: boolean, edit?: Speech) {
  const nd = $<HTMLDialogElement>("speech-dialog");
  const name = $<HTMLInputElement>("nd-name");
  const area = $<HTMLTextAreaElement>("nd-text");
  if (!nd || !name || !area) return;
  ndFromLive = fromLive;
  ndEditing = edit?.id ?? "";
  setText("nd-title", edit ? "Rename speech" : "New speech");
  area.required = !edit;
  area.hidden = !!edit;
  $("nd-load")?.toggleAttribute("hidden", !!edit);
  name.value = edit?.title ?? text.trim().split("\n")[0].trim().slice(0, 60);
  area.value = text;
  setNewSpeechColor(edit ? speechColor(edit) : SPEECH_COLORS[speeches.length % SPEECH_COLORS.length][0]);
  nd.returnValue = "";
  nd.showModal();
  (text || edit ? name : area).focus();
  if (edit) name.select();
}

function practiceSpeech(id: string) {
  openSpeech(id);
  showView("live");
  toggleScript(true);
}

// --- Profile & settings --------------------------------------------------------

// localStorage can throw (blocked storage); settings then just don't persist.
function load(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

// Custom dropdown over a native <select>. The select stays in the DOM (hidden)
// as the source of truth, so .value and "change" listeners keep working; call
// syncSelect after setting .value from code.
function enhanceSelect(sel: HTMLSelectElement) {
  const dd = document.createElement("div");
  dd.className = "dd";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "ctl dd-btn";
  btn.setAttribute("aria-haspopup", "listbox");
  btn.setAttribute("aria-expanded", "false");
  btn.dataset.name = sel.getAttribute("aria-label") ?? sel.closest(".set-row")?.querySelector("span")?.firstChild?.textContent ?? "";
  const list = document.createElement("div");
  list.className = "dd-list";
  list.setAttribute("role", "listbox");
  sel.replaceWith(dd);
  sel.hidden = true;
  dd.append(sel, btn, list);
  syncSelect(sel);

  const opts = () => [...list.querySelectorAll<HTMLElement>("[role=option]")];
  const open = () => {
    list.innerHTML = [...sel.options]
      .map((o, i) => `<div role="option" tabindex="-1" data-i="${i}" aria-selected="${i === sel.selectedIndex}">${escapeHtml(o.text)}</div>`)
      .join("");
    setDropdownOpen(dd, true);
    opts()[Math.max(0, sel.selectedIndex)]?.focus();
  };
  const pick = (i: number) => {
    if (i !== sel.selectedIndex) {
      sel.selectedIndex = i;
      syncSelect(sel);
      sel.dispatchEvent(new Event("change", { bubbles: true }));
    }
    setDropdownOpen(dd, false);
    if (btn.isConnected) btn.focus();
  };

  btn.addEventListener("click", () => (dd.classList.contains("open") ? setDropdownOpen(dd, false) : open()));
  list.addEventListener("click", (e) => {
    const o = (e.target as Element).closest<HTMLElement>("[role=option]");
    if (o) pick(Number(o.dataset.i));
  });
  dd.addEventListener("keydown", (e) => {
    const all = opts();
    const at = all.indexOf(document.activeElement as HTMLElement);
    if (at < 0) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        open();
      }
      return;
    }
    const move = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: all.length - 1 }[e.key];
    if (move !== undefined) all[Math.min(all.length - 1, Math.max(0, move))].focus();
    else if (e.key === "Enter" || e.key === " ") pick(at);
    else if (e.key === "Escape") {
      setDropdownOpen(dd, false);
      btn.focus();
    } else return;
    e.preventDefault();
  });
  // Tabbing out closes; pointer clicks outside are handled document-wide below.
  dd.addEventListener("focusout", (e) => {
    if (e.relatedTarget && !dd.contains(e.relatedTarget as Node)) setDropdownOpen(dd, false);
  });
}

function syncSelect(sel: HTMLSelectElement) {
  const btn = sel.parentElement?.querySelector<HTMLElement>(".dd-btn");
  if (!btn) return;
  btn.textContent = sel.selectedOptions[0]?.text ?? "";
  btn.setAttribute("aria-label", `${btn.dataset.name}: ${btn.textContent}`);
}

function setDropdownOpen(dd: Element, on: boolean) {
  dd.classList.toggle("open", on);
  dd.querySelector(".dd-btn")?.setAttribute("aria-expanded", String(on));
}

document.addEventListener("pointerdown", (e) => {
  document.querySelectorAll(".dd.open").forEach((dd) => {
    if (!dd.contains(e.target as Node)) setDropdownOpen(dd, false);
  });
});

const NAME_KEY = "speech.name";
const THEME_KEY = "speech.theme";
const DRILL_KEY = "speech.drillSecs";
const GAIN_KEY = "speech.levelGain";
const PAUSE_KEY = "speech.pauseMs";
const HOVER_KEY = "speech.hoverFx";
const COUNTDOWN_KEY = "speech.timerCountdown";
const CORRECT_KEY = "speech.accurateCorrection";
let accurateCorrection = true;

// Whisper models (see models.rs). tiny.en and medium.en are bundled, so they're
// the fallbacks when a chosen model is deleted; the rest download on demand.
const LIVE_MODEL_KEY = "speech.liveModel";
const CORRECT_MODEL_KEY = "speech.correctionModel";
let liveModel = "tiny.en";
let correctionModel = "medium.en";
type ModelInfo = { id: string; size_mb: number; installed: boolean; bundled: boolean };
let models: ModelInfo[] = [];
const MODEL_NOTES: Record<string, string> = {
  "tiny.en": "Fastest, roughest",
  "base.en": "Fast, a little cleaner",
  "small.en": "Balanced",
  "medium.en": "Accurate",
  "large-v3-turbo": "Most accurate, slowest",
};
const downloading = new Map<string, number>(); // id → percent done
const modelErrors = new Map<string, string>(); // id → last download failure

function renderModelChain() {
  setText("model-chain", accurateCorrection ? `${liveModel} → ${correctionModel}` : liveModel);
}

// Swaps the backend's live model. It loads and warms the new one before
// switching, so the old model keeps serving until then.
async function setLiveModel(id: string) {
  liveModel = id;
  save(LIVE_MODEL_KEY, id);
  renderModelChain();
  try {
    await invoke("set_live_model", { id });
  } catch (e) {
    appendError(String(e));
  }
}

// Fills both model pickers with installed models and redraws the Models list.
async function refreshModels() {
  models = await invoke<ModelInfo[]>("list_models");
  const installed = models.filter((m) => m.installed);
  if (!installed.some((m) => m.id === liveModel)) void setLiveModel("tiny.en");
  if (!installed.some((m) => m.id === correctionModel)) correctionModel = "medium.en";
  for (const [id, value] of [["set-live-model", liveModel], ["set-correct-model", correctionModel]]) {
    const sel = $<HTMLSelectElement>(id);
    if (!sel) continue;
    sel.innerHTML = installed.map((m) => `<option value="${m.id}">${m.id}</option>`).join("");
    sel.value = value;
    syncSelect(sel);
  }
  renderModelChain();
  const list = $("model-list");
  if (!list) return;
  list.innerHTML = models
    .map((m) => {
      const pct = downloading.get(m.id);
      const action = m.bundled
        ? `<span class="sub">Bundled</span>`
        : pct !== undefined
          ? `<button class="ctl" type="button" data-progress="${m.id}" disabled>${pct}%</button>`
          : m.installed
            ? `<button class="ctl" type="button" data-delete="${m.id}">Delete</button>`
            : `<button class="ctl" type="button" data-download="${m.id}">${modelErrors.has(m.id) ? "Retry" : "Download"}</button>`;
      const note = modelErrors.has(m.id) ? `Download failed: ${escapeHtml(modelErrors.get(m.id)!)}` : `${MODEL_NOTES[m.id] ?? ""} · ${m.size_mb} MB`;
      return `<div class="set-row"><span>${m.id}<span class="sub">${note}</span></span>${action}</div>`;
    })
    .join("");
}

async function downloadModel(id: string) {
  downloading.set(id, 0);
  modelErrors.delete(id);
  void refreshModels();
  try {
    await invoke("download_model", { id });
  } catch (e) {
    modelErrors.set(id, String(e));
  }
  downloading.delete(id);
  await refreshModels();
}

// Consecutive calendar days with at least one session, ending today (or
// yesterday, so the streak survives until you've had a chance to practise today).
function dayStreak(timestamps: number[], now = Date.now()): number {
  const days = new Set(timestamps.map((ts) => new Date(ts).toDateString()));
  const d = new Date(now);
  if (!days.has(d.toDateString())) d.setDate(d.getDate() - 1);
  let n = 0;
  while (days.has(d.toDateString())) {
    n++;
    d.setDate(d.getDate() - 1);
  }
  return n;
}

function fmtHours(ms: number): string {
  if (ms < 3_600_000) return formatTimestamp(ms);
  const m = Math.round(ms / 60_000);
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

const DAY_MS = 86_400_000;

// Trend chart metrics (profile). `max` pins the y-axis top; otherwise it's the
// data max rounded up to a nice number. Every axis starts at zero.
// `scriptOnly` plots only sessions read against a script.
const METRICS: Record<string, { name: string; get: (h: SessionSummary) => number; fmt: (v: number) => string; max?: number; scriptOnly?: boolean }> = {
  score: { name: "Score", get: (h) => h.scores.overall, fmt: (v) => String(Math.round(v)), max: 100 },
  wpm: { name: "Pace (wpm)", get: (h) => h.wpm, fmt: (v) => `${Math.round(v)} wpm` },
  fillers: { name: "Fillers/min", get: (h) => h.fillersPerMin, fmt: (v) => `${v.toFixed(1)}/min` },
  pauses: { name: "Pauses/min", get: (h) => h.pausesPerMin, fmt: (v) => `${v.toFixed(1)}/min` },
  pitch: { name: "Pitch range", get: (h) => h.pitchRange, fmt: (v) => `${v.toFixed(1)} st` },
  speaking: { name: "Speaking time", get: (h) => h.durationMs / 60_000, fmt: (v) => formatTimestamp(v * 60_000) },
  words: { name: "Words", get: (h) => h.words, fmt: (v) => `${Math.round(v)} words` },
  accuracy: { name: "Script accuracy", get: (h) => h.script?.accuracy ?? 0, fmt: (v) => `${Math.round(v)}%`, max: 100, scriptOnly: true },
};
const RANGES: Array<[string, number]> = [["7D", 7], ["30D", 30], ["90D", 90], ["1Y", 365], ["All", Infinity]];
let profMetric = "score";
let profRange = "All";
let profSpeech = ""; // "" = all sessions, else a saved speech id

function niceCeil(v: number): number {
  if (v <= 0) return 1;
  const step = 10 ** Math.floor(Math.log10(v));
  return Math.ceil(v / step) * step;
}

// All-time chart: one point per session, on a real time axis for the range.
function trendHtml(): string {
  const m = METRICS[profMetric];
  const days = RANGES.find((r) => r[0] === profRange)?.[1] ?? Infinity;
  const now = Date.now();
  if (!speechTitle(profSpeech)) profSpeech = ""; // deleted since
  const pts = history.filter(
    (h) => now - h.ts <= days * DAY_MS && (!profSpeech || h.speech === profSpeech) && (!m.scriptOnly || h.script),
  );
  const t0 = days === Infinity ? (pts[0]?.ts ?? now) : now - days * DAY_MS;
  const ranges = RANGES.map(([r]) => `<button type="button" role="radio" data-range="${r}" aria-checked="${r === profRange}">${r}</button>`).join("");
  const opts = Object.entries(METRICS)
    .map(([k, v]) => `<option value="${k}"${k === profMetric ? " selected" : ""}>${v.name}</option>`)
    .join("");
  const speechSel = speeches.length
    ? `<select id="prof-speech" class="ctl" aria-label="Speech"><option value="">All sessions</option>` +
      speeches.map((sp) => `<option value="${sp.id}"${sp.id === profSpeech ? " selected" : ""}>${escapeHtml(sp.title)}</option>`).join("") +
      `</select>`
    : "";
  return (
    `<div class="panel"><div class="panel-head"><span class="label">${escapeHtml(speechTitle(profSpeech) ?? "All time")} · ${m.name}</span>` +
    `<div class="trend-filters">${speechSel}<select id="prof-metric" class="ctl" aria-label="Metric">${opts}</select>` +
    `<div class="seg" role="radiogroup" aria-label="Time frame">${ranges}</div></div></div>${plotHtml(pts, m, t0, now)}</div>`
  );
}

// One point per session on a time axis from t0 to now.
function plotHtml(pts: SessionSummary[], m: (typeof METRICS)[string], t0: number, now: number): string {
  const span = Math.max(1, now - t0);
  const hi = m.max ?? niceCeil(Math.max(0, ...pts.map(m.get)));
  const x = (ts: number) => ((ts - t0) / span) * 100;
  const y = (v: number) => 100 - (Math.min(v, hi) / hi) * 100;
  const line = pts.map((h) => `${x(h.ts).toFixed(2)},${y(m.get(h)).toFixed(2)}`).join(" ");
  const dots = pts
    .map(
      (h) =>
        `<div class="pt" style="left:${x(h.ts).toFixed(2)}%;top:${y(m.get(h)).toFixed(2)}%" data-tip="${fmtDate(h.ts)} · ${escapeHtml(PRESETS[h.preset]?.name ?? String(h.preset))} — ${m.fmt(m.get(h))}"></div>`,
    )
    .join("");
  const yLabels = [hi, hi / 2, 0].map((v) => `<span>${m.fmt(v)}</span>`).join("");
  const xLabels = [t0, t0 + span / 2, now].map((t) => `<span>${fmtDate(t)}</span>`).join("");
  return pts.length
    ? `<div class="pace-grid trend-grid"><div class="y-labels">${yLabels}</div><div class="pace-plot">` +
      `<div class="trend-plot${pts.length > 60 ? " dense" : ""}"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">` +
      `<line x1="0" y1="50" x2="100" y2="50" stroke="currentColor" stroke-opacity="0.06" vector-effect="non-scaling-stroke" />` +
      `<line x1="0" y1="100" x2="100" y2="100" stroke="currentColor" stroke-opacity="0.1" vector-effect="non-scaling-stroke" />` +
      `<polyline points="${line}" fill="none" style="stroke:var(--blue)" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round" />` +
      `</svg>${dots}</div><div class="x-labels">${xLabels}</div></div></div>`
    : `<div class="empty">No sessions in this range.</div>`;
}

// GitHub-style grid: 53 weeks of days (columns = weeks, rows = Sun–Sat),
// shaded by speaking time relative to your biggest day.
function activityHtml(): string {
  const byDay = new Map<string, { n: number; ms: number }>();
  for (const h of history) {
    const k = new Date(h.ts).toDateString();
    const d = byDay.get(k) ?? { n: 0, ms: 0 };
    d.n++;
    d.ms += h.durationMs;
    byDay.set(k, d);
  }
  const maxMs = Math.max(1, ...[...byDay.values()].map((d) => d.ms));
  const today = new Date();
  const d = new Date(today.getFullYear(), today.getMonth(), today.getDate() - today.getDay() - 52 * 7);
  let cells = "";
  let months = "";
  let active = 0;
  for (let i = 0; d <= today; i++, d.setDate(d.getDate() + 1)) {
    const a = byDay.get(d.toDateString());
    const col = Math.floor(i / 7) + 2; // column 1 holds the weekday labels
    const lvl = a ? Math.max(1, Math.ceil((a.ms / maxMs) * 4)) : 0;
    if (a) active++;
    const tip = a
      ? `${fmtDate(d.getTime())} — ${a.n} session${a.n === 1 ? "" : "s"} · ${formatTimestamp(a.ms)} speaking`
      : `${fmtDate(d.getTime())} — no practice`;
    cells += `<div class="day l${lvl}" style="grid-area:${d.getDay() + 1}/${col}" data-tip="${tip}"></div>`;
    if (d.getDate() === 1) months += `<span style="grid-column:${col}">${d.toLocaleDateString(undefined, { month: "short" })}</span>`;
  }
  const days = ["Mon", "Wed", "Fri"].map((w, i) => `<span style="grid-area:${i * 2 + 2}/1">${w}</span>`).join("");
  const legend = [0, 1, 2, 3, 4].map((l) => `<div class="day l${l}"></div>`).join("");
  return (
    `<div class="panel"><div class="panel-head"><span class="label">Activity · ${active} day${active === 1 ? "" : "s"} in the last year</span>` +
    `<div class="act-legend">Less${legend}More</div></div>` +
    `<div class="act-months">${months}</div><div class="act-grid">${days}${cells}</div></div>`
  );
}

function statTile(value: string, unit: string, sub: string, tip: string): string {
  return `<div class="tile" data-tip="${escapeHtml(tip)}"><div class="card-top"><span class="big">${value}</span><span class="unit">${unit}</span></div><span class="delta flat">${sub}</span></div>`;
}

// What a session was read against: a saved speech (links to its page), an
// unsaved script, or nothing (free speaking). Script sessions show accuracy.
function practisedCell(h: SessionSummary): string {
  const acc = h.script ? ` <span class="sub">${h.script.accuracy}%</span>` : "";
  const sp = speeches.find((x) => x.id === h.speech);
  if (sp)
    return `<button type="button" class="link-btn" data-goto="${sp.id}" data-tip="Open this speech"><span class="speech-dot" style="--c:${speechColor(sp)}"></span>${escapeHtml(sp.title)}</button>${acc}`;
  return h.script ? `Script${acc}` : `<span class="sub">Free</span>`;
}

function renderProfile() {
  const body = $("prof-body");
  if (!body) return;
  const n = history.length;
  setText("prof-since", n ? `Practising since ${fmtDate(history[0].ts)} · ${n} session${n === 1 ? "" : "s"}` : "");
  if (!n) {
    body.innerHTML = `<div class="panel"><span class="sub">No sessions yet — press ${keyName(recordKey)} on the Live tab to record your first.</span></div>`;
    return;
  }
  const scores = history.map((h) => h.scores.overall);
  const avg = Math.round(mean(scores));
  const tile = statTile;
  const speakingMs = history.reduce((a, h) => a + h.durationMs, 0);
  const words = history.reduce((a, h) => a + h.words, 0);
  const tiles =
    `<div class="tiles">` +
    tile(fmtHours(speakingMs), "speaking", `${formatTimestamp(speakingMs / n)} per session`, "Total time spent actually talking, pauses excluded") +
    tile(`~${words.toLocaleString()}`, "words", `${Math.round(words / Math.max(1, speakingMs / 60_000))} wpm lifetime`, "Estimated words spoken — counted from the on-device transcript") +
    tile(String(dayStreak(history.map((h) => h.ts))), "day streak", `${n} sessions`, "Consecutive days with at least one session") +
    tile(`${avg} ${grade(avg)}`, "avg score", "all sessions", "Mean delivery score across all sessions") +
    tile(Math.min(...history.map((h) => h.fillersPerMin)).toFixed(1), "fillers/min", "personal best", "Lowest filler rate in a session") +
    `</div>`;
  const bests = Object.entries(PRESETS)
    .map(([key, p]) => {
      const hs = history.filter((h) => h.preset === key);
      if (!hs.length) return tile("—", p.name, "no sessions yet", `No ${p.name.toLowerCase()} sessions yet`);
      const b = hs.reduce((a, h) => (h.scores.overall > a.scores.overall ? h : a));
      return tile(`${b.scores.overall} ${grade(b.scores.overall)}`, p.name, `best of ${hs.length} · ${fmtDate(b.ts)}`, `Best ${p.name.toLowerCase()} score`);
    })
    .join("");
  const rows = history
    .slice(-10)
    .reverse()
    .map(
      (h) =>
        `<tr><td>${fmtDate(h.ts)}</td><td>${practisedCell(h)}</td><td>${escapeHtml(PRESETS[h.preset]?.name ?? String(h.preset))}</td><td>${formatTimestamp(h.durationMs)}</td>` +
        `<td>${Math.round(h.wpm)}</td><td>${h.fillersPerMin.toFixed(1)}</td><td>${h.scores.overall} ${grade(h.scores.overall)}</td></tr>`,
    )
    .join("");
  body.innerHTML =
    tiles +
    `<div class="panel"><span class="label">Best by mode</span><div class="prof-tiles">${bests}</div></div>` +
    activityHtml() +
    trendHtml() +
    `<div class="panel"><span class="label">Recent sessions</span><table class="prof-table">` +
    `<tr><th>Date</th><th>Practised</th><th>Context</th><th>Speaking</th><th>WPM</th><th>Fillers/min</th><th>Score</th></tr>${rows}</table></div>`;
  for (const id of ["prof-speech", "prof-metric"]) {
    const sel = $<HTMLSelectElement>(id);
    if (sel) enhanceSelect(sel);
  }
}

// --- History ---------------------------------------------------------------
// Every saved session, newest first, filtered; a row opens that session's
// stats. Only the summary is stored, so its page has no pace plot or transcript.
let histPage = 0; // ts of the open session; 0 = the list
let histSaved = false; // starred only
let histPreset = ""; // "" = all contexts
let histSpeech = ""; // "" = all, "free" = no script, else a speech id
let histRange = "All";
let histDetail: Detail | null = null; // the open session's per-line data, if still kept

async function openHistorySession(ts: number) {
  histPage = ts;
  if (histDetail?.ts !== ts) {
    histDetail = null;
    try {
      const json = await invoke<string | null>("load_session_detail", { ts });
      if (json && histPage === ts) histDetail = parseDetail(ts, json);
    } catch (e) {
      appendError(`Couldn't load session: ${e}`);
    }
  }
  if (histPage === ts) renderHistory();
}

// The open session's pace series and per-line data, for the chart's hover/click.
function histPace(): { pace: PaceData; L?: Lines } | null {
  const s = history.find((h) => h.ts === histPage);
  const L = histDetail?.ts === histPage ? histDetail.L : undefined;
  const pace = s?.pace ?? (L && paceData(L));
  return pace ? { pace, L } : null;
}

function renderHistory() {
  const el = $("history");
  if (!el) return;
  const i = history.findIndex((h) => h.ts === histPage);
  if (i < 0) histPage = 0; // cleared since
  el.innerHTML = i < 0 ? historyListHtml() : historyPageHtml(i);
  for (const id of ["hist-preset", "hist-speech"]) {
    const sel = $<HTMLSelectElement>(id);
    if (sel) enhanceSelect(sel);
  }
}

function historyListHtml(): string {
  if (!history.length)
    return `<div class="panel"><span class="label">History</span><div class="empty">No sessions yet — press ${keyName(recordKey)} on the Live tab to record your first.</div></div>`;
  if (histSpeech && histSpeech !== "free" && !speechTitle(histSpeech)) histSpeech = ""; // deleted since
  const days = RANGES.find((r) => r[0] === histRange)?.[1] ?? Infinity;
  const now = Date.now();
  const rows = history
    .filter(
      (h) =>
        (!histSaved || h.saved) &&
        (!histPreset || h.preset === histPreset) &&
        (!histSpeech || (histSpeech === "free" ? !h.script : h.speech === histSpeech)) &&
        now - h.ts <= days * DAY_MS,
    )
    .reverse();
  const opt = (v: string, label: string, cur: string) => `<option value="${v}"${v === cur ? " selected" : ""}>${escapeHtml(label)}</option>`;
  const presetSel =
    `<select id="hist-preset" class="ctl" aria-label="Context">${opt("", "All contexts", histPreset)}` +
    Object.entries(PRESETS).map(([k, p]) => opt(k, p.name, histPreset)).join("") +
    `</select>`;
  const speechSel =
    `<select id="hist-speech" class="ctl" aria-label="Practised">${opt("", "Anything practised", histSpeech)}${opt("free", "Free speaking", histSpeech)}` +
    speeches.map((sp) => opt(sp.id, sp.title, histSpeech)).join("") +
    `</select>`;
  const seg = (attr: string, items: Array<[string, string, boolean]>, label: string) =>
    `<div class="seg" role="radiogroup" aria-label="${label}">` +
    items.map(([v, text, on]) => `<button type="button" role="radio" ${attr}="${v}" aria-checked="${on}">${text}</button>`).join("") +
    `</div>`;
  const filters =
    `<div class="speech-filters">` +
    seg("data-hsaved", [["0", "All", !histSaved], ["1", "★ Saved", histSaved]], "Saved") +
    presetSel +
    speechSel +
    seg("data-range", RANGES.map(([r]) => [r, r, r === histRange]), "Time frame") +
    `</div>`;
  const body = rows
    .map(
      (h) =>
        `<tr data-open="${h.ts}" data-tip="Open this session's stats"><td>${starHtml(h.ts)}</td>` +
        `<td>${new Date(h.ts).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</td>` +
        `<td>${practisedCell(h)}</td><td>${escapeHtml(PRESETS[h.preset]?.name ?? String(h.preset))}</td><td>${formatTimestamp(h.durationMs)}</td>` +
        `<td>${Math.round(h.wpm)}</td><td>${h.fillersPerMin.toFixed(1)}</td><td>${h.scores.overall} ${grade(h.scores.overall)}</td></tr>`,
    )
    .join("");
  return (
    `<div class="panel"><div class="panel-head"><span class="label">History · ${rows.length} of ${history.length} session${history.length === 1 ? "" : "s"}</span></div>${filters}` +
    `<table class="prof-table"><tr><th></th><th>Date</th><th>Practised</th><th>Context</th><th>Speaking</th><th>WPM</th><th>Fillers/min</th><th>Score</th></tr>` +
    (body || `<tr><td colspan="8" class="empty">No sessions match.</td></tr>`) +
    `</table></div>`
  );
}

// One saved session. Recent and starred ones keep their per-line data and get
// the full Stats report; older ones get what the summary holds: the report's
// hero, pace chart and tiles, the sub-score breakdown and tips.
function historyPageHtml(i: number): string {
  const s = history[i];
  const back = `<div class="panel"><div class="panel-head"><button type="button" class="ctl" data-back>← History</button></div></div>`;
  if (histDetail?.ts === s.ts)
    return (
      back +
      `<div class="hist-report">${reportBodyHtml(s, histDetail.L, s.pace ?? paceData(histDetail.L))}</div>` +
      `<section class="panel rep-transcript">${transcriptCopyHtml(histDetail.lines)}</section>`
    );
  const dims = DIMENSIONS.filter(([k]) => typeof s.scores[k] === "number")
    .map(([k, name]) => {
      const v = s.scores[k] ?? 0;
      return `<div class="fw-row" data-tip="${name} — ${v} of 100"><span>${name}</span><div class="fw-bar"><div style="width:${v}%"></div></div><span class="n">${v}</span></div>`;
    })
    .join("");
  return (
    back +
    reportHeroHtml(s, history[i - 1], i + 1, s.pace?.lengthMs ?? s.durationMs) +
    (s.pace ? paceChartHtml(s.pace, PRESETS[s.preset]) : "") +
    reportTilesHtml(s, history[i - 1]) +
    `<div class="panel"><span class="label">Score breakdown</span><div class="fw-list dims">${dims}</div></div>` +
    `<div class="panel"><span class="label">What to work on</span><ul class="overview">` +
    generateTips(s).map((t) => `<li>${escapeHtml(t)}</li>`).join("") +
    `</ul></div>`
  );
}

// Report → transcript: switch back to the live view and flash the line.
function jumpToLine(index: number, ms: number) {
  showView("live");
  toggleScript(false);
  const el = segmentEls.get(index);
  if (el) flashLine(el, index, ms);
}

// --- Practice ----------------------------------------------------------------

const PROMPTS = [
  "Describe your ideal weekend in detail.",
  "Explain how to make your favourite meal.",
  "Argue for or against remote work.",
  "Tell the story of a time you overcame a challenge.",
  "Describe a place that means a lot to you and why.",
  "Pitch your favourite app to someone who's never used it.",
  "Explain a complex topic you know well to a 10-year-old.",
  "What would you change about your city, and how?",
  "Describe your morning routine step by step.",
  "Convince a friend to try your favourite hobby.",
  "Summarise a book or film you enjoyed recently.",
  "What advice would you give your younger self?",
];

let drillMs = 0;

window.addEventListener("DOMContentLoaded", () => {
  const ribbonCanvas = document.querySelector<HTMLCanvasElement>("#ribbon");
  if (ribbonCanvas) ribbon = new Ribbon(ribbonCanvas, "idle");
  const shardsCanvas = document.querySelector<HTMLCanvasElement>("#shards");
  if (shardsCanvas) new Shards(shardsCanvas, shardsCanvas.parentElement!);
  // The scope drags the window only once a held pointer actually moves.
  // data-tauri-drag-region starts a WM move on every press and maximizes on
  // double-click, so spam-clicking the scope's click effect stalled the app.
  const scope = shardsCanvas?.parentElement;
  let pressAt: { x: number; y: number } | null = null;
  scope?.addEventListener("pointerdown", (e) => {
    pressAt = e.button === 0 && (e.target === scope || (e.target as Element).matches(".ribbon, .countdown")) ? { x: e.clientX, y: e.clientY } : null;
  });
  scope?.addEventListener("pointerup", () => (pressAt = null));
  scope?.addEventListener("pointerleave", () => (pressAt = null));
  scope?.addEventListener("pointermove", (e) => {
    if (pressAt && e.buttons === 1 && Math.hypot(e.clientX - pressAt.x, e.clientY - pressAt.y) > 4) {
      pressAt = null;
      getCurrentWindow().startDragging();
    }
  });
  const sparkEl = $("pace-spark");
  if (sparkEl) new ResizeObserver(() => drawSpark()).observe(sparkEl);

  recordBtn = $<HTMLButtonElement>("record-btn");
  statusEl = $("status");
  transcriptEl = $("transcript");

  scriptInput = document.querySelector("#script-input");
  scriptFile = document.querySelector("#script-file");
  scriptDisplay = document.querySelector("#script-display");
  scriptProgress = document.querySelector("#script-progress");

  initTooltips();

  recordBtn?.addEventListener("click", toggleRecording);
  $("reset-btn")?.addEventListener("click", resetSession);
  $("script-btn")?.addEventListener("click", () => toggleScript());
  $("stats-btn")?.addEventListener("click", () => showView("report"));
  $("live-btn")?.addEventListener("click", () => showView("live"));
  $("profile-btn")?.addEventListener("click", () => showView("profile"));
  $("settings-btn")?.addEventListener("click", () => showView("settings"));

  // Hotkey rebinding: the button arms capture, the next non-modifier key
  // becomes the binding (Escape cancels).
  const bindBtn = $<HTMLButtonElement>("set-record-key");
  let capturing = false;
  const syncBind = () => {
    if (bindBtn) bindBtn.textContent = capturing ? "Press a key…" : keyName(recordKey);
    setRecordingUi(recording);
  };
  recordKey = load(RECORD_KEY) || recordKey;
  syncBind();
  bindBtn?.addEventListener("click", () => {
    capturing = true;
    syncBind();
  });
  bindBtn?.addEventListener("blur", () => {
    capturing = false;
    syncBind();
  });

  // The hotkey starts/stops recording from anywhere except text fields. Blur first
  // so a focused button doesn't also get "clicked" by the same keypress on keyup.
  document.addEventListener("keydown", (e) => {
    if (capturing) {
      if (["Shift", "Control", "Alt", "Meta"].includes(e.key)) return;
      e.preventDefault();
      if (e.code && e.code !== "Escape") {
        recordKey = e.code;
        save(RECORD_KEY, recordKey);
      }
      bindBtn?.blur(); // ends capture via the blur handler
      return;
    }
    // Esc on a speech's page goes back to the list (the popup and open
    // dropdowns handle their own Esc).
    if (
      e.key === "Escape" &&
      document.body.dataset.view === "speeches" &&
      speechPage &&
      !(e.target as Element).closest("dialog, .dd")
    ) {
      speechPage = "";
      renderSpeeches();
      return;
    }
    // Tab / Shift+Tab step through Live, Stats (when enabled), Speeches and History.
    // Text fields and the popup keep Tab for moving focus. Matched on
    // e.code: GTK reports Shift+Tab's key as "ISO_Left_Tab".
    if (
      e.code === "Tab" &&
      !e.ctrlKey && !e.altKey && !e.metaKey &&
      !(e.target as Element).closest("input:not([type=checkbox], [type=range]), textarea, dialog, [contenteditable]")
    ) {
      e.preventDefault();
      const tabs = TAB_VIEWS.filter(([, id]) => !$<HTMLButtonElement>(id)?.disabled);
      const at = tabs.findIndex(([v]) => v === document.body.dataset.view);
      const n = tabs.length;
      // From Profile/Settings, Tab enters at the first page, Shift+Tab at the last.
      const next = at < 0 ? (e.shiftKey ? n - 1 : 0) : (at + (e.shiftKey ? -1 : 1) + n) % n;
      $(tabs[next][1])?.click();
      return;
    }
    if (e.code !== recordKey || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    // Starts only from the live tab; a running recording can be stopped from anywhere.
    if (!recording && document.body.dataset.view !== "live") return;
    if ((e.target as Element).closest("input, textarea, select, .dd, .timer-menu, dialog, [contenteditable]")) return;
    e.preventDefault();
    (document.activeElement as HTMLElement | null)?.blur();
    if (!recordBtn?.disabled) void toggleRecording();
  });

  const profBody = $("prof-body");
  profBody?.addEventListener("click", (e) => {
    const go = (e.target as Element).closest<HTMLElement>("[data-goto]")?.dataset.goto;
    if (go) {
      speechPage = go;
      return showView("speeches");
    }
    const r = (e.target as Element).closest<HTMLElement>("[data-range]")?.dataset.range;
    if (r) {
      profRange = r;
      renderProfile();
    }
  });
  profBody?.addEventListener("change", (e) => {
    const t = e.target as HTMLSelectElement;
    if (t.id === "prof-metric") profMetric = t.value;
    else if (t.id === "prof-speech") profSpeech = t.value;
    else return;
    renderProfile();
  });

  const nameInput = $<HTMLInputElement>("prof-name");
  if (nameInput) {
    nameInput.value = load(NAME_KEY) ?? "";
    nameInput.addEventListener("input", () => save(NAME_KEY, nameInput.value));
  }

  document.querySelectorAll<HTMLSelectElement>(".view-settings select").forEach(enhanceSelect);

  const themeSel = $<HTMLSelectElement>("set-theme");
  const applyTheme = (t: string) => {
    document.documentElement.dataset.theme = t;
    if (themeSel) {
      themeSel.value = t;
      syncSelect(themeSel);
    }
  };
  applyTheme(load(THEME_KEY) ?? "graphite");
  themeSel?.addEventListener("change", () => {
    save(THEME_KEY, themeSel.value);
    applyTheme(themeSel.value);
  });

  drillMs = (Number(load(DRILL_KEY)) || 0) * 1000;
  const syncDrill = tickClock;
  // Clicking the timer opens a length picker. The picked length applies to
  // every recording started afterwards; Off records until stopped.
  const timerMenu = $("timer-menu");
  const timerCustom = $<HTMLInputElement>("timer-custom");
  const closeTimerMenu = () => timerMenu?.setAttribute("hidden", "");
  const setDrill = (secs: number) => {
    if (!(secs >= 0)) return;
    drillMs = secs && Math.round(Math.min(3600, Math.max(5, secs))) * 1000;
    save(DRILL_KEY, String(drillMs / 1000));
    syncDrill();
    closeTimerMenu();
  };
  $("drill-btn")?.addEventListener("click", (e) => {
    if (!timerMenu) return;
    if (!timerMenu.hidden) return closeTimerMenu();
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    timerMenu.style.left = `${r.left}px`;
    timerMenu.style.top = `${r.bottom + 6}px`;
    timerMenu.querySelectorAll<HTMLElement>("[data-s]").forEach((b) =>
      b.setAttribute("aria-current", String(Number(b.dataset.s) * 1000 === drillMs)),
    );
    if (timerCustom) timerCustom.value = "";
    timerMenu.removeAttribute("hidden");
    timerMenu.querySelector<HTMLElement>("[data-s]")?.focus();
  });
  timerMenu?.addEventListener("click", (e) => {
    const b = (e.target as Element).closest<HTMLElement>("[data-s]");
    if (b) setDrill(Number(b.dataset.s));
  });
  timerCustom?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && timerCustom.value) setDrill(Number(timerCustom.value));
  });
  timerMenu?.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeTimerMenu();
      $("drill-btn")?.focus();
    }
  });
  document.addEventListener("pointerdown", (e) => {
    const t = e.target as Element;
    if (timerMenu && !timerMenu.contains(t) && !t.closest("#drill-btn")) closeTimerMenu();
  });
  syncDrill();

  const gainInput = $<HTMLInputElement>("set-gain");
  const savedGain = Number(load(GAIN_KEY));
  if (savedGain > 0) levelGain = savedGain;
  if (gainInput) {
    gainInput.value = String(levelGain);
    gainInput.addEventListener("input", () => {
      levelGain = Number(gainInput.value);
      save(GAIN_KEY, gainInput.value);
    });
  }

  const micBtn = $<HTMLButtonElement>("mic-test-btn");
  micBtn?.addEventListener("click", async () => {
    micBtn.disabled = true;
    await (micTesting ? stopMicTest(true) : startMicTest());
    micBtn.disabled = false;
  });

  const pauseSel = $<HTMLSelectElement>("set-pause");
  const savedPause = Number(load(PAUSE_KEY));
  if (savedPause > 0) INTER_PAUSE_MS = savedPause;
  if (pauseSel) {
    pauseSel.value = String(INTER_PAUSE_MS);
    syncSelect(pauseSel);
    pauseSel.addEventListener("change", () => {
      INTER_PAUSE_MS = Number(pauseSel.value);
      save(PAUSE_KEY, pauseSel.value);
      renderStats();
    });
  }

  // Boolean toggles: stored as "0"/"1", default `def`.
  const bindToggle = (id: string, key: string, apply: (on: boolean) => void, def = true) => {
    const box = $<HTMLInputElement>(id);
    const on = (load(key) ?? (def ? "1" : "0")) !== "0";
    apply(on);
    if (!box) return;
    box.checked = on;
    box.addEventListener("change", () => {
      save(key, box.checked ? "1" : "0");
      apply(box.checked);
    });
  };
  bindToggle("set-correct", CORRECT_KEY, (on) => {
    accurateCorrection = on;
    renderModelChain();
  });

  // The backend always starts on tiny.en; swap in the saved pick if different.
  correctionModel = load(CORRECT_MODEL_KEY) ?? correctionModel;
  liveModel = load(LIVE_MODEL_KEY) ?? liveModel;
  void refreshModels().then(() => {
    if (liveModel !== "tiny.en") void setLiveModel(liveModel);
  });
  const liveSel = $<HTMLSelectElement>("set-live-model");
  liveSel?.addEventListener("change", () => void setLiveModel(liveSel.value));
  const correctSel = $<HTMLSelectElement>("set-correct-model");
  correctSel?.addEventListener("change", () => {
    correctionModel = correctSel.value;
    save(CORRECT_MODEL_KEY, correctionModel);
    renderModelChain();
  });
  $("model-list")?.addEventListener("click", async (e) => {
    const btn = (e.target as Element).closest<HTMLButtonElement>("button");
    if (btn?.dataset.download) void downloadModel(btn.dataset.download);
    if (btn?.dataset.delete) {
      btn.disabled = true;
      try {
        await invoke("delete_model", { id: btn.dataset.delete });
      } catch (e) {
        appendError(String(e));
      }
      await refreshModels();
    }
  });
  listen<{ id: string; done: number; total: number }>("model_download", ({ payload: p }) => {
    const pct = p.total ? Math.floor((p.done / p.total) * 100) : 0;
    downloading.set(p.id, pct);
    const btn = document.querySelector<HTMLButtonElement>(`[data-progress="${p.id}"]`);
    if (btn) btn.textContent = p.total ? `${pct}%` : `${Math.round(p.done / 1e6)} MB`;
  });
  bindToggle("set-hover", HOVER_KEY, (on) => document.body.classList.toggle("no-hover-fx", !on));
  bindToggle("set-countdown", COUNTDOWN_KEY, (on) => document.body.classList.toggle("timer-countdown", on), false);

  // Two-click confirm: first click arms the button for 3 s.
  const clearBtn = $<HTMLButtonElement>("set-clear");
  let clearArmed = 0;
  clearBtn?.addEventListener("click", async () => {
    if (!clearArmed) {
      clearBtn.textContent = "Click again to delete";
      clearArmed = window.setTimeout(() => {
        clearArmed = 0;
        clearBtn.textContent = "Clear history";
      }, 3000);
      return;
    }
    clearTimeout(clearArmed);
    clearArmed = 0;
    try {
      await invoke("clear_sessions");
      history = [];
      renderSessionLabel();
      clearBtn.textContent = "Cleared";
    } catch (e) {
      clearBtn.textContent = "Clear history";
      appendError(String(e));
    }
  });

  $("report-body")?.addEventListener("click", (e) => {
    const star = (e.target as Element).closest<HTMLElement>("[data-star]");
    if (star) return void toggleSaved(Number(star.dataset.star));
    const m = (e.target as Element).closest<HTMLElement>(".moment");
    if (m) jumpToLine(Number(m.dataset.index), Number(m.dataset.ms));
    const report = $("report");
    if (report) clickPace(e, report, paceData(), live);
  });
  $("report-body")?.addEventListener("mousemove", (e) => hoverPace(e, paceData(), live));

  // Editing a saved speech updates it in place.
  scriptInput?.addEventListener("input", () => {
    setScript(scriptInput?.value ?? "");
    const sp = speeches.find((s) => s.id === currentSpeech);
    if (sp) {
      sp.text = scriptText;
      saveSpeeches();
    }
  });
  // A loaded file starts out unsaved rather than overwriting the open speech.
  scriptFile?.addEventListener("change", async () => {
    const file = scriptFile?.files?.[0];
    if (!file) return;
    currentSpeech = "";
    saveSpeeches();
    renderSpeechPicker();
    loadScript(await file.text());
  });

  try {
    const parsed = JSON.parse(load(SPEECHES_KEY) ?? "[]");
    if (Array.isArray(parsed)) speeches = parsed;
  } catch {
    // corrupt entry; start with no saved speeches
  }
  currentSpeech = speechTitle(load(SPEECH_KEY) ?? "") ? load(SPEECH_KEY)! : "";
  const speechSel = $<HTMLSelectElement>("speech-sel");
  if (speechSel) enhanceSelect(speechSel);
  renderSpeechPicker();
  speechSel?.addEventListener("change", () => openSpeech(speechSel.value));
  speechPage = load(SPEECH_PAGE_KEY) ?? "";
  $("speeches-btn")?.addEventListener("click", () => showView("speeches"));
  $("history-btn")?.addEventListener("click", () => showView("history"));
  const histEl = $("history");
  histEl?.addEventListener("click", (e) => {
    const b = (e.target as Element).closest<HTMLElement>("[data-star], [data-goto], [data-open], [data-back], [data-hsaved], [data-range]");
    const d = b?.dataset;
    if (!b || !d) return;
    if (d.star) return void toggleSaved(Number(d.star));
    if (d.goto) {
      speechPage = d.goto;
      return showView("speeches");
    }
    if (d.open) {
      histEl.scrollTop = 0;
      return void openHistorySession(Number(d.open));
    }
    if (d.back !== undefined) histPage = 0;
    else if (d.hsaved) histSaved = d.hsaved === "1";
    else if (d.range) histRange = d.range;
    renderHistory();
    if (d.back !== undefined) histEl.scrollTop = 0;
  });
  // The open session's report: key moments and pace clicks jump within its own
  // transcript copy (the live one belongs to another session).
  histEl?.addEventListener("click", (e) => {
    const hp = histPace();
    if (!hp?.L) return;
    const m = (e.target as Element).closest<HTMLElement>(".moment");
    if (m) jumpInCopy(histEl, Number(m.dataset.index), Number(m.dataset.ms), hp.L);
    clickPace(e, histEl, hp.pace, hp.L);
  });
  histEl?.addEventListener("mousemove", (e) => {
    const hp = histPace();
    if (hp) hoverPace(e, hp.pace, hp.L);
  });
  histEl?.addEventListener("change", (e) => {
    const t = e.target as HTMLSelectElement;
    if (t.id === "hist-preset") histPreset = t.value;
    else if (t.id === "hist-speech") histSpeech = t.value;
    else return;
    renderHistory();
  });
  const speechesEl = $("speeches");
  speechesEl?.addEventListener("click", (e) => {
    const b = (e.target as Element).closest<HTMLElement>(
      "[data-new], [data-rename], [data-practice], [data-open], [data-back], [data-delete], [data-smetric], [data-cfilter]",
    );
    const d = b?.dataset;
    if (!b || !d) return;
    if (d.practice) return practiceSpeech(d.practice);
    if (d.new !== undefined) return openSpeechDialog("", false);
    if (d.rename !== undefined) {
      const sp = speeches.find((x) => x.id === speechPage);
      return sp && openSpeechDialog("", false, sp);
    }
    if (d.delete) {
      if (!confirmClick(b)) return;
      deleteSpeech(d.delete);
      speechPage = "";
    } else if (d.open) speechPage = d.open;
    else if (d.back !== undefined) speechPage = "";
    else if (d.smetric) speechMetric = d.smetric;
    else if (d.cfilter !== undefined) speechFilter = d.cfilter;
    renderSpeeches();
    if (d.open || d.back !== undefined) speechesEl.scrollTop = 0;
  });
  speechesEl?.addEventListener("change", (e) => {
    const t = e.target as HTMLSelectElement;
    if (t.id !== "speech-sort") return;
    speechSort = t.value;
    renderSpeeches();
  });
  speechesEl?.addEventListener("input", (e) => {
    const t = e.target as HTMLInputElement;
    // Searching swaps just the rows so the search box keeps focus.
    if (t.id === "speech-q") {
      speechQuery = t.value;
      const table = $("speech-table");
      if (table) table.innerHTML = speechRowsHtml();
    }
  });
  // Save opens the new-speech popup with the script box's text; on a saved
  // speech the button is Clear instead, emptying the box for a new script.
  $("speech-save")?.addEventListener("click", () => {
    if (currentSpeech) openSpeech("");
    else openSpeechDialog(scriptText, true);
  });

  // New-speech popup. Saving from the Live tab puts the speech in the
  // read-along; saving from the Speeches list opens its page.
  const nd = $<HTMLDialogElement>("speech-dialog");
  $("nd-cancel")?.addEventListener("click", () => nd?.close());
  $("nd-colors")?.addEventListener("click", (e) => {
    const c = (e.target as Element).closest<HTMLElement>("[data-ndcolor]")?.dataset.ndcolor;
    if (c) setNewSpeechColor(c);
  });
  $<HTMLInputElement>("nd-file")?.addEventListener("change", async (e) => {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    const text = $<HTMLTextAreaElement>("nd-text");
    if (file && text) text.value = await file.text();
    input.value = "";
  });
  nd?.addEventListener("close", () => {
    const edited = speeches.find((x) => x.id === ndEditing);
    if (edited) {
      if (nd.returnValue !== "save") return;
      edited.title = $<HTMLInputElement>("nd-name")?.value.trim() || edited.title;
      edited.color = ndColor;
      saveSpeeches();
      renderSpeechPicker();
      renderSpeeches();
      return;
    }
    const text = $<HTMLTextAreaElement>("nd-text")?.value ?? "";
    if (nd.returnValue !== "save" || !text.trim()) return;
    const name = $<HTMLInputElement>("nd-name")?.value.trim() || text.trim().split("\n")[0].trim().slice(0, 60);
    const id = String(Date.now());
    speeches.push({ id, title: name, text, color: ndColor });
    saveSpeeches();
    if (ndFromLive) openSpeech(id);
    else {
      renderSpeechPicker();
      speechPage = id;
      renderSpeeches();
    }
  });

  // Restore a saved script, if any.
  try {
    const savedScript = localStorage.getItem(SCRIPT_KEY);
    if (savedScript) {
      if (scriptInput) scriptInput.value = savedScript;
      setScript(savedScript);
    }
  } catch {
    // storage unavailable; no script to restore
  }

  // Context preset (segmented control on the scope). Re-scores the live dock
  // and, if shown, the report.
  try {
    const savedPreset = localStorage.getItem(PRESET_KEY);
    if (savedPreset && PRESETS[savedPreset]) currentPreset = savedPreset;
  } catch {
    // storage unavailable; default preset stands
  }
  const presetButtons = [...document.querySelectorAll<HTMLButtonElement>("#preset-seg button")];
  const presetSel = $<HTMLSelectElement>("set-preset");
  const syncPreset = () => {
    presetButtons.forEach((b) => b.setAttribute("aria-checked", String(b.dataset.preset === currentPreset)));
    if (presetSel) {
      presetSel.value = currentPreset;
      syncSelect(presetSel);
    }
  };
  const setPreset = (p: string | undefined) => {
    if (!p || !PRESETS[p]) return;
    currentPreset = p;
    save(PRESET_KEY, p);
    syncPreset();
    renderStats();
  };
  presetButtons.forEach((b) => b.addEventListener("click", () => setPreset(b.dataset.preset)));
  presetSel?.addEventListener("change", () => setPreset(presetSel.value));
  syncPreset();
  renderStats();

  // Session numbering and "vs last" come from the saved history.
  renderSessionLabel();
  invoke<string>("list_sessions")
    .then((json) => {
      history = parseHistory(json);
      renderSessionLabel();
    })
    .catch(() => {
      // no history available; numbering starts at 1
    });

  // Practice: random prompt, timer.
  const promptEl = $("practice-prompt");
  $("prompt-btn")?.addEventListener("click", () => {
    if (!promptEl) return;
    promptEl.textContent = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];
    promptEl.removeAttribute("hidden");
  });

  listen<TranscriptSegment>("transcript_segment", (event) => {
    const segment = withPreviewPrefix(event.payload);
    upsertSegment(segment);
    updateScriptFromSegment(segment);
  });

  listen<UtteranceAnalysis>("utterance_analysis", (event) => {
    const analysis = event.payload;
    analysisByIndex.set(analysis.index, analysis);
    refreshWaveform(analysis.index); // draws now if the line exists, else on commit
    renderStats();
  });

  listen<number>("correction_started", (event) => onCorrectionStarted(event.payload));

  listen("corrections_done", () => {
    resetCountdown();
    // Lines the pass didn't replace — medium heard only noise ("[BLANK_AUDIO]"),
    // its decode failed, or correction is off — never get a refined segment, so
    // settle them here: the fast text is final.
    for (const el of segmentEls.values()) el.classList.replace("draft", "final");
    if (reportPending) {
      reportPending = false;
      renderReport();
      if (reportVisible && document.body.dataset.view === "live") showView("report");
    }
    flushSave();
    if (!recording) setStatus("Idle");
  });

  listen<string>("transcription_error", (event) => {
    appendError(event.payload);
  });

  // Mic RMS → 0..1 drive for the ribbon and the dock's level meter. Speech RMS
  // is small (~0.02–0.1) so a plain multiply barely moves the quiet end; sqrt is
  // a perceptual curve that lifts soft speech into a visible range. Bump the
  // gain if it still reacts weakly.
  listen<boolean>("mic_test_speech", (event) => $("mic-speech")?.classList.toggle("on", event.payload));
  listen<number>("audio_level", (event) => {
    const level = Math.min(1, Math.sqrt(event.payload * levelGain));
    ribbon?.setLevel(level);
    setLevelMeter(level);
    setText("rms-label", `rms ${event.payload.toFixed(2)} · 16 kHz`);
  });
});

let levelGain = 8;

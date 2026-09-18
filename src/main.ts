import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Ribbon } from "./ribbon";

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
const INTER_PAUSE_MS = 600;

let recording = false;
let recordBtn: HTMLButtonElement | null;
let statusEl: HTMLElement | null;
let transcriptEl: HTMLElement | null;
let wpmEl: HTMLElement | null;
let wordsEl: HTMLElement | null;
let fillersEl: HTMLElement | null;
let pausesEl: HTMLElement | null;
let timeEl: HTMLElement | null;

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
const timedIndices = new Set<number>();

// Per-utterance acoustic analysis keyed by index, for the waveform under each
// line and the pause tally. `timingByIndex` keeps each utterance's start/end so
// the gaps *between* utterances can be counted as pauses too. Both are cleared
// per session (indices restart at 1).
const analysisByIndex = new Map<number, UtteranceAnalysis>();
const timingByIndex = new Map<number, { start: number; end: number }>();

// Bumped on every new recording. Captured when a contextual review is scheduled
// so a review that resolves after the user has restarted is discarded instead
// of writing into the new session's transcript or stats.
let sessionId = 0;

// Last script-alignment result (null until a script is used), so the end-of-
// session report can score articulation. Updated every re-align in renderScriptMatch.
let lastScriptResult:
  | { total: number; hits: number; misses: number; subs: number; accuracy: number }
  | null = null;

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
    .replace(/>/g, "&gt;");
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
    entry.innerHTML = `<span class="timestamp">${formatTimestamp(startMs)}</span> ${renderHighlighted(text, ranges)}`;
    refreshWaveform(index); // innerHTML rewrite wiped it
    // Practice: in filler-free mode, flag any newly counted fillers on this line.
    if (fillerFreeMode && ranges.length > (fillersByIndex.get(index) ?? 0)) flashFillerAlert();
  }
  // Adjust the running total by the delta for this line, so a re-review of a
  // corrected line replaces its earlier filler count rather than stacking on it.
  totalFillers += ranges.length - (fillersByIndex.get(index) ?? 0);
  fillersByIndex.set(index, ranges.length);
  renderStats();
}

function resetStats() {
  totalWords = 0;
  totalFillers = 0;
  speakingMs = 0;
  wordsByIndex.clear();
  fillersByIndex.clear();
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

function renderStats() {
  const speakingMinutes = speakingMs / 60000;
  const wpm = speakingMinutes > 0 ? Math.round(totalWords / speakingMinutes) : 0;
  if (wpmEl) wpmEl.textContent = String(wpm);
  if (wordsEl) wordsEl.textContent = String(totalWords);
  if (fillersEl) fillersEl.textContent = String(totalFillers);
  if (pausesEl) pausesEl.textContent = String(countPauses());
  if (timeEl) timeEl.textContent = formatTimestamp(speakingMs);
  // Keep the report in sync as deferred corrections/analyses trickle in after stop.
  if (reportVisible) renderReport();
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
  entry.insertAdjacentHTML("beforeend", buildWaveformSvg(analysis));
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

function perSecondPace(): SecondPace[] {
  const idx = [...timingByIndex.keys()].sort((a, b) => a - b);
  if (idx.length === 0) return [];
  const utts = idx.map((i) => {
    const t = timingByIndex.get(i)!;
    return {
      start: t.start,
      end: t.end,
      wordsPerMs: (wordsByIndex.get(i) ?? 0) / Math.max(1, t.end - t.start),
      fillers: fillersByIndex.get(i) ?? 0,
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
  };

  // Keep the current word in view within the panel.
  if (currentSpan) {
    scriptDisplay.scrollTop = currentSpan.offsetTop - scriptDisplay.clientHeight / 2;
  }
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
    entry = document.createElement("p");
    entry.className = "segment";
    transcriptEl.appendChild(entry);
    segmentEls.set(segment.index, entry);
  }
  // Visual state of the line, so the two-pass pipeline is legible:
  //   partial   — live interim from the fast (tiny) model, still being spoken
  //   draft     — fast model's committed text, correction pending
  //   corrected — replaced by the accurate (medium) model; flashes once on change
  // A refined segment is also is_final, so check it first.
  entry.classList.remove("partial", "draft", "corrected");
  if (!segment.is_final) {
    entry.classList.add("partial");
  } else if (segment.refined) {
    entry.classList.add("corrected");
  } else {
    entry.classList.add("draft");
  }
  // Tier-1 render: plain text, no filler flags. This paints immediately; all
  // filler flagging happens in the deferred review below, so the text always
  // appears before any flagging.
  entry.innerHTML = `<span class="timestamp">${formatTimestamp(segment.start_ms)}</span> ${escapeHtml(segment.text)}`;
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
}

async function toggleRecording() {
  if (!recordBtn) return;

  recordBtn.disabled = true;
  try {
    if (!recording) {
      // The backend restarts utterance indices each session; drop stale
      // line references so a new session's index 1 starts a fresh line.
      segmentEls.clear();
      finalized.clear();
      refinedIndices.clear();
      tokensByIndex.clear();
      renderScriptMatch(); // reset read-along highlights to the start
      resetStats();
      // Clear the previous session's report; it's rebuilt on stop.
      reportVisible = false;
      document.querySelector("#report")?.setAttribute("hidden", "");
      await invoke("start_recording");
      recording = true;
      recordBtn.textContent = "Stop recording";
      recordBtn.classList.add("recording");
      ribbon?.setMode("listening");
      setStatus("Recording…");
    } else {
      await invoke("stop_recording");
      recording = false;
      recordBtn.textContent = "Start recording";
      recordBtn.classList.remove("recording");
      ribbon?.setMode("idle");
      ribbon?.setLevel(0); // no more level events once stopped; settle to rest
      setStatus("Idle");
      // Build the report. It keeps refreshing via renderStats as any trailing
      // corrections land.
      renderReport();
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
  segmentEls.clear();
  finalized.clear();
  refinedIndices.clear();
  tokensByIndex.clear();
  renderScriptMatch();
  resetStats();
  reportVisible = false;
  document.querySelector("#report")?.setAttribute("hidden", "");
  if (transcriptEl)
    transcriptEl.innerHTML =
      '<p class="placeholder">Your transcript will appear here as you speak.</p>';
  setStatus("Idle");
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
  script: { total: number; hits: number; misses: number; subs: number; accuracy: number } | null;
  preset: string;
  scores: Scores;
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

// Analyses ordered by utterance index, for the per-sentence report charts.
function orderedAnalyses(): UtteranceAnalysis[] {
  return [...analysisByIndex.keys()]
    .sort((a, b) => a - b)
    .map((i) => analysisByIndex.get(i)!);
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

  const script = lastScriptResult && scriptTokens.length > 0 ? { ...lastScriptResult } : null;
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
    ts: Date.now(),
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

// Generic per-sentence/per-session bar chart as an inline SVG string. Reused for
// the pace, pitch and volume report charts (the live WPM chart stays its own).
function barChartSvg(values: number[], title: (i: number, v: number) => string): string {
  const W = 300;
  const H = 70;
  const n = values.length;
  if (n === 0) return `<svg viewBox="0 0 ${W} ${H}" class="report-chart-svg"></svg>`;
  const max = Math.max(...values, 1);
  const slot = W / n;
  const barW = slot * 0.7;
  let bars = "";
  values.forEach((v, i) => {
    const h = (v / max) * H;
    const x = i * slot + (slot - barW) / 2;
    bars += `<rect class="report-bar" x="${x.toFixed(2)}" y="${(H - h).toFixed(2)}" width="${barW.toFixed(2)}" height="${Math.max(0.5, h).toFixed(2)}"><title>${title(i, v)}</title></rect>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="report-chart-svg">${bars}</svg>`;
}

// A point on the WPM line. `x01` is its position along the x-axis in [0,1];
// `filler` marks a filler onset (drawn as a red ×); `title` is the hover text.
interface LinePoint {
  x01: number;
  wpm: number;
  filler: boolean;
  title: string;
}
interface XTick {
  at01: number;
  label: string;
  anchor: "start" | "middle" | "end";
}

// A "nice" tick spacing (1/2/5 × 10^k) so an axis lands ~targetTicks round marks
// that adapt to the data range, rather than a fixed peak/0 pair.
function niceStep(range: number, targetTicks: number): number {
  const raw = Math.max(range, 1) / Math.max(1, targetTicks);
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / pow;
  return (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * pow;
}

// Monkeytype-style WPM line graph: a continuous polyline through `points` with
// adaptive gridline ticks on both axes (y = WPM in round steps, x from the
// caller). Every point carries a transparent hover target with its `title`;
// filler onsets also draw a visible red ×.
function lineGraphSvg(points: LinePoint[], dataMax: number, xTicks: XTick[]): string {
  const W = 600;
  const H = 150;
  const PAD_L = 40; // y-axis tick labels
  const PAD_T = 22; // "WPM" unit + headroom
  const PAD_B = 26; // x-axis labels
  if (points.length === 0) return `<svg viewBox="0 0 ${W} ${H}" class="report-chart-svg"></svg>`;
  const top = PAD_T;
  const bottom = H - PAD_B;
  const plotW = W - PAD_L - 4;
  const yStep = niceStep(dataMax, 4);
  const axisMax = Math.max(yStep, Math.ceil(dataMax / yStep) * yStep);
  const X = (x01: number) => PAD_L + Math.max(0, Math.min(1, x01)) * plotW;
  const Y = (v: number) => bottom - (v / axisMax) * (bottom - top);
  const pts = points.map((p) => `${X(p.x01).toFixed(2)},${Y(p.wpm).toFixed(2)}`).join(" ");

  // Adaptive y-axis: a gridline + WPM value at each round step.
  let yAxis = "";
  for (let v = 0; v <= axisMax + 1e-6; v += yStep) {
    const y = Y(v).toFixed(2);
    yAxis +=
      `<line class="axis-grid" x1="${PAD_L}" y1="${y}" x2="${W - 2}" y2="${y}" />` +
      `<text class="axis-label" x="${PAD_L - 5}" y="${(Y(v) + 3.5).toFixed(2)}" text-anchor="end">${v}</text>`;
  }
  // Adaptive x-axis: a short tick + label at each caller-supplied mark.
  const xAxis = xTicks
    .map((t) => {
      const x = X(t.at01).toFixed(2);
      return (
        `<line class="axis-grid" x1="${x}" y1="${bottom}" x2="${x}" y2="${bottom + 3}" />` +
        `<text class="axis-label" x="${x}" y="${H - 6}" text-anchor="${t.anchor}">${escapeHtml(t.label)}</text>`
      );
    })
    .join("");

  const r = 5;
  const fillerMarks = points
    .filter((p) => p.filler)
    .map((p) => {
      const x = X(p.x01);
      const y = Y(p.wpm);
      return (
        `<path class="wpm-filler-mark" d="M${(x - r).toFixed(2)},${(y - r).toFixed(2)} ` +
        `l${(2 * r).toFixed(2)},${(2 * r).toFixed(2)} M${(x + r).toFixed(2)},${(y - r).toFixed(2)} ` +
        `l${(-2 * r).toFixed(2)},${(2 * r).toFixed(2)}" />`
      );
    })
    .join("");
  // One hover target per point: transparent, reveals a dot on hover (CSS) and
  // shows the point's WPM + fillers as a native tooltip.
  const hits = points
    .map((p) => `<circle class="trend-hit" cx="${X(p.x01).toFixed(2)}" cy="${Y(p.wpm).toFixed(2)}" r="4"><title>${escapeHtml(p.title)}</title></circle>`)
    .join("");

  return (
    `<svg viewBox="0 0 ${W} ${H}" class="report-chart-svg trend">` +
    yAxis +
    `<text class="axis-unit" x="2" y="14">WPM</text>` +
    `<polyline points="${pts}" fill="none" />` +
    fillerMarks +
    `<g class="trend-hits">${hits}</g>` +
    xAxis +
    `</svg>`
  );
}

// Per-session pace: WPM sampled every second over the session's elapsed time
// (x = 0:00 → end, adaptive time ticks). Hover any second for its WPM + running
// filler count; filler onsets mark as ×.
function paceLineSvg(): string {
  const series = perSecondPace();
  if (series.length === 0) return `<svg viewBox="0 0 600 150" class="report-chart-svg"></svg>`;
  const totalSec = series.length - 1;
  const at01 = (sec: number) => (totalSec > 0 ? sec / totalSec : 0.5);
  const yMax = Math.max(...series.map((s) => s.wpm), 1);
  const points: LinePoint[] = series.map((s, sec) => ({
    x01: at01(sec),
    wpm: s.wpm,
    filler: s.fillerHere,
    title: `${formatTimestamp(sec * 1000)} · ${s.wpm} WPM · ${s.fillersCum} filler${s.fillersCum === 1 ? "" : "s"}`,
  }));
  const xStep = Math.max(1, Math.round(niceStep(totalSec, 4)));
  const xTicks: XTick[] = [];
  for (let sec = 0; sec <= totalSec; sec += xStep)
    xTicks.push({ at01: at01(sec), label: formatTimestamp(sec * 1000), anchor: sec === 0 ? "start" : "middle" });
  return lineGraphSvg(points, yMax, xTicks);
}

// A titled chart with caption. Pass `axis` to frame the plot with a y-axis
// (peak value + unit at top, 0 at bottom) and an x-axis label — used by the
// report bar charts so each has readable ticks/units.
function chartBlock(
  label: string,
  svg: string,
  caption: string,
  axis?: { max: number; unit: string; xLabel: string },
): string {
  let plot = `<div class="report-chart-box">${svg}</div>`;
  if (axis) {
    const unit = axis.unit ? ` ${axis.unit}` : "";
    plot =
      `<div class="chart-plot">` +
      `<div class="y-axis"><span>${axis.max}${escapeHtml(unit)}</span><span>0</span></div>` +
      plot +
      `</div><div class="x-axis-label">${escapeHtml(axis.xLabel)}</div>`;
  }
  return `<div class="report-chart"><span class="mode-label">${label}</span>${plot}<div class="chart-caption">${escapeHtml(caption)}</div></div>`;
}

function renderReport() {
  const body = document.querySelector("#report-body");
  const section = document.querySelector("#report");
  if (!body || !section) return;
  if (totalWords === 0) {
    section.setAttribute("hidden", "");
    reportVisible = false;
    return;
  }
  reportVisible = true;
  section.removeAttribute("hidden");

  const s = computeSummary();
  const c = s.scores;
  const subscore = (label: string, v: number) =>
    `<div class="subscore"><span class="subscore-label">${label}</span><div class="meter"><div class="meter-fill" style="width:${v}%"></div></div><span class="subscore-val">${v}</span></div>`;

  const subs = [
    subscore("Pace", c.pace),
    subscore("Fillers", c.fillers),
    subscore("Pauses", c.pauses),
    subscore("Pitch", c.pitch),
    subscore("Volume", c.volume),
    ...(c.articulation !== null ? [subscore("Articulation", c.articulation)] : []),
  ].join("");

  const ordered = orderedAnalyses();
  const pitchData = ordered.map((a) => Number(a.f0_range_semitones.toFixed(1)));
  const volumeData = ordered.map((a) => Math.round(a.rms_level * 1000));

  // Headline chart: continuous WPM-over-time line for the session (Monkeytype
  // style), followed by the per-sentence pitch/volume bars.
  const paceBlock =
    `<div class="report-chart">` +
    `<span class="mode-label">Pace over time (WPM)</span>` +
    `<div class="report-chart-box large">${paceLineSvg()}</div>` +
    `<div class="chart-legend">` +
    `<span class="legend-item"><svg class="legend-mark" viewBox="0 0 16 10" aria-hidden="true"><line x1="0" y1="5" x2="16" y2="5" /><circle cx="8" cy="5" r="2.2" /></svg>WPM</span>` +
    `<span class="legend-item"><svg class="legend-mark filler" viewBox="0 0 16 10" aria-hidden="true"><path d="M5,1 L11,9 M11,1 L5,9" /></svg>filler used</span>` +
    `</div>` +
    `<div class="chart-caption">${escapeHtml(`avg ${s.wpm} WPM · target ${PRESETS[s.preset].wpmLow}–${PRESETS[s.preset].wpmHigh}`)}</div>` +
    `</div>`;

  const peak = (d: number[]) => Math.max(...d, 1);
  const charts =
    paceBlock +
    chartBlock("Pitch inflection per sentence (semitones)", barChartSvg(pitchData, (i, v) => `Sentence ${i + 1}: ${v} st`), `avg ${s.pitchRange.toFixed(1)} st${s.pitchRange < 3 ? " · monotone" : ""}`, { max: peak(pitchData), unit: "st", xLabel: "sentence →" }) +
    chartBlock("Volume per sentence", barChartSvg(volumeData, (i) => `Sentence ${i + 1}`), s.trailingOff < 0.7 ? `trails off to ${Math.round(s.trailingOff * 100)}% at ends` : "steady", { max: peak(volumeData), unit: "", xLabel: "sentence →" });

  const tips = generateTips(s).map((t) => `<li>${escapeHtml(t)}</li>`).join("");

  body.innerHTML = `
    <div class="score-hero">
      <div class="score-ring score-${grade(c.overall).toLowerCase()}">
        <span class="score-num">${c.overall}</span>
        <span class="score-grade">${grade(c.overall)}</span>
      </div>
      <div class="score-meta">
        <div>${s.words} words · ${formatTimestamp(s.durationMs)} speaking · ${s.wpm} WPM</div>
        <div>${s.fillers} fillers · ${s.pauses} pauses${s.script ? ` · ${s.script.accuracy}% script` : ""}</div>
      </div>
    </div>
    <div class="subscores">${subs}</div>
    <div class="tips"><span class="mode-label">What to work on</span><ol>${tips}</ol></div>
    <div class="report-charts">${charts}</div>
  `;
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

let fillerFreeMode = false;

function flashFillerAlert() {
  document.body.classList.add("filler-flash");
  setTimeout(() => document.body.classList.remove("filler-flash"), 350);
}

const DRILL_MS = 60_000;

window.addEventListener("DOMContentLoaded", () => {
  const ribbonCanvas = document.querySelector<HTMLCanvasElement>("#ribbon");
  if (ribbonCanvas) ribbon = new Ribbon(ribbonCanvas, "idle");

  recordBtn = document.querySelector("#record-btn");
  statusEl = document.querySelector("#status");
  transcriptEl = document.querySelector("#transcript");
  wpmEl = document.querySelector("#stat-wpm");
  wordsEl = document.querySelector("#stat-words");
  fillersEl = document.querySelector("#stat-fillers");
  pausesEl = document.querySelector("#stat-pauses");
  timeEl = document.querySelector("#stat-time");

  scriptInput = document.querySelector("#script-input");
  scriptFile = document.querySelector("#script-file");
  scriptDisplay = document.querySelector("#script-display");
  scriptProgress = document.querySelector("#script-progress");

  recordBtn?.addEventListener("click", toggleRecording);
  document.querySelector("#reset-btn")?.addEventListener("click", resetSession);

  scriptInput?.addEventListener("input", () => setScript(scriptInput?.value ?? ""));
  scriptFile?.addEventListener("change", async () => {
    const file = scriptFile?.files?.[0];
    if (!file) return;
    const text = await file.text();
    if (scriptInput) scriptInput.value = text;
    setScript(text);
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

  // Context preset selector (re-scores the currently shown report on change).
  const presetSelect = document.querySelector<HTMLSelectElement>("#report-preset");
  try {
    const savedPreset = localStorage.getItem(PRESET_KEY);
    if (savedPreset && PRESETS[savedPreset]) currentPreset = savedPreset;
  } catch {
    // storage unavailable; default preset stands
  }
  if (presetSelect) {
    presetSelect.value = currentPreset;
    presetSelect.addEventListener("change", () => {
      currentPreset = presetSelect.value;
      try {
        localStorage.setItem(PRESET_KEY, currentPreset);
      } catch {
        // ignore
      }
      if (reportVisible) renderReport();
    });
  }

  // Practice: random prompt + timed drill + filler-free mode.
  const promptEl = document.querySelector("#practice-prompt");
  const showPrompt = () => {
    if (promptEl) promptEl.textContent = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];
  };
  document.querySelector("#prompt-btn")?.addEventListener("click", showPrompt);
  document.querySelector("#drill-btn")?.addEventListener("click", () => {
    showPrompt();
    if (!recording) {
      const drillSession = sessionId + 1; // resetStats (in toggleRecording) bumps to this
      void toggleRecording();
      // Auto-stop after the drill window, unless the user already stopped or
      // started another session.
      setTimeout(() => {
        if (recording && sessionId === drillSession) void toggleRecording();
      }, DRILL_MS);
    }
  });
  document.querySelector<HTMLInputElement>("#filler-free")?.addEventListener("change", (e) => {
    fillerFreeMode = (e.target as HTMLInputElement).checked;
  });

  listen<TranscriptSegment>("transcript_segment", (event) => {
    upsertSegment(event.payload);
    updateScriptFromSegment(event.payload);
  });

  listen<UtteranceAnalysis>("utterance_analysis", (event) => {
    const analysis = event.payload;
    analysisByIndex.set(analysis.index, analysis);
    refreshWaveform(analysis.index); // draws now if the line exists, else on commit
    renderStats();
  });

  listen<string>("transcription_error", (event) => {
    appendError(event.payload);
  });

  // Mic RMS → 0..1 ribbon drive. Speech RMS is small (~0.02–0.1) so a plain
  // multiply barely moves the quiet end; sqrt is a perceptual curve that lifts
  // soft speech into a visible range. Bump the gain if it still reacts weakly.
  listen<number>("audio_level", (event) => {
    ribbon?.setLevel(Math.sqrt(event.payload * RIBBON_LEVEL_GAIN));
  });
});

const RIBBON_LEVEL_GAIN = 8;

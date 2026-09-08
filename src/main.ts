import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

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
let wpmChartSection: HTMLElement | null;
let wpmChartEl: HTMLElement | null;
let wpmChartCaptionEl: HTMLElement | null;

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

    for (let i = 0; i < terms.length; ) {
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
  renderWpmChart();
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

// --- Pace-per-sentence chart -------------------------------------------------
// One WPM value per committed utterance, from the same per-index word counts and
// timing the headline stats use — so a correction re-tallying a line's words
// moves that bar too. Rebuilt whole on each stats render (cheap; it's a handful
// of <rect>s). WPM here is per-sentence articulation rate (words over that
// utterance's own duration), so short utterances read spiky by nature.

interface SentencePace {
  wpm: number;
}

function perSentenceWpm(): SentencePace[] {
  const out: SentencePace[] = [];
  for (const index of [...timingByIndex.keys()].sort((a, b) => a - b)) {
    const t = timingByIndex.get(index)!;
    const words = wordsByIndex.get(index) ?? 0;
    const minutes = (t.end - t.start) / 60000;
    if (words > 0 && minutes > 0) out.push({ wpm: Math.round(words / minutes) });
  }
  return out;
}

function renderWpmChart() {
  if (!wpmChartSection || !wpmChartEl) return;
  const data = perSentenceWpm();
  if (data.length === 0) {
    wpmChartSection.setAttribute("hidden", "");
    return;
  }
  wpmChartSection.removeAttribute("hidden");

  const W = 300;
  const H = 100;
  const values = data.map((d) => d.wpm);
  const peak = Math.max(...values);
  const scaleMax = Math.max(peak, 1);
  const n = data.length;
  const slot = W / n;
  const barW = slot * 0.7;

  let bars = "";
  data.forEach((d, i) => {
    const h = (d.wpm / scaleMax) * H;
    const x = i * slot + (slot - barW) / 2;
    const y = H - h;
    bars +=
      `<rect class="wpm-bar" x="${x.toFixed(2)}" y="${y.toFixed(2)}" ` +
      `width="${barW.toFixed(2)}" height="${Math.max(0.5, h).toFixed(2)}">` +
      `<title>Sentence ${i + 1}: ${d.wpm} WPM</title></rect>`;
  });

  // Session-average reference line, matching the headline WPM stat.
  const speakingMinutes = speakingMs / 60000;
  const avg = speakingMinutes > 0 ? Math.round(totalWords / speakingMinutes) : 0;
  let avgLine = "";
  if (avg > 0) {
    const y = H - (Math.min(avg, scaleMax) / scaleMax) * H;
    avgLine = `<line class="wpm-avg" x1="0" y1="${y.toFixed(2)}" x2="${W}" y2="${y.toFixed(2)}" />`;
  }

  wpmChartEl.innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" ` +
    `role="img" aria-label="Words per minute for each sentence">${avgLine}<g>${bars}</g></svg>`;

  if (wpmChartCaptionEl) {
    wpmChartCaptionEl.textContent =
      `avg ${avg} · fastest ${peak} · slowest ${Math.min(...values)} WPM`;
  }
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
  let currentSpan: HTMLElement | null = null;
  for (let i = 0; i < scriptSpans.length; i++) {
    let state: string;
    if (matched[i] || substituted[i]) {
      // Substitutions (a different/misheard word) count the same as exact hits.
      state = "script-hit";
      hits++;
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
      await invoke("start_recording");
      recording = true;
      recordBtn.textContent = "Stop recording";
      recordBtn.classList.add("recording");
      setStatus("Recording…");
    } else {
      await invoke("stop_recording");
      recording = false;
      recordBtn.textContent = "Start recording";
      recordBtn.classList.remove("recording");
      setStatus("Idle");
    }
  } catch (e) {
    setStatus("Error");
    appendError(String(e));
  } finally {
    recordBtn.disabled = false;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  recordBtn = document.querySelector("#record-btn");
  statusEl = document.querySelector("#status");
  transcriptEl = document.querySelector("#transcript");
  wpmEl = document.querySelector("#stat-wpm");
  wordsEl = document.querySelector("#stat-words");
  fillersEl = document.querySelector("#stat-fillers");
  pausesEl = document.querySelector("#stat-pauses");
  timeEl = document.querySelector("#stat-time");
  wpmChartSection = document.querySelector("#wpm-chart-section");
  wpmChartEl = document.querySelector("#wpm-chart");
  wpmChartCaptionEl = document.querySelector("#wpm-chart-caption");

  scriptInput = document.querySelector("#script-input");
  scriptFile = document.querySelector("#script-file");
  scriptDisplay = document.querySelector("#script-display");
  scriptProgress = document.querySelector("#script-progress");

  recordBtn?.addEventListener("click", toggleRecording);

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
});

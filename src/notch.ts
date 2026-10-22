// The live notch's payload (macOS; src-tauri/src/notch.rs opens the window,
// notch.html renders it). Pure; checked by notch.check.ts.

export interface NotchState {
  rec: boolean;
  time: string | null; // null when Timer/stopwatch is off
  wpm: number;
  band: "in" | "out" | ""; // WPM against the preset's target band; "" before any words
  fpm: string; // fillers/min, 1 decimal
  script: string | null; // null = Script off; "" = on but nothing ahead
}

export function notchState(i: {
  rec: boolean;
  time: string | null;
  wpm: number;
  low: number;
  high: number;
  fpm: number;
  script: string | null;
}): NotchState {
  const band = i.wpm === 0 ? "" : i.wpm >= i.low && i.wpm <= i.high ? "in" : "out";
  return { rec: i.rec, time: i.time, wpm: i.wpm, band, fpm: i.fpm.toFixed(1), script: i.script };
}

// The next `n` script words from the read-along cursor, as written (punctuation
// between them kept), whitespace collapsed to single spaces.
export function scriptAhead(text: string, tokens: { start: number; end: number }[], cursor: number, n = 8): string {
  if (cursor >= tokens.length) return "";
  const last = tokens[Math.min(tokens.length, cursor + n) - 1];
  return text.slice(tokens[cursor].start, last.end).replace(/\s+/g, " ");
}

// Runs `fn` one call at a time, in call order: a quick Show on → off (or a
// Script toggle mid-build) can't overtake an earlier notch_show. A failed call
// doesn't block the ones after it.
export function serial<A extends unknown[]>(fn: (...a: A) => Promise<void>): (...a: A) => Promise<void> {
  let queue: Promise<void> = Promise.resolve();
  return (...a: A) => (queue = queue.catch(() => {}).then(() => fn(...a)));
}

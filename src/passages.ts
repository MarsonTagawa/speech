// Groups per-word skip counts (summed across a speech's attempts) into
// sentences, so the speech page can quote the passages that trip you up.
// Sentences end at . ! ? or a newline. Pure; checked by passages.check.ts.

export interface Spot {
  start: number; // char range in the script, whitespace-trimmed
  end: number;
  skips: number;
}

export function troubleSpots(
  text: string,
  tokens: Array<{ start: number; end: number }>,
  counts: number[],
  limit = 2,
): Spot[] {
  const spots: Spot[] = [];
  for (const m of text.matchAll(/[^.!?\n]+[.!?]*/g)) {
    const start = (m.index ?? 0) + (m[0].length - m[0].trimStart().length);
    const end = (m.index ?? 0) + m[0].trimEnd().length;
    let skips = 0;
    tokens.forEach((t, i) => {
      if (t.start >= start && t.end <= end) skips += counts[i] ?? 0;
    });
    if (skips) spots.push({ start, end, skips });
  }
  // Stable sort keeps script order among ties.
  return spots.sort((a, b) => b.skips - a.skips).slice(0, limit);
}

// Splits a speech into memorisation chunks: paragraphs (runs of non-blank
// lines), each cut into whole sentences grouped up to `maxWords` words. A
// sentence longer than that is a chunk on its own. Pure; checked by chunks.check.ts.

export interface Chunk {
  start: number; // char range in the text, whitespace-trimmed
  end: number;
}

export function chunkSpeech(text: string, maxWords = 40): Chunk[] {
  const chunks: Chunk[] = [];
  for (const p of text.matchAll(/(?:[^\n]*\S[^\n]*(?:\n|$))+/g)) {
    let cur: Chunk | null = null;
    let words = 0;
    for (const m of p[0].matchAll(/[^.!?]+[.!?]*/g)) {
      const at = (p.index ?? 0) + (m.index ?? 0);
      const start = at + (m[0].length - m[0].trimStart().length);
      const end = at + m[0].trimEnd().length;
      if (end <= start) continue;
      const w = (m[0].match(/\S+/g) ?? []).length;
      if (cur && words + w > maxWords) {
        chunks.push(cur);
        cur = null;
      }
      if (cur) {
        cur.end = end;
        words += w;
      } else {
        cur = { start, end };
        words = w;
      }
    }
    if (cur) chunks.push(cur);
  }
  return chunks;
}

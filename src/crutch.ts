// Word-choice habits over a session's normalized tokens: stutter repeats, hedges,
// and overused words. Pure; checked by crutch.check.ts. Kept disjoint from the
// filler detector in main.ts (which already owns "kind of", "actually", "like"…)
// so nothing counts twice.
//
// ponytail: whisper (esp. the medium correction model) tidies stutters out of its
// text, so `repeats` undercounts; a signal, not a measurement.

export interface Crutch {
  repeats: number;
  hedges: [string, number][]; // most frequent first
  overused: [string, number][]; // top 3, most frequent first
}

const HEDGES = [
  "i think", "i guess", "i suppose", "i believe", "maybe", "perhaps", "probably",
  "a bit", "a little", "somewhat", "pretty much",
].map((h) => h.split(" "));

// Doubles that are often grammatical ("I know that that's…", "she had had…").
const LEGIT_DOUBLES = new Set(["that", "had", "is", "do"]);

const STOPWORDS = new Set(
  ("a an the and or but so if then than that this these those it its it's i i'm i've i'd me my we " +
    "we're our you you're your he she they they're them their his her him is are was were be been being " +
    "am do does did done have has had having to of in on at by for with from as into about up down out " +
    "over not no yes there here what which who whom when where why how all any some can could would " +
    "should will just like well right okay ok yeah um uh er ah hmm mm also very too more most one").split(" "),
);

const isHesitation = (w: string) => /^(u[hm]+|e+r+m*|a+h+|h+m+|m+)$/.test(w);

export function crutch(words: string[]): Crutch {
  let repeats = 0;
  const hedges = new Map<string, number>();
  const counts = new Map<string, number>();
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (i > 0 && w === words[i - 1] && !LEGIT_DOUBLES.has(w) && !isHesitation(w)) repeats++;
    const h = HEDGES.find((p) => p.every((t, k) => words[i + k] === t));
    if (h) hedges.set(h.join(" "), (hedges.get(h.join(" ")) ?? 0) + 1);
    if (w.length > 2 && !STOPWORDS.has(w) && !isHesitation(w) && !/^\d+$/.test(w))
      counts.set(w, (counts.get(w) ?? 0) + 1);
  }
  const min = Math.max(4, words.length * 0.02);
  const byCount = (m: Map<string, number>) => [...m].sort((a, b) => b[1] - a[1]);
  return {
    repeats,
    hedges: byCount(hedges),
    overused: byCount(counts).filter(([, n]) => n >= min).slice(0, 3),
  };
}

export const hedgeCount = (c: Crutch) => c.hedges.reduce((a, [, n]) => a + n, 0);

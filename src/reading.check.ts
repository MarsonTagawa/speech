// Run: bun src/reading.check.ts
import assert from "node:assert";
import { type Reading, frameAt, groupScreens, isRead, reading, runsOf, slideRows, slideScore, slideSpans, withoutSpoken } from "./reading";

const w = (s: string) => s.toLowerCase().split(/\s+/).filter(Boolean);
const shown = w("Revenue grew 40 percent in Q3 driven by enterprise renewals");
const verbatim = w("revenue grew 40 um percent in q3 driven by enterprise renewals");
const around = w("so the big story this quarter is that our biggest customers kept coming back");
const at = (index: number, slide: number, ms: number, said: string[], s = shown) => ({ index, slide, ms, dur: 2000, said, shown: s });

// Slide 2 read verbatim (an "um" doesn't break a run) over two lines; slide 3 talked around.
const r = reading([at(1, 1, 0, verbatim), at(2, 1, 2500, verbatim), at(3, 2, 6000, around)]);
assert.ok(r);
assert.equal(r.frac, 16 / (16 + 12));
assert.deepStrictEqual(r.slides, [
  { slide: 2, frac: 1, runs: 16, index: 1, ms: 0, spokeMs: 4000 },
  { slide: 3, frac: 0, runs: 12, index: 3, ms: 6000, spokeMs: 2000 },
]);
assert.deepStrictEqual(r.slides.map(isRead), [true, false]);

// A slide with no text (image) still gets a row, with no runs; it isn't judged.
const img = reading([at(1, 1, 0, verbatim), at(2, 1, 2500, verbatim), at(3, 4, 6000, around, [])]);
assert.deepStrictEqual(img?.slides[1], { slide: 5, frac: 0, runs: 0, index: 3, ms: 6000, spokeMs: 2000 });
assert.equal(isRead(img!.slides[1]), false);

// Too few runs on a slide: frac 0, not judged.
const few = reading([at(1, 1, 0, verbatim), at(2, 1, 2500, verbatim), at(3, 2, 6000, w("revenue grew 40 percent"))]);
assert.deepStrictEqual(few?.slides[1], { slide: 3, frac: 0, runs: 2, index: 3, ms: 6000, spokeMs: 2000 });

// Too little speech over text slides to judge reading: rows still come back
// (times and grades need them), with no overall share. Nothing said: no result.
const imageDeck = reading([at(1, 0, 0, w("a b c d e f g h i j k l m n"), [])]);
assert.equal(imageDeck?.frac, null);
assert.deepStrictEqual(imageDeck?.slides.map((x) => x.slide), [1]);
assert.equal(reading([at(1, 1, 0, w("revenue grew 40 percent"))])?.frac, null);
assert.equal(reading([]), null);

// The first cut saved `slides` as slide numbers: no rows, no throw.
assert.deepStrictEqual(slideRows({ frac: 0.5, slides: [2, 3] } as unknown as Reading), []);
assert.equal(slideRows(r).length, 2);

// --- Screen OCR source -------------------------------------------------------

// The app's own transcript on the recorded display: runs already spoken are cut,
// and no new run spans a cut.
const spoken = new Set(runsOf(w("we grew a lot this year")));
assert.deepStrictEqual(withoutSpoken(w("Transcript we grew a lot this year Revenue grew 40 percent"), spoken), w("transcript ¦ revenue grew 40 percent"));

const slideA = w("Revenue grew 40 percent in Q3 driven by enterprise renewals");
const slideB = w("Hiring plan two new teams in Berlin and Lisbon next spring");
// Frames carry their cut text (`text`, see withoutSpoken) and the OCR as read (`raw`).
const fr = (ms: number, text: string[], raw = text) => ({ ms, text, raw });
const g = groupScreens([
  fr(0, slideA),
  fr(3000, [...slideA, "12:01"]), // clock ticked: same slide
  fr(6000, w("ok")), // blank (too few words)
  fr(9000, slideB),
  fr(15000, slideA), // back to the first slide
]);
assert.deepStrictEqual(g.map((x) => x.slide), [0, 0, -1, 1, 0]);
assert.deepStrictEqual(g[1].text, slideA); // a slide's text is its first frame's

// Review focus 3: the same app chrome/transcript on every frame, once cut, doesn't merge slides.
const chrome = w("live transcript so the big story this quarter");
const said = new Set(runsOf(chrome));
const g2 = groupScreens([slideA, slideB].map((t, i) => fr(i * 5000, withoutSpoken([...chrome, ...t], said), [...chrome, ...t])));
assert.deepStrictEqual(g2.map((x) => x.slide), [0, 1]);

// Review I-1: text on every frame that was never said (a browser's menus, a deck
// footer) doesn't make every slide one slide.
const menus = w("file edit view insert format slide arrange tools extensions help");
const title = w("Q3 results Acme Corp board meeting");
const g3 = groupScreens([title, slideA, slideB].map((t, i) => fr(i * 5000, [...menus, ...t])));
assert.deepStrictEqual(g3.map((x) => x.slide), [0, 1, 2]);

// …even when one slide is most of the frames: its own text isn't mistaken for chrome.
const g4 = groupScreens([slideA, slideA, slideA, slideA, slideB].map((t, i) => fr(i * 5000, t)));
assert.deepStrictEqual(g4.map((x) => x.slide), [0, 0, 0, 0, 1]);

// Review I-2: back on a slide already read aloud, its cut text is gone, but the
// frame is still that slide, scored against its first frame's text.
const g5 = groupScreens([fr(0, slideA), fr(5000, slideB), fr(9000, w("¦"), slideA)]);
assert.deepStrictEqual(g5.map((x) => x.slide), [0, 1, 0]);
assert.deepStrictEqual(g5[2].text, slideA);

// The frame on screen when an utterance started: the last at or before it.
const frames = [{ ms: 0 }, { ms: 3000 }, { ms: 9000 }];
assert.equal(frameAt(frames, 2999), frames[0]);
assert.equal(frameAt(frames, 3000), frames[1]);
assert.equal(frameAt([{ ms: 500 }], 100), undefined);

// --- Slide times and grades --------------------------------------------------

// Slide changes on the session clock → spans; repeats merge, -1 (no slide) ends one.
assert.deepStrictEqual(
  slideSpans([[0, 0], [4000, 1], [4000, 1], [9000, -1], [12000, 0], [15000, 0]], 20000),
  [
    { slide: 0, start: 0, end: 4000 },
    { slide: 1, start: 4000, end: 9000 },
    { slide: 0, start: 12000, end: 20000 },
  ],
);
assert.deepStrictEqual(slideSpans([], 5000), []);

// A slide's delivery: clean pace, few fillers and pauses, talked around → high;
// read word for word with fillers → low. No reading data just leaves it out.
const good = slideScore({ pace: 100, fillersPerMin: 0.5, pausesPerMin: 2, read: 0 });
const bad = slideScore({ pace: 100, fillersPerMin: 10, pausesPerMin: 2, read: 1 });
assert.equal(good, 100);
assert.ok(bad < 50, String(bad));
assert.equal(slideScore({ pace: 100, fillersPerMin: 0.5, pausesPerMin: 2, read: null }), 100);
console.log("reading ok");

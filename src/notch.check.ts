// Run: bun src/notch.check.ts
import assert from "node:assert";
import { notchState, scriptAhead, serial } from "./notch";

// scriptAhead: the next n script words from the cursor, original text, whitespace collapsed.
const text = "Good  morning everyone, thanks\nfor coming today.";
const toks = [...text.matchAll(/[\w']+/g)].map((m) => ({ start: m.index!, end: m.index! + m[0].length }));
assert.strictEqual(scriptAhead(text, toks, 0, 3), "Good morning everyone");
assert.strictEqual(scriptAhead(text, toks, 3), "thanks for coming today"); // n clamps at the end
assert.strictEqual(scriptAhead(text, toks, 6, 8), "today");
assert.strictEqual(scriptAhead(text, toks, 7), ""); // read to the end
assert.strictEqual(scriptAhead("", [], 0), ""); // no script

// notchState: band, fpm formatting, and toggled-off fields.
const base = { rec: true, time: "1:05", wpm: 130, low: 120, high: 160, fpm: 1.25, script: "next words" };
assert.deepStrictEqual(notchState(base), { rec: true, time: "1:05", wpm: 130, band: "in", fpm: "1.3", script: "next words" });
assert.strictEqual(notchState({ ...base, wpm: 190 }).band, "out");
assert.strictEqual(notchState({ ...base, wpm: 100 }).band, "out");
assert.strictEqual(notchState({ ...base, wpm: 160 }).band, "in"); // band is inclusive
assert.strictEqual(notchState({ ...base, wpm: 0 }).band, ""); // nothing said yet: neutral
assert.strictEqual(notchState({ ...base, time: null }).time, null);
assert.strictEqual(notchState({ ...base, script: null }).script, null); // Script off: row gone
assert.strictEqual(notchState({ ...base, script: "" }).script, ""); // on, nothing ahead: row kept, empty
// serial: overlapping calls run one at a time, in call order, even when an
// earlier one is slower (Show on, then quickly off, must end off).
const log: string[] = [];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const run = serial(async (tag: string, ms: number) => {
  log.push(`start ${tag}`);
  await sleep(ms);
  log.push(`end ${tag}`);
});
void run("on", 30);
await run("off", 0);
assert.deepStrictEqual(log, ["start on", "end on", "start off", "end off"]);
// A failed call doesn't wedge the queue.
const flaky = serial(async (fail: boolean) => {
  if (fail) throw new Error("boom");
});
await assert.rejects(flaky(true));
await flaky(false);
console.log("notch ok");

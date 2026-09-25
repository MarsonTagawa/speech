// Run: bun src/passages.check.ts
import assert from "node:assert";
import { troubleSpots } from "./passages";

const text = "Hello there. We gather today!\nThanks all";
const tokens = [...text.matchAll(/\w+/g)].map((m) => ({ start: m.index!, end: m.index! + m[0].length }));
// Hello there | We gather today | Thanks all
const counts = [0, 1, 2, 0, 1, 0, 3];
const spots = troubleSpots(text, tokens, counts);
assert.deepStrictEqual(
  spots.map((s) => [text.slice(s.start, s.end), s.skips]),
  [["We gather today!", 3], ["Thanks all", 3]],
);
assert.deepStrictEqual(troubleSpots(text, tokens, counts.map(() => 0)), []);
console.log("passages ok");

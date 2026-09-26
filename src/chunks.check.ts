// Run: bun src/chunks.check.ts
import assert from "node:assert";
import { chunkSpeech } from "./chunks";

const slices = (text: string, max?: number) => chunkSpeech(text, max).map((c) => text.slice(c.start, c.end));

// Sentences group up to the word limit; paragraphs never merge.
const text = "One two three. Four five. Six seven eight nine.\n\n  Ten eleven!\nTwelve.";
assert.deepStrictEqual(slices(text, 5), ["One two three. Four five.", "Six seven eight nine.", "Ten eleven!\nTwelve."]);
assert.deepStrictEqual(slices(text), ["One two three. Four five. Six seven eight nine.", "Ten eleven!\nTwelve."]);
// An over-long sentence stands alone; no trailing punctuation is fine.
assert.deepStrictEqual(slices("a b c d e f. g h", 3), ["a b c d e f.", "g h"]);
assert.deepStrictEqual(chunkSpeech("   \n\n "), []);
console.log("chunks ok");

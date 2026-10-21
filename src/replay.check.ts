// Run: bun src/replay.check.ts
import assert from "node:assert";
import { fitRect, bubbleRect, due } from "./replay";

// 16:9 into 16:9 fills; 4:3 into 16:9 pillarboxes; portrait letterboxes centred.
assert.deepEqual(fitRect(1280, 720, 640, 360), { x: 0, y: 0, w: 640, h: 360 });
assert.deepEqual(fitRect(1024, 768, 640, 360), { x: 80, y: 0, w: 480, h: 360 });
assert.deepEqual(fitRect(1080, 1920, 640, 360), { x: 219, y: 0, w: 203, h: 360 });
// Bubble: 22% wide, 16:9, inset 2% from the bottom-right corner.
assert.deepEqual(bubbleRect(1000, 562), { x: 760, y: 418, w: 220, h: 124 });
// Fetch when never fetched, when far enough from the last fetch, or after seeking back.
assert.equal(due(-1, 0, 250), true);
assert.equal(due(1000, 1100, 250), false);
assert.equal(due(1000, 1250, 250), true);
assert.equal(due(5000, 1000, 250), true);
console.log("replay ok");

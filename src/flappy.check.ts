// Run: bun src/flappy.check.ts
import assert from "node:assert";
import { flap, newState, step } from "./flappy";

const W = 400, H = 200;

// waits for the first click, then gravity pulls it down after a flap
let s = newState(H);
step(s, 0.5, W, H);
assert.equal(s.mode, "ready");
assert.equal(s.y, H / 2);
flap(s, H);
assert(s.mode === "play" && s.vy < 0);
for (let i = 0; i < 40; i++) step(s, 0.02, W, H);
assert(s.vy > 0, `falls after the flap, vy ${s.vy}`);

// clearing a pipe scores; hitting one crashes
s = { ...newState(H), mode: "play", y: 100, pipes: [{ x: 60, gapY: 100, scored: false }] };
step(s, 0.02, W, H);
assert.equal(s.score, 1);
assert.equal(s.mode, "play");
s = { ...newState(H), mode: "play", y: 30, pipes: [{ x: 95, gapY: 120, scored: false }] };
step(s, 0.02, W, H);
assert.equal(s.mode, "dead");

// hitting the floor crashes
s = { ...newState(H), mode: "play", y: H - 7, vy: 200 };
step(s, 0.02, W, H);
assert.equal(s.mode, "dead");

// a click right after the crash is ignored; a later one restarts, best kept
s = { ...newState(H), mode: "play", y: H - 7, vy: 200, score: 3, best: 3 };
step(s, 0.02, W, H);
flap(s, H);
assert.equal(s.mode, "dead");
step(s, 0.6, W, H);
flap(s, H);
assert(s.mode === "play" && s.score === 0 && s.best === 3 && s.pipes.length === 0);

console.log("flappy ok");

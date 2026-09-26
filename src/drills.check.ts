// Run: bun src/drills.check.ts
import assert from "node:assert";
import { drillById, type DrillStats } from "./drills";

const base: DrillStats = { durationMs: 40_000, wpm: 140, fillers: 0, fillersPerMin: 0, pausesPerMin: 5, trailingOff: 0.95, script: null };
const band: [number, number] = [120, 160];
const pass = (id: string, s: Partial<DrillStats>) => drillById(id)!.judge({ ...base, ...s }, band).pass;

assert(pass("filler-free", {}));
assert(!pass("filler-free", { fillers: 1 }));
assert(!pass("filler-free", { durationMs: 10_000 })); // too short to count
assert(pass("pace-lock", {}));
assert(!pass("pace-lock", { wpm: 170 }));
assert(pass("pause-not-um", {}));
assert(!pass("pause-not-um", { pausesPerMin: 2 }));
assert(!pass("pause-not-um", { fillersPerMin: 1.5 }));
assert(pass("twisters", { script: { accuracy: 92 }, durationMs: 5_000 }));
assert(!pass("twisters", { script: null }));
assert(!pass("finish-strong", { trailingOff: 0.7 }));
assert(pass("memorise", { script: { accuracy: 90 }, durationMs: 3_000 }));
assert(!pass("memorise", { script: { accuracy: 80 } }));
console.log("drills ok");

// Run: bun src/drills.check.ts
import assert from "node:assert";
import { DRILLS, drillById, type DrillStats } from "./drills";
import type { Presence } from "./presence";

const base: DrillStats = { durationMs: 40_000, wpm: 140, fillers: 0, fillersPerMin: 0, pausesPerMin: 5, trailingOff: 0.95, pitchRange: 4.5, uptalk: 0, script: null };
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
assert(pass("vary-pitch", {}));
assert(!pass("vary-pitch", { pitchRange: 2.5 }));
assert(pass("land-it", { uptalk: 1 }));
assert(!pass("land-it", { uptalk: 2 }));
assert(pass("impromptu", { durationMs: 60_000 }));
assert(!pass("impromptu", { durationMs: 60_000, fillersPerMin: 2.5 }));
assert(!pass("impromptu", { durationMs: 60_000, wpm: 100 }));
assert(!pass("impromptu", {})); // 40 s voiced is under half of 90
const cam = (o: Partial<Presence>): Partial<DrillStats> => ({
  durationMs: 40_000,
  presence: { eyeContact: 0.8, gestureFrac: 0.3, gestures: { list: 1, palmsup: 4, growth: 1 }, ...o } as Presence,
});
assert(pass("eyes-hands", cam({})));
assert(!pass("eyes-hands", {})); // camera off
assert(!pass("eyes-hands", cam({ eyeContact: 0.5 })));
assert(!pass("eyes-hands", cam({ gestureFrac: 0.05 }))); // hands still
assert(!pass("eyes-hands", cam({ gestureFrac: 0.9 }))); // flailing
assert(!pass("eyes-hands", cam({ gestures: { list: 3, palmsup: 2 } }))); // only 2 kinds
assert(!pass("eyes-hands", cam({ gestures: undefined }))); // session from before gestures
assert(!drillById("pace-lock")!.video && drillById("eyes-hands")!.video);
console.log("drills ok");

// Every drill has its own icon.
const icons = DRILLS.map((d) => d.icon);
assert(icons.every(Boolean) && new Set(icons).size === icons.length);

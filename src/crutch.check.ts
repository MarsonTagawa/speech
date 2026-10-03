// Run: bun src/crutch.check.ts
import assert from "node:assert";
import { crutch, hedgeCount } from "./crutch";

const w = (s: string) => s.toLowerCase().split(/\s+/);
const c = crutch(w("i think the the plan is really really good and i think maybe that that is fine um um really really"));
assert.equal(c.repeats, 3); // the the, really really ×2; not "that that", not "um um"
assert.deepStrictEqual(c.hedges, [["i think", 2], ["maybe", 1]]);
assert.equal(hedgeCount(c), 3);
assert.deepStrictEqual(c.overused, [["really", 4]]);
assert.deepStrictEqual(crutch([]), { repeats: 0, hedges: [], overused: [] });
console.log("crutch ok");

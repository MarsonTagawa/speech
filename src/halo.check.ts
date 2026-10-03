// Run: bun src/halo.check.ts
import assert from "node:assert";
import { expressionFromDefinition, renderAvatarDefinition, renderAvatarExpression } from "@bible-strong/avatar-core";
import { silhouette } from "./halo";
import strobi from "./strobi.avatar.json";

const g = renderAvatarDefinition(strobi as never).geometry;
const sh = silhouette([...g.backPaths, g.headPath, ...g.frontPaths]);
const at = (deg: number) => sh[Math.round(((deg % 360) + 360) % 360)];
assert(Math.abs(at(270) - 122.31 / 120) < 0.01, `top is the head circle, got ${at(270)}`); // svg y is down
assert(at(165) > 1.25 && at(15) > 1.25, `arms stick out at the sides, got ${at(165)} ${at(15)}`);
assert.deepStrictEqual([...silhouette([])], new Array(360).fill(1)); // no paths yet: round

// Tipped/turned (cursor below or to the side): foreshortened arms have sparse
// path points; the outline must still reach their true edge, sampled densely.
const N = expressionFromDefinition("neutral", (strobi as never as { expressions: { neutral: never } }).expressions.neutral);
for (const [headX, headY] of [[-35, 0], [-35, 30], [35, 0], [0, 40]]) {
  const t = renderAvatarExpression(strobi as never, { ...N, headX, headY, headZ: 0 }).geometry;
  const ds = [...t.backPaths, t.headPath, ...t.frontPaths];
  const out = silhouette(ds);
  for (const d of ds.filter((d) => !d.includes("A"))) {
    let px = 0, py = 0;
    for (const c of d.match(/[MC][^MCZ]*/g) ?? []) {
      const n = c.slice(1).trim().split(/[\s,]+/).map(Number);
      if (c[0] === "M") { [px, py] = n; continue; }
      for (let k = 0; k + 5 < n.length; k += 6) {
        const [x1, y1, x2, y2, x, y] = n.slice(k, k + 6);
        for (let s = 0; s <= 1; s += 0.02) {
          const u = 1 - s, X = u * u * u * px + 3 * u * u * s * x1 + 3 * u * s * s * x2 + s * s * s * x, Y = u * u * u * py + 3 * u * u * s * y1 + 3 * u * s * s * y2 + s * s * s * y;
          const i = Math.round(((Math.atan2(Y, X) * 180) / Math.PI + 360) % 360) % 360;
          assert(out[i] * 120 > Math.hypot(X, Y) - 1.5, `head ${headX},${headY}: arm pokes ${(Math.hypot(X, Y) - out[i] * 120).toFixed(1)} units past the outline at ${i}°`);
        }
        [px, py] = [x, y];
      }
    }
  }
}
console.log("halo ok");

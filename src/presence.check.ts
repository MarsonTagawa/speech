// Run: bun src/presence.check.ts
import assert from "node:assert";
import { headAngles, toFrame, scoreLinear, type RawResults } from "./presence";

const near = (a: number, b: number, eps = 0.01) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const rad = (d: number) => (d * Math.PI) / 180;
// 4×4 flattened column-major (MediaPipe's layout) from a 3×3 rotation, with translation.
const mat = (R: number[][]): number[] => {
  const m = new Array(16).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m[c * 4 + r] = R[r][c];
  m[12] = 1; m[13] = 2; m[14] = -30; m[15] = 1;
  return m;
};
const yawM = (d: number) => mat([[Math.cos(rad(d)), 0, Math.sin(rad(d))], [0, 1, 0], [-Math.sin(rad(d)), 0, Math.cos(rad(d))]]);
const pitchM = (d: number) => mat([[1, 0, 0], [0, Math.cos(rad(d)), -Math.sin(rad(d))], [0, Math.sin(rad(d)), Math.cos(rad(d))]]);
const transpose = (m: number[]) => m.map((_, i) => m[(i % 4) * 4 + Math.floor(i / 4)]);

// scoreLinear (moved from main.ts): unchanged behaviour.
assert.equal(scoreLinear(1, 1, 12), 100);
assert.equal(scoreLinear(12, 1, 12), 0);
assert.equal(scoreLinear(6.5, 1, 12), 50);
assert.equal(scoreLinear(3, 5, 5), 100);

// headAngles: magnitudes are right in either flattening order.
near(headAngles(mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).yaw, 0);
near(Math.abs(headAngles(yawM(30)).yaw), 30);
near(Math.abs(headAngles(yawM(30)).pitch), 0);
near(Math.abs(headAngles(pitchM(20)).pitch), 20);
near(Math.abs(headAngles(transpose(pitchM(20))).pitch), 20);
near(Math.abs(headAngles(transpose(yawM(30))).yaw), 30);

// toFrame
const facing: Record<string, number> = { mouthSmileLeft: 0.4, mouthSmileRight: 0.6, browInnerUp: 0.2, browOuterUpLeft: 0.3, eyeLookDownLeft: 0.2 };
const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5 }));
pose[0] = { x: 0.5, y: 0.3 }; // nose
pose[11] = { x: 0.4, y: 0.5 }; // left shoulder
pose[12] = { x: 0.6, y: 0.5 }; // right shoulder
const raw = (o: Partial<RawResults> = {}): RawResults => ({ blend: facing, matrix: yawM(5), pose, wrists: [], ...o });

const none = toFrame(raw({ blend: null, matrix: null, pose: null }), 640, 360, 0);
assert.deepStrictEqual(none, { t: 0, face: false, looking: false, smile: 0, brows: 0, body: null, wrists: [] });

const f = toFrame(raw({ wrists: [{ x: 0.5, y: 0.8 }] }), 640, 360, 42);
assert.equal(f.t, 42);
assert.equal(f.face, true);
assert.equal(f.looking, true); // 5° yaw, eyes 0.2 < 0.5
near(f.smile, 0.5);
near(f.brows, 0.3);
// shoulders 128 px apart; midpoint (320,180) px → (2.5, 1.40625) shoulder widths
near(f.body![0][0], 2.5);
near(f.body![0][1], 1.40625);
near(f.body![1][1], 108 / 128); // nose y = 0.3·360
near(f.wrists[0][1], 288 / 128); // wrist y = 0.8·360

assert.equal(toFrame(raw({ matrix: yawM(30) }), 640, 360, 0).looking, false); // head turned
assert.equal(toFrame(raw({ blend: { ...facing, eyeLookOutLeft: 0.7 } }), 640, 360, 0).looking, false); // eyes away
assert.equal(toFrame(raw({ matrix: null }), 640, 360, 0).looking, false); // no matrix → can't tell → not looking

// Degenerate pose (shoulders coincide): no body, no wrists, nothing infinite.
const flat = pose.map((p) => ({ ...p }));
flat[12] = { ...flat[11] };
const d = toFrame(raw({ pose: flat, wrists: [{ x: 0.5, y: 0.8 }] }), 640, 360, 0);
assert.equal(d.body, null);
assert.deepStrictEqual(d.wrists, []);

console.log("presence part 1 ok");

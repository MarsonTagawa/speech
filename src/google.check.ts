// Run: bun src/google.check.ts
import assert from "node:assert";
import { googleFile } from "./google";

assert.deepEqual(googleFile("https://docs.google.com/presentation/d/1AbC-d_9/edit#slide=id.p"), { kind: "presentation", id: "1AbC-d_9" });
assert.deepEqual(googleFile("docs.google.com/document/d/XYZ/edit?usp=sharing"), { kind: "document", id: "XYZ" });
assert.deepEqual(googleFile("https://docs.google.com/spreadsheets/d/S1/edit"), { kind: "spreadsheets", id: "S1" });
assert.deepEqual(googleFile("https://drive.google.com/file/d/F1le/view?usp=drive_link"), { kind: "file", id: "F1le" });
assert.deepEqual(googleFile("https://drive.google.com/open?id=F2"), { kind: "file", id: "F2" });
assert.deepEqual(googleFile("https://drive.google.com/uc?export=download&id=F3"), { kind: "file", id: "F3" });
assert.equal(googleFile("https://example.com/presentation/d/1"), null);
assert.equal(googleFile("https://drive.google.com/drive/folders/abc"), null);
console.log("google ok");

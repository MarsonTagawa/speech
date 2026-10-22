// Camera presence: the camera and all vision inference run in Rust
// (src-tauri/src/vision) so the UI thread stays free — doing it here blocked
// animations for ~55 ms a tick. This file turns Rust's messages into Frames
// and paints its JPEG preview.
import { Channel, invoke } from "@tauri-apps/api/core";
import { toFrame, type Frame, type Pt } from "./presence";

export const FRAME_W = 640;
export const FRAME_H = 360;
// MediaPipe's 21-landmark hand skeleton (HandLandmarker.HAND_CONNECTIONS).
export const HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
].map(([start, end]) => ({ start, end }));

interface VisionFrame {
  t: number;
  blend: Record<string, number> | null;
  head: { yaw: number; pitch: number } | null;
  pose: Pt[] | null;
  hands: Pt[][];
  handed: number[];
  world: number[][][];
}
type VisionMsg = VisionFrame | { ended: true };

let gen = 0; // bumps on every start/stop so a stale session's messages are ignored
const REPORT_MS = 5000;
let recent: Omit<VisionFrame, "blend">[] = []; // the last REPORT_MS of raw results, for a gesture report

// The last few seconds as Rust sent them (no blendshapes), to replay through the gesture code.
export const recentVision = () => recent;

// onFrame gets the scored Frame plus each hand's 21 normalised landmarks (for
// the live highlight only; never stored).
export async function startCamera(view: HTMLCanvasElement, onFrame: (f: Frame, hands: Pt[][]) => void, onEnded: () => void): Promise<void> {
  const my = ++gen;
  const results = new Channel<VisionMsg>();
  let last: Frame | undefined; // smooths each hand's palm direction frame to frame
  recent = [];
  results.onmessage = (m) => {
    if (my !== gen) return;
    if ("ended" in m) return onEnded();
    const { blend: _, ...raw } = m;
    recent.push(raw);
    recent = recent.filter((r) => m.t - r.t <= REPORT_MS);
    last = toFrame({ blend: m.blend, head: m.head, pose: m.pose, hands: m.hands, handed: m.handed, world: m.world }, FRAME_W, FRAME_H, m.t, last);
    onFrame(last, m.hands);
  };
  const preview = new Channel<ArrayBuffer>();
  const cx = view.getContext("2d")!;
  let [got, drawn] = [0, 0]; // decodes can finish out of order at 30 fps: never paint an older frame over a newer one
  preview.onmessage = async (buf) => {
    if (my !== gen) return;
    const n = ++got;
    const bmp = await createImageBitmap(new Blob([buf], { type: "image/jpeg" })); // decoded off the main thread
    if (n < drawn) return bmp.close();
    drawn = n;
    if (view.width !== bmp.width || view.height !== bmp.height) [view.width, view.height] = [bmp.width, bmp.height];
    cx.drawImage(bmp, 0, 0);
    bmp.close();
  };
  await invoke("start_vision", { results, preview });
}

export function stopCamera() {
  gen++;
  void invoke("stop_vision");
}

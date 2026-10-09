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
}
type VisionMsg = VisionFrame | { ended: true };

let gen = 0; // bumps on every start/stop so a stale session's messages are ignored

// onFrame gets the scored Frame plus each hand's 21 normalised landmarks (for
// the live highlight only; never stored).
export async function startCamera(view: HTMLCanvasElement, onFrame: (f: Frame, hands: Pt[][]) => void, onEnded: () => void): Promise<void> {
  const my = ++gen;
  const results = new Channel<VisionMsg>();
  results.onmessage = (m) => {
    if (my !== gen) return;
    if ("ended" in m) return onEnded();
    onFrame(toFrame({ blend: m.blend, head: m.head, pose: m.pose, wrists: m.hands.map((h) => h[0]) }, FRAME_W, FRAME_H, m.t), m.hands);
  };
  const preview = new Channel<ArrayBuffer>();
  const cx = view.getContext("2d")!;
  preview.onmessage = async (buf) => {
    if (my !== gen) return;
    const bmp = await createImageBitmap(new Blob([buf], { type: "image/jpeg" })); // decoded off the main thread
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

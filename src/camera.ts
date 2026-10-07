// Webcam capture + MediaPipe inference for presence grading. Pure scoring is in
// presence.ts; this file only turns camera frames into Frames. Assets are
// served from public/mediapipe/ (see README). Nothing here loads until the
// camera is first switched on.
import { FaceLandmarker, FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import { toFrame, type Frame } from "./presence";

const BASE = "/mediapipe";
// Inference size. The camera delivers 1080p whatever we ask; feeding that
// straight in halves the frame rate (spike, 2026-10-07).
const W = 640;
const H = 360;
const STEP_MS = 80; // ~12 fps

interface Models {
  face: FaceLandmarker;
  pose: PoseLandmarker;
  hand: HandLandmarker;
}

async function create(delegate: "GPU" | "CPU"): Promise<Models> {
  const files = await FilesetResolver.forVisionTasks(BASE);
  const base = (file: string) => ({ baseOptions: { modelAssetPath: `${BASE}/${file}`, delegate }, runningMode: "VIDEO" as const });
  // Sequential: the spike created them one after another; untested in parallel.
  const face = await FaceLandmarker.createFromOptions(files, {
    ...base("face_landmarker.task"),
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
  const pose = await PoseLandmarker.createFromOptions(files, base("pose_landmarker_lite.task"));
  const hand = await HandLandmarker.createFromOptions(files, { ...base("hand_landmarker.task"), numHands: 2 });
  return { face, pose, hand };
}

// Loaded once per app run; GPU first, CPU (~10 fps) if WebGL init fails.
let models: Promise<Models> | null = null;
function loadModels(): Promise<Models> {
  models ??= create("GPU")
    .catch(() => create("CPU"))
    .catch((e) => {
      models = null; // let a later toggle retry
      throw new Error(`Camera models missing — see README (${e})`);
    });
  return models;
}

let stream: MediaStream | null = null;
let raf = 0;
let gen = 0; // bumps on every start/stop so a slow start can't outlive a stop

export async function startCamera(video: HTMLVideoElement, onFrame: (f: Frame) => void, onEnded: () => void): Promise<void> {
  stopCamera();
  const my = ++gen;
  const m = await loadModels();
  const s = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: W }, height: { ideal: H }, frameRate: { ideal: 30 } },
  });
  if (my !== gen) {
    s.getTracks().forEach((t) => t.stop());
    return;
  }
  stream = s;
  s.getVideoTracks()[0]?.addEventListener("ended", () => {
    if (my === gen) {
      stopCamera();
      onEnded();
    }
  });
  video.srcObject = s;
  await video.play();

  const cv = document.createElement("canvas");
  cv.width = W;
  cv.height = H;
  const cx = cv.getContext("2d")!;
  let last = 0;
  const tick = (now: number) => {
    if (my !== gen) return;
    raf = requestAnimationFrame(tick);
    if (now - last < STEP_MS || video.readyState < 2) return;
    last = now;
    cx.drawImage(video, 0, 0, W, H);
    const f = m.face.detectForVideo(cv, now);
    const p = m.pose.detectForVideo(cv, now);
    const h = m.hand.detectForVideo(cv, now);
    const shapes = f.faceBlendshapes[0]?.categories;
    onFrame(
      toFrame(
        {
          blend: shapes ? Object.fromEntries(shapes.map((c) => [c.categoryName, c.score])) : null,
          matrix: f.facialTransformationMatrixes[0]?.data ?? null,
          pose: p.landmarks[0] ?? null,
          wrists: h.landmarks.map((l) => l[0]),
        },
        W,
        H,
        now,
      ),
    );
  };
  raf = requestAnimationFrame(tick);
}

export function stopCamera() {
  gen++;
  cancelAnimationFrame(raf);
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
}

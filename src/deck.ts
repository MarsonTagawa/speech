// Slide deck to present from: PDF pages or image files, one slide each,
// painted into a single canvas (CSS object-fit letterboxes it). WebKitGTK has
// no built-in PDF viewer, so PDFs go through pdf.js.
/// <reference types="vite/client" />
import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

interface Slide {
  w: number;
  h: number;
  text: string; // the slide's own words (PDF text layer); "" for images
  // `track`: a render paintDeck may cancel (thumbnails aren't).
  draw: (cx: CanvasRenderingContext2D, scale: number, track: boolean) => Promise<void>;
}

let slides: Slide[] = [];
let at = 0;
let task: pdfjs.RenderTask | undefined;
let paints = 0;

async function pdfSlides(data: Uint8Array): Promise<Slide[]> {
  const doc = await pdfjs.getDocument({ data }).promise;
  const out: Slide[] = [];
  for (let n = 1; n <= doc.numPages; n++) {
    const page = await doc.getPage(n);
    const { width, height } = page.getViewport({ scale: 1 });
    const { items } = await page.getTextContent();
    out.push({
      w: width,
      h: height,
      text: items.map((i) => ("str" in i ? i.str : "")).join(" "),
      draw: async (cx, scale, track) => {
        const t = page.render({ canvas: cx.canvas, canvasContext: cx, viewport: page.getViewport({ scale }) });
        if (track) task = t;
        await t.promise;
      },
    });
  }
  return out;
}

async function imageSlide(file: File): Promise<Slide> {
  const img = await createImageBitmap(file);
  return { w: img.width, h: img.height, text: "", draw: async (cx, s) => cx.drawImage(img, 0, 0, img.width * s, img.height * s) };
}

// Files in picker order; a PDF adds all its pages. Throws on anything else.
export async function loadDeck(files: File[]): Promise<number> {
  const out: Slide[] = [];
  for (const f of files) {
    if (f.type === "application/pdf" || /\.pdf$/i.test(f.name)) out.push(...(await pdfSlides(new Uint8Array(await f.arrayBuffer()))));
    else if (f.type.startsWith("image/")) out.push(await imageSlide(f));
    else throw new Error(`${f.name}: export it as PDF first (PowerPoint, Keynote, Google Slides and Docs all can)`);
  }
  return use(out);
}

// A PDF already in memory (downloaded from Google).
export const loadPdf = async (data: Uint8Array) => use(await pdfSlides(data));

function use(out: Slide[]) {
  if (!out.length) throw new Error("no pages in that file");
  task?.cancel();
  slides = out;
  at = 0;
  return out.length;
}

export const deckSize = () => slides.length;
export const deckAt = () => at;
export const deckText = () => slides[at]?.text ?? "";

// Slide `n` at `width` px wide as a JPEG data URL, for the report's slide list.
export async function thumbDeck(n: number, width = 160): Promise<string> {
  const slide = slides[n];
  if (!slide) return "";
  const s = width / slide.w;
  const c = document.createElement("canvas");
  [c.width, c.height] = [width, Math.max(1, Math.round(slide.h * s))];
  const cx = c.getContext("2d");
  if (!cx) return "";
  cx.fillStyle = "#fff"; // JPEG has no alpha; transparent images would go black
  cx.fillRect(0, 0, c.width, c.height);
  await slide.draw(cx, s, false);
  return c.toDataURL("image/jpeg", 0.7);
}
export function closeDeck() {
  task?.cancel();
  slides = [];
}
// Moves by `by` slides, clamped to the deck; false if already at that end.
export function stepDeck(by: number): boolean {
  const was = at;
  at = Math.max(0, Math.min(slides.length - 1, at + by));
  return at !== was;
}

// Paints the current slide sharp at the box's on-screen size. It renders
// off-screen and swaps in when done, so the previous slide stays up meanwhile
// (resizing the visible canvas would blank it). A newer paint cancels one
// still rendering (fast arrow presses, resizes).
export async function paintDeck(canvas: HTMLCanvasElement) {
  const slide = slides[at];
  if (!slide) return;
  const my = ++paints;
  task?.cancel();
  const s = Math.min(canvas.clientWidth / slide.w, canvas.clientHeight / slide.h) * (devicePixelRatio || 1);
  if (!(s > 0)) return; // hidden
  const off = document.createElement("canvas");
  off.width = Math.round(slide.w * s);
  off.height = Math.round(slide.h * s);
  const cx = off.getContext("2d");
  if (!cx) return;
  try {
    await slide.draw(cx, s, true);
  } catch (e) {
    if (my === paints && !(e instanceof pdfjs.RenderingCancelledException)) throw e;
    return;
  }
  if (my !== paints) return; // a newer paint owns the canvas
  [canvas.width, canvas.height] = [off.width, off.height];
  canvas.getContext("2d")?.drawImage(off, 0, 0);
}

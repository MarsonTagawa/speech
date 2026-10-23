// Animation loops that only run while their element is on screen. A rAF loop
// keeps firing (and running its JS) for a canvas that's display:none, detached
// or scrolled away; the webview only throttles it when the whole window hides.

// Calls `on` each time `el` starts or stops being visible: displayed, in the
// viewport, and in a window that isn't minimised. Starts out not visible.
// Detached elements count as hidden and are picked up again on re-attach.
export function watchVisible(el: Element, on: (visible: boolean) => void): () => void {
  let seen = false, was = false;
  const sync = () => {
    const now = seen && !document.hidden;
    if (now !== was) on((was = now));
  };
  const io = new IntersectionObserver((es) => {
    seen = es[es.length - 1].isIntersecting;
    sync();
  });
  io.observe(el);
  document.addEventListener("visibilitychange", sync);
  return () => {
    io.disconnect();
    document.removeEventListener("visibilitychange", sync);
  };
}

// Settings → Performance mode: every VisibleLoop's frame-rate cap (0 = the
// display's rate) and the most canvas pixels per CSS pixel.
export const perf = { fps: 0, maxDpr: 2 };

// A rAF loop that runs while `el` is visible and it isn't `held`. dt is in
// seconds, capped so a resume doesn't jump the animation.
export class VisibleLoop {
  private raf = 0;
  private last = 0;
  private visible = false;
  private hold = false;
  private unwatch: () => void;

  constructor(el: Element, private frame: (dt: number, now: number) => void) {
    this.unwatch = watchVisible(el, (v) => {
      this.visible = v;
      this.sync();
    });
  }

  get held() {
    return this.hold;
  }
  set held(v: boolean) {
    this.hold = v;
    this.sync();
  }

  destroy() {
    this.unwatch();
    this.held = true;
  }

  private tick = (now: number) => {
    // Under a cap, skip display frames until the next one is due (1 ms slack for rAF jitter).
    if (perf.fps && now - this.last < 1000 / perf.fps - 1) {
      this.raf = requestAnimationFrame(this.tick);
      return;
    }
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    this.frame(dt, now);
    this.raf = requestAnimationFrame(this.tick);
  };

  private sync() {
    const run = this.visible && !this.hold;
    if (run && !this.raf) {
      this.last = performance.now();
      this.raf = requestAnimationFrame(this.tick);
    } else if (!run && this.raf) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
    }
  }
}

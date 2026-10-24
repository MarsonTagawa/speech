// Window controls in the platform's own style, since decorations are off.
// macOS: traffic lights atop the nav rail (close, minimize, fullscreen;
// option-click the green one to zoom). Windows and Linux: a slim title bar
// with minimize, maximize and close at the right, Windows 11 and GNOME looks.

import { getCurrentWindow } from "@tauri-apps/api/window";

export type Os = "mac" | "windows" | "linux";

export function currentOs(): Os {
  const ua = navigator.userAgent;
  return ua.includes("Mac") ? "mac" : ua.includes("Windows") ? "windows" : "linux";
}

const svg = (body: string) => `<svg viewBox="0 0 10 10" aria-hidden="true">${body}</svg>`;

const GLYPHS: Record<Os, Record<"close" | "min" | "max" | "restore", string>> = {
  mac: {
    close: svg('<path d="M2.5 2.5l5 5M7.5 2.5l-5 5" />'),
    min: svg('<path d="M2 5h6" />'),
    max: svg('<path class="fill" d="M2.6 7.4V3.6l3.8 3.8zM7.4 2.6v3.8L3.6 2.6z" />'), // outward corners
    restore: svg('<path class="fill" d="M1.6 5.4h3v3zM8.4 4.6h-3v-3z" />'), // inward corners
  },
  windows: {
    close: svg('<path d="M0.5 0.5l9 9M9.5 0.5l-9 9" />'),
    min: svg('<path d="M0 5h10" />'),
    max: svg('<rect x="0.5" y="0.5" width="9" height="9" rx="1" />'),
    restore: svg('<rect x="0.5" y="2.5" width="7" height="7" rx="1" /><path d="M2.5 2.5V1.5a1 1 0 0 1 1-1h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-1" />'),
  },
  linux: {
    close: svg('<path d="M2.5 2.5l5 5M7.5 2.5l-5 5" />'),
    min: svg('<path d="M2.5 7.5h5" />'),
    max: svg('<rect x="2.5" y="2.5" width="5" height="5" />'),
    restore: svg('<rect x="2" y="3.5" width="4.5" height="4.5" /><path d="M3.5 3.5V2h4.5v4.5H6.5" />'),
  },
};

function button(os: Os, kind: "close" | "min" | "max", label: string): HTMLButtonElement {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `wc-btn wc-${kind}`;
  b.setAttribute("aria-label", label);
  b.innerHTML = GLYPHS[os][kind];
  return b;
}

export function initWindowControls() {
  const os = currentOs();
  const win = getCurrentWindow();
  document.body.dataset.os = os;

  const group = document.createElement("div");
  group.className = "wc";
  const close = button(os, "close", "Close");
  const min = button(os, "min", "Minimize");
  const max = button(os, "max", os === "mac" ? "Enter full screen" : "Maximize");
  // Native order: close first on macOS, last elsewhere.
  group.append(...(os === "mac" ? [close, min, max] : [min, max, close]));

  close.addEventListener("click", () => void win.close());
  min.addEventListener("click", () => void win.minimize());
  max.addEventListener("click", async (e) => {
    if (os !== "mac" || e.altKey) return void win.toggleMaximize();
    void win.setFullscreen(!(await win.isFullscreen()));
  });

  if (os === "mac") {
    document.querySelector(".nav")?.prepend(group);
  } else {
    // A title bar to hold them, which also drags the window (main.ts) and
    // maximizes on double-click like a native one.
    const bar = document.createElement("div");
    bar.className = "titlebar";
    bar.append(group);
    bar.addEventListener("dblclick", (e) => {
      if (e.target === bar) void win.toggleMaximize();
    });
    document.querySelector(".app")?.before(bar);
  }

  // The middle button's glyph follows the window: restore when maximized
  // (fullscreen on macOS). Both arrive as resizes.
  const sync = async () => {
    const on = await (os === "mac" ? win.isFullscreen() : win.isMaximized()).catch(() => false);
    max.innerHTML = GLYPHS[os][on ? "restore" : "max"];
    max.setAttribute("aria-label", os === "mac" ? (on ? "Exit full screen" : "Enter full screen") : on ? "Restore" : "Maximize");
  };
  void sync();
  void win.onResized(() => void sync());
  // Greyed out while another window has focus, as native controls are.
  void win.onFocusChanged(({ payload }) => document.body.classList.toggle("win-blurred", !payload));
}

# Mac live notch — design

Date: 2026-10-21
Status: approved in conversation, pending spec review

## Problem

While presenting, the main window is hidden behind Keynote, PowerPoint or a
fullscreen browser, so the live cues it shows (pace, fillers, the clock, the
read-along script) are out of sight right when they matter. On a MacBook the
notch sits next to the camera, where the speaker is supposed to be looking
anyway.

## Goals and success criteria

- On macOS, a **Show notch** setting opens a small overlay wrapped around the
  MacBook notch, like the Dynamic Island. It shows:
  - always: a recording dot, **WPM** (green inside the preset's target band,
    amber outside it, neutral at 0) and **fillers/min**;
  - with **Timer/stopwatch** on: the drill countdown when a timer is set, the
    "Think m:ss" prep countdown during a drill's prep, and the elapsed stopwatch
    otherwise. These are the same labels as `#clock` and `#countdown`;
  - with **Script** on and a script loaded: the next ~8 words from the
    read-along cursor, in a row that drops below the notch.
- It stays visible over fullscreen apps and on every Space, sits above the menu
  bar, never takes focus and lets every click through.
- On a Mac with no notch (or an external display only), it's a pill centred at
  the top of the main screen.
- Updates arrive at most 250 ms behind the main window.
- Turning **Show notch** off removes the window. The setting is remembered, and
  the notch reopens on launch.

## Non-goals

- Linux and Windows. The Settings group is hidden there and the command does
  nothing (decided in conversation; you test on your own Mac).
- Controls in the notch (start/stop, slide changes). It's read-only and
  click-through.
- Hover to expand, or Dynamic Island-style animations.
- Re-placing the notch when displays change mid-session. Toggling it off and on
  places it again. Marked `ponytail:`.

## Architecture

A second Tauri webview window (`notch`), opened and closed from Rust. The main
window stays the only owner of state and pushes one small payload to it.

### Rust: `src-tauri/src/notch.rs`

- `#[tauri::command] notch_show(app, on: bool, script: bool) -> Result<(), String>`
  - On macOS, `on`: if a `notch` window exists, re-place it (height changes with
    `script`). Otherwise build it with `WebviewWindowBuilder::new(app, "notch",
    WebviewUrl::App("notch.html"))`, set to `transparent(true)`,
    `decorations(false)`, `focusable(false)`, `shadow(false)`,
    `resizable(false)`, `skip_taskbar(true)`, `visible_on_all_workspaces(true)`
    and `always_on_top(true)`. After building it:
    - `set_ignore_cursor_events(true)`;
    - objc2 on `ns_window()`: `setLevel(NSStatusWindowLevel)` (25, above the
      menu bar; at Tauri's default floating level macOS pushes the window below
      the menu bar), then `setCollectionBehavior(CanJoinAllSpaces | Stationary
      | FullScreenAuxiliary | IgnoresCycle)`;
    - `setFrame` to the rect from `notch_frame` (below), in AppKit's
      bottom-left screen coordinates.
  - On macOS, `!on`: close the `notch` window if there is one.
  - Elsewhere: `Ok(())`.
- Screen choice (macOS): the first `NSScreen` whose `safeAreaInsets.top > 0`
  (the built-in notched display), else `NSScreen.mainScreen`.
- `fn notch_frame(screen: Rect, notch: Option<(f64, f64)>, bar_h: f64, script: bool) -> Rect`
  is pure and not behind any `cfg`, so its test runs on Linux.
  - `notch = Some((left, right))`: the x-range of the notch gap, i.e. the right
    edge of `auxiliaryTopLeftArea` to the left edge of `auxiliaryTopRightArea`.
    The frame spans `left - WING .. right + WING` (`WING = 110.0` pt), top-aligned
    to the screen. Its height is `bar_h` (`safeAreaInsets.top`), plus `ROW = 28.0`
    when `script` is on.
  - `notch = None`: a `PILL_W = 360.0` wide pill, centred horizontally and
    top-aligned. Its height is `PILL_H = 32.0`, plus `ROW` when `script` is on.
- Registered in `lib.rs`'s `generate_handler!`.

### Config

- `tauri.conf.json`: `app.macOSPrivateApi: true`. Transparent windows on macOS
  need it, and it rules out the Mac App Store, which is fine here.
- `Cargo.toml`: add the `macos-private-api` feature to `tauri`. Add the
  `NSWindow`, `NSScreen`, `NSResponder` and `NSGraphics` features to the
  existing macOS `objc2-app-kit` dependency, as needed for `setLevel`,
  `setCollectionBehavior`, `setFrame` and the screen queries.
- `capabilities/default.json`: `"windows": ["main", "notch"]`. `core:default`
  already covers `listen` and `emit_to`.
- `vite.config.ts`: `build.rollupOptions.input = { main: "index.html", notch: "notch.html" }`.

### Frontend

- `notch.html` + `src/notch.ts`: a page with a transparent background that
  listens for the `notch` event and renders it. Two black wings (rounded bottom
  corners) either side of an empty gap the width of the notch, and an optional
  black script row below with rounded bottom corners. It uses the CSS variables
  for the green and amber colours from `styles.css`. It has no state of its own,
  so if the main window stops sending, it simply keeps the last frame.
- `src/notch.ts` also exports the pure payload builder used by `main.ts`:

  ```ts
  export interface NotchState {
    rec: boolean;
    time: string | null;      // null when Timer/stopwatch is off
    wpm: number;
    band: "in" | "out" | "";  // "" when wpm is 0
    fpm: string;              // fillersPerMin.toFixed(1)
    script: string | null;    // null when Script is off or no script is loaded
  }
  export function notchState(...): NotchState
  export function scriptAhead(text: string, tokens: {start:number,end:number}[], cursor: number, n = 8): string
  ```

- `main.ts`:
  - `renderStats` stores the latest `wpm`, the band and `fillersPerMin` in
    module variables that are already in scope (no recompute).
  - `renderScriptMatch` stores the latest `cursor`.
  - `pushNotch()` builds the payload, compares it with the last one it sent
    (`JSON.stringify`), and only on a change calls
    `emitTo("notch", "notch", state)`. It's called from `tickClock`, which runs
    every 250 ms while recording or prepping and once on stop or reset, and
    from `renderStats` and `renderScriptMatch` so idle edits still reach the
    notch.
  - When Show notch is off, `pushNotch` does nothing.
- Settings: a new "Notch" group with three `set-row` switches,
  `set-notch`, `set-notch-time` and `set-notch-script`, saved as
  `speech.notch` / `speech.notch.time` / `speech.notch.script` through the
  existing `load`/`save`. Changing Show notch or Script calls
  `invoke("notch_show", { on, script })`, then `pushNotch()` after it resolves.
  The group is hidden unless `navigator.userAgent` includes `"Mac"`. At startup,
  if `speech.notch` is `"1"` on a Mac, it opens the notch.

## Error handling

- If `notch_show` returns `Err`: `console.error`, flip `set-notch` back off and
  don't save "1".
- `emitTo` a window that doesn't exist resolves without effect, so `pushNotch`
  needs no guard beyond the setting check (`.catch(() => {})`).
- The user closing the notch window isn't possible: it can't take focus and has
  no decorations. If the app quits, both windows go together.
- objc2 calls are `unsafe` and must run on the main thread. Wrap them in
  `app.run_on_main_thread`, like `screenshot.rs`'s `with_webview`, which runs
  on the main thread and uses `MainThreadMarker::new_unchecked()`.

## Testing

- **Rust**: a `#[cfg(test)]` test for `notch_frame` covering a notched screen
  (with and without the script row: the wings straddle the gap, the frame is
  top-aligned and has the right height) and a non-notched screen (the pill is
  centred). Runs on Linux with
  `CARGO_TARGET_DIR=<scratchpad> cargo test notch`, which avoids the shared
  `target/`.
- **TS**: `src/notch.check.ts` covers `scriptAhead` (mid-script, near the end,
  past the end giving `""`, an empty script) and `notchState` (band in, out and
  empty at 0; `time` and `script` null when their toggle is off).
- **Manual, on the Mac**:
  1. Show notch on: the wings hug the notch and nothing covers it.
  2. It sits over the menu bar, not below it.
  3. It's visible over fullscreen Keynote and a fullscreen Safari tab, and
     after switching Spaces.
  4. Clicks on the wings reach the menu bar or app underneath, and Keynote
     keeps focus.
  5. Record: the dot, stopwatch, WPM colour and fillers/min update. With a
     drill timer it counts down, and "Think" shows during prep.
  6. With a script loaded and Script on, the row shows upcoming words and
     advances while reading.
  7. External display only (lid closed), toggled off and on: the pill is centred.
  8. Turning it off removes the window, and relaunching restores it.
- `bun run build` and `cargo check` pass on Linux, so the macOS-only code stays
  inside its `cfg`.

## Risks

- **`focusable(false)` + `set_ignore_cursor_events` on macOS**: if the window
  still activates the app on show, set `NSWindow.setIgnoresMouseEvents` directly
  and order it front with `orderFrontRegardless` instead of `show()`.
- **Window level**: some fullscreen apps sit above `NSStatusWindowLevel`. If
  Keynote's fullscreen covers it, try `NSPopUpMenuWindowLevel` (101) next.
- **Notch API availability**: `auxiliaryTopLeftArea` and `safeAreaInsets` are
  macOS 12+. On older systems the pill fallback applies (check
  `respondsToSelector`).

import { defineConfig } from "vite";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/
// ajv (via @bible-strong/avatar-core) registers its 2020-12 meta-schema with a
// call Rollup wrongly treeshakes away, so the bundle throws "no schema with key
// or ref" at startup and nothing in the app works. Keep ajv untouched.
const ajvNoTreeshake = {
  name: "ajv-no-treeshake",
  transform(code: string, id: string) {
    if (/node_modules\/ajv\/dist\//.test(id)) return { code, map: null, moduleSideEffects: "no-treeshake" as const };
  },
};

export default defineConfig(async () => ({
  plugins: [ajvNoTreeshake],

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // Two pages: the app and the macOS notch overlay (src-tauri/src/notch.rs).
  build: {
    rollupOptions: { input: { main: "index.html", notch: "notch.html" } },
  },
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
}));

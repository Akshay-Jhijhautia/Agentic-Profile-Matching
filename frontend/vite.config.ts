// Builds the UI into the Python package, where `profile-match web` serves it.
// In development (`npm run dev`), API calls go to a running `profile-match web`; start
// both with the same PROFILE_MATCH_TOKEN so the proxy can authenticate (see README.md).
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const api = process.env.PROFILE_MATCH_API ?? "http://127.0.0.1:8765";
const token = process.env.PROFILE_MATCH_TOKEN ?? "";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "../profile_matching/web/static", emptyOutDir: true },
  server: {
    proxy: {
      "/api": { target: api, changeOrigin: true, headers: { "X-Profile-Match-Token": token } },
    },
  },
  test: { environment: "jsdom", globals: true, setupFiles: [] },
});

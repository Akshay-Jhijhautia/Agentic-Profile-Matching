// Builds the UI into the Python package, where `resume-shortlister web` serves it.
// In development (`npm run dev`), API calls go to a running `resume-shortlister web`; start
// both with the same RESUME_SHORTLISTER_TOKEN so the proxy can authenticate (see README.md).
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const api = process.env.RESUME_SHORTLISTER_API ?? "http://127.0.0.1:8765";
const token = process.env.RESUME_SHORTLISTER_TOKEN ?? "";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "../resume_shortlister/web/static", emptyOutDir: true },
  server: {
    proxy: {
      "/api": { target: api, changeOrigin: true, headers: { "X-Resume-Shortlister-Token": token } },
    },
  },
  test: { environment: "jsdom", globals: true, setupFiles: [] },
});

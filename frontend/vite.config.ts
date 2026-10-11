import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the UI runs on :5173 and proxies /api to the FastAPI backend on :8000.
const api = process.env.CARDIOSAGE_API ?? "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true, proxy: { "/api": api } },
  preview: { port: 4173, strictPort: true, proxy: { "/api": api } },
  build: { chunkSizeWarningLimit: 900 },
});

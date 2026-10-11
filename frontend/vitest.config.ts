import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
export default defineConfig({ plugins: [react()], test: { environment: "jsdom", testTimeout: 60000, include: ["src/test/**/*.test.tsx", "src/test/**/*.test.ts"] } });

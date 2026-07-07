import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(rootDir, "..");

export default defineConfig({
  plugins: [react()],
  base: "./",
  resolve: {
    alias: {
      "@": path.resolve(rootDir, "./src"),
    },
  },
  server: {
    fs: {
      allow: [repoRoot],
    },
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});

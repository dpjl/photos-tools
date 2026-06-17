import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// En dev, on proxifie l'API vers le serveur FastAPI (port 8077 par défaut).
const API_TARGET = process.env.API_TARGET || "http://127.0.0.1:8077";

export default defineConfig({
  plugins: [svelte()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
      "/healthz": { target: API_TARGET, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});

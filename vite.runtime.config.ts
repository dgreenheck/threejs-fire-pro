import { defineConfig } from 'vite';
export default defineConfig({
  publicDir: false,
  build: {
    outDir: 'dist-runtime',
    emptyOutDir: true,
    lib: { entry: 'editor/runtime.ts', formats: ['es'], fileName: () => 'runtime.js' },
    minify: false,
  },
});

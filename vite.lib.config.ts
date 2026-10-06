import { defineConfig } from 'vite';

export default defineConfig({
  publicDir: false,
  build: {
    outDir: 'dist-lib',
    lib: {
      entry: 'src/index.ts',
      name: 'ThreeJSFirePro',
      formats: ['es'],
      fileName: 'threejs-fire-pro',
    },
    rollupOptions: { external: [/^three(?:\/|$)/] },
  },
});

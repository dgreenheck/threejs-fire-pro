import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { readFile } from 'node:fs/promises';
import { readdirSync } from 'node:fs';
import { resolve } from 'node:path';

const examples = readdirSync('examples', { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => resolve('examples', entry.name, 'index.html'));

export default defineConfig({
  // GitHub Pages serves the editor from /threejs-fire-pro/ (.github/workflows/pages.yml).
  base: process.env.BASE_PATH ?? '/',
  optimizeDeps: { include: ['three', 'three/webgpu', 'three/tsl'] },
  plugins: [
    react(),
    {
      name: 'standalone-export-runtime',
      configureServer(server) {
        server.middlewares.use('/export-runtime/runtime.js', async (_req, res) => {
          try {
            res.setHeader('Content-Type', 'text/javascript');
            res.end(await readFile('dist-runtime/runtime.js'));
          } catch {
            res.statusCode = 503;
            res.end('Run npm run build:runtime.');
          }
        });
      },
      async generateBundle() {
        this.emitFile({
          type: 'asset',
          fileName: 'export-runtime/runtime.js',
          source: await readFile('dist-runtime/runtime.js'),
        });
      },
    },
  ],
  build: { rollupOptions: { input: [resolve('index.html'), ...examples] } },
});

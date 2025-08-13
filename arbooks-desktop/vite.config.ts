import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path, { resolve } from 'path';
import electron from 'vite-plugin-electron';
import renderer from 'vite-plugin-electron-renderer';
import { copyFileSync, mkdirSync, readdirSync, statSync, existsSync } from 'fs';

export default defineConfig({
  base: process.env.ELECTRON_RENDERER_URL ? undefined : "./",
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
  },
  build: {
    outDir: 'dist-electron',
  },
  plugins: [
    electron([
      {
        // Main-Process entry file of the Electron App.
        entry: 'electron/main.ts',
      },
      {
        entry: 'electron/preload.ts',
        onstart(options) {
          // Notify the Renderer-Process to reload the page when the Preload-Scripts build is complete, 
          // instead of restarting the entire Electron App.
          options.reload()
        },
      },
    ]),
    renderer(),
    react(),
    {
      name: 'copy-assets',
      writeBundle() {
        // Ensure directories exist in the dist folder
        const copyDirectory = (src: string, dest: string) => {
          // Check if source directory exists
          if (!existsSync(src)) {
            console.log(`Source directory ${src} does not exist, skipping copy`);
            return;
          }

          // Ensure the destination directory exists
          mkdirSync(dest, { recursive: true });

          readdirSync(src).forEach((file) => {
            const srcPath = resolve(src, file);
            const destPath = resolve(dest, file);

            // Check if it's a file or directory
            const stat = statSync(srcPath);

            if (stat.isDirectory()) {
              // Recursively copy the contents of subdirectories
              copyDirectory(srcPath, destPath);
            } else {
              // Copy file
              copyFileSync(srcPath, destPath);
            }
          });
        };

        // Only copy bin directory if it exists
        if (existsSync('bin')) {
          copyDirectory('bin', 'dist-electron/bin');
        }
      },
    },
  ],
});
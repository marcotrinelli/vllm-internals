import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  // relative, so the build can be served from any path
  base: './',
  plugins: [react()],
  server: { port: 5274 },
  preview: { port: 5274 },
  test: {
    environment: 'node',
    include: ['src/**/*.test.{ts,tsx}'],
  },
});

import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const proxyTarget = env.VITE_API_BASE || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      port: 3000,
      // Proxy all /api/* requests to the backend (Colab URL or localhost)
      // This avoids CORS issues when developing locally.
      proxy: {
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
          // Pass ngrok header so it skips the browser warning page
          headers: {
            'ngrok-skip-browser-warning': 'true',
          },
        },
      },
    },
  }
})

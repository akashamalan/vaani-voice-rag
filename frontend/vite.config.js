import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Vite rejects requests whose Host header it does not recognise. Behind a
    // cloudflared tunnel the Host is *.trycloudflare.com, which would otherwise
    // return "Blocked request. This host is not allowed."
    allowedHosts: true,
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
  },
})

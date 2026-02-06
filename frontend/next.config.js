/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: {
    unoptimized: true,
  },
  env: {
    NEXT_PUBLIC_EDGE_API_URL: process.env.NEXT_PUBLIC_EDGE_API_URL || 'http://localhost:8080',
    NEXT_PUBLIC_CLOUD_API_URL: process.env.NEXT_PUBLIC_CLOUD_API_URL || 'https://ona-diagnostic-production.up.railway.app',
  },
}

module.exports = nextConfig

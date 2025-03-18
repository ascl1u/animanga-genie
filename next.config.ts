import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  /* config options here */
  images: {
    domains: [
      's4.anilist.co',  // Main AniList CDN domain
      'media.kitsu.io',  // Additional CDN sometimes used by AniList
      'img.anili.st',    // Another alternative AniList CDN
    ],
  },
  // Add configuration for ONNX Runtime WASM files
  webpack: (config) => {
    // Add rule for WASM files
    config.experiments = { ...config.experiments, asyncWebAssembly: true };
    
    // Add fallback for ONNX runtime
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
    };
    
    return config;
  },
  // Define proper MIME types and caching for WASM files
  async headers() {
    return [
      {
        // Apply to all WASM files
        source: '/:path*.wasm',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/wasm',
          },
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      {
        // Apply to ONNX model files
        source: '/:path*.onnx',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/octet-stream',
          },
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      {
        // Add cross-origin headers for model files
        source: '/models/:path*',
        headers: [
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'require-corp',
          },
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin',
          },
        ],
      },
    ];
  },
};

export default nextConfig;

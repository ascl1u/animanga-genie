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
};

export default nextConfig;

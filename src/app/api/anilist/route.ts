import { NextRequest, NextResponse } from 'next/server';

/**
 * POST /api/anilist
 * Proxy for AniList GraphQL API to avoid CORS issues
 * Includes rate limit handling with a 60-second cooldown
 */

// Track the last time we hit a rate limit
let lastRateLimitTime = 0;
const RATE_LIMIT_COOLDOWN = 60000; // 60 seconds in milliseconds

// Type definition for AniList error response
interface AnilistError {
  message?: string;
  extensions?: {
    code?: string;
  };
}

export async function POST(request: NextRequest) {
  try {
    const now = Date.now();
    
    // Check if we're in a cooldown period after hitting a rate limit
    if (lastRateLimitTime > 0 && now - lastRateLimitTime < RATE_LIMIT_COOLDOWN) {
      const remainingCooldown = Math.ceil((RATE_LIMIT_COOLDOWN - (now - lastRateLimitTime)) / 1000);
      console.log(`[ANILIST] Still in rate limit cooldown. ${remainingCooldown}s remaining.`);
      
      return NextResponse.json(
        { 
          errors: [{ 
            message: `AniList API rate limit reached. Please try again in ${remainingCooldown} seconds.`,
            extensions: { code: 'RATE_LIMITED', retryAfter: remainingCooldown }
          }] 
        },
        { status: 429 }
      );
    }
    
    // Get the request body (GraphQL query and variables)
    const body = await request.json();
    
    // Forward the request to AniList GraphQL API
    const response = await fetch('https://graphql.anilist.co', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    // Get the response data
    const data = await response.json();
    
    // Check if the response indicates a rate limit issue
    if (
      response.status === 429 || 
      (data.errors && data.errors.some((e: AnilistError) => 
        e.message?.includes('rate limit') || 
        e.message?.includes('too many requests') ||
        e.extensions?.code === 'RATE_LIMITED'
      ))
    ) {
      console.log('[ANILIST] Rate limit reached. Setting cooldown for 60 seconds.');
      lastRateLimitTime = now;
      
      return NextResponse.json(
        { 
          errors: [{ 
            message: 'AniList API rate limit reached. Please try again in 60 seconds.',
            extensions: { code: 'RATE_LIMITED', retryAfter: 60 }
          }] 
        },
        { status: 429 }
      );
    }
    
    // Return the regular response from AniList
    return NextResponse.json(data);
  } catch (error) {
    console.error('[ANILIST] Error proxying request to AniList:', error);
    return NextResponse.json(
      { errors: [{ message: 'Error proxying request to AniList' }] },
      { status: 500 }
    );
  }
} 
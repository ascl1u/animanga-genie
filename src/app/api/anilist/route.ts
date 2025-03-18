import { NextRequest, NextResponse } from 'next/server';

/**
 * POST /api/anilist
 * Proxy for AniList GraphQL API to avoid CORS issues
 */
export async function POST(request: NextRequest) {
  try {
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
    
    // Return the response from AniList
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error proxying request to AniList:', error);
    return NextResponse.json(
      { errors: [{ message: 'Error proxying request to AniList' }] },
      { status: 500 }
    );
  }
} 
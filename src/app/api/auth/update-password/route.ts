import { NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { NextRequest } from 'next/server';

/**
 * POST /api/auth/update-password
 * Updates a user's password (for server-side operations)
 * 
 * Note: For client-side password resets after clicking a reset link, 
 * we use supabase.auth.updateUser() directly in the client component.
 * This endpoint is primarily maintained for completeness and server-side operations.
 */
export async function POST(request: NextRequest) {
  try {
    const { password } = await request.json();

    // Validate required fields
    if (!password) {
      return NextResponse.json(
        { error: 'Password is required' },
        { status: 400 }
      );
    }

    if (!process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY) {
      throw new Error('Missing Supabase environment variables');
    }

    // Use server client to access cookies
    const supabase = await createClient();
    
    // Verify the user is authenticated
    const { data: userData, error: userError } = await supabase.auth.getUser();
    
    if (userError || !userData.user) {
      console.error('User auth error:', userError);
      return NextResponse.json(
        { error: 'You must be logged in to update your password' },
        { status: 401 }
      );
    }
    
    // Update the password for logged-in user
    const { error } = await supabase.auth.updateUser({
      password: password
    });
    
    if (error) {
      console.error('Password update error:', error);
      return NextResponse.json({ error: error.message }, { status: 400 });
    }
    
    return NextResponse.json({ 
      success: true, 
      message: 'Password updated successfully' 
    });
  } catch (error) {
    console.error('Password update error:', error);
    return NextResponse.json(
      { error: 'Failed to update password' },
      { status: 500 }
    );
  }
} 
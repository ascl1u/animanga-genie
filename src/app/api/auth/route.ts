import { NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { NextRequest } from 'next/server';

/**
 * POST /api/auth
 * Handles all auth-related requests such as signup, login, logout, and password reset
 */
export async function POST(request: NextRequest) {
  try {
    const supabase = await createClient();
    const body = await request.json();
    const { action, email, password, fullName, redirectTo } = body;

    if (!process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY) {
      throw new Error('Missing Supabase environment variables');
    }

    switch (action) {
      case 'signUp': {
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: {
              full_name: fullName,
            },
            emailRedirectTo: redirectTo || `${process.env.NEXT_PUBLIC_SITE_URL}/auth/callback`,
          },
        });

        if (error) {
          return NextResponse.json({ error: error.message }, { status: 400 });
        }

        // We no longer need to manually create user preferences
        // The database trigger will handle this automatically

        return NextResponse.json({ 
          success: true, 
          message: 'Signup successful. Please check your email for verification.', 
          user: data.user 
        });
      }

      case 'signIn': {
        const { data, error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });

        if (error) {
          return NextResponse.json({ error: error.message }, { status: 400 });
        }

        return NextResponse.json({ success: true, session: data.session, user: data.user });
      }

      case 'signOut': {
        const { error } = await supabase.auth.signOut();

        if (error) {
          return NextResponse.json({ error: error.message }, { status: 500 });
        }

        return NextResponse.json({ success: true }, { status: 200 });
      }

      case 'resetPassword': {
        // Ensure we have the proper site URL for redirects
        const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || '';
        let resetPasswordUrl = redirectTo;
        
        // If no redirect URL is provided, or it doesn't include our domain, use the default
        if (!resetPasswordUrl || !resetPasswordUrl.includes(siteUrl)) {
          resetPasswordUrl = `${siteUrl}/auth/reset-password`;
        }
        
        console.log(`Password reset requested for ${email}, redirecting to ${resetPasswordUrl}`);
        
        const { error } = await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: resetPasswordUrl,
        });

        if (error) {
          console.error('Password reset error:', error);
          return NextResponse.json({ error: error.message }, { status: 400 });
        }

        return NextResponse.json({ 
          success: true, 
          message: 'Password reset email sent successfully' 
        });
      }

      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
    }
  } catch (error) {
    console.error('Authentication error:', error);
    return NextResponse.json({ error: 'Authentication failed' }, { status: 500 });
  }
}

/**
 * GET /api/auth
 * Returns the current session information
 */
export async function GET() {
  try {
    const supabase = await createClient();
    
    // Get session (this is the most reliable way to check auth status)
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();
    
    if (sessionError) {
      console.error('Session error:', sessionError);
      return NextResponse.json({ error: sessionError.message }, { status: 500 });
    }
    
    // If no session, return early
    if (!session) {
      return NextResponse.json({ user: null, session: null }, { status: 200 });
    }
    
    // If we have a session, get the user
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    
    if (userError) {
      console.error('User error:', userError);
      return NextResponse.json({ error: userError.message }, { status: 500 });
    }
    
    // Return user and session (filtering sensitive info)
    return NextResponse.json({
      user: user ? {
        id: user.id,
        email: user.email,
        emailConfirmed: !!user.email_confirmed_at,
      } : null,
      session: session ? {
        expiresAt: session.expires_at,
      } : null,
    }, { status: 200 });
  } catch (error) {
    console.error('Auth API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 
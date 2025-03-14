import { NextResponse } from "next/server";
import { createClient } from '@/utils/supabase/server';

/**
 * GET /auth/callback
 * Handles auth callback from Supabase - email verification, etc.
 * Also ensures user is created in our application tables
 */
export async function GET(request: Request) {
  try {
    const requestUrl = new URL(request.url);
    const code = requestUrl.searchParams.get("code");
    
    // If there is no code, redirect to home page
    if (!code) {
      return NextResponse.redirect(new URL("/", request.url));
    }
    
    // Use server client to ensure cookies are properly handled
    const supabase = await createClient();
    
    // Exchange the code for a session (this handles PKCE verification internally)
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    
    if (error) {
      console.error('Exchange code for session error:', error);
      return NextResponse.redirect(
        new URL(`/?error=${encodeURIComponent(error.message)}`, request.url)
      );
    }
    
    // Get the user AFTER session exchange
    const { data: { user } } = await supabase.auth.getUser();
    
    if (user) {
      try {
        // Check if user exists in our public.users table
        const { data: existingUser, error: checkError } = await supabase
          .from('users')
          .select('id')
          .eq('id', user.id)
          .single();
        
        if (checkError && checkError.code !== 'PGRST116') { // PGRST116 is "no rows returned"
          console.error('Error checking for existing user:', checkError);
        }
        
        // If user doesn't exist in our table, create them
        if (!existingUser) {
          console.log('User not found in public tables. Creating...');
          
          // Create user record
          const { error: userError } = await supabase
            .from('users')
            .insert([{ 
              id: user.id, 
              email: user.email,
              created_at: new Date().toISOString()
            }]);
          
          if (userError) {
            console.error('Error creating user record:', userError);
          } else {
            console.log('Successfully created user record');
            
            // Create user preferences
            const { error: prefsError } = await supabase
              .from('user_preferences')
              .insert([{
                user_id: user.id,
                preferred_genres: [],
                watch_history: []
              }]);
            
            if (prefsError) {
              console.error('Error creating user preferences:', prefsError);
            } else {
              console.log('Successfully created user preferences');
            }
          }
        }
      } catch (dbError) {
        console.error('Database operations error:', dbError);
        // Continue the flow even if DB operations fail - don't block the user
      }
    }
    
    // For all auth flows (signup verification, etc.), redirect to home
    return NextResponse.redirect(new URL("/", request.url));
  } catch (error) {
    console.error('Auth callback error:', error);
    return NextResponse.redirect(
      new URL("/?error=auth_callback_failed", request.url)
    );
  }
} 
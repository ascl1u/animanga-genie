'use client';

import { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { createClient } from '@/utils/supabase/client';
import { User, Session } from '@supabase/supabase-js';
import { syncLocalDataWithDatabase } from '@/services/authDataSyncService';
import { toast } from 'react-hot-toast';

type AuthState = {
  user: User | null;
  session: Session | null;
  isLoading: boolean;
  isAuthenticated: boolean;
};

type AuthContextType = AuthState & {
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>;
  signUp: (email: string, password: string) => Promise<{ error: Error | null }>;
  signOut: () => Promise<void>;
  refreshAuth: () => Promise<void>;
};

// Create context with default values
const AuthContext = createContext<AuthContextType>({
  user: null,
  session: null,
  isLoading: true,
  isAuthenticated: false,
  // These will be properly implemented in the provider
  signIn: async () => ({ error: new Error('Not implemented') }),
  signUp: async () => ({ error: new Error('Not implemented') }),
  signOut: async () => {},
  refreshAuth: async () => {},
});

// Hook for components to access auth state and methods
export function useAuth() {
  return useContext(AuthContext);
}

// Component to only be rendered on the client side
export function SimpleAuthProvider({ children }: { children: React.ReactNode }) {
  // Auth state
  const [state, setState] = useState<AuthState>({
    user: null,
    session: null,
    isLoading: true,
    isAuthenticated: false,
  });

  // Client-side safety
  const [isMounted, setIsMounted] = useState(false);
  const supabase = createClient();
  
  // Refresh auth state - wrapped in useCallback to avoid dependency issues
  const refreshAuth = useCallback(async () => {
    // Skip if not mounted yet
    if (!isMounted) return;
    
    try {
      setState(current => ({ ...current, isLoading: true }));

      // Check for session
      const { data: sessionData, error: sessionError } = await supabase.auth.getSession();
      
      if (sessionError) {
        console.error('Session error:', sessionError);
        setState({
          user: null,
          session: null,
          isLoading: false,
          isAuthenticated: false,
        });
        return;
      }
      
      // If we have a session, get the user
      if (sessionData.session) {
        const { data: userData, error: userError } = await supabase.auth.getUser();
        
        if (userError) {
          console.error('User error:', userError);
          setState({
            user: null,
            session: sessionData.session,
            isLoading: false,
            isAuthenticated: false,
          });
          return;
        }
        
        setState({
          user: userData.user,
          session: sessionData.session,
          isLoading: false,
          isAuthenticated: true,
        });
      } else {
        // No session
        setState({
          user: null,
          session: null,
          isLoading: false,
          isAuthenticated: false,
        });
      }
    } catch (error) {
      console.error('Auth refresh error:', error);
      setState({
        user: null,
        session: null,
        isLoading: false,
        isAuthenticated: false,
      });
    }
  }, [isMounted, supabase.auth]);

  // Sign in
  const signIn = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      
      if (!error) {
        // Attempt to sync local data to database after successful login
        try {
          const syncResult = await syncLocalDataWithDatabase();
          if (syncResult.migrated > 0) {
            // Notify user that their data has been migrated
            toast.success(`${syncResult.migrated} items from your watch history have been saved to your account!`);
          }
          if (syncResult.duplicates > 0) {
            // Just log this, no need to notify the user
            console.log(`${syncResult.duplicates} items were already in your account`);
          }
        } catch (syncError) {
          console.error('Error syncing local data to database:', syncError);
          // Don't return an error here, the login was still successful
        }
      }
      
      return { error };
    } catch (error) {
      console.error('Sign in error:', error);
      return { error: error instanceof Error ? error : new Error('Unknown error') };
    }
  };

  // Sign up
  const signUp = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signUp({
        email,
        password,
      });
      
      if (!error) {
        // Attempt to sync local data to database after successful signup
        try {
          const syncResult = await syncLocalDataWithDatabase();
          if (syncResult.migrated > 0) {
            // Notify user that their data has been migrated
            toast.success(`${syncResult.migrated} items from your watch history have been saved to your account!`);
          }
        } catch (syncError) {
          console.error('Error syncing local data to database:', syncError);
          // Don't return an error here, the signup was still successful
        }
      }
      
      return { error };
    } catch (error) {
      console.error('Sign up error:', error);
      return { error: error instanceof Error ? error : new Error('Unknown error') };
    }
  };

  // Sign out
  const signOut = async () => {
    try {
      await supabase.auth.signOut();
      setState({
        user: null,
        session: null,
        isLoading: false,
        isAuthenticated: false,
      });
    } catch (error) {
      console.error('Sign out error:', error);
    }
  };

  // Set up auth state handling
  useEffect(() => {
    // Only run on client
    setIsMounted(true);
    
    // Initialize auth state
    refreshAuth();
    
    // Subscribe to auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        setState(current => ({
          ...current,
          session,
          isAuthenticated: !!session,
          isLoading: false,
        }));
        
        // Get user when session changes
        if (session) {
          const { data, error } = await supabase.auth.getUser();
          
          if (error) {
            console.error('Error getting user:', error);
            return;
          }
          
          setState(current => ({
            ...current,
            user: data.user,
            isAuthenticated: true
          }));
          
          // If this is a new sign in (not just a token refresh), sync local data
          if (event === 'SIGNED_IN') {
            try {
              const syncResult = await syncLocalDataWithDatabase();
              if (syncResult.migrated > 0) {
                // Notify user that their data has been migrated
                toast.success(`${syncResult.migrated} items from your watch history have been saved to your account!`);
              }
            } catch (syncError) {
              console.error('Error syncing local data to database:', syncError);
            }
          }
        }
      }
    );
    
    return () => {
      subscription.unsubscribe();
    };
  }, [refreshAuth, supabase.auth]);

  // For hydration safety, return children directly during SSR or initial render
  if (!isMounted) {
    return <>{children}</>;
  }

  // Context value
  const value = {
    ...state,
    signIn,
    signUp,
    signOut,
    refreshAuth,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
} 
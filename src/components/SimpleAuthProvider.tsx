'use client';

import { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { createClient } from '@/utils/supabase/client';
import { User, Session } from '@supabase/supabase-js';
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
export interface SimpleAuthProviderProps {
  children: React.ReactNode;
  onAuthStateChange?: (isAuthenticated: boolean, event?: string) => void;
}

export function SimpleAuthProvider({ 
  children, 
  onAuthStateChange 
}: SimpleAuthProviderProps) {
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

        // Call the callback if provided
        if (onAuthStateChange) {
          onAuthStateChange(true);
        }
      } else {
        // No session
        setState({
          user: null,
          session: null,
          isLoading: false,
          isAuthenticated: false,
        });
        
        // Call the callback if provided
        if (onAuthStateChange) {
          onAuthStateChange(false);
        }
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
  }, [isMounted, supabase.auth, onAuthStateChange]);

  // Sign in
  const signIn = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      
      if (!error) {
        toast.success('Logged in successfully!');
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
        toast.success('Account created! Please check your email to confirm your account.');
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
          
          // Call the callback if provided with the auth event for more granular control
          if (onAuthStateChange) {
            onAuthStateChange(true, event);
          }

          console.log(`[AUTH] Auth state changed to ${event} for user: ${data.user?.id || 'unknown'}`);
        } else {
          // Call the callback if provided - user is not authenticated
          if (onAuthStateChange) {
            onAuthStateChange(false, event);
          }
          
          console.log(`[AUTH] Auth state changed to ${event}, no session.`);
        }
      }
    );
    
    // Clean up subscription
    return () => {
      subscription.unsubscribe();
    };
  }, [refreshAuth, supabase.auth, onAuthStateChange]);

  return (
    <AuthContext.Provider
      value={{
        ...state,
        signIn,
        signUp,
        signOut,
        refreshAuth,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
} 
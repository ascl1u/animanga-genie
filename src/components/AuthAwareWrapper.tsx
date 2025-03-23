'use client';

import React, { useState, useEffect } from 'react';
import { useAuth } from '@/components/SimpleAuthProvider';
import { clearLocalStorage } from '@/services/localStorageService';

interface AuthAwareWrapperProps {
  children: React.ReactNode;
}

/**
 * AuthAwareWrapper
 * 
 * A component that forces its children to remount when the authentication
 * state changes. This is useful for components that need to reset their
 * state entirely when a user logs in or out.
 * 
 * This component sets a unique key on the wrapped component that changes
 * with the auth state, forcing React to unmount and remount it.
 */
export const AuthAwareWrapper: React.FC<AuthAwareWrapperProps> = ({ children }) => {
  const { isAuthenticated, user } = useAuth();
  const [key, setKey] = useState(`auth-${isAuthenticated}-${user?.id || 'none'}`);
  const [prevAuthState, setPrevAuthState] = useState(isAuthenticated);
  
  // Reset key when auth state changes to force remounting
  useEffect(() => {
    // Clear localStorage if transitioning from unauthenticated to authenticated
    if (!prevAuthState && isAuthenticated) {
      console.log('[AuthAwareWrapper] Transitioning from unauthenticated to authenticated state, clearing localStorage');
      clearLocalStorage();
    }
    
    setPrevAuthState(isAuthenticated);
    setKey(`auth-${isAuthenticated}-${user?.id || 'none'}-${Date.now()}`);
  }, [isAuthenticated, user?.id, prevAuthState]);
  
  return (
    <React.Fragment key={key}>
      {children}
    </React.Fragment>
  );
};

export default AuthAwareWrapper;

/**
 * Version with delayed remounting and loading state
 * This can be useful when transitioning between auth states needs to show a loading indicator
 */
interface AuthAwareWrapperWithLoadingProps extends AuthAwareWrapperProps {
  authenticatedKey?: string;
  unauthenticatedKey?: string;
  loadingComponent?: React.ReactNode;
}

export function AuthAwareWrapperWithLoading({
  children,
  authenticatedKey = 'auth',
  unauthenticatedKey = 'unauth',
  loadingComponent,
}: AuthAwareWrapperWithLoadingProps) {
  const { isAuthenticated } = useAuth();
  const [prevAuthState, setPrevAuthState] = useState(isAuthenticated);
  const [isTransitioning, setIsTransitioning] = useState(false);
  
  // Set up a key based on auth state to force remounting when it changes
  const [authKey, setAuthKey] = useState(
    isAuthenticated ? authenticatedKey : unauthenticatedKey
  );
  
  useEffect(() => {
    // If auth state changed
    if (prevAuthState !== isAuthenticated) {
      // Clear localStorage if transitioning from unauthenticated to authenticated
      if (!prevAuthState && isAuthenticated) {
        console.log('[AuthAwareWrapperWithLoading] Transitioning from unauthenticated to authenticated state, clearing localStorage');
        clearLocalStorage();
      }
      
      setIsTransitioning(true);
      
      // Short delay before remounting to allow for transitions
      const timer = setTimeout(() => {
        setAuthKey(isAuthenticated ? authenticatedKey : unauthenticatedKey);
        setIsTransitioning(false);
        setPrevAuthState(isAuthenticated);
      }, 500);
      
      return () => clearTimeout(timer);
    }
  }, [authenticatedKey, isAuthenticated, prevAuthState, unauthenticatedKey]);
  
  if (isTransitioning) {
    return (
      <div className="transition-opacity duration-300 ease-in-out">
        {loadingComponent || (
          <div className="flex justify-center items-center p-8">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-indigo-500"></div>
          </div>
        )}
      </div>
    );
  }
  
  return (
    <div key={authKey}>
      {children}
    </div>
  );
} 
'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useAuth } from './SimpleAuthProvider';

export default function ClientNavigation() {
  const [mounted, setMounted] = useState(false);
  
  // Only run on client
  useEffect(() => {
    setMounted(true);
  }, []);
  
  if (!mounted) {
    return (
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Link href="/" className="text-xl font-bold text-indigo-600">
                  AnimeManga Genie
                </Link>
              </div>
              <div className="ml-6 flex items-center space-x-4">
                <Link href="/" className="text-gray-700 hover:text-indigo-500">
                  Home
                </Link>
                <Link href="/about" className="text-gray-700 hover:text-indigo-500">
                  About
                </Link>
              </div>
            </div>
          </div>
        </div>
      </nav>
    );
  }
  
  // Client side rendering with auth
  const AuthenticatedNav = () => {
    const { user, isLoading, isAuthenticated, signOut } = useAuth();
    
    return (
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Link href="/" className="text-xl font-bold text-indigo-600">
                  AnimeManga Genie
                </Link>
              </div>
              <div className="ml-6 flex items-center space-x-4">
                <Link href="/" className="text-gray-700 hover:text-indigo-500">
                  Home
                </Link>
                <Link href="/about" className="text-gray-700 hover:text-indigo-500">
                  About
                </Link>
                <Link href="/explore" className="text-gray-700 hover:text-indigo-500">
                  Explore
                </Link>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {isLoading ? (
                <span className="text-gray-500">Loading...</span>
              ) : isAuthenticated ? (
                <>
                  <Link href="/profile" className="text-gray-700 hover:text-indigo-500">
                    Profile
                  </Link>
                  <span className="text-sm text-gray-500">{user?.email}</span>
                  <button
                    onClick={() => signOut()}
                    className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700"
                  >
                    Sign Out
                  </button>
                </>
              ) : (
                <>
                  <Link href="/login" className="text-gray-700 hover:text-indigo-500">
                    Login
                  </Link>
                  <Link 
                    href="/signup" 
                    className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700"
                  >
                    Sign Up
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      </nav>
    );
  };

  return <AuthenticatedNav />;
} 
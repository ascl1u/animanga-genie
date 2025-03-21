'use client';

import { useState, useEffect, useRef } from 'react';
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
                  AniManga Genie
                </Link>
              </div>
              <div className="ml-6 flex items-center space-x-4">
                <Link href="/search" className="text-gray-700 hover:text-indigo-500">
                  Search
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
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);
    
    // Close dropdown when clicking outside
    useEffect(() => {
      const handleClickOutside = (event: MouseEvent) => {
        if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
          setDropdownOpen(false);
        }
      };
      
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }, []);

    // Get user initials for avatar
    const getUserInitials = () => {
      if (!user || !user.email) return '?';
      const email = user.email;
      const nameParts = email.split('@')[0].split(/[._-]/);
      return nameParts.map(part => part[0]?.toUpperCase() || '').join('').substring(0, 2);
    };
    
    return (
      <nav className="bg-white shadow-sm relative z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Link href="/" className="text-xl font-bold text-indigo-600">
                  AniManga Genie
                </Link>
              </div>
              <div className="ml-6 flex items-center space-x-4">
                <Link href="/search" className="text-gray-700 hover:text-indigo-500">
                  Search
                </Link>
                <Link href="/recommendations" className="text-gray-700 hover:text-indigo-500">
                  Recommendations
                </Link>
                <Link href="/my-anime" className="text-gray-700 hover:text-indigo-500">
                  My Anime
                </Link>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {isLoading ? (
                <span className="text-gray-500">Loading...</span>
              ) : isAuthenticated ? (
                <div className="relative" ref={dropdownRef}>
                  <button
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                    className="flex items-center focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 rounded-full"
                    aria-expanded={dropdownOpen}
                    aria-haspopup="true"
                  >
                    <div className="w-10 h-10 rounded-full bg-indigo-600 text-white flex items-center justify-center font-medium shadow-md">
                      {getUserInitials()}
                    </div>
                  </button>
                  
                  {dropdownOpen && (
                    <div className="fixed inset-0 z-30 bg-black bg-opacity-25 md:bg-transparent md:inset-auto md:absolute">
                      <div className="absolute right-0 mt-2 w-64 rounded-lg shadow-xl bg-white ring-1 ring-black ring-opacity-5 py-1 z-40 border border-gray-200">
                        <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 rounded-t-lg">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {user?.email}
                          </p>
                        </div>
                        <Link
                          href="/profile"
                          className="block px-4 py-3 text-sm text-gray-700 hover:bg-gray-100 transition-colors duration-150"
                          onClick={() => setDropdownOpen(false)}
                        >
                          <div className="flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                            Profile
                          </div>
                        </Link>
                        <Link
                          href="/my-anime"
                          className="block px-4 py-3 text-sm text-gray-700 hover:bg-gray-100 transition-colors duration-150"
                          onClick={() => setDropdownOpen(false)}
                        >
                          <div className="flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                            My Anime
                          </div>
                        </Link>
                        <button
                          onClick={() => {
                            signOut();
                            setDropdownOpen(false);
                          }}
                          className="block w-full text-left px-4 py-3 text-sm text-gray-700 hover:bg-gray-100 transition-colors duration-150 rounded-b-lg"
                        >
                          <div className="flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                            </svg>
                            Sign Out
                          </div>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
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
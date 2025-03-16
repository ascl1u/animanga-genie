'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/components/SimpleAuthProvider';
import WatchHistoryForm from '@/components/WatchHistoryForm';
import WatchHistoryList from '@/components/WatchHistoryList';
import Link from 'next/link';

export default function MyAnimePage() {
  const { user, isLoading, isAuthenticated } = useAuth();
  const [mounted, setMounted] = useState(false);
  const router = useRouter();
  
  // Client-side only code
  useEffect(() => {
    setMounted(true);
  }, []);
  
  // Redirect to login if not authenticated after loading
  useEffect(() => {
    if (mounted && !isLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [mounted, isLoading, isAuthenticated, router]);
  
  // Show loading state
  if (isLoading || !mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Loading...</h1>
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500 mx-auto"></div>
        </div>
      </div>
    );
  }
  
  // Show my anime page if authenticated
  if (isAuthenticated && user) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">My Anime</h1>
              <p className="mt-1 text-sm text-gray-500">
                Track your anime watch history and ratings
              </p>
            </div>
            <Link
              href="/profile"
              className="px-4 py-2 text-sm font-medium text-indigo-600 bg-white border border-indigo-600 rounded-md hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
            >
              Back to Profile
            </Link>
          </div>
          
          {/* Watch History Form */}
          <WatchHistoryForm />
          
          {/* Watch History List */}
          <WatchHistoryList />
        </div>
      </div>
    );
  }
  
  // Fallback - should not reach here due to redirect
  return null;
} 
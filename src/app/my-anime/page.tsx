'use client';

import { useState, useEffect, useRef } from 'react';
import { useAuth } from '@/components/SimpleAuthProvider';
import WatchHistoryForm from '@/components/WatchHistoryForm';
import WatchHistoryList from '@/components/WatchHistoryList';
import WatchHistoryImport from '@/components/WatchHistoryImport';
import Link from 'next/link';
import { AnimeWatchHistoryItem } from '@/types/watchHistory';

export default function MyAnimePage() {
  const { isLoading, isAuthenticated } = useAuth();
  const [mounted, setMounted] = useState(false);
  const watchHistoryListRef = useRef<{ addAnime?: (anime: AnimeWatchHistoryItem) => void }>({}); 
  
  // Client-side only code
  useEffect(() => {
    setMounted(true);
  }, []);
  
  // Handler for when anime is added via the form
  const handleAnimeAdded = (anime: AnimeWatchHistoryItem) => {
    // If the WatchHistoryList component has exposed an addAnime method, call it
    if (watchHistoryListRef.current && watchHistoryListRef.current.addAnime) {
      watchHistoryListRef.current.addAnime(anime);
    }
  };
  
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
  
  // Show my anime page (for both authenticated and non-authenticated users)
  return (
    <div className="min-h-screen py-12">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">My Anime</h1>
            <p className="mt-1 text-sm text-gray-700">
              Track your anime watch history and ratings
            </p>
            {!isAuthenticated && (
              <p className="mt-2 text-sm text-orange-600">
                You are using AniManga Genie without an account.
                <Link href="/signup" className="ml-2 font-medium underline text-indigo-600">
                  Sign up
                </Link>
                <span className="mx-1">or</span>
                <Link href="/login" className="font-medium underline text-indigo-600">
                  Log in
                </Link>
                <span className="ml-1">to save your data permanently.</span>
              </p>
            )}
          </div>
        </div>
        
        {/* Import Options - Only show for authenticated users */}
        {isAuthenticated && <WatchHistoryImport />}
        
        {/* Watch History Form */}
        <WatchHistoryForm onAnimeAdded={handleAnimeAdded} />
        
        {/* Gap between form and list */}
        <div className="mt-6"></div>
        
        {/* Watch History List */}
        <WatchHistoryList ref={watchHistoryListRef} />
      </div>
    </div>
  );
} 
'use client';

import RecommendationCard from '@/components/RecommendationCard';
import { useRecommendations } from '@/context/RecommendationsContext';
import { useModelContext } from '@/context/ModelContext';
import { useAuth } from '@/components/SimpleAuthProvider';
import Link from 'next/link';
import type { MouseEvent } from 'react';
import { useState } from 'react';

export default function RecommendationsPage() {
  const { isAuthenticated } = useAuth();
  const { isModelLoading } = useModelContext();
  const {
    recommendations,
    watchHistory,
    isLoading,
    isError,
    errorMessage,
    generateRecommendations,
    refreshRecommendations,
    isInitialized,
    hasWatchHistory,
    isWatchHistoryLoaded,
    loadAttempts,
    modelLoadingProgress,
    feedback,
    addFeedback,
    isCollaborativeFilteringEnabled,
    watchHistoryChanged
  } = useRecommendations();

  // State for watch history dropdown
  const [showWatchHistory, setShowWatchHistory] = useState(false);

  const handleLike = (animeId: number) => {
    addFeedback(animeId, 'like');
  };

  const handleDislike = (animeId: number) => {
    addFeedback(animeId, 'dislike');
  };

  // Handle normal recommendation generation
  const handleGenerateRecommendations = (e: MouseEvent) => {
    e.preventDefault();
    generateRecommendations();
  };
  
  // Handle retry
  const handleRetry = (e: MouseEvent) => {
    e.preventDefault();
    refreshRecommendations();
  };

  // If no watch history, show message
  if (!isLoading && isWatchHistoryLoaded && !hasWatchHistory) {
    return (
      <div className="min-h-screen py-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-white p-8 rounded-lg shadow-md text-center">
            <h1 className="text-3xl font-bold text-gray-900 mb-4">Recommendations</h1>
            <div className="my-8">
              <p className="text-lg text-gray-700 mb-4">
                You haven&apos;t added any anime to your watch history yet. Add some anime to get personalized recommendations!
              </p>
              <Link 
                href="/my-anime"
                className="px-6 py-3 bg-indigo-600 text-white font-medium rounded-md shadow-sm hover:bg-indigo-700"
              >
                Add Anime to Watch History
              </Link>
            </div>
            {!isAuthenticated && (
              <div className="mt-8 p-4 bg-blue-50 rounded-lg">
                <p className="text-sm text-blue-600">
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
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-6">Your Recommendations</h1>
      
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <p className="text-gray-600 mb-4">
          These recommendations are personalized based on your preferences and watch history.
          The more anime you rate, the better your recommendations will become!
        </p>
        
        <div className="mb-4 p-3 bg-indigo-50 border border-indigo-200 text-indigo-700 rounded">
          <p className="font-medium flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 mr-2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
            </svg>
            Important: How recommendations work
          </p>
          <p className="text-sm mt-1">
            New recommendations are only generated when you add new anime or update ratings in your watch history. 
            {isAuthenticated 
              ? ' Your recommendations are saved to your account and can be accessed on any device.'
              : ' Your watch history and recommendations are stored locally in your browser.'}
          </p>
        </div>
        
        {isCollaborativeFilteringEnabled && (
          <div className="mb-4 p-3 bg-blue-50 border border-blue-200 text-blue-700 rounded">
            <p className="font-medium flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 mr-2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z" />
              </svg>
              Collaborative Filtering Enabled
            </p>
            <p className="text-sm mt-1">
              Your recommendations are enhanced using data from users with similar taste patterns.
            </p>
          </div>
        )}
        
        {!isLoading && (
          <div className="flex flex-col items-center mt-4">
            {!isWatchHistoryLoaded ? (
              <div className="mb-4 p-3 bg-blue-50 border border-blue-200 text-blue-700 rounded w-full flex items-center">
                <div className="w-5 h-5 mr-3">
                  <div className="w-full h-full border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
                <p>Loading your watch history...</p>
              </div>
            ) : !hasWatchHistory ? (
              <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 text-yellow-700 rounded w-full">
                <p>You don&apos;t have any watch history yet. Recommendations will use default preferences.</p>
              </div>
            ) : null}
            
            <button
              onClick={handleGenerateRecommendations}
              disabled={isModelLoading || isLoading || (isInitialized && !watchHistoryChanged)}
              className="px-6 py-3 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 
                        focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2
                        disabled:opacity-70 disabled:cursor-not-allowed transition-colors"
            >
              {isModelLoading 
                ? 'Loading Model...' 
                : isLoading
                  ? 'Generating...'
                  : isInitialized && !watchHistoryChanged
                    ? 'Up to Date'
                    : isInitialized
                      ? 'Generate New Recommendations'
                      : 'Generate Recommendations'}
            </button>
            
            {isInitialized && !watchHistoryChanged && (
              <p className="mt-2 text-sm text-amber-600">
                Your recommendations are up to date with your current watch history.
                Add or rate more anime to get new recommendations.
              </p>
            )}
          </div>
        )}
        
        {Object.keys(feedback).length > 0 && (
          <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded">
            <p className="text-green-700 text-sm">
              Thank you for your feedback! We&apos;ll use it to improve your recommendations.
            </p>
            <p className="text-green-700 text-xs mt-1">
              You&apos;ve provided feedback on {Object.keys(feedback).length} recommendation(s).
            </p>
          </div>
        )}
      </div>
      
      {isLoading && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <p className="text-gray-600">
            {isModelLoading 
              ? 'Loading the recommendation model... This might take a moment.' 
              : 'Generating your personalized recommendations... Almost there!'}
          </p>
          
          <div className="w-full h-4 bg-gray-200 rounded-full mt-4 overflow-hidden">
            <div 
              className={`h-full bg-indigo-600 rounded-full transition-all duration-300 
                ${modelLoadingProgress === 100 ? '' : 'animate-pulse'}`}
              style={{ 
                width: `${isModelLoading ? modelLoadingProgress : 100}%`
              }}
            ></div>
          </div>
          
          {loadAttempts > 1 && (
            <p className="text-sm text-gray-500 mt-2">
              Attempt {loadAttempts} - {isModelLoading ? 'Loading model...' : 'Running inference...'}
            </p>
          )}
          
          {/* Add notification for non-authenticated users */}
          {!isAuthenticated && (
            <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <p className="text-sm text-blue-600">
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
            </div>
          )}
        </div>
      )}
      
      {isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg shadow p-6 mb-6">
          <h3 className="text-lg font-semibold text-red-700 mb-2">Error</h3>
          <p className="text-red-700 mb-2">
            {errorMessage || 'Failed to load recommendations. Please try again.'}
          </p>
          
          {loadAttempts > 1 && (
            <p className="text-sm text-red-600 mb-3">
              Previous attempts: {loadAttempts - 1}
            </p>
          )}
          
          <button 
            className="mt-2 bg-red-600 text-white py-2 px-4 rounded hover:bg-red-700 transition-colors"
            onClick={handleRetry}
          >
            Retry
          </button>
          
          {/* Add notification for non-authenticated users */}
          {!isAuthenticated && (
            <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <p className="text-sm text-blue-600">
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
            </div>
          )}
        </div>
      )}
      
      {!isLoading && isInitialized && (
        <div className="p-3 bg-green-50 border border-green-200 text-green-700 rounded w-full mb-4">
          <p className="font-medium flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {watchHistoryChanged 
              ? 'Recommendations are being generated'
              : 'Recommendations loaded from your saved preferences'}
          </p>
          {!watchHistoryChanged && watchHistory.length > 0 && (
            <p className="text-sm ml-7 mt-1">
              Update your ratings or add more anime to get new recommendations.
            </p>
          )}
        </div>
      )}
      
      {recommendations.length > 0 && !isLoading && !isError && (
        <>
          <div className="mb-6">
            <h2 className="text-2xl font-semibold mb-2">
              Your Top Recommendations
            </h2>
            <p className="text-gray-600">
              Based on {watchHistory.length} anime in your watch history
            </p>
            
            {/* Add notification for non-authenticated users */}
            {!isAuthenticated && (
              <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <p className="text-sm text-blue-600">
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
              </div>
            )}
            
            {watchHistory.length > 0 && (
              <div className="mt-2 text-sm bg-blue-50 p-3 rounded border border-blue-200">
                <div 
                  className="flex justify-between items-center cursor-pointer"
                  onClick={() => setShowWatchHistory(!showWatchHistory)}
                >
                  <p className="font-medium text-blue-700">Recommending anime based on:</p>
                  <div className="text-blue-700">
                    {showWatchHistory ? '▲' : '▼'}
                  </div>
                </div>
                {showWatchHistory && (
                  <ul className="list-disc ml-5 mt-2 text-gray-800">
                    {watchHistory.map(item => (
                      <li key={item.id}>
                        {item.title} - Rated: {item.rating}/10
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {recommendations.map((anime) => (
              <div key={anime.id} className="relative">
                {feedback[anime.id] && (
                  <div className={`absolute top-2 right-2 z-10 px-2 py-1 text-xs font-bold rounded ${
                    feedback[anime.id] === 'like' 
                      ? 'bg-green-500 text-white' 
                      : 'bg-red-500 text-white'
                  }`}>
                    {feedback[anime.id] === 'like' ? 'Liked' : 'Disliked'}
                  </div>
                )}
                <RecommendationCard
                  anime={anime}
                  onLike={() => handleLike(anime.id)}
                  onDislike={() => handleDislike(anime.id)}
                />
              </div>
            ))}
          </div>
        </>
      )}
      
      {recommendations.length === 0 && !isLoading && !isError && isInitialized && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg shadow p-6">
          <p className="text-yellow-700">
            No recommendations found. Try updating your preferences in your profile.
          </p>
          <button 
            className="mt-4 bg-yellow-600 text-white py-2 px-4 rounded hover:bg-yellow-700"
            onClick={handleGenerateRecommendations}
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
} 
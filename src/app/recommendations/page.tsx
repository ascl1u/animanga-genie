'use client';

import { useRecommendations } from '@/hooks/useRecommendations';
import RecommendationCard from '@/components/RecommendationCard';
import { useState } from 'react';
import { useModelContext } from '@/context/ModelContext';
import type { MouseEvent } from 'react';

export default function RecommendationsPage() {
  const { isModelLoaded, isModelLoading } = useModelContext();
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
    modelLoadingProgress
  } = useRecommendations();

  const [feedback, setFeedback] = useState<Record<number, 'like' | 'dislike' | null>>({});
  // New state for debug mode
  const [showDebug, setShowDebug] = useState(false);
  // State for test limit
  const [testLimit, setTestLimit] = useState(3);

  const handleLike = (animeId: number) => {
    setFeedback(prev => ({
      ...prev,
      [animeId]: 'like'
    }));
    // Here you would also send the feedback to your backend
    console.log(`Liked anime ${animeId}`);
  };

  const handleDislike = (animeId: number) => {
    setFeedback(prev => ({
      ...prev,
      [animeId]: 'dislike'
    }));
    // Here you would also send the feedback to your backend
    console.log(`Disliked anime ${animeId}`);
  };

  const handleMoreInfo = (animeId: number) => {
    // Implementation for showing more info about the anime
    console.log(`Show more info for anime ${animeId}`);
  };

  // Generate test recommendations with limited count
  const handleTestRecommendations = (e: MouseEvent) => {
    e.preventDefault();
    // Call the hook with limited count
    console.log(`Generating test recommendations with limit: ${testLimit}`);
    generateRecommendations(testLimit);
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

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-6">Your Recommendations</h1>
      
      {/* Debug Toggle */}
      <div className="mb-4 flex justify-end">
        <button
          onClick={() => setShowDebug(!showDebug)}
          className="px-3 py-1 bg-gray-200 text-gray-800 rounded-md text-sm hover:bg-gray-300"
        >
          {showDebug ? 'Hide Debug' : 'Show Debug'}
        </button>
      </div>
      
      {/* Debug info panel */}
      {showDebug && (
        <div className="bg-gray-100 rounded-lg shadow p-4 mb-6 text-xs font-mono overflow-auto max-h-80">
          <h3 className="font-bold mb-2">Debug Information:</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p><strong>Model Loaded:</strong> {isModelLoaded ? 'Yes' : 'No'}</p>
              <p><strong>Model Loading:</strong> {isModelLoading ? 'Yes' : 'No'}</p>
              <p><strong>Status:</strong> {isLoading ? 'Loading' : isError ? 'Error' : isInitialized ? 'Ready' : 'Not Initialized'}</p>
              <p><strong>Watch History Items:</strong> {watchHistory.length}</p>
              <p><strong>Load Attempts:</strong> {loadAttempts}</p>
              <p><strong>Recommendations:</strong> {recommendations.length}</p>
            </div>
            <div>
              <div className="mb-2">
                <p className="font-bold mb-1">Test Options:</p>
                <div className="flex items-center space-x-2">
                  <label className="whitespace-nowrap">
                    Limit:
                    <input 
                      type="number" 
                      min="1" 
                      max="10" 
                      value={testLimit}
                      onChange={(e) => setTestLimit(Number(e.target.value))}
                      className="ml-2 w-16 px-2 py-1 border rounded"
                    />
                  </label>
                  <button
                    onClick={handleTestRecommendations}
                    className="px-3 py-1 bg-indigo-600 text-white text-xs rounded-md hover:bg-indigo-700"
                    disabled={isLoading}
                  >
                    Test ({testLimit})
                  </button>
                </div>
              </div>
              
              {watchHistory.length > 0 && (
                <div>
                  <p className="font-bold mb-1">Watch History:</p>
                  <ul className="list-disc pl-4">
                    {watchHistory.map((item) => (
                      <li key={item.id}>
                        {item.title} (ID: {item.anilist_id}) - Rated: {item.rating}/10
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <p className="text-gray-600 mb-4">
          These recommendations are personalized based on your preferences and watch history.
          The more anime you rate, the better your recommendations will become!
        </p>
        
        {!isInitialized && !isLoading && (
          <div className="flex flex-col items-center mt-4">
            {!isWatchHistoryLoaded ? (
              <div className="mb-4 p-3 bg-blue-50 border border-blue-200 text-blue-700 rounded w-full flex items-center">
                <div className="w-5 h-5 mr-3">
                  <div className="w-full h-full border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
                <p>Loading your watch history...</p>
              </div>
            ) : hasWatchHistory ? (
              <div className="mb-4 p-3 bg-blue-50 border border-blue-200 text-blue-700 rounded w-full">
                <p className="font-medium">Based on your watch history:</p>
                <ul className="list-disc pl-5 mt-2">
                  {watchHistory.slice(0, 3).map(item => (
                    <li key={item.id}>{item.title} - Rated: {item.rating}/10</li>
                  ))}
                  {watchHistory.length > 3 && <li>and {watchHistory.length - 3} more...</li>}
                </ul>
              </div>
            ) : (
              <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 text-yellow-700 rounded w-full">
                <p>You don&apos;t have any watch history yet. Recommendations will use default preferences.</p>
              </div>
            )}
            
            <button
              onClick={handleGenerateRecommendations}
              disabled={isModelLoading}
              className="px-6 py-3 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 
                        focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2
                        disabled:opacity-70 disabled:cursor-not-allowed transition-colors"
            >
              {isModelLoading 
                ? 'Loading Model...' 
                : isModelLoaded 
                  ? 'Generate Recommendations' 
                  : 'Load Model & Generate'}
            </button>
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
        </div>
      )}
      
      {isInitialized && !isLoading && !isError && (
        <>
          {recommendations.length > 0 ? (
            <>
              <div className="mb-6">
                <h2 className="text-2xl font-semibold text-gray-800 mb-2">
                  Your Top {recommendations.length} Recommendations
                </h2>
                <p className="text-gray-600">
                  Based on {watchHistory.length} anime in your watch history
                </p>
                {watchHistory.length > 0 && (
                  <div className="mt-2 text-sm bg-blue-50 p-3 rounded border border-blue-200">
                    <p className="font-medium text-blue-700">Recommending anime similar to:</p>
                    <ul className="list-disc ml-5 mt-1">
                      {watchHistory.map(item => (
                        <li key={item.id}>
                          {item.title} - Rated: {item.rating}/10
                        </li>
                      ))}
                    </ul>
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
                      onMoreInfo={() => handleMoreInfo(anime.id)}
                    />
                  </div>
                ))}
              </div>
            </>
          ) : (
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
        </>
      )}
    </div>
  );
} 
import { useState, useEffect, useCallback, useRef } from 'react';
import { getRecommendations, AnimeData } from '@/services/recommendationService';
import { useModelContext } from '@/context/ModelContext';
import { AnimeWatchHistoryItem } from '@/types/watchHistory';
import { dataAccessService } from '@/services/dataAccessService';
import { useAuth } from '@/components/SimpleAuthProvider';
import { WATCH_HISTORY_CHANGED_EVENT } from '@/services/watchHistoryService';
import { clearLocalStorage } from '@/services/localStorageService';

interface UseRecommendationsOptions {
  preferredGenres?: string[];
  preferredTags?: string[];
  limit?: number;
  autoLoad?: boolean;
}

/**
 * Simplified hook for managing anime recommendations
 * 
 * This hook handles generating anime recommendations based on the user's watch history.
 * It only persists recommendations for authenticated users and uses a request-based model
 * instead of complex effect dependencies.
 */
export function useRecommendations({
  preferredGenres = ['Action', 'Adventure'],
  preferredTags = ['magic', 'fantasy'],
  limit = 9,
  autoLoad = false,
}: UseRecommendationsOptions = {}) {
  const { isModelLoaded, isModelLoading, loadModel, loadingProgress } = useModelContext();
  const { isAuthenticated, user } = useAuth();
  
  // Core state - minimized to only what's essential
  const [recommendations, setRecommendations] = useState<AnimeData[]>([]);
  const [watchHistory, setWatchHistory] = useState<AnimeWatchHistoryItem[]>([]);
  const [similarUsers, setSimilarUsers] = useState<{ userId: string, similarity: number }[]>([]);
  
  // Single status object for all status flags
  const [status, setStatus] = useState({
    isLoading: false,
    isError: false,
    errorMessage: '',
    isInitialized: false,
    watchHistoryLoaded: false,
    loadAttempts: 0,
    collaborativeFilteringEnabled: false,
    watchHistoryChanged: false
  });
  
  // Reference to track if a recommendation operation is in progress
  const operationInProgressRef = useRef(false);
  
  /**
   * Fetch watch history from the appropriate source
   */
  const fetchWatchHistory = useCallback(async () => {
    try {
      const history = await dataAccessService.getWatchHistory();
      setWatchHistory(history);
      setStatus(prev => ({ 
        ...prev, 
        watchHistoryLoaded: true,
        watchHistoryChanged: prev.watchHistoryLoaded ? true : false 
      }));
      return history;
    } catch (error) {
      console.error('Error fetching watch history:', error);
      setStatus(prev => ({ ...prev, watchHistoryLoaded: true }));
      return [];
    }
  }, []);
  
  /**
   * Reset state on authentication changes
   */
  useEffect(() => {
    // Reset state on auth change
    setRecommendations([]);
    setWatchHistory([]);
    setSimilarUsers([]);
    
    setStatus({
      isLoading: false,
      isError: false,
      errorMessage: '',
      isInitialized: false,
      watchHistoryLoaded: false,
      loadAttempts: 0,
      collaborativeFilteringEnabled: false,
      watchHistoryChanged: false
    });
    
    // Fetch watch history after auth change
    fetchWatchHistory();
    
    // For unauthenticated users, set up cleanup on tab close
    if (!isAuthenticated) {
      const handleBeforeUnload = () => {
        clearLocalStorage();
      };
      
      window.addEventListener('beforeunload', handleBeforeUnload);
      return () => {
        window.removeEventListener('beforeunload', handleBeforeUnload);
      };
    }
  }, [isAuthenticated, user?.id, fetchWatchHistory]);
  
  /**
   * Core function to generate recommendations
   * This uses a request-based model rather than effect dependencies
   */
  const generateRecommendations = useCallback(async (customLimit?: number) => {
    // Don't start if we're already generating
    if (operationInProgressRef.current) return;
    operationInProgressRef.current = true;
    
    try {
      // If no watch history available, nothing to do
      if (watchHistory.length === 0) {
        setStatus(prev => ({ ...prev, isInitialized: true }));
        operationInProgressRef.current = false;
        return;
      }
      
      // Update loading state
      setStatus(prev => ({
        ...prev,
        isLoading: true,
        loadAttempts: prev.loadAttempts + 1,
        isError: false,
        errorMessage: '',
        watchHistoryChanged: false
      }));
      
      // Ensure model is loaded
      if (!isModelLoaded && !isModelLoading) {
        await loadModel();
      }
      
      const userId = user?.id || 'anonymous-user';
      const recommendationLimit = customLimit || limit;
      
      // Generate recommendations
      const result = await getRecommendations(
        userId,
        preferredGenres,
        preferredTags,
        recommendationLimit
      );
      
      if (result.status === 'success') {
        setRecommendations(result.recommendations);
        
        // Only authenticated users get recommendations persistently stored
        if (isAuthenticated && result.recommendations.length > 0) {
          const watchHistoryHash = dataAccessService.createWatchHistoryHash(watchHistory);
          await dataAccessService.saveRecommendations(result.recommendations, watchHistoryHash);
        }
        
        // Update collaborative filtering info if available
        if (result.debugInfo?.collaborativeFiltering?.used) {
          setStatus(prev => ({ ...prev, collaborativeFilteringEnabled: true }));
          
          const collaborativeData = result.debugInfo.collaborativeFiltering as Record<string, unknown>;
          if (collaborativeData.similarUsers && Array.isArray(collaborativeData.similarUsers)) {
            setSimilarUsers(collaborativeData.similarUsers as { userId: string, similarity: number }[]);
          }
        }
        
        // Update status
        setStatus(prev => ({
          ...prev,
          isLoading: false,
          isInitialized: true
        }));
      } else {
        // Handle error
        setStatus(prev => ({
          ...prev,
          isLoading: false,
          isError: true,
          errorMessage: result.error || 'Failed to generate recommendations',
          isInitialized: true
        }));
      }
    } catch (error) {
      console.error('Error getting recommendations:', error);
      setStatus(prev => ({
        ...prev,
        isLoading: false,
        isError: true,
        errorMessage: (error as Error).message,
        isInitialized: true
      }));
    } finally {
      operationInProgressRef.current = false;
    }
  }, [
    watchHistory, 
    isModelLoaded, 
    isModelLoading, 
    loadModel, 
    preferredGenres, 
    preferredTags, 
    limit,
    user?.id,
    isAuthenticated
  ]);
  
  /**
   * Refresh recommendations - convenience method that's just an alias
   */
  const refreshRecommendations = useCallback((customLimit?: number) => {
    return generateRecommendations(customLimit);
  }, [generateRecommendations]);
  
  /**
   * Setup watch history change listener
   */
  useEffect(() => {
    // Set up watch history change listener
    const handleWatchHistoryChange = () => {
      fetchWatchHistory();
    };
    
    // Listen for manual recommendation generation requests
    const handleManualRecommendationGeneration = () => {
      generateRecommendations();
    };
    
    // Add event listeners
    window.addEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
    window.addEventListener('manual-recommendation-generation', handleManualRecommendationGeneration);
    
    // Clean up
    return () => {
      window.removeEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
      window.removeEventListener('manual-recommendation-generation', handleManualRecommendationGeneration);
    };
  }, [fetchWatchHistory, generateRecommendations]);
  
  /**
   * Initial load of recommendations if autoLoad is enabled
   */
  useEffect(() => {
    // Only load if autoLoad is enabled, we have watch history, and haven't initialized yet
    if (autoLoad && !status.isInitialized && status.watchHistoryLoaded && watchHistory.length > 0) {
      generateRecommendations();
    }
  }, [autoLoad, status.isInitialized, status.watchHistoryLoaded, watchHistory, generateRecommendations]);
  
  /**
   * Load recommendations for authenticated users if watch history changes
   */
  useEffect(() => {
    // For authenticated users, if watch history changes and we already have recommendations,
    // regenerate them automatically
    if (isAuthenticated && status.watchHistoryChanged && status.isInitialized) {
      generateRecommendations();
    }
  }, [isAuthenticated, status.watchHistoryChanged, status.isInitialized, generateRecommendations]);
  
  // Return a consistent API matching what RecommendationsContext expects
  return {
    recommendations,
    watchHistory,
    isLoading: status.isLoading,
    isError: status.isError,
    errorMessage: status.errorMessage,
    isInitialized: status.isInitialized,
    hasWatchHistory: watchHistory.length > 0,
    isWatchHistoryLoaded: status.watchHistoryLoaded,
    loadAttempts: status.loadAttempts,
    modelLoadingProgress: loadingProgress,
    generateRecommendations,
    refreshRecommendations,
    isCollaborativeFilteringEnabled: status.collaborativeFilteringEnabled,
    similarUsers,
    watchHistoryChanged: status.watchHistoryChanged
  };
} 
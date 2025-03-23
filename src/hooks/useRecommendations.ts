import { useState, useEffect, useCallback } from 'react';
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

// Debug logging utility
const debugLog = (message: string) => {
  console.log(`[RECOMMENDATIONS] ${message}`);
};

/**
 * Simplified hook for managing anime recommendations
 * 
 * This hook handles generating anime recommendations based on the user's watch history.
 * It only persists recommendations for authenticated users and uses a streamlined state model.
 */
export function useRecommendations({
  preferredGenres = ['Action', 'Adventure'],
  preferredTags = ['magic', 'fantasy'],
  limit = 9,
  autoLoad = false,
}: UseRecommendationsOptions = {}) {
  const { isModelLoaded, isModelLoading, loadModel, loadingProgress } = useModelContext();
  const { isAuthenticated, user } = useAuth();
  
  // Simplified core state model
  const [recommendations, setRecommendations] = useState<AnimeData[]>([]);
  const [watchHistory, setWatchHistory] = useState<AnimeWatchHistoryItem[]>([]);
  const [similarUsers, setSimilarUsers] = useState<{ userId: string, similarity: number }[]>([]);
  
  // Combined status object with essential flags
  const [status, setStatus] = useState({
    isLoading: false,
    isError: false,
    errorMessage: '',
    isInitialized: false,
    loadAttempts: 0,
    collaborativeFilteringEnabled: false,
    watchHistoryLoaded: false
  });
  
  // Fetch watch history from appropriate source
  const fetchWatchHistory = useCallback(async () => {
    try {
      debugLog('Fetching watch history');
      const history = await dataAccessService.getWatchHistory();
      debugLog(`Fetched ${history.length} items from dataAccessService`);
      
      setWatchHistory(history);
      setStatus(prev => ({ ...prev, watchHistoryLoaded: true }));
      
      return history;
    } catch (error) {
      console.error('Error fetching watch history:', error);
      setStatus(prev => ({ ...prev, watchHistoryLoaded: true }));
      return [];
    }
  }, []);

  // Reset state on auth changes
  useEffect(() => {
    debugLog(`Auth state changed: ${isAuthenticated ? 'authenticated' : 'not authenticated'}`);
    
    // Reset all state on auth change
    setRecommendations([]);
    setWatchHistory([]);
    setSimilarUsers([]);
    
    setStatus({
      isLoading: false,
      isError: false,
      errorMessage: '',
      isInitialized: false,
      loadAttempts: 0,
      collaborativeFilteringEnabled: false,
      watchHistoryLoaded: false
    });
    
    // Fetch watch history after auth change
    fetchWatchHistory();
  }, [isAuthenticated, user?.id, fetchWatchHistory]);
  
  // Primary function to generate recommendations
  const fetchRecommendations = useCallback(async (customLimit?: number) => {
    // If no watch history available, nothing to do
    if (watchHistory.length === 0) {
      debugLog('No watch history available, skipping recommendation generation');
      setStatus(prev => ({ ...prev, isInitialized: true }));
      return;
    }
    
    // Update loading state
    setStatus(prev => ({
      ...prev,
      isLoading: true,
      loadAttempts: prev.loadAttempts + 1,
      isError: false,
      errorMessage: ''
    }));
    
    // Ensure model is loaded
    if (!isModelLoaded && !isModelLoading) {
      debugLog('Loading model first');
      await loadModel();
    }
    
    const userId = user?.id || 'anonymous-user';
    const recommendationLimit = customLimit || limit;
    
    try {
      // Get recommendations
      const result = await getRecommendations(
        userId,
        preferredGenres,
        preferredTags,
        recommendationLimit
      );
      
      if (result.status === 'success') {
        debugLog(`Recommendations generated successfully: ${result.recommendations.length} items`);
        setRecommendations(result.recommendations);
        
        // Only save recommendations for authenticated users
        if (isAuthenticated && result.recommendations.length > 0) {
          // Create a hash for recommendations tied to current watch history
          const watchHistoryHash = dataAccessService.createWatchHistoryHash(watchHistory);
          debugLog('Saving recommendations for authenticated user');
          await dataAccessService.saveRecommendations(result.recommendations, watchHistoryHash);
        }
        
        // Update collaborative filtering state if available
        if (result.debugInfo?.collaborativeFiltering?.used) {
          setStatus(prev => ({ ...prev, collaborativeFilteringEnabled: true }));
          
          // Handle similar users data if present
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
        debugLog(`Error in recommendation generation: ${result.error}`);
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

  // Initial setup & event listeners
  useEffect(() => {
    // Automatically load recommendations if specified
    if (autoLoad && !status.isInitialized && status.watchHistoryLoaded && watchHistory.length > 0) {
      debugLog('Auto-loading recommendations');
      fetchRecommendations();
    }
    
    // Set up watch history change listener
    const handleWatchHistoryChange = () => {
      debugLog('Watch history change event detected');
      fetchWatchHistory().then(freshHistory => {
        if (freshHistory.length > 0 && status.isInitialized) {
          // If we already had recommendations, refresh them with the new history
          fetchRecommendations();
        }
      });
    };
    
    // Listen for manual recommendation generation requests
    const handleManualRecommendationGeneration = () => {
      debugLog('Manual recommendation generation requested');
      fetchRecommendations();
    };
    
    // Add event listeners
    window.addEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
    window.addEventListener('manual-recommendation-generation', handleManualRecommendationGeneration);
    
    // For unauthenticated users, clear watch history on tab close
    if (!isAuthenticated) {
      const handleBeforeUnload = () => {
        debugLog('Tab closing - clearing localStorage for unauthenticated user');
        clearLocalStorage();
      };
      
      window.addEventListener('beforeunload', handleBeforeUnload);
      
      // Clean up
      return () => {
        window.removeEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
        window.removeEventListener('manual-recommendation-generation', handleManualRecommendationGeneration);
        window.removeEventListener('beforeunload', handleBeforeUnload);
      };
    }
    
    // Clean up
    return () => {
      window.removeEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
      window.removeEventListener('manual-recommendation-generation', handleManualRecommendationGeneration);
    };
  }, [
    autoLoad,
    status.isInitialized,
    status.watchHistoryLoaded,
    watchHistory,
    fetchWatchHistory,
    fetchRecommendations,
    isAuthenticated
  ]);

  // For authenticated users only, try loading saved recommendations
  useEffect(() => {
    const loadSavedRecommendations = async () => {
      // Only load saved recommendations for authenticated users
      if (!isAuthenticated || !status.watchHistoryLoaded || status.isInitialized || status.isLoading || watchHistory.length === 0) {
        return;
      }
      
      try {
        const watchHistoryHash = dataAccessService.createWatchHistoryHash(watchHistory);
        if (!watchHistoryHash) return;
        
        debugLog('Authenticated user - checking for saved recommendations');
        const savedRecommendations = await dataAccessService.getRecommendations(watchHistoryHash);
        
        if (savedRecommendations && savedRecommendations.length > 0) {
          debugLog(`Found ${savedRecommendations.length} saved recommendations`);
          setRecommendations(savedRecommendations);
          setStatus(prev => ({ ...prev, isInitialized: true }));
        } else if (autoLoad) {
          debugLog('No saved recommendations found - generating new ones');
          fetchRecommendations();
        }
      } catch (error) {
        console.error('Error loading saved recommendations:', error);
      }
    };
    
    loadSavedRecommendations();
  }, [
    isAuthenticated,
    status.watchHistoryLoaded,
    status.isInitialized,
    status.isLoading,
    watchHistory,
    autoLoad,
    fetchRecommendations
  ]);

  return {
    recommendations,
    watchHistory,
    isLoading: status.isLoading || isModelLoading,
    isError: status.isError,
    errorMessage: status.errorMessage,
    isInitialized: status.isInitialized,
    isModelLoaded,
    loadAttempts: status.loadAttempts,
    modelLoadingProgress: loadingProgress,
    generateRecommendations: fetchRecommendations,
    refreshRecommendations: fetchRecommendations,
    hasWatchHistory: watchHistory.length > 0,
    isWatchHistoryLoaded: status.watchHistoryLoaded,
    isCollaborativeFilteringEnabled: status.collaborativeFilteringEnabled,
    similarUsers,
    watchHistoryChanged: !status.isInitialized // For compatibility
  };
} 
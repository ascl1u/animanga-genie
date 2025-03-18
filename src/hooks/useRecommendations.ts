import { useState, useEffect, useCallback } from 'react';
import { getRecommendations, AnimeData } from '@/services/recommendationService';
import { useModelContext } from '@/context/ModelContext';
import { logOnnxServiceState } from '@/services/onnxModelService';
import { AnimeWatchHistoryItem } from '@/types/watchHistory';
import { createClient } from '@/utils/supabase/client';
import { getUserWatchHistory, WATCH_HISTORY_CHANGED_EVENT } from '@/services/watchHistoryService';
import { collaborativeFilteringService } from '@/services/collaborativeFilteringService';
import { saveRecommendations, loadRecommendations } from '@/services/recommendationPersistenceService';

interface UseRecommendationsOptions {
  userId?: string;
  preferredGenres?: string[];
  preferredTags?: string[];
  limit?: number;
  autoLoad?: boolean;
}

// Helper function to create a hash of the watch history for comparison
const createWatchHistoryHash = (history: AnimeWatchHistoryItem[]): string => {
  return history
    .map(item => `${item.anilist_id}-${item.rating}`)
    .sort()
    .join('|');
};

// Debug logging utility
const debugLog = (message: string) => {
  console.log(`[RECOMMENDATIONS] ${message}`);
};

export function useRecommendations({
  userId,
  preferredGenres = ['Action', 'Adventure'],
  preferredTags = ['magic', 'fantasy'],
  limit = 9,
  autoLoad = false, // Default to not loading automatically
}: UseRecommendationsOptions = {}) {
  const { isModelLoaded, isModelLoading, loadModel, loadingProgress } = useModelContext();
  const supabase = createClient();
  
  const [status, setStatus] = useState<{
    isLoading: boolean;
    isError: boolean;
    errorMessage: string;
    isInitialized: boolean; // Track if recommendations have been initialized
    loadAttempts: number; // Track number of load attempts
    collaborativeFilteringEnabled: boolean; // Track if collaborative filtering is enabled
  }>({
    isLoading: false,
    isError: false,
    errorMessage: '',
    isInitialized: false,
    loadAttempts: 0,
    collaborativeFilteringEnabled: false
  });

  const [recommendations, setRecommendations] = useState<AnimeData[]>([]);
  const [watchHistory, setWatchHistory] = useState<AnimeWatchHistoryItem[]>([]);
  const [watchHistoryLoaded, setWatchHistoryLoaded] = useState<boolean>(false);
  const [similarUsers, setSimilarUsers] = useState<{ userId: string, similarity: number }[]>([]);
  // Track the hash of the current watch history to detect changes
  const [watchHistoryHash, setWatchHistoryHash] = useState<string>('');
  // Track if watch history has changed since last recommendation
  const [watchHistoryChanged, setWatchHistoryChanged] = useState<boolean>(true);

  // Get current user ID if not provided
  const getUserId = useCallback(async () => {
    if (userId) return userId;
    
    try {
      const { data: { user } } = await supabase.auth.getUser();
      return user?.id || '1'; // Fallback to default
    } catch (error) {
      debugLog(`Error getting user ID: ${error instanceof Error ? error.message : String(error)}`);
      return '1'; // Fallback to default
    }
  }, [supabase.auth, userId]);

  // Fetch watch history separately
  const fetchWatchHistory = useCallback(async () => {
    try {
      debugLog('Fetching watch history...');
      const history = await getUserWatchHistory();
      debugLog(`Fetched watch history: ${history.length} items`);
      
      // Create a hash of the new watch history
      const newHash = createWatchHistoryHash(history);
      
      // Check if watch history has changed from the current one
      if (newHash !== watchHistoryHash) {
        debugLog('Watch history has changed, enabling new recommendations');
        setWatchHistoryChanged(true);
        setWatchHistoryHash(newHash);
      }
      
      setWatchHistory(history);
      setWatchHistoryLoaded(true);
      return history;
    } catch (error) {
      debugLog(`Error fetching watch history: ${error instanceof Error ? error.message : String(error)}`);
      setWatchHistoryLoaded(true);
      return [];
    }
  }, [watchHistoryHash]);

  // Memoized function to fetch recommendations
  const fetchRecommendations = useCallback(async (customLimit?: number) => {
    // Check if watch history has changed since last recommendation
    if (!watchHistoryChanged && status.isInitialized) {
      debugLog('Watch history unchanged since last recommendation, skipping generation');
      return;
    }
    
    const currentUserId = await getUserId();
    debugLog(`Using user ID: ${currentUserId}`);

    // Use custom limit if provided, otherwise use the default
    const recommendationLimit = customLimit || limit;
    debugLog(`Using recommendation limit: ${recommendationLimit}`);

    // Increment attempt counter
    setStatus(prev => ({
      ...prev,
      loadAttempts: prev.loadAttempts + 1,
      isLoading: true,
      isError: false,
      errorMessage: '',
    }));

    // Make sure model is loaded first
    if (!isModelLoaded) {
      debugLog('Model not loaded yet, loading model first...');
      logOnnxServiceState();
      
      try {
        // Load the model first and wait for it to complete
        const success = await loadModel();
        
        if (!success) {
          throw new Error('Failed to load model - operation returned false');
        }
        
        debugLog('Model loaded successfully, now will fetch recommendations');
      } catch (error) {
        debugLog(`Error loading model: ${error instanceof Error ? error.message : String(error)}`);
        logOnnxServiceState();
        setStatus(prev => ({
          ...prev,
          isLoading: false,
          isError: true,
          errorMessage: `Failed to load model: ${error instanceof Error ? error.message : String(error)}`,
        }));
        return;
      }
    }
    
    debugLog('Starting recommendation generation');
    
    // Pre-train collaborative filtering model
    try {
      // Only attempt to train if we have watch history
      if (watchHistory.length >= 5) {
        debugLog('Pre-training collaborative filtering model...');
        await collaborativeFilteringService.trainModel();
        
        // Get similar users to display in UI
        const similarUsersResult = await collaborativeFilteringService.getSimilarUsers(currentUserId, 5);
        setSimilarUsers(similarUsersResult);
        
        if (similarUsersResult.length > 0) {
          debugLog(`Found ${similarUsersResult.length} similar users through collaborative filtering`);
          setStatus(prev => ({
            ...prev,
            collaborativeFilteringEnabled: true
          }));
        }
      }
    } catch (error) {
      debugLog(`Error pre-training collaborative filtering model: ${error instanceof Error ? error.message : String(error)}`);
      // Continue with regular recommendations - this is just an enhancement
    }
    
    try {
      const result = await getRecommendations(
        currentUserId,
        preferredGenres,
        preferredTags,
        recommendationLimit
      );
      
      if (result.status === 'success') {
        debugLog(`Recommendations generated successfully: ${result.recommendations.length} items`);
        setRecommendations(result.recommendations);
        
        // Save watch history if available
        if (result.userWatchHistory) {
          debugLog(`Received watch history with ${result.userWatchHistory.length} items`);
          const newHistory = result.userWatchHistory;
          setWatchHistory(newHistory);
          setWatchHistoryLoaded(true);
          
          // Update the watch history hash and mark as unchanged
          const newHash = createWatchHistoryHash(newHistory);
          setWatchHistoryHash(newHash);
          setWatchHistoryChanged(false);
          
          // Save recommendations to Supabase
          if (result.recommendations.length > 0) {
            debugLog('Saving recommendations to database');
            saveRecommendations(result.recommendations, newHash);
          }
        }
        
        // Log any debug info
        if (result.debugInfo) {
          debugLog(`Debug info: ${JSON.stringify(result.debugInfo, null, 2)}`);
          
          // Check if collaborative filtering was used
          if (result.debugInfo.collaborativeFiltering?.used) {
            setStatus(prev => ({
              ...prev,
              collaborativeFilteringEnabled: true
            }));
          }
        }
        
        setStatus(prev => ({
          ...prev,
          isLoading: false,
          isError: false,
          errorMessage: '',
          isInitialized: true
        }));
      } else {
        debugLog(`Error in recommendation status: ${result.error}`);
        setStatus(prev => ({
          ...prev,
          isLoading: false,
          isError: true,
          errorMessage: result.error || 'Failed to generate recommendations',
          isInitialized: true,
        }));
      }
    } catch (error) {
      console.error('Error getting recommendations:', error);
      setStatus(prev => ({
        ...prev,
        isLoading: false,
        isError: true,
        errorMessage: (error as Error).message,
        isInitialized: true,
      }));
    }
  }, [getUserId, isModelLoaded, limit, loadModel, preferredGenres, preferredTags, status.isInitialized, watchHistory, watchHistoryChanged]);

  // Effect to auto-load if specified
  useEffect(() => {
    if (autoLoad && !status.isInitialized && !status.isLoading) {
      debugLog('Auto-loading recommendations');
      fetchRecommendations();
    }
  }, [autoLoad, fetchRecommendations, status.isInitialized, status.isLoading]);

  // Effect to fetch watch history on mount and when watch history changes
  useEffect(() => {
    if (!watchHistoryLoaded) {
      fetchWatchHistory();
    }
    
    // Listen for watch history change events
    const handleWatchHistoryChange = () => {
      debugLog('Watch history change event received');
      fetchWatchHistory();
    };
    
    // Add event listener
    window.addEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
    
    // Clean up
    return () => {
      window.removeEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
    };
  }, [fetchWatchHistory, watchHistoryLoaded]);
  
  // Effect to load saved recommendations when watch history is loaded
  useEffect(() => {
    // Function to load saved recommendations if available
    const loadSavedRecommendations = async () => {
      if (watchHistoryLoaded && watchHistoryHash && !status.isInitialized && !status.isLoading) {
        debugLog('Attempting to load saved recommendations');
        const savedRecommendations = await loadRecommendations(watchHistoryHash);
        
        if (savedRecommendations && savedRecommendations.length > 0) {
          debugLog(`Loaded ${savedRecommendations.length} saved recommendations`);
          setRecommendations(savedRecommendations);
          setStatus(prev => ({
            ...prev,
            isInitialized: true,
            isError: false,
            errorMessage: '',
          }));
          // Mark watch history as unchanged since we just loaded matching recommendations
          setWatchHistoryChanged(false);
        } else if (autoLoad) {
          // If no saved recommendations and autoLoad is true, generate new ones
          debugLog('No saved recommendations found - generating new recommendations');
          fetchRecommendations();
        }
      }
    };
    
    if (watchHistoryLoaded && !status.isInitialized && !status.isLoading) {
      loadSavedRecommendations();
    }
  }, [autoLoad, fetchRecommendations, status.isInitialized, status.isLoading, watchHistoryHash, watchHistoryLoaded]);

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
    isWatchHistoryLoaded: watchHistoryLoaded,
    isCollaborativeFilteringEnabled: status.collaborativeFilteringEnabled,
    similarUsers,
    watchHistoryChanged
  };
} 
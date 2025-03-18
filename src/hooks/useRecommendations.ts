import { useState, useEffect, useCallback } from 'react';
import { getRecommendations, AnimeData } from '@/services/recommendationService';
import { useModelContext } from '@/context/ModelContext';
import { logOnnxServiceState } from '@/services/onnxModelService';
import { AnimeWatchHistoryItem } from '@/types/watchHistory';
import { createClient } from '@/utils/supabase/client';
import { getUserWatchHistory } from '@/services/watchHistoryService';

interface UseRecommendationsOptions {
  userId?: string;
  preferredGenres?: string[];
  preferredTags?: string[];
  limit?: number;
  autoLoad?: boolean;
}

function debugLog(message: string): void {
  console.log(`[HOOK] ${message}`);
}

export function useRecommendations({
  userId,
  preferredGenres = ['Action', 'Adventure'],
  preferredTags = ['magic', 'fantasy'],
  limit = 10,
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
  }>({
    isLoading: false,
    isError: false,
    errorMessage: '',
    isInitialized: false,
    loadAttempts: 0
  });

  const [recommendations, setRecommendations] = useState<AnimeData[]>([]);
  const [watchHistory, setWatchHistory] = useState<AnimeWatchHistoryItem[]>([]);
  const [watchHistoryLoaded, setWatchHistoryLoaded] = useState<boolean>(false);

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
      setWatchHistory(history);
      setWatchHistoryLoaded(true);
      return history;
    } catch (error) {
      debugLog(`Error fetching watch history: ${error instanceof Error ? error.message : String(error)}`);
      setWatchHistoryLoaded(true);
      return [];
    }
  }, []);

  // Memoized function to fetch recommendations
  const fetchRecommendations = useCallback(async (customLimit?: number) => {
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
          setWatchHistory(result.userWatchHistory);
          setWatchHistoryLoaded(true);
        }
        
        // Log any debug info
        if (result.debugInfo) {
          debugLog(`Debug info: ${JSON.stringify(result.debugInfo, null, 2)}`);
        }
        
        setStatus({
          isLoading: false,
          isError: false,
          errorMessage: '',
          isInitialized: true,
          loadAttempts: status.loadAttempts
        });
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
  }, [getUserId, preferredGenres, preferredTags, limit, isModelLoaded, loadModel, status.loadAttempts]);

  // Effect to auto-load if specified
  useEffect(() => {
    if (autoLoad && !status.isInitialized && !status.isLoading) {
      debugLog('Auto-loading recommendations');
      fetchRecommendations();
    }
  }, [autoLoad, fetchRecommendations, status.isInitialized, status.isLoading]);

  // Effect to fetch watch history on mount
  useEffect(() => {
    if (!watchHistoryLoaded) {
      fetchWatchHistory();
    }
  }, [fetchWatchHistory, watchHistoryLoaded]);

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
  };
} 
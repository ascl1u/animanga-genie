import { useState, useEffect, useCallback, useRef } from 'react';
import { getRecommendations, AnimeData } from '@/services/recommendationService';
import { useModelContext } from '@/context/ModelContext';
import { AnimeWatchHistoryItem } from '@/types/watchHistory';
import { dataAccessService } from '@/services/dataAccessService';
import { useAuth } from '@/components/SimpleAuthProvider';
import { WATCH_HISTORY_CHANGED_EVENT } from '@/services/watchHistoryService';

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
 * This hook handles fetching, generating, and caching anime recommendations
 * based on the user's watch history. It automatically adapts to the user's
 * authentication state using the dataAccessService.
 */
export function useRecommendations({
  preferredGenres = ['Action', 'Adventure'],
  preferredTags = ['magic', 'fantasy'],
  limit = 9,
  autoLoad = false,
}: UseRecommendationsOptions = {}) {
  const { isModelLoaded, isModelLoading, loadModel, loadingProgress } = useModelContext();
  const { isAuthenticated, user } = useAuth();
  
  // Basic state for recommendations
  const [recommendations, setRecommendations] = useState<AnimeData[]>([]);
  const [watchHistory, setWatchHistory] = useState<AnimeWatchHistoryItem[]>([]);
  const [watchHistoryLoaded, setWatchHistoryLoaded] = useState<boolean>(false);
  const [watchHistoryHash, setWatchHistoryHash] = useState<string>('');
  const [watchHistoryChanged, setWatchHistoryChanged] = useState<boolean>(true);
  const [similarUsers, setSimilarUsers] = useState<{ userId: string, similarity: number }[]>([]);
  
  // State for tracking loading and errors
  const [status, setStatus] = useState({
    isLoading: false,
    isError: false,
    errorMessage: '',
    isInitialized: false,
    loadAttempts: 0,
    collaborativeFilteringEnabled: false
  });

  // Simple session tracking to avoid race conditions
  const sessionIdRef = useRef<string>(Math.random().toString(36).substring(2));
  
  // Fetch watch history using the dataAccessService
  const fetchWatchHistory = useCallback(async () => {
    try {
      debugLog('Fetching watch history');
      
      // Get history from the appropriate source via dataAccessService
      const history = await dataAccessService.getWatchHistory();
      debugLog(`Fetched ${history.length} items from dataAccessService`);
      
      // Update state with the fetched history
      setWatchHistory(history);
      setWatchHistoryLoaded(true);
      
      // Create a hash to detect changes using dataAccessService
      const newHash = dataAccessService.createWatchHistoryHash(history);
      
      // Check if watch history has changed
      if (newHash !== watchHistoryHash) {
        debugLog('Watch history has changed, enabling new recommendations');
        setWatchHistoryChanged(true);
        setWatchHistoryHash(newHash);
      }
      
      return history;
    } catch (error) {
      console.error('Error fetching watch history:', error);
      setWatchHistoryLoaded(true);
      return [];
    }
  }, [watchHistoryHash]);

  // Effect to handle auth state changes by resetting state
  useEffect(() => {
    // Generate a new session ID for new auth state
    sessionIdRef.current = Math.random().toString(36).substring(2);
    
    // Reset all state on auth change to ensure clean slate
    setRecommendations([]);
    setWatchHistory([]);
    setWatchHistoryHash('');
    setWatchHistoryLoaded(false);
    setWatchHistoryChanged(true);
    
    // Reset status
    setStatus({
      isLoading: false,
      isError: false,
      errorMessage: '',
      isInitialized: false,
      loadAttempts: 0,
      collaborativeFilteringEnabled: false
    });
    
    debugLog(`Auth state changed: ${isAuthenticated ? 'authenticated' : 'not authenticated'}`);
  }, [isAuthenticated, user?.id]);

  // Fetch recommendations function
  const fetchRecommendations = useCallback(async (customLimit?: number) => {
    // Skip if watch history unchanged and we already have recommendations
    if (!watchHistoryChanged && status.isInitialized) {
      debugLog('Watch history unchanged since last recommendation, skipping generation');
      return;
    }
    
    // Skip if no watch history available
    if (watchHistory.length === 0) {
      debugLog('No watch history available, skipping recommendation generation');
      setStatus(prev => ({
        ...prev,
        isInitialized: true
      }));
      return;
    }
    
    // Start loading
    setStatus(prev => ({
      ...prev,
      isLoading: true,
      loadAttempts: prev.loadAttempts + 1
    }));
    
    // Ensure model is loaded
    if (!isModelLoaded && !isModelLoading) {
      debugLog('Loading model first');
      await loadModel();
    }
    
    // Use actual user ID or fallback to default
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
        
        // Save recommendations using dataAccessService
        if (result.recommendations.length > 0) {
          debugLog('Saving recommendations');
          await dataAccessService.saveRecommendations(result.recommendations, watchHistoryHash);
        }
        
        // Update collaborative filtering state if available
        if (result.debugInfo?.collaborativeFiltering?.used) {
          setStatus(prev => ({
            ...prev,
            collaborativeFilteringEnabled: true
          }));
          
          // Update similar users if they're included in the response
          // Handle object with potentially unknown properties
          const collaborativeData = result.debugInfo.collaborativeFiltering as Record<string, unknown>;
          if (collaborativeData.similarUsers && 
              Array.isArray(collaborativeData.similarUsers)) {
            setSimilarUsers(collaborativeData.similarUsers as { userId: string, similarity: number }[]);
          }
        }
        
        // Finish loading
        setStatus(prev => ({
          ...prev,
          isLoading: false,
          isError: false,
          errorMessage: '',
          isInitialized: true
        }));
        
        // Mark watch history as processed
        setWatchHistoryChanged(false);
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
    watchHistoryChanged, 
    watchHistoryHash,
    status.isInitialized, 
    isModelLoaded, 
    isModelLoading, 
    loadModel, 
    preferredGenres, 
    preferredTags, 
    limit,
    user?.id
  ]);

  // Load watch history and attempt to load saved recommendations
  useEffect(() => {
    if (!watchHistoryLoaded) {
      fetchWatchHistory();
    }
    
    // Listen for watch history changes
    const handleWatchHistoryChange = () => {
      debugLog('Watch history change event detected, refreshing watch history');
      fetchWatchHistory();
    };
    
    // Listen for manual recommendation generation requests
    const handleManualRecommendationGeneration = () => {
      debugLog('Manual recommendation generation requested');
      fetchRecommendations();
    };
    
    // Add event listeners
    window.addEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
    window.addEventListener('manual-recommendation-generation', handleManualRecommendationGeneration);
    
    // Clean up
    return () => {
      window.removeEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
      window.removeEventListener('manual-recommendation-generation', handleManualRecommendationGeneration);
    };
  }, [fetchWatchHistory, watchHistoryLoaded, fetchRecommendations]);
  
  // Load saved recommendations effect
  useEffect(() => {
    const loadSavedRecommendations = async () => {
      // Only try to load saved recommendations if we have watch history and haven't initialized yet
      if (watchHistoryLoaded && watchHistoryHash && !status.isInitialized && !status.isLoading) {
        debugLog('Attempting to load saved recommendations');
        
        try {
          // Try to load recommendations from appropriate storage
          const savedRecommendations = await dataAccessService.getRecommendations(watchHistoryHash);
          
          if (savedRecommendations && savedRecommendations.length > 0) {
            debugLog(`Found ${savedRecommendations.length} saved recommendations`);
            setRecommendations(savedRecommendations);
            setStatus(prev => ({
              ...prev,
              isInitialized: true
            }));
            // Mark watch history as unchanged since we just loaded matching recommendations
            setWatchHistoryChanged(false);
          } else if (autoLoad) {
            // Generate new recommendations if none found and autoLoad is true
            debugLog('No saved recommendations found - generating new ones');
            fetchRecommendations();
          } else {
            debugLog('No saved recommendations found and autoLoad is false - not generating new ones');
          }
        } catch (error) {
          console.error('Error loading saved recommendations:', error);
        }
      }
    };
    
    if (watchHistoryLoaded && !status.isInitialized && !status.isLoading) {
      loadSavedRecommendations();
    }
  }, [
    autoLoad, 
    fetchRecommendations, 
    status.isInitialized, 
    status.isLoading, 
    watchHistoryHash, 
    watchHistoryLoaded
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
    isWatchHistoryLoaded: watchHistoryLoaded,
    isCollaborativeFilteringEnabled: status.collaborativeFilteringEnabled,
    similarUsers,
    watchHistoryChanged
  };
} 
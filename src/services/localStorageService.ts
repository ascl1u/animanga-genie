import { AnimeWatchHistoryItem, WatchHistoryFormData } from '@/types/watchHistory';
import { v4 as uuidv4 } from 'uuid';
import { AnimeData } from '@/services/recommendationService';
import { WATCH_HISTORY_CHANGED_EVENT } from '@/services/watchHistoryService';
import { createClient } from '@/utils/supabase/client';

// Constants for local storage keys
const WATCH_HISTORY_KEY = 'animanga-genie-watch-history';
const RECOMMENDATIONS_KEY = 'animanga-genie-recommendations';
const WATCH_HISTORY_HASH_KEY = 'animanga-genie-watch-history-hash';

// Helper function to generate a UUID for local items
const generateUUID = () => uuidv4();

/**
 * Helper function to notify that watch history has changed
 * This ensures both authenticated and non-authenticated users 
 * trigger the same events when watch history changes
 */
const notifyWatchHistoryChanged = () => {
  if (typeof window !== 'undefined') {
    // Add stack trace to debug where this is being called from
    console.log('[LocalStorage] About to dispatch watch history changed event');
    console.trace('[LocalStorage] TRACE: Watch history event dispatch stack trace');
    
    // Check auth state before dispatching
    const supabase = createClient();
    supabase.auth.getUser().then(({ data: { user } }) => {
      if (user) {
        console.log(`[LocalStorage] WARNING: Watch history change event being dispatched for authenticated user: ${user.id}`);
      } else {
        console.log('[LocalStorage] Watch history change event being dispatched for non-authenticated user');
      }
    });
    
    // Create and dispatch the same event used in watchHistoryService
    const event = new CustomEvent(WATCH_HISTORY_CHANGED_EVENT);
    window.dispatchEvent(event);
    console.log('[LocalStorage] Dispatched watch history changed event');
  }
};

/**
 * Get watch history from local storage
 */
export const getLocalWatchHistory = (): AnimeWatchHistoryItem[] => {
  if (typeof window === 'undefined') return [];
  
  try {
    const storedData = localStorage.getItem(WATCH_HISTORY_KEY);
    if (!storedData) return [];
    
    return JSON.parse(storedData);
  } catch (error) {
    console.error('Error reading watch history from local storage:', error);
    return [];
  }
};

/**
 * Add an anime to local watch history
 */
export const addToLocalWatchHistory = (watchHistoryData: WatchHistoryFormData): AnimeWatchHistoryItem => {
  try {
    // Get existing watch history
    const existingHistory = getLocalWatchHistory();
    
    // Check if anime already exists
    const existingIndex = existingHistory.findIndex(item => item.anilist_id === watchHistoryData.anilist_id);
    
    // Create new entry with UUID and timestamps
    const now = new Date().toISOString();
    const newEntry: AnimeWatchHistoryItem = {
      ...watchHistoryData,
      id: generateUUID(),
      user_id: 'local-user', // Add user_id to satisfy type requirement
      created_at: now,
      updated_at: now
    };
    
    // Update existing entry or add new one
    let updatedHistory: AnimeWatchHistoryItem[];
    if (existingIndex >= 0) {
      // Update existing entry
      updatedHistory = [...existingHistory];
      updatedHistory[existingIndex] = {
        ...newEntry,
        id: existingHistory[existingIndex].id // Keep original ID
      };
    } else {
      // Add new entry
      updatedHistory = [newEntry, ...existingHistory];
    }
    
    // Save to local storage
    localStorage.setItem(WATCH_HISTORY_KEY, JSON.stringify(updatedHistory));
    
    // Notify that watch history has changed (same as in watchHistoryService)
    notifyWatchHistoryChanged();
    
    return newEntry;
  } catch (error) {
    console.error('Error adding to local watch history:', error);
    throw error;
  }
};

/**
 * Update a local watch history item's rating
 */
export const updateLocalWatchHistoryRating = (
  id: string,
  rating: number
): AnimeWatchHistoryItem => {
  try {
    const existingHistory = getLocalWatchHistory();
    const itemIndex = existingHistory.findIndex(item => item.id === id);
    
    if (itemIndex === -1) {
      throw new Error('Watch history item not found');
    }
    
    // Update the rating and updated_at timestamp
    const updatedItem = {
      ...existingHistory[itemIndex],
      rating,
      updated_at: new Date().toISOString()
    };
    
    const updatedHistory = [...existingHistory];
    updatedHistory[itemIndex] = updatedItem;
    
    // Save to local storage
    localStorage.setItem(WATCH_HISTORY_KEY, JSON.stringify(updatedHistory));
    
    // Notify that watch history has changed
    notifyWatchHistoryChanged();
    
    return updatedItem;
  } catch (error) {
    console.error('Error updating local watch history rating:', error);
    throw error;
  }
};

/**
 * Delete a local watch history item
 */
export const deleteLocalWatchHistoryItem = (id: string): void => {
  try {
    const existingHistory = getLocalWatchHistory();
    const updatedHistory = existingHistory.filter(item => item.id !== id);
    
    // Save to local storage
    localStorage.setItem(WATCH_HISTORY_KEY, JSON.stringify(updatedHistory));
    
    // Notify that watch history has changed
    notifyWatchHistoryChanged();
  } catch (error) {
    console.error('Error deleting local watch history item:', error);
    throw error;
  }
};

/**
 * Save the current watch history hash with recommendations
 * This is used to validate if saved recommendations are still valid
 */
export const saveLocalWatchHistoryHash = (hash: string): void => {
  if (typeof window === 'undefined') return;
  
  try {
    localStorage.setItem(WATCH_HISTORY_HASH_KEY, hash);
    console.log('[LocalStorage] Saved watch history hash:', hash);
  } catch (error) {
    console.error('Error saving watch history hash to local storage:', error);
  }
};

/**
 * Get the saved watch history hash
 */
export const getLocalWatchHistoryHash = (): string | null => {
  if (typeof window === 'undefined') return null;
  
  try {
    return localStorage.getItem(WATCH_HISTORY_HASH_KEY);
  } catch (error) {
    console.error('Error reading watch history hash from local storage:', error);
    return null;
  }
};

/**
 * Save recommendations to local storage
 */
export const saveLocalRecommendations = (recommendations: AnimeData[], watchHistoryHash: string): void => {
  if (typeof window === 'undefined') return;
  
  try {
    // Check if this is being called for an authenticated user (which shouldn't happen)
    const supabase = createClient();
    supabase.auth.getUser().then(({ data: { user } }) => {
      if (user) {
        console.error('[LocalStorage] WARNING: saveLocalRecommendations called for authenticated user!', user.id);
        console.trace('[LocalStorage] Stack trace for saveLocalRecommendations call:');
      }
    });
    
    localStorage.setItem(RECOMMENDATIONS_KEY, JSON.stringify(recommendations));
    saveLocalWatchHistoryHash(watchHistoryHash);
    console.log('[LocalStorage] Saved recommendations with hash:', watchHistoryHash);
    console.log(`[LocalStorage] Saved ${recommendations.length} recommendations to localStorage`);
  } catch (error) {
    console.error('Error saving recommendations to local storage:', error);
  }
};

/**
 * Get recommendations from local storage
 */
export const getLocalRecommendations = (): AnimeData[] => {
  if (typeof window === 'undefined') return [];
  
  try {
    const storedData = localStorage.getItem(RECOMMENDATIONS_KEY);
    if (!storedData) return [];
    
    return JSON.parse(storedData);
  } catch (error) {
    console.error('Error reading recommendations from local storage:', error);
    return [];
  }
};

/**
 * Check if local recommendations are valid for the current watch history hash
 */
export const areLocalRecommendationsValid = (currentHash: string): boolean => {
  const savedHash = getLocalWatchHistoryHash();
  
  if (!savedHash) {
    console.log('[LocalStorage] No saved watch history hash found');
    return false;
  }
  
  const isValid = savedHash === currentHash;
  console.log(`[LocalStorage] Recommendations valid: ${isValid} (current: ${currentHash}, saved: ${savedHash})`);
  return isValid;
};

/**
 * Clear local storage
 * @returns Object with information about what was cleared
 */
export const clearLocalStorage = (): { 
  watchHistoryCleared: boolean; 
  recommendationsCleared: boolean; 
  hashCleared: boolean; 
} => {
  if (typeof window === 'undefined') return {
    watchHistoryCleared: false,
    recommendationsCleared: false,
    hashCleared: false
  };
  
  try {
    console.log('[LocalStorage] Starting clearLocalStorage operation...');
    
    // Check what exists first to report what was cleared
    const hadWatchHistory = !!localStorage.getItem(WATCH_HISTORY_KEY);
    const hadRecommendations = !!localStorage.getItem(RECOMMENDATIONS_KEY);
    const hadHash = !!localStorage.getItem(WATCH_HISTORY_HASH_KEY);
    
    console.log(`[LocalStorage] Found items before clearing: watchHistory=${hadWatchHistory}, recommendations=${hadRecommendations}, hash=${hadHash}`);
    
    if (hadWatchHistory) {
      const watchHistoryCount = JSON.parse(localStorage.getItem(WATCH_HISTORY_KEY) || '[]').length;
      console.log(`[LocalStorage] Watch history contains ${watchHistoryCount} items before clearing`);
    }
    
    if (hadRecommendations) {
      const recommendationsCount = JSON.parse(localStorage.getItem(RECOMMENDATIONS_KEY) || '[]').length;
      console.log(`[LocalStorage] Recommendations contains ${recommendationsCount} items before clearing`);
      // Add details about the recommendations
      const recommendations = JSON.parse(localStorage.getItem(RECOMMENDATIONS_KEY) || '[]');
      console.log(`[LocalStorage] Recommendation keys: ${JSON.stringify(Object.keys(recommendations[0] || {}))}`);
    }
    
    if (hadHash) {
      console.log(`[LocalStorage] Current watch history hash before clearing: ${localStorage.getItem(WATCH_HISTORY_HASH_KEY)}`);
    }
    
    // Clear storage
    console.log(`[LocalStorage] Calling removeItem for ${WATCH_HISTORY_KEY}`);
    localStorage.removeItem(WATCH_HISTORY_KEY);
    
    console.log(`[LocalStorage] Calling removeItem for ${RECOMMENDATIONS_KEY}`);
    localStorage.removeItem(RECOMMENDATIONS_KEY);
    
    console.log(`[LocalStorage] Calling removeItem for ${WATCH_HISTORY_HASH_KEY}`);
    localStorage.removeItem(WATCH_HISTORY_HASH_KEY);
    
    // Verify items were removed
    const verifyWatchHistory = localStorage.getItem(WATCH_HISTORY_KEY);
    const verifyRecommendations = localStorage.getItem(RECOMMENDATIONS_KEY);
    const verifyHash = localStorage.getItem(WATCH_HISTORY_HASH_KEY);
    
    console.log(`[LocalStorage] Verification after clearing: watchHistory=${!!verifyWatchHistory}, recommendations=${!!verifyRecommendations}, hash=${!!verifyHash}`);
    console.log(`[LocalStorage] Raw verification values: watchHistory=${verifyWatchHistory}, recommendations=${verifyRecommendations}, hash=${verifyHash}`);
    
    // Check for any localStorage keys that might contain recommendations
    const totalKeys = localStorage.length;
    console.log(`[LocalStorage] Total localStorage keys after clearing: ${totalKeys}`);
    if (totalKeys > 0) {
      console.log(`[LocalStorage] All remaining localStorage keys:`);
      for (let i = 0; i < totalKeys; i++) {
        const key = localStorage.key(i);
        console.log(`[LocalStorage] Key ${i}: ${key}`);
      }
    }
    
    const result = {
      watchHistoryCleared: hadWatchHistory,
      recommendationsCleared: hadRecommendations,
      hashCleared: hadHash
    };
    
    console.log(`[LocalStorage] Cleared local storage: `, result);
    return result;
  } catch (error) {
    console.error('Error clearing local storage:', error);
    return {
      watchHistoryCleared: false,
      recommendationsCleared: false,
      hashCleared: false
    };
  }
};

/**
 * Remove only recommendations from local storage
 * @returns boolean indicating if recommendations were removed
 */
export const removeLocalRecommendations = (): boolean => {
  if (typeof window === 'undefined') return false;
  
  try {
    console.log('[LocalStorage] Removing local recommendations...');
    
    // Check if recommendations exist
    const hadRecommendations = !!localStorage.getItem(RECOMMENDATIONS_KEY);
    
    if (hadRecommendations) {
      const recommendationsCount = JSON.parse(localStorage.getItem(RECOMMENDATIONS_KEY) || '[]').length;
      console.log(`[LocalStorage] Removing ${recommendationsCount} recommendations from local storage`);
    }
    
    // Remove recommendations and hash
    localStorage.removeItem(RECOMMENDATIONS_KEY);
    localStorage.removeItem(WATCH_HISTORY_HASH_KEY);
    
    // Verify items were removed
    const verifyRecommendations = localStorage.getItem(RECOMMENDATIONS_KEY);
    const verifyHash = localStorage.getItem(WATCH_HISTORY_HASH_KEY);
    
    console.log(`[LocalStorage] Verification after removing: recommendations=${!!verifyRecommendations}, hash=${!!verifyHash}`);
    
    return hadRecommendations && !verifyRecommendations;
  } catch (error) {
    console.error('Error removing local recommendations:', error);
    return false;
  }
}; 
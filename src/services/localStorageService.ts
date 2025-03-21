import { AnimeWatchHistoryItem, WatchHistoryFormData } from '@/types/watchHistory';
import { v4 as uuidv4 } from 'uuid';
import { AnimeData } from '@/services/recommendationService';

// Constants for local storage keys
const WATCH_HISTORY_KEY = 'animanga-genie-watch-history';
const RECOMMENDATIONS_KEY = 'animanga-genie-recommendations';

// Helper function to generate a UUID for local items
const generateUUID = () => uuidv4();

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
  } catch (error) {
    console.error('Error deleting local watch history item:', error);
    throw error;
  }
};

/**
 * Save recommendations to local storage
 */
export const saveLocalRecommendations = (recommendations: AnimeData[]): void => {
  if (typeof window === 'undefined') return;
  
  try {
    localStorage.setItem(RECOMMENDATIONS_KEY, JSON.stringify(recommendations));
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
 * Clear local storage
 */
export const clearLocalStorage = (): void => {
  if (typeof window === 'undefined') return;
  
  try {
    localStorage.removeItem(WATCH_HISTORY_KEY);
    localStorage.removeItem(RECOMMENDATIONS_KEY);
  } catch (error) {
    console.error('Error clearing local storage:', error);
  }
}; 
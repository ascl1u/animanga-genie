import { AnimeWatchHistoryItem, WatchHistoryFormData } from '@/types/watchHistory';
import { AnimeData } from '@/services/recommendationService';
import { createClient } from '@/utils/supabase/client';
import { 
  getLocalWatchHistory, 
  addToLocalWatchHistory, 
  updateLocalWatchHistoryRating, 
  deleteLocalWatchHistoryItem
} from './localStorageService';
import {
  getUserWatchHistory,
  addToWatchHistory,
  updateWatchHistoryRating,
  deleteWatchHistoryItem as deleteDbWatchHistoryItem
} from './watchHistoryService';
import {
  saveRecommendations,
  loadRecommendations
} from './recommendationPersistenceService';

/**
 * Unified Data Access Service
 * 
 * This service provides a consistent API for data access operations,
 * automatically selecting the appropriate storage mechanism based on auth state.
 * It separates the storage concerns from the business logic.
 */
class DataAccessService {
  private supabase = createClient();

  /**
   * Check if user is authenticated
   */
  private async isAuthenticated(): Promise<boolean> {
    const { data } = await this.supabase.auth.getUser();
    return !!data.user;
  }

  /**
   * Get watch history based on auth state
   */
  async getWatchHistory(): Promise<AnimeWatchHistoryItem[]> {
    const authenticated = await this.isAuthenticated();
    
    if (authenticated) {
      try {
        // Get from database for authenticated users
        return await getUserWatchHistory();
      } catch (error) {
        console.error('Error getting watch history from database:', error);
        return [];
      }
    } else {
      // Get from localStorage for unauthenticated users
      return getLocalWatchHistory();
    }
  }

  /**
   * Save watch history item based on auth state
   */
  async saveWatchHistory(item: WatchHistoryFormData): Promise<AnimeWatchHistoryItem> {
    const authenticated = await this.isAuthenticated();
    
    if (authenticated) {
      // Save to database for authenticated users
      return await addToWatchHistory(item);
    } else {
      // Save to localStorage for unauthenticated users
      return addToLocalWatchHistory(item);
    }
  }

  /**
   * Update watch history rating based on auth state
   */
  async updateWatchHistoryRating(id: string, rating: number): Promise<AnimeWatchHistoryItem> {
    const authenticated = await this.isAuthenticated();
    
    if (authenticated) {
      // Update in database for authenticated users
      return await updateWatchHistoryRating({ id, rating });
    } else {
      // Update in localStorage for unauthenticated users
      return updateLocalWatchHistoryRating(id, rating);
    }
  }

  /**
   * Delete watch history item based on auth state
   */
  async deleteWatchHistory(id: string): Promise<void> {
    const authenticated = await this.isAuthenticated();
    
    if (authenticated) {
      // Delete from database for authenticated users
      await deleteDbWatchHistoryItem(id);
    } else {
      // Delete from localStorage for unauthenticated users
      deleteLocalWatchHistoryItem(id);
    }
  }

  /**
   * Get recommendations based on auth state
   */
  async getRecommendations(watchHistoryHash?: string): Promise<AnimeData[]> {
    const authenticated = await this.isAuthenticated();
    
    if (authenticated) {
      try {
        // Get from database for authenticated users
        const recommendations = await loadRecommendations(watchHistoryHash || '');
        return recommendations || [];
      } catch (error) {
        console.error('Error getting recommendations from database:', error);
        return [];
      }
    } else {
      // No longer retrieving from localStorage for unauthenticated users
      return [];
    }
  }

  /**
   * Save recommendations based on auth state
   * @param recommendations Recommendations to save
   * @param watchHistoryHash Optional hash of watch history for validation
   */
  async saveRecommendations(recommendations: AnimeData[], watchHistoryHash: string): Promise<void> {
    const authenticated = await this.isAuthenticated();
    
    if (authenticated) {
      // Save to database for authenticated users
      await saveRecommendations(recommendations, watchHistoryHash);
    }
    // No longer handling localStorage for unauthenticated users
  }

  /**
   * Check if the current user has any saved data
   */
  async hasSavedData(): Promise<boolean> {
    const watchHistory = await this.getWatchHistory();
    return watchHistory.length > 0;
  }

  /**
   * Create hash for watch history to track changes
   * @param watchHistory The watch history to hash
   */
  createWatchHistoryHash(watchHistory: AnimeWatchHistoryItem[]): string {
    if (!watchHistory.length) return '';
    
    // Sort by ID to ensure consistent hash
    const sorted = [...watchHistory].sort((a, b) => a.anilist_id - b.anilist_id);
    
    // Create a string with anime IDs and ratings
    const hashString = sorted.map(item => `${item.anilist_id}:${item.rating}`).join('|');
    
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < hashString.length; i++) {
      const char = hashString.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    
    return hash.toString(16);
  }
}

// Create a singleton instance
export const dataAccessService = new DataAccessService(); 
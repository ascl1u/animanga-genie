import { createClient } from '@/utils/supabase/client';
import { AnimeData } from './recommendationService';

const supabase = createClient();

/**
 * Interface for stored recommendations
 */
interface StoredRecommendations {
  id: string;
  user_id: string;
  recommendations: AnimeData[];
  watch_history_hash: string;
  created_at: string;
  updated_at: string;
}

/**
 * Save recommendations to Supabase
 * @param recommendations Array of recommendation objects
 * @param watchHistoryHash Hash representing watch history state
 * @returns The stored recommendation data or null if there was an error
 */
export const saveRecommendations = async (
  recommendations: AnimeData[],
  watchHistoryHash: string
): Promise<StoredRecommendations | null> => {
  try {
    const { data: { user } } = await supabase.auth.getUser();
    
    if (!user) {
      console.error('Cannot save recommendations: User not authenticated');
      return null;
    }
    
    console.log(`[DB] Saving recommendations for user ${user.id}`);
    console.log(`[DB] Found ${recommendations.length} recommendations to save`);
    
    // Keep only necessary data to reduce storage size
    const simplifiedRecommendations = recommendations.map(rec => ({
      id: rec.id,
      title: rec.title,
      score: rec.score,
      averageScore: rec.averageScore,
      cover_image: rec.cover_image,
      genres: rec.genres,
      description: rec.description
    }));
    
    // Check localStorage to ensure we're not accidentally saving here too
    const localStorageCheck = localStorage.getItem('animanga-genie-recommendations');
    console.log(`[DB] Before DB save, localStorage has recommendations: ${!!localStorageCheck}`);
    
    const { data, error } = await supabase
      .from('anime_recommendations')
      .upsert({
        user_id: user.id,
        recommendations: simplifiedRecommendations,
        watch_history_hash: watchHistoryHash
      })
      .select()
      .single();
    
    if (error) {
      console.error('[DB] Error saving recommendations:', error);
      return null;
    }
    
    console.log('[DB] Successfully saved recommendations to database');
    
    // Verify localStorage still doesn't have recommendations after DB save
    const afterSaveCheck = localStorage.getItem('animanga-genie-recommendations');
    console.log(`[DB] After DB save, localStorage has recommendations: ${!!afterSaveCheck}`);
    
    if (!!afterSaveCheck) {
      console.log('[DB] WARNING: Found recommendations in localStorage after DB save!');
    }
    
    return data;
  } catch (error) {
    console.error('[DB] Error in saveRecommendations:', error);
    return null;
  }
};

/**
 * Load recommendations from Supabase
 * @param watchHistoryHash Current watch history hash to check if recommendations are still valid
 * @returns Array of recommendation objects or null if not found
 */
export const loadRecommendations = async (
  watchHistoryHash: string
): Promise<AnimeData[] | null> => {
  try {
    const { data: { user } } = await supabase.auth.getUser();
    
    if (!user) {
      console.error('[DB] Cannot load recommendations: User not authenticated');
      return null;
    }
    
    console.log(`[DB] Loading recommendations for user ${user.id}`);
    console.log(`[DB] Looking for watch history hash: ${watchHistoryHash}`);
    
    const { data, error } = await supabase
      .from('anime_recommendations')
      .select('*')
      .eq('user_id', user.id)
      .single();
    
    if (error) {
      if (error.code === 'PGRST116') {
        // No data found - this is not an error, just no recommendations yet
        console.log('[DB] No saved recommendations found in database');
      } else {
        console.error('[DB] Error loading recommendations:', error);
      }
      return null;
    }
    
    const storedData = data as StoredRecommendations;
    
    // Check if the recommendations are still valid based on watch history hash
    if (storedData.watch_history_hash !== watchHistoryHash) {
      console.log(`[DB] Watch history has changed - recommendations need to be regenerated`);
      console.log(`[DB] Stored hash: ${storedData.watch_history_hash}, Current hash: ${watchHistoryHash}`);
      return null;
    }
    
    console.log(`[DB] Loaded ${storedData.recommendations.length} recommendations from database`);
    
    // Check localStorage to ensure we're not accidentally saving here too
    const localStorageCheck = localStorage.getItem('animanga-genie-recommendations');
    console.log(`[DB] After DB load, localStorage has recommendations: ${!!localStorageCheck}`);
    
    return storedData.recommendations;
  } catch (error) {
    console.error('[DB] Error in loadRecommendations:', error);
    return null;
  }
};

/**
 * Check if the user has stored recommendations
 * @returns Boolean indicating if recommendations exist
 */
export const hasStoredRecommendations = async (): Promise<boolean> => {
  try {
    const { data: { user } } = await supabase.auth.getUser();
    
    if (!user) {
      return false;
    }
    
    const { count, error } = await supabase
      .from('anime_recommendations')
      .select('*', { count: 'exact', head: true })
      .eq('user_id', user.id);
    
    if (error) {
      console.error('Error checking for recommendations:', error);
      return false;
    }
    
    return (count || 0) > 0;
  } catch (error) {
    console.error('Error in hasStoredRecommendations:', error);
    return false;
  }
}; 
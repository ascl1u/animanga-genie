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
    
    console.log(`Saving recommendations for user ${user.id}`);
    console.log(`Found ${recommendations.length} recommendations to save`);
    
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
    
    // First, check if user already has recommendations stored
    const { data: existingData } = await supabase
      .from('anime_recommendations')
      .select('id')
      .eq('user_id', user.id)
      .single();
    
    let result;
    
    if (existingData) {
      // Update the existing record
      console.log('Updating existing recommendations record');
      result = await supabase
        .from('anime_recommendations')
        .update({
          recommendations: simplifiedRecommendations,
          watch_history_hash: watchHistoryHash,
          updated_at: new Date().toISOString()
        })
        .eq('id', existingData.id)
        .select()
        .single();
    } else {
      // Create a new record
      console.log('Creating new recommendations record');
      result = await supabase
        .from('anime_recommendations')
        .insert({
          user_id: user.id,
          recommendations: simplifiedRecommendations,
          watch_history_hash: watchHistoryHash
        })
        .select()
        .single();
    }
    
    if (result.error) {
      console.error('Error saving recommendations:', result.error);
      return null;
    }
    
    console.log('Successfully saved recommendations');
    return result.data;
  } catch (error) {
    console.error('Error in saveRecommendations:', error);
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
      console.error('Cannot load recommendations: User not authenticated');
      return null;
    }
    
    console.log(`Loading recommendations for user ${user.id}`);
    
    const { data, error } = await supabase
      .from('anime_recommendations')
      .select('*')
      .eq('user_id', user.id)
      .single();
    
    if (error) {
      if (error.code === 'PGRST116') {
        // No data found - this is not an error, just no recommendations yet
        console.log('No saved recommendations found');
      } else {
        console.error('Error loading recommendations:', error);
      }
      return null;
    }
    
    const storedData = data as StoredRecommendations;
    
    // Check if the recommendations are still valid based on watch history hash
    if (storedData.watch_history_hash !== watchHistoryHash) {
      console.log(`Watch history has changed - recommendations need to be regenerated. 
        Current hash: ${watchHistoryHash}, 
        Stored hash: ${storedData.watch_history_hash}`);
      return null;
    }
    
    console.log(`Loaded ${storedData.recommendations.length} recommendations`);
    return storedData.recommendations;
  } catch (error) {
    console.error('Error in loadRecommendations:', error);
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

/**
 * Load the latest recommendations for a user regardless of watch history hash
 * This is useful when a user logs in to immediately show their personalized recommendations
 * @returns Array of recommendation objects or null if not found
 */
export const getLatestUserRecommendations = async (): Promise<{
  recommendations: AnimeData[] | null;
  watchHistoryHash: string | null;
}> => {
  try {
    const { data: { user } } = await supabase.auth.getUser();
    
    if (!user) {
      console.error('Cannot load recommendations: User not authenticated');
      return { recommendations: null, watchHistoryHash: null };
    }
    
    console.log(`Loading latest recommendations for user ${user.id}`);
    
    const { data, error } = await supabase
      .from('anime_recommendations')
      .select('*')
      .eq('user_id', user.id)
      .order('updated_at', { ascending: false })
      .limit(1)
      .single();
    
    if (error) {
      if (error.code === 'PGRST116') {
        // No data found - this is not an error, just no recommendations yet
        console.log('No saved recommendations found');
      } else {
        console.error('Error loading recommendations:', error);
      }
      return { recommendations: null, watchHistoryHash: null };
    }
    
    const storedData = data as StoredRecommendations;
    
    console.log(`Loaded ${storedData.recommendations.length} recommendations (hash: ${storedData.watch_history_hash})`);
    return { 
      recommendations: storedData.recommendations,
      watchHistoryHash: storedData.watch_history_hash
    };
  } catch (error) {
    console.error('Error in getLatestUserRecommendations:', error);
    return { recommendations: null, watchHistoryHash: null };
  }
}; 
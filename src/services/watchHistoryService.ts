import { createClient } from '@/utils/supabase/client';
import type { AnimeWatchHistoryItem, UpdateWatchHistoryParams, WatchHistoryFormData } from '@/types/watchHistory';

const supabase = createClient();

// Custom event for watch history changes
export const WATCH_HISTORY_CHANGED_EVENT = 'watch-history-changed';

// Function to trigger a watch history change event
export const notifyWatchHistoryChanged = () => {
  // Create a custom event that can be listened to by components
  const event = new CustomEvent(WATCH_HISTORY_CHANGED_EVENT);
  window.dispatchEvent(event);
};

/**
 * Get all watch history items for the current user
 */
export const getUserWatchHistory = async (): Promise<AnimeWatchHistoryItem[]> => {
  try {
    // First, get the current user
    const { data: { user } } = await supabase.auth.getUser();
    
    if (!user) {
      throw new Error('User not authenticated');
    }
    
    console.log('Fetching watch history for user:', user.id);
    
    // Check if user_preferences table has any data
    const { data: prefData, error: prefError } = await supabase
      .from('user_preferences')
      .select('watch_history')
      .eq('user_id', user.id)
      .single();
      
    if (prefData && !prefError) {
      console.log('Found data in user_preferences.watch_history:', 
        Array.isArray(prefData.watch_history) ? prefData.watch_history.length : 'None');
    }
    
    // Fetch from anime_watch_history table
    const { data, error } = await supabase
      .from('anime_watch_history')
      .select('*')
      .eq('user_id', user.id)
      .order('updated_at', { ascending: false });

    if (error) {
      console.error('Error fetching watch history:', error);
      throw error;
    }

    console.log('Fetched watch history from anime_watch_history table. Count:', data?.length || 0);
    return data || [];
  } catch (error) {
    console.error('Error in getUserWatchHistory:', error);
    throw error;
  }
};

/**
 * Add an anime to the user's watch history
 */
export const addToWatchHistory = async (watchHistoryData: WatchHistoryFormData): Promise<AnimeWatchHistoryItem> => {
  try {
    const { data: { user } } = await supabase.auth.getUser();
    
    if (!user) {
      console.error('Cannot add to watch history: User not authenticated');
      throw new Error('User not authenticated');
    }
    
    console.log('Current authenticated user:', user.id);

    // Add user_id to the data
    const dataWithUserId = {
      ...watchHistoryData,
      user_id: user.id
    };
    
    console.log('Inserting data into anime_watch_history:', dataWithUserId);

    // Upsert the data to the anime_watch_history table
    const { data, error } = await supabase
      .from('anime_watch_history')
      .upsert(dataWithUserId)
      .select()
      .single();

    if (error) {
      console.error('Error adding to watch history:', error);
      throw error;
    }

    console.log('Successfully added to watch history:', data);
    
    // Verify the data was actually inserted
    const { data: verifyData, error: verifyError } = await supabase
      .from('anime_watch_history')
      .select('*')
      .eq('user_id', user.id)
      .eq('anilist_id', watchHistoryData.anilist_id)
      .single();
      
    if (verifyError) {
      console.error('Error verifying watch history entry:', verifyError);
    } else {
      console.log('Verified watch history entry exists:', verifyData);
    }
    
    // Notify that watch history has changed
    notifyWatchHistoryChanged();
    
    return data;
  } catch (error) {
    console.error('Error in addToWatchHistory:', error);
    throw error;
  }
};

/**
 * Update an existing watch history item's rating
 */
export const updateWatchHistoryRating = async (
  params: UpdateWatchHistoryParams
): Promise<AnimeWatchHistoryItem> => {
  try {
    const { id, rating } = params;
    
    const { data, error } = await supabase
      .from('anime_watch_history')
      .update({ rating })
      .eq('id', id)
      .select()
      .single();

    if (error) {
      console.error('Error updating watch history rating:', error);
      throw error;
    }

    // Notify that watch history has changed
    notifyWatchHistoryChanged();
    
    return data;
  } catch (error) {
    console.error('Error in updateWatchHistoryRating:', error);
    throw error;
  }
};

/**
 * Delete a watch history item
 */
export const deleteWatchHistoryItem = async (id: string): Promise<void> => {
  try {
    const { error } = await supabase
      .from('anime_watch_history')
      .delete()
      .eq('id', id);

    if (error) {
      console.error('Error deleting watch history item:', error);
      throw error;
    }
    
    // Notify that watch history has changed
    notifyWatchHistoryChanged();
  } catch (error) {
    console.error('Error in deleteWatchHistoryItem:', error);
    throw error;
  }
}; 
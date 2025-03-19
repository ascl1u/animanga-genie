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

/**
 * Import watch history from AniList
 * This will import a user's AniList anime entries and merge with existing watch history
 */
export const importAnilistWatchHistory = async (username: string): Promise<{
  added: number;
  updated: number;
  unchanged: number;
  total: number;
}> => {
  try {
    // Get current authenticated user
    const { data: { user } } = await supabase.auth.getUser();
    
    if (!user) {
      throw new Error('User not authenticated');
    }
    
    // Import only works if we have a username
    if (!username || username.trim() === '') {
      throw new Error('AniList username is required');
    }
    
    // Get the user's anime list from AniList
    const { getUserAnimeList } = await import('@/utils/anilistClient');
    const anilistLists = await getUserAnimeList(username);
    
    if (!anilistLists || anilistLists.length === 0) {
      throw new Error('No anime lists found for this AniList user');
    }
    
    // Get existing watch history
    const existingHistory = await getUserWatchHistory();
    const existingMap = new Map<number, AnimeWatchHistoryItem>();
    existingHistory.forEach(item => existingMap.set(item.anilist_id, item));
    
    // Track statistics
    let added = 0;
    let updated = 0;
    let unchanged = 0;
    let total = 0;
    
    // Prepare items to be added/updated
    const itemsToProcess = [];
    
    // Process each list (Completed, Watching, etc.)
    for (const list of anilistLists) {
      // Skip empty lists
      if (!list.entries || list.entries.length === 0) continue;
      
      // Process each entry in the list
      for (const entry of list.entries) {
        total++;
        const media = entry.media;
        
        // Default NULL scores to 5, or use existing score if valid
        // AniList returns scores from 1-10, matching our system
        // If score is null or 0, we'll use 5 as a default neutral rating
        const rating = entry.score ? entry.score : 5;
        
        // Create watch history item
        const watchItem: WatchHistoryFormData = {
          anilist_id: media.id,
          title: media.title.english || media.title.romaji,
          // Only include cover image if it's a known safe domain
          cover_image: media.coverImage?.medium?.includes('s4.anilist.co') 
            ? media.coverImage.medium 
            : undefined,
          rating // Use the calculated rating
        };
        
        // Check if this anime already exists in the user's watch history
        const existing = existingMap.get(media.id);
        
        // Always process all entries - we're always overwriting
        if (existing) {
          // If this is an update to an existing entry
          if (existing.rating !== rating) {
            watchItem.id = existing.id; // Keep the same ID for updating
            itemsToProcess.push(watchItem);
            updated++;
          } else {
            unchanged++;
          }
        } else {
          // This is a new entry
          itemsToProcess.push(watchItem);
          added++;
        }
      }
    }
    
    // Now perform the database operations
    if (itemsToProcess.length > 0) {
      // Generate UUID function - use the same format as Supabase
      const generateUUID = () => {
        // Simple UUID v4 implementation
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
          const r = Math.random() * 16 | 0;
          const v = c === 'x' ? r : (r & 0x3 | 0x8);
          return v.toString(16);
        });
      };
      
      // Add user_id and required fields to all items
      const itemsWithUserId = itemsToProcess.map(item => ({
        ...item,
        user_id: user.id,
        // Ensure ID is always present for new items
        id: item.id || generateUUID(),
        // Add timestamps to ensure required fields are present
        created_at: item.id ? undefined : new Date().toISOString(),
        updated_at: new Date().toISOString()
      }));
      
      // Process items in smaller batches to avoid payload size issues
      const BATCH_SIZE = 5; // Reduced batch size further to minimize potential issues
      const batches = [];
      
      for (let i = 0; i < itemsWithUserId.length; i += BATCH_SIZE) {
        batches.push(itemsWithUserId.slice(i, i + BATCH_SIZE));
      }
      
      console.log(`Processing ${itemsWithUserId.length} items in ${batches.length} batches`);
      
      // Process each batch
      for (const batch of batches) {
        try {
          // Log each batch for debugging
          console.log('Processing batch:', JSON.stringify(batch));
          
          const { error } = await supabase
            .from('anime_watch_history')
            .upsert(batch, {
              onConflict: 'user_id,anilist_id', // Define the conflict resolution strategy
              ignoreDuplicates: false           // Update existing records
            });
          
          if (error) {
            console.error('Error during AniList import batch operation:', error);
            throw error;
          }
        } catch (error) {
          console.error('Batch processing error:', error);
          console.error('Problematic batch data:', JSON.stringify(batch, null, 2));
          throw error;
        }
      }
    }
    
    // Notify that watch history has changed
    notifyWatchHistoryChanged();
    
    // Return statistics about the operation
    return {
      added,
      updated,
      unchanged,
      total
    };
  } catch (error) {
    console.error('Error in importAnilistWatchHistory:', error);
    throw error;
  }
}; 
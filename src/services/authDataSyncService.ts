import { getLocalWatchHistory, clearLocalStorage } from '@/services/localStorageService';
import { addToWatchHistory } from '@/services/watchHistoryService';

/**
 * Sync local data with database after user authentication
 * This checks if there's local watch history from a non-authenticated session
 * and adds it to the user's database
 * @returns Promise with the sync results
 */
export const syncLocalDataWithDatabase = async (): Promise<{
  migrated: number;
  duplicates: number;
  errors: number;
  success: boolean;
}> => {
  try {
    console.log('Starting local data sync with database');
    
    // Get local watch history
    const localWatchHistory = getLocalWatchHistory();
    
    // If there's no local data, nothing to sync
    if (!localWatchHistory || localWatchHistory.length === 0) {
      console.log('No local watch history found, nothing to sync');
      return {
        migrated: 0,
        duplicates: 0,
        errors: 0,
        success: true,
      };
    }
    
    console.log(`Found ${localWatchHistory.length} local watch history items to sync`);
    
    // Stats for tracking the sync process
    let migrated = 0;
    let duplicates = 0;
    let errors = 0;
    
    // Process each local item
    for (const item of localWatchHistory) {
      try {
        // Create the database entry from the local item
        await addToWatchHistory({
          anilist_id: item.anilist_id,
          title: item.title,
          cover_image: item.cover_image,
          rating: item.rating,
        });
        
        migrated++;
        console.log(`Migrated: ${item.title} (ID: ${item.anilist_id})`);
      } catch (error) {
        // Check if this is a duplicate entry error - this is expected
        // and not a real error since some items might already exist
        if (error instanceof Error && error.message.includes('duplicate')) {
          duplicates++;
          console.log(`Duplicate (already in database): ${item.title}`);
        } else {
          errors++;
          console.error(`Error migrating item ${item.title}:`, error);
        }
      }
    }
    
    // If we successfully migrated at least some items, clear local storage
    if (migrated > 0 || duplicates > 0) {
      clearLocalStorage();
      console.log('Cleared local storage after successful migration');
    }
    
    console.log(`Data sync complete. Migrated: ${migrated}, Duplicates: ${duplicates}, Errors: ${errors}`);
    
    return {
      migrated,
      duplicates,
      errors,
      success: true,
    };
  } catch (error) {
    console.error('Error syncing local data with database:', error);
    return {
      migrated: 0,
      duplicates: 0,
      errors: 1,
      success: false,
    };
  }
}; 
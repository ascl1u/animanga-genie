'use client';

import { useState, useEffect, useCallback, useRef, forwardRef, useImperativeHandle } from 'react';
import Image from 'next/image';
import { useAuth } from '@/components/SimpleAuthProvider';
import { toast } from 'react-hot-toast';
import { createClient } from '@/utils/supabase/client';
import { AnimeWatchHistoryItem } from '@/types/watchHistory';
import { getUserWatchHistory, updateWatchHistoryRating, deleteWatchHistoryItem } from '@/services/watchHistoryService';

export interface WatchHistoryListRef {
  addAnime?: (anime: AnimeWatchHistoryItem) => void;
}

const WatchHistoryList = forwardRef<WatchHistoryListRef>((props, ref) => {
  const { user } = useAuth();
  const [watchHistory, setWatchHistory] = useState<AnimeWatchHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editItemId, setEditItemId] = useState<string | null>(null);
  const [editRating, setEditRating] = useState<number>(0);
  const [isUpdating, setIsUpdating] = useState(false);
  const [isDeleting, setIsDeleting] = useState<Record<string, boolean>>({});
  const supabase = createClient();
  const channelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);

  // Expose methods to parent components through the ref
  useImperativeHandle(ref, () => ({
    // Method to manually add an anime to the watch history list
    addAnime: (anime: AnimeWatchHistoryItem) => {
      console.log('Manually adding anime to watch history list:', anime);
      setWatchHistory(prev => [anime, ...prev]);
    }
  }));

  const fetchWatchHistory = useCallback(async () => {
    if (!user) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await getUserWatchHistory();
      setWatchHistory(data);
    } catch (err) {
      console.error('Error fetching watch history:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      toast.error('Failed to load watch history');
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  // Improved real-time subscription setup
  useEffect(() => {
    if (!user) return;
    
    // Only set up subscription if we have a user
    const setupRealtimeSubscription = () => {
      // Clean up any existing subscription
      if (channelRef.current) {
        supabase.removeChannel(channelRef.current);
      }
      
      // Set up a new subscription
      channelRef.current = supabase
        .channel('anime_watch_history_changes')
        .on(
          'postgres_changes',
          {
            event: '*',
            schema: 'public',
            table: 'anime_watch_history',
            filter: `user_id=eq.${user.id}`,
          },
          (payload) => {
            console.log('Real-time update received:', payload);
            
            // Handle different types of changes
            if (payload.eventType === 'INSERT') {
              // Add new item to the list
              const newItem = payload.new as AnimeWatchHistoryItem;
              setWatchHistory(prev => {
                // Check if the item is already in the list (to avoid duplicates)
                if (prev.some(item => item.id === newItem.id)) {
                  return prev;
                }
                return [newItem, ...prev];
              });
              toast.success('New anime added to your watch history');
            } 
            else if (payload.eventType === 'DELETE') {
              // Remove deleted item from the list
              const deletedItem = payload.old as AnimeWatchHistoryItem;
              setWatchHistory(prev => 
                prev.filter(item => item.id !== deletedItem.id)
              );
            }
            else if (payload.eventType === 'UPDATE') {
              // Update the modified item in the list
              const updatedItem = payload.new as AnimeWatchHistoryItem;
              setWatchHistory(prev => 
                prev.map(item => 
                  item.id === updatedItem.id ? updatedItem : item
                )
              );
            }
          }
        )
        .subscribe((status) => {
          console.log('Subscription status:', status);
          if (status !== 'SUBSCRIBED') {
            // If subscription fails, fall back to polling
            fetchWatchHistory();
          }
        });
    };

    // Initial data fetch
    fetchWatchHistory();
    
    // Set up real-time subscription
    setupRealtimeSubscription();

    // Clean up subscription when component unmounts or user changes
    return () => {
      if (channelRef.current) {
        supabase.removeChannel(channelRef.current);
        channelRef.current = null;
      }
    };
  }, [user, supabase, fetchWatchHistory]);

  const handleEdit = (item: AnimeWatchHistoryItem) => {
    setEditItemId(item.id);
    setEditRating(item.rating);
  };

  const cancelEdit = () => {
    setEditItemId(null);
    setEditRating(0);
  };

  // Optimistic UI update for deletion
  const handleDelete = async (id: string) => {
    // Optimistically remove the item from the UI first
    const itemToDelete = watchHistory.find(item => item.id === id);
    if (!itemToDelete) return;
    
    // Save the current state for rollback if needed
    const previousWatchHistory = [...watchHistory];
    
    // Update UI immediately
    setWatchHistory(prev => prev.filter(item => item.id !== id));
    
    // Mark as deleting
    setIsDeleting(prev => ({ ...prev, [id]: true }));
    
    try {
      // Actually delete from database
      await deleteWatchHistoryItem(id);
      toast.success('Removed from watch history');
    } catch (error) {
      console.error('Error deleting watch history item:', error);
      toast.error('Failed to remove from watch history');
      
      // Rollback on error
      setWatchHistory(previousWatchHistory);
    } finally {
      setIsDeleting(prev => ({ ...prev, [id]: false }));
    }
  };

  // Optimistic UI update for rating changes
  const saveRating = async (id: string) => {
    if (editRating < 1 || editRating > 10) {
      toast.error('Rating must be between 1 and 10');
      return;
    }

    // Save the current state for rollback if needed
    const previousWatchHistory = [...watchHistory];
    
    // Update UI immediately
    setWatchHistory(prev => 
      prev.map(item => 
        item.id === id ? { ...item, rating: editRating } : item
      )
    );
    
    setIsUpdating(true);
    
    try {
      await updateWatchHistoryRating({ id, rating: editRating });
      toast.success('Rating updated successfully');
      setEditItemId(null);
    } catch (error) {
      console.error('Error updating rating:', error);
      toast.error('Failed to update rating');
      
      // Rollback on error
      setWatchHistory(previousWatchHistory);
    } finally {
      setIsUpdating(false);
    }
  };

  if (isLoading) {
    return (
      <div className="py-8 flex justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 p-4 rounded-md">
        <p className="text-red-600">Error loading watch history: {error}</p>
      </div>
    );
  }

  if (watchHistory.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <h3 className="font-medium text-gray-700">No anime in your watch history yet</h3>
        <p className="mt-2 text-sm text-gray-600">
          Add anime to your watch history using the form above.
        </p>
      </div>
    );
  }

  return (
    <div className="mt-8">
      <h2 className="text-xl font-semibold mb-4 text-gray-900">Your Watch History</h2>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {watchHistory.map((item) => (
          <div 
            key={item.id} 
            className="bg-white p-4 rounded-lg shadow-sm border border-gray-200"
          >
            <div className="flex">
              {item.cover_image ? (
                <div className="flex-shrink-0 h-20 w-14 mr-4 relative">
                  <Image
                    src={item.cover_image}
                    alt={item.title}
                    className="rounded-sm object-cover"
                    fill
                    sizes="56px"
                    onError={(e) => {
                      // Replace broken image with placeholder
                      const target = e.target as HTMLImageElement;
                      target.style.display = 'none';
                    }}
                  />
                </div>
              ) : (
                <div className="flex-shrink-0 h-20 w-14 mr-4 bg-gray-200 rounded-sm flex items-center justify-center">
                  <span className="text-xs text-gray-700">No image</span>
                </div>
              )}
              
              <div className="flex-1">
                <h3 className="font-medium text-gray-900 line-clamp-2">{item.title}</h3>
                
                <div className="mt-2 flex items-center">
                  {editItemId === item.id ? (
                    <div className="flex items-center space-x-1">
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((star) => (
                        <button
                          key={star}
                          type="button"
                          onClick={() => setEditRating(star)}
                          className={`text-sm ${star <= editRating ? 'text-yellow-500' : 'text-gray-400'} hover:text-yellow-400 transition-colors`}
                          aria-label={`Rate ${star} out of 10`}
                        >
                          {star}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="flex items-center">
                      <span className="text-yellow-500 mr-1">★</span>
                      <span className="text-sm font-semibold text-gray-800">{item.rating}/10</span>
                    </div>
                  )}
                  
                  {!editItemId && (
                    <div className="ml-4 text-xs text-gray-700">
                      {new Date(item.created_at).toLocaleDateString()}
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            <div className="mt-4 flex justify-end space-x-2">
              {editItemId === item.id ? (
                <>
                  <button
                    onClick={cancelEdit}
                    className="px-2 py-1 text-xs text-gray-700 hover:text-gray-900"
                    disabled={isUpdating}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => saveRating(item.id)}
                    className="px-2 py-1 text-xs bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
                    disabled={isUpdating}
                  >
                    {isUpdating ? 'Saving...' : 'Save'}
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => handleEdit(item)}
                    className="px-2 py-1 text-xs text-indigo-600 hover:text-indigo-800"
                  >
                    Edit Rating
                  </button>
                  <button
                    onClick={() => handleDelete(item.id)}
                    className="px-2 py-1 text-xs text-red-600 hover:text-red-800 disabled:opacity-50"
                    disabled={isDeleting[item.id]}
                  >
                    {isDeleting[item.id] ? 'Deleting...' : 'Delete'}
                  </button>
                </>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
})

// Display name for debugging purposes
WatchHistoryList.displayName = 'WatchHistoryList';

export default WatchHistoryList; 
'use client';

import { useState, useEffect, useCallback, useRef, forwardRef, useImperativeHandle } from 'react';
import Image from 'next/image';
import { useAuth } from '@/components/SimpleAuthProvider';
import { toast } from 'react-hot-toast';
import { createClient } from '@/utils/supabase/client';
import { AnimeWatchHistoryItem } from '@/types/watchHistory';
import { dataAccessService } from '@/services/dataAccessService';
import { WATCH_HISTORY_CHANGED_EVENT } from '@/services/watchHistoryService';
import { AuthAwareWrapper } from '@/components/AuthAwareWrapper';

export interface WatchHistoryListRef {
  addAnime?: (anime: AnimeWatchHistoryItem) => void;
}

// Inner component that will be remounted when auth state changes
const WatchHistoryListInner = forwardRef<WatchHistoryListRef>((props, ref) => {
  const { user, isAuthenticated } = useAuth();
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
    setIsLoading(true);
    setError(null);
    
    try {
      // Use dataAccessService to fetch watch history regardless of auth state
      const data = await dataAccessService.getWatchHistory();
      setWatchHistory(data);
    } catch (err) {
      console.error('Error fetching watch history:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      toast.error('Failed to load watch history');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Improved real-time subscription setup - only for authenticated users
  useEffect(() => {
    // For authenticated users, set up real-time updates
    if (isAuthenticated && user) {
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

      // Set up real-time subscription
      setupRealtimeSubscription();

      // Clean up subscription when component unmounts or user changes
      return () => {
        if (channelRef.current) {
          supabase.removeChannel(channelRef.current);
          channelRef.current = null;
        }
      };
    }
    
    // For both authenticated and non-authenticated users, listen for watch history changes
    const handleWatchHistoryChange = () => {
      console.log('Watch history changed event detected');
      fetchWatchHistory();
    };
    
    // Listen for custom watch history change events
    window.addEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
    
    return () => {
      window.removeEventListener(WATCH_HISTORY_CHANGED_EVENT, handleWatchHistoryChange);
    };
  }, [isAuthenticated, user, supabase, fetchWatchHistory]);

  // Initial data fetch
  useEffect(() => {
    fetchWatchHistory();
  }, [fetchWatchHistory]);

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
      // Use dataAccessService to delete watch history item regardless of auth state
      await dataAccessService.deleteWatchHistory(id);
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
      // Use dataAccessService to update rating regardless of auth state
      await dataAccessService.updateWatchHistoryRating(id, editRating);
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
      <div className="p-4 bg-red-50 rounded-lg mt-4">
        <p className="text-red-700">Error loading watch history: {error}</p>
        <button 
          onClick={() => fetchWatchHistory()}
          className="mt-2 px-3 py-1 bg-red-100 text-red-800 rounded-md hover:bg-red-200"
        >
          Retry
        </button>
      </div>
    );
  }

  if (watchHistory.length === 0) {
    return (
      <div className="py-12 border border-gray-200 rounded-lg bg-gray-50 text-center">
        <p className="text-gray-500">Your watch history is empty.</p>
        <p className="text-gray-500 mt-1">Use the form above to add anime you&apos;ve watched.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4">
      {watchHistory.map((item) => (
        <div key={item.id} className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden p-4">
          <div className="flex items-start gap-4">
            {/* Cover image */}
            <div className="flex-shrink-0">
              {item.cover_image ? (
                <Image
                  src={item.cover_image}
                  alt={item.title}
                  width={80}
                  height={120}
                  className="rounded-md object-cover"
                />
              ) : (
                <div className="bg-gray-200 w-20 h-30 rounded-md flex items-center justify-center">
                  <span className="text-gray-500 text-xs">No Image</span>
                </div>
              )}
            </div>
            
            {/* Content */}
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-semibold text-gray-900 truncate">{item.title}</h3>
              
              <div className="mt-2">
                {editItemId === item.id ? (
                  <div className="flex items-center">
                    <div className="flex space-x-1">
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((num) => (
                        <button
                          key={num}
                          type="button"
                          onClick={() => setEditRating(num)}
                          className={`text-lg ${num <= editRating ? 'text-yellow-500' : 'text-gray-300'} hover:text-yellow-400`}
                          aria-label={`Rate ${num} out of 10`}
                        >
                          {num}
                        </button>
                      ))}
                    </div>
                    <div className="ml-4 space-x-2">
                      <button
                        onClick={() => saveRating(item.id)}
                        disabled={isUpdating}
                        className="px-2 py-1 text-sm bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
                      >
                        {isUpdating ? 'Saving...' : 'Save'}
                      </button>
                      <button
                        onClick={cancelEdit}
                        className="px-2 py-1 text-sm bg-gray-200 text-gray-800 rounded hover:bg-gray-300"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center">
                    <div className="flex space-x-1">
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((num) => (
                        <span
                          key={num}
                          className={`text-lg ${num <= item.rating ? 'text-yellow-500' : 'text-gray-300'}`}
                        >
                          {num}
                        </span>
                      ))}
                    </div>
                    <div className="ml-4 text-sm text-gray-700">
                      Rating: {item.rating}/10
                    </div>
                  </div>
                )}
              </div>
              
              <div className="mt-3 flex space-x-2">
                {!editItemId && (
                  <button
                    onClick={() => handleEdit(item)}
                    className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                  >
                    Edit Rating
                  </button>
                )}
                
                <button
                  onClick={() => handleDelete(item.id)}
                  disabled={!!isDeleting[item.id]}
                  className="px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50"
                >
                  {isDeleting[item.id] ? 'Removing...' : 'Remove'}
                </button>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
});

// Main component wrapper with AuthAwareWrapper
const WatchHistoryList = forwardRef<WatchHistoryListRef>((props, ref) => {
  return (
    <AuthAwareWrapper>
      <WatchHistoryListInner ref={ref} />
    </AuthAwareWrapper>
  );
});

WatchHistoryList.displayName = 'WatchHistoryList';
WatchHistoryListInner.displayName = 'WatchHistoryListInner';

export default WatchHistoryList; 
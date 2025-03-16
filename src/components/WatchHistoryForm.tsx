'use client';

import { useState } from 'react';
import { useAuth } from '@/components/SimpleAuthProvider';
import AnimeSearch from './AnimeSearch';
import { toast } from 'react-hot-toast';
import { type AnimeSearchResult } from '@/utils/anilistClient';
import { WatchHistoryFormData, AnimeWatchHistoryItem } from '@/types/watchHistory';
import { addToWatchHistory } from '@/services/watchHistoryService';

interface WatchHistoryFormProps {
  onAnimeAdded?: (anime: AnimeWatchHistoryItem) => void;
}

export default function WatchHistoryForm({ onAnimeAdded }: WatchHistoryFormProps) {
  const { user } = useAuth();
  const [selectedAnime, setSelectedAnime] = useState<AnimeSearchResult | null>(null);
  const [rating, setRating] = useState<number>(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedAnime) {
      toast.error('Please select an anime first');
      return;
    }
    
    if (rating === 0) {
      toast.error('Please provide a rating');
      return;
    }
    
    if (!user) {
      toast.error('You must be logged in to add to your watch history');
      return;
    }
    
    setIsSubmitting(true);
    setErrorDetails(null);
    
    try {
      // Create a safer version of the cover image URL
      // Only include the cover image URL if it's from an allowed domain
      const coverImage = selectedAnime.coverImage?.medium;
      const allowedDomains = ['s4.anilist.co', 'media.kitsu.io', 'img.anili.st'];
      const isSafeDomain = coverImage && 
        allowedDomains.some(domain => coverImage.includes(domain));
      
      // Prepare the data for our new normalized table
      const formData: WatchHistoryFormData = {
        anilist_id: selectedAnime.id,
        title: selectedAnime.title.english || selectedAnime.title.romaji,
        cover_image: isSafeDomain ? coverImage : undefined,
        rating: rating
      };
      
      // Call our service to add the anime to watch history
      const addedAnime = await addToWatchHistory(formData);
      
      toast.success('Added to watch history!');
      
      // Call the callback function if provided
      if (onAnimeAdded) {
        onAnimeAdded(addedAnime);
      }
      
      // Reset form
      setSelectedAnime(null);
      setRating(0);
    } catch (error) {
      console.error('Error saving watch history:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setErrorDetails(errorMessage);
      toast.error('Failed to save watch history. Please check details below.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Add to Watch History</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="animeTitle" className="block text-sm font-medium text-gray-800 mb-2">
            Anime Title
          </label>
          <AnimeSearch 
            onSelect={(anime) => setSelectedAnime(anime)} 
            placeholder="Search for an anime..." 
          />
          {selectedAnime && (
            <div className="mt-2 flex items-center">
              <span className="text-sm text-gray-700">Selected: </span>
              <span className="ml-1 text-sm font-medium text-indigo-700">
                {selectedAnime.title.english || selectedAnime.title.romaji}
              </span>
            </div>
          )}
        </div>
        
        <div className="mb-6">
          <label htmlFor="rating" className="block text-sm font-medium text-gray-800 mb-2">
            Rating (1-10)
          </label>
          <div className="flex items-center space-x-1">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((star) => (
              <button
                key={star}
                type="button"
                onClick={() => setRating(star)}
                className={`text-xl ${star <= rating ? 'text-yellow-500' : 'text-gray-400'} hover:text-yellow-400 transition-colors`}
                aria-label={`Rate ${star} out of 10`}
              >
                {star}
              </button>
            ))}
          </div>
          <div className="mt-1 text-sm text-gray-700">
            {rating > 0 ? `Your rating: ${rating}/10` : 'Select a rating'}
          </div>
        </div>
        
        {errorDetails && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-600 font-mono overflow-auto max-h-32">
              {errorDetails}
            </p>
          </div>
        )}
        
        <button
          type="submit"
          disabled={isSubmitting || !selectedAnime || rating === 0}
          className="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSubmitting ? 'Saving...' : 'Add to History'}
        </button>
      </form>
    </div>
  );
} 
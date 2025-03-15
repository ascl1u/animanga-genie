'use client';

import { useState } from 'react';

export default function WatchHistoryForm() {
  const [animeTitle, setAnimeTitle] = useState('');
  const [rating, setRating] = useState<number>(0);
  const [watchStatus, setWatchStatus] = useState('completed');
  const [watchDate, setWatchDate] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // This will be implemented with Supabase integration
    console.log({
      animeTitle,
      rating,
      watchStatus,
      watchDate: watchDate || null,
    });
    
    // Reset form
    setAnimeTitle('');
    setRating(0);
    setWatchStatus('completed');
    setWatchDate('');
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">Add to Watch History</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="animeTitle" className="block text-sm font-medium text-gray-700 mb-1">
            Anime Title
          </label>
          <input
            type="text"
            id="animeTitle"
            value={animeTitle}
            onChange={(e) => setAnimeTitle(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            placeholder="Search for an anime..."
            required
          />
        </div>
        
        <div className="mb-4">
          <label htmlFor="rating" className="block text-sm font-medium text-gray-700 mb-1">
            Rating (1-5)
          </label>
          <div className="flex items-center space-x-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                type="button"
                onClick={() => setRating(star)}
                className={`text-2xl ${star <= rating ? 'text-yellow-500' : 'text-gray-300'}`}
                aria-label={`Rate ${star} stars`}
              >
                ★
              </button>
            ))}
          </div>
        </div>
        
        <div className="mb-4">
          <label htmlFor="watchStatus" className="block text-sm font-medium text-gray-700 mb-1">
            Watch Status
          </label>
          <select
            id="watchStatus"
            value={watchStatus}
            onChange={(e) => setWatchStatus(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            required
          >
            <option value="completed">Completed</option>
            <option value="watching">Currently Watching</option>
            <option value="plan_to_watch">Plan to Watch</option>
            <option value="dropped">Dropped</option>
          </select>
        </div>
        
        <div className="mb-6">
          <label htmlFor="watchDate" className="block text-sm font-medium text-gray-700 mb-1">
            Date Completed (Optional)
          </label>
          <input
            type="date"
            id="watchDate"
            value={watchDate}
            onChange={(e) => setWatchDate(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
        
        <button
          type="submit"
          className="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
        >
          Add to History
        </button>
      </form>
    </div>
  );
} 
'use client';

import { useState } from 'react';

interface RatingEditorProps {
  initialRating: number;
  onSave: (newRating: number) => void;
  onCancel: () => void;
}

export default function RatingEditor({ initialRating, onSave, onCancel }: RatingEditorProps) {
  const [rating, setRating] = useState<number>(initialRating);

  return (
    <div className="p-3 bg-gray-50 rounded-md border border-gray-200">
      <div className="mb-3">
        <div className="flex items-center space-x-1">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((star) => (
            <button
              key={star}
              type="button"
              onClick={() => setRating(star)}
              className={`text-xl ${star <= rating ? 'text-yellow-500' : 'text-gray-300'} hover:text-yellow-400 transition-colors`}
              aria-label={`Rate ${star} out of 10`}
            >
              {star}
            </button>
          ))}
        </div>
        <div className="mt-1 text-sm text-gray-500">
          {rating > 0 ? `Your rating: ${rating}/10` : 'Select a rating'}
        </div>
      </div>
      <div className="flex space-x-2">
        <button
          onClick={() => onSave(rating)}
          className="px-3 py-1 bg-indigo-600 text-white text-sm rounded-md hover:bg-indigo-700"
        >
          Save
        </button>
        <button
          onClick={onCancel}
          className="px-3 py-1 bg-gray-200 text-gray-700 text-sm rounded-md hover:bg-gray-300"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
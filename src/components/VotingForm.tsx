'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Poll } from '@/types/polls';
import { submitVote } from '@/services/pollService';

interface VotingFormProps {
  poll: Poll;
  userId: string;
}

export function VotingForm({ poll, userId }: VotingFormProps) {
  const router = useRouter();
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [isSuccess, setIsSuccess] = useState(false);
  
  const handleVote = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (selectedOption === null) {
      setError('Please select an option to vote');
      return;
    }
    
    setIsSubmitting(true);
    setError('');
    
    try {
      await submitVote(poll.id, userId, selectedOption);
      // Show success animation before refreshing
      setIsSuccess(true);
      
      // Wait a moment to show the success animation before refreshing
      setTimeout(() => {
        router.refresh();
      }, 600);
    } catch (err) {
      setError('Failed to submit your vote. Please try again.');
      console.error('Vote submission error:', err);
      setIsSubmitting(false);
    }
  };
  
  if (isSuccess) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 shadow-md p-6 mb-6 text-center">
        <div className="animate-pulse flex flex-col items-center justify-center py-4">
          <svg className="w-12 h-12 text-green-500 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <h3 className="text-lg font-medium text-gray-900">Vote Recorded!</h3>
          <p className="text-gray-600 mt-1">Loading results...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-md p-6 mb-6">
      <form onSubmit={handleVote}>
        <fieldset disabled={isSubmitting}>
          <legend className="sr-only">Poll options</legend>
          
          <div className="space-y-3">
            {poll.options.map((option, index) => (
              <div key={index} className="flex items-center">
                <input
                  id={`option-${index}`}
                  name="poll-option"
                  type="radio"
                  className="h-4 w-4 border-gray-300 text-blue-600 focus:ring-blue-500"
                  value={index}
                  checked={selectedOption === index}
                  onChange={() => setSelectedOption(index)}
                />
                <label
                  htmlFor={`option-${index}`}
                  className="ml-3 block text-sm font-medium text-gray-700"
                >
                  {option}
                </label>
              </div>
            ))}
          </div>
          
          {error && (
            <p className="mt-2 text-sm text-red-600">{error}</p>
          )}
          
          <button
            type="submit"
            className="mt-6 w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 transition-colors"
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Submitting...
              </span>
            ) : 'Submit Vote'}
          </button>
        </fieldset>
      </form>
    </div>
  );
} 
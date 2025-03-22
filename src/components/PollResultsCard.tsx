'use client';

import { PollWithResults } from '@/types/polls';

interface PollResultsCardProps {
  poll: PollWithResults;
}

export function PollResultsCard({ poll }: PollResultsCardProps) {
  // Sort results with user's vote at the top if it exists
  const sortedResults = poll.userVote !== undefined
    ? [...poll.results].sort((a, b) => {
        const aIndex = poll.results.indexOf(a);
        const bIndex = poll.results.indexOf(b);
        if (aIndex === poll.userVote) return -1;
        if (bIndex === poll.userVote) return 1;
        return 0;
      })
    : poll.results;

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-md p-6">
      <div className="mb-4">
        <p className="text-sm text-gray-500">
          Total votes: <span className="font-medium">{poll.totalVotes}</span>
        </p>
      </div>
      
      <div className="space-y-5">
        {sortedResults.map((result) => {
          const originalIndex = poll.results.indexOf(result);
          const isUserVote = poll.userVote === originalIndex;
          
          return (
            <div 
              key={originalIndex} 
              className={`space-y-2 p-3 rounded-lg ${isUserVote ? 'bg-blue-50 border border-blue-100' : ''}`}
            >
              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  {isUserVote && (
                    <span className="inline-block w-4 h-4 rounded-full bg-blue-600 mr-2 flex-shrink-0">
                      <span className="sr-only">Your vote</span>
                    </span>
                  )}
                  <span className={`text-sm font-medium ${isUserVote ? 'text-blue-800' : 'text-gray-800'}`}>
                    {result.option} {isUserVote && <span className="text-blue-600 ml-1">(Your vote)</span>}
                  </span>
                </div>
                <span className={`text-sm font-bold ${isUserVote ? 'text-blue-700' : 'text-gray-900'}`}>
                  {result.percentage}%
                </span>
              </div>
              
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className={`${isUserVote ? 'bg-blue-600' : 'bg-gray-500'} h-2.5 rounded-full transition-all duration-500 ease-out`}
                  style={{ width: `${result.percentage}%` }}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={result.percentage}
                  role="progressbar"
                />
              </div>
              
              <p className={`text-xs ${isUserVote ? 'text-blue-600' : 'text-gray-500'}`}>
                {result.votes} vote{result.votes !== 1 ? 's' : ''}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
} 
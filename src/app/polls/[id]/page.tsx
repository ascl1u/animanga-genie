import { createClient } from '@/utils/supabase/server';
import { PollResultsCard } from '@/components/PollResultsCard';
import { VotingForm } from '@/components/VotingForm';
import { notFound } from 'next/navigation';
import Link from 'next/link';

interface PollPageProps {
  params: {
    id: string;
  };
}

export async function generateMetadata({ params }: PollPageProps) {
  // Await the params
  const resolvedParams = await Promise.resolve(params);
  const pollId = parseInt(resolvedParams.id);
  
  if (isNaN(pollId)) {
    return {
      title: 'Poll Not Found - Animanga Genie',
    };
  }
  
  const supabase = await createClient();
  const { data: poll } = await supabase
    .from('polls')
    .select('question')
    .eq('id', pollId)
    .single();
  
  if (!poll) {
    return {
      title: 'Poll Not Found - Animanga Genie',
    };
  }
  
  return {
    title: `${poll.question} - Animanga Genie Polls`,
    description: `Vote on "${poll.question}" in our anime community poll`,
  };
}

export default async function PollPage({ params }: PollPageProps) {
  // Await the params
  const resolvedParams = await Promise.resolve(params);
  const pollId = parseInt(resolvedParams.id);
  
  if (isNaN(pollId)) {
    notFound();
  }
  
  const supabase = await createClient();
  
  // Get the current user if they're logged in
  const {
    data: { user },
  } = await supabase.auth.getUser();
  
  // Get the poll
  const { data: poll, error: pollError } = await supabase
    .from('polls')
    .select('*')
    .eq('id', pollId)
    .single();
  
  if (pollError || !poll) {
    notFound();
  }
  
  // Get votes for this poll
  const { data: votes, error: votesError } = await supabase
    .from('votes')
    .select('option_index')
    .eq('poll_id', pollId);
  
  if (votesError) {
    console.error('Error fetching votes:', votesError);
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-6">{poll.question}</h1>
        <p className="text-red-500">Error loading poll results. Please try again later.</p>
      </div>
    );
  }
  
  // Check if user has voted
  let userVote;
  if (user) {
    const { data: userVoteData } = await supabase
      .from('votes')
      .select('option_index')
      .eq('poll_id', pollId)
      .eq('user_id', user.id)
      .maybeSingle();
    
    if (userVoteData) {
      userVote = userVoteData.option_index;
    }
  }
  
  // Calculate results
  const votesArray = votes || [];
  const totalVotes = votesArray.length;
  
  const results = poll.options.map((option: string, index: number) => {
    const voteCount = votesArray.filter((vote) => vote.option_index === index).length;
    const percentage = totalVotes > 0 ? Math.round((voteCount / totalVotes) * 100) : 0;
    
    return {
      option,
      votes: voteCount,
      percentage
    };
  });
  
  const pollWithResults = {
    ...poll,
    results,
    totalVotes,
    userVote
  };
  
  return (
    <div className="container mx-auto px-4 py-8">
      <Link 
        href="/polls"
        className="inline-block mb-6 text-blue-600 hover:text-blue-800"
      >
        ← Back to all polls
      </Link>
      
      <h1 className="text-2xl font-bold mb-6">{poll.question}</h1>
      
      {user && userVote !== undefined ? (
        <>
          <div className="mb-4 bg-green-50 p-4 rounded-md border border-green-200">
            <p className="text-green-800">Thanks for voting! Here are the current results:</p>
          </div>
          <PollResultsCard poll={pollWithResults} />
        </>
      ) : (
        <>
          {user ? (
            <>
              <p className="mb-4 text-gray-700">Vote to see the current results!</p>
              <VotingForm poll={poll} userId={user.id} />
            </>
          ) : (
            <div className="bg-white rounded-lg border border-gray-200 text-gray-900 shadow-md p-6 mb-6">
              <p className="mb-4">You need to be logged in to vote in polls and see results.</p>
              <Link 
                href="/login"
                className="inline-block px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Log In to Vote
              </Link>
            </div>
          )}
        </>
      )}
    </div>
  );
} 
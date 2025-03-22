import { createClient } from '@/utils/supabase/client';
import { Poll, PollWithResults, Vote } from '@/types/polls';

export const fetchActivePolls = async (): Promise<Poll[]> => {
  const supabase = createClient();
  
  const { data, error } = await supabase
    .from('polls')
    .select('*')
    .eq('active', true)
    .order('created_at', { ascending: false });
  
  if (error) {
    console.error('Error fetching active polls:', error);
    throw error;
  }
  
  return data || [];
};

export const fetchPollWithResults = async (
  pollId: number,
  userId?: string
): Promise<PollWithResults | null> => {
  const supabase = createClient();
  
  // Get the poll
  const { data: poll, error: pollError } = await supabase
    .from('polls')
    .select('*')
    .eq('id', pollId)
    .single();
  
  if (pollError) {
    console.error('Error fetching poll:', pollError);
    throw pollError;
  }
  
  if (!poll) return null;
  
  // Get votes for this poll
  const { data: votes, error: votesError } = await supabase
    .from('votes')
    .select('option_index')
    .eq('poll_id', pollId);
  
  if (votesError) {
    console.error('Error fetching votes:', votesError);
    throw votesError;
  }
  
  // Check if user has voted
  let userVote: number | undefined;
  
  if (userId) {
    const { data: userVoteData, error: userVoteError } = await supabase
      .from('votes')
      .select('option_index')
      .eq('poll_id', pollId)
      .eq('user_id', userId)
      .maybeSingle();
    
    if (!userVoteError && userVoteData) {
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
  
  return {
    ...poll,
    results,
    totalVotes,
    userVote
  };
};

export const submitVote = async (
  pollId: number,
  userId: string,
  optionIndex: number
): Promise<Vote> => {
  const supabase = createClient();
  
  const { data, error } = await supabase
    .from('votes')
    .insert({
      poll_id: pollId,
      user_id: userId,
      option_index: optionIndex
    })
    .select()
    .single();
  
  if (error) {
    console.error('Error submitting vote:', error);
    throw error;
  }
  
  return data;
};

export const createPoll = async (
  question: string,
  options: string[]
): Promise<Poll> => {
  const supabase = createClient();
  
  const { data, error } = await supabase
    .from('polls')
    .insert({
      question,
      options
    })
    .select()
    .single();
  
  if (error) {
    console.error('Error creating poll:', error);
    throw error;
  }
  
  return data;
}; 
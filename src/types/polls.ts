export interface Poll {
  id: number;
  question: string;
  options: string[];
  created_at: string;
  active: boolean;
}

export interface Vote {
  id: number;
  poll_id: number;
  user_id: string;
  option_index: number;
  voted_at: string;
}

export interface PollWithResults extends Poll {
  results: {
    option: string;
    votes: number;
    percentage: number;
  }[];
  totalVotes: number;
  userVote?: number;
} 
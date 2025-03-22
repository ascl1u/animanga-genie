import { createClient } from '@/utils/supabase/server';
import Link from 'next/link';
import { Poll } from '@/types/polls';

export const metadata = {
  title: 'Polls - Animanga Genie',
  description: 'Vote on your favorite anime and manga in our community polls',
};

export default async function PollsPage() {
  const supabase = await createClient();
  
  const { data: polls, error } = await supabase
    .from('polls')
    .select('*')
    .eq('active', true)
    .order('created_at', { ascending: false });
    
  if (error) {
    console.error('Error fetching polls:', error);
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-6">Anime Community Polls</h1>
        <p className="text-red-500">Error loading polls. Please try again later.</p>
      </div>
    );
  }
  
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Anime Community Polls</h1>
      
      {polls && polls.length > 0 ? (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {polls.map((poll: Poll) => (
            <Link 
              key={poll.id}
              href={`/polls/${poll.id}`}
              className="block p-6 bg-white rounded-lg border border-gray-200 shadow-md hover:bg-gray-100 transition-colors"
            >
              <h2 className="text-xl font-semibold mb-3 text-gray-900">{poll.question}</h2>
              <p className="text-sm text-gray-600">
                {poll.options.length} options • Created {new Date(poll.created_at).toLocaleDateString()}
              </p>
              <div className="mt-4">
                <span className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                  Vote Now
                </span>
              </div>
            </Link>
          ))}
        </div>
      ) : (
        <p className="text-gray-600">No active polls at the moment. Check back later!</p>
      )}
    </div>
  );
} 
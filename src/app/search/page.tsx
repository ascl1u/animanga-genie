import { Metadata } from 'next';
import SearchBar from '@/components/SearchBar';

export const metadata: Metadata = {
  title: 'Search - AniManga Genie',
  description: 'Search for anime and manga series',
};

export default function SearchPage() {
  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-6">Search Anime</h1>
      
      <div className="max-w-2xl mx-auto">
        <div className="mb-8">
          <SearchBar />
        </div>
        
        <div className="bg-white rounded-lg shadow p-6 mb-4">
          <p className="text-gray-600 text-center">
            Enter a search term to find anime and manga titles.
          </p>
        </div>
      </div>
    </div>
  );
} 
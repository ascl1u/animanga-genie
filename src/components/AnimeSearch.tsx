'use client';

import { useState, useEffect, useRef } from 'react';
import Image from 'next/image';
import { searchAnime, type AnimeSearchResult } from '@/utils/anilistClient';
import { useDebounce } from '@/utils/hooks';

type AnimeSearchProps = {
  onSelect: (anime: AnimeSearchResult) => void;
  placeholder?: string;
};

export default function AnimeSearch({ onSelect, placeholder = "Search for an anime..." }: AnimeSearchProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState<AnimeSearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [imageErrors, setImageErrors] = useState<Record<number, boolean>>({});
  const debouncedSearch = useDebounce(searchQuery, 500);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Search anime when debouncedSearch changes
  useEffect(() => {
    const fetchAnime = async () => {
      if (debouncedSearch.trim().length < 2) {
        setResults([]);
        return;
      }

      setIsLoading(true);
      try {
        const data = await searchAnime(debouncedSearch);
        setResults(data);
        // Reset image errors when new results come in
        setImageErrors({});
      } catch (error) {
        console.error('Error fetching anime:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnime();
  }, [debouncedSearch]);

  // Handle click outside to close dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleSelectAnime = (anime: AnimeSearchResult) => {
    onSelect(anime);
    setSearchQuery(anime.title.english || anime.title.romaji);
    setIsOpen(false);
  };

  const handleImageError = (animeId: number) => {
    setImageErrors((prev) => ({ ...prev, [animeId]: true }));
  };

  return (
    <div className="relative w-full" ref={wrapperRef}>
      <input
        type="text"
        value={searchQuery}
        onChange={(e) => {
          setSearchQuery(e.target.value);
          setIsOpen(true);
        }}
        onFocus={() => setIsOpen(true)}
        className="w-full p-2 border border-gray-300 rounded-md text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        placeholder={placeholder}
      />
      
      {isLoading && (
        <div className="absolute right-3 top-2.5">
          <div className="animate-spin h-5 w-5 border-2 border-indigo-500 rounded-full border-t-transparent"></div>
        </div>
      )}
      
      {isOpen && results.length > 0 && (
        <div className="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-sm overflow-auto">
          {results.map((anime) => (
            <div
              key={anime.id}
              className="flex items-center px-4 py-2 hover:bg-gray-100 cursor-pointer"
              onClick={() => handleSelectAnime(anime)}
            >
              {anime.coverImage?.medium && !imageErrors[anime.id] ? (
                <div className="flex-shrink-0 h-10 w-10 mr-3 relative">
                  <Image
                    src={anime.coverImage.medium}
                    alt={anime.title.english || anime.title.romaji}
                    className="rounded-sm object-cover"
                    fill
                    sizes="40px"
                    onError={() => handleImageError(anime.id)}
                  />
                </div>
              ) : (
                <div className="flex-shrink-0 h-10 w-10 mr-3 flex items-center justify-center bg-gray-200 rounded-sm">
                  <span className="text-xs text-gray-500">No image</span>
                </div>
              )}
              <div>
                <div className="font-medium text-gray-900">
                  {anime.title.english || anime.title.romaji}
                </div>
                <div className="text-xs text-gray-500">
                  {anime.format} • {anime.seasonYear} • Score: {anime.averageScore ? (anime.averageScore / 10).toFixed(1) : 'N/A'}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {isOpen && searchQuery.trim().length >= 2 && !isLoading && results.length === 0 && (
        <div className="absolute z-10 mt-1 w-full bg-white shadow-lg rounded-md py-2 px-4 text-sm">
          No results found
        </div>
      )}
    </div>
  );
} 
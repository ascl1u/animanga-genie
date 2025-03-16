'use client';

import { useState } from 'react';
import { toast } from 'react-hot-toast';

export default function WatchHistoryImport() {
  const [malUsername, setMalUsername] = useState('');
  const [anilistUsername, setAnilistUsername] = useState('');
  const [isImportingMal, setIsImportingMal] = useState(false);
  const [isImportingAnilist, setIsImportingAnilist] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const handleMalImport = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsImportingMal(true);
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // This is where you would implement the actual API call
      toast.success('MyAnimeList import feature coming soon!');
    } catch (error) {
      console.error('Error importing from MyAnimeList:', error);
    } finally {
      setIsImportingMal(false);
    }
  };

  const handleAnilistImport = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsImportingAnilist(true);
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // This is where you would implement the actual API call
      toast.success('AniList import feature coming soon!');
    } catch (error) {
      console.error('Error importing from AniList:', error);
    } finally {
      setIsImportingAnilist(false);
    }
  };

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="bg-white rounded-lg shadow p-4 mb-6 transition-all duration-300">
      {/* Collapsible header */}
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={toggleExpanded}
      >
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Import Watch History</h2>
          <p className="text-sm text-gray-700">
            Import your anime watch history from external services
          </p>
        </div>
        <div className="text-gray-700 hover:text-gray-900 text-xl">
          {isExpanded ? '▲' : '▼'}
        </div>
      </div>
      
      {/* Collapsible content */}
      <div 
        className={`mt-4 overflow-hidden transition-all duration-300 ${
          isExpanded ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          {/* MyAnimeList Import */}
          <div className="border border-gray-200 rounded-md p-4">
            <h3 className="font-medium text-gray-800 mb-2">MyAnimeList</h3>
            <div className="flex flex-col space-y-3">
              <input 
                type="text" 
                value={malUsername}
                onChange={(e) => setMalUsername(e.target.value)}
                placeholder="MyAnimeList Username"
                className="px-3 py-2 border border-gray-300 rounded-md text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
              <button
                onClick={handleMalImport}
                disabled={isImportingMal || !malUsername.trim()}
                className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isImportingMal ? 'Importing...' : 'Import from MyAnimeList'}
              </button>
              <div className="text-xs text-amber-700 font-medium">
                Coming Soon
              </div>
            </div>
          </div>
          
          {/* AniList Import */}
          <div className="border border-gray-200 rounded-md p-4">
            <h3 className="font-medium text-gray-800 mb-2">AniList</h3>
            <div className="flex flex-col space-y-3">
              <input 
                type="text" 
                value={anilistUsername}
                onChange={(e) => setAnilistUsername(e.target.value)}
                placeholder="AniList Username"
                className="px-3 py-2 border border-gray-300 rounded-md text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
              <button
                onClick={handleAnilistImport}
                disabled={isImportingAnilist || !anilistUsername.trim()}
                className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isImportingAnilist ? 'Importing...' : 'Import from AniList'}
              </button>
              <div className="text-xs text-amber-700 font-medium">
                Coming Soon
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 
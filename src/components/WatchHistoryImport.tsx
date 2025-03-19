'use client';

import { useState } from 'react';
import { toast } from 'react-hot-toast';
import { importAnilistWatchHistory } from '@/services/watchHistoryService';
import { Dialog } from '@/components/ui/Dialog';

export default function WatchHistoryImport() {
  const [malUsername, setMalUsername] = useState('');
  const [anilistUsername, setAnilistUsername] = useState('');
  const [isImportingMal, setIsImportingMal] = useState(false);
  const [isImportingAnilist, setIsImportingAnilist] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [importResults, setImportResults] = useState<{
    added: number;
    updated: number;
    unchanged: number;
    total: number;
  } | null>(null);

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

  const handleConfirmAnilistImport = async () => {
    setShowConfirmDialog(false);
    setIsImportingAnilist(true);
    setImportResults(null);
    
    try {
      // Call the service to import watch history from AniList
      const results = await importAnilistWatchHistory(anilistUsername);
      
      // Show success message with stats
      toast.success(`Successfully synced ${results.total} anime from AniList!`);
      
      // Show detailed results
      setImportResults(results);
    } catch (error) {
      console.error('Error importing from AniList:', error);
      
      // Extract more detailed error message
      let errorMessage = 'Failed to import from AniList';
      
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (typeof error === 'object' && error !== null) {
        // Try to extract Supabase error details if available
        const supabaseError = error as { message?: string; details?: string; hint?: string; code?: string };
        if (supabaseError.message) {
          errorMessage = supabaseError.message;
          
          // Add more context if available
          if (supabaseError.details) {
            errorMessage += `: ${supabaseError.details}`;
          } else if (supabaseError.hint) {
            errorMessage += ` (${supabaseError.hint})`;
          }
        }
      }
      
      toast.error(errorMessage);
    } finally {
      setIsImportingAnilist(false);
    }
  };

  const startAnilistImport = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!anilistUsername.trim()) {
      toast.error('Please enter an AniList username');
      return;
    }
    
    // Show confirmation dialog
    setShowConfirmDialog(true);
  };

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  const cancelImport = () => {
    setShowConfirmDialog(false);
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
                onClick={startAnilistImport}
                disabled={isImportingAnilist || !anilistUsername.trim()}
                className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isImportingAnilist ? 'Importing...' : 'Import from AniList'}
              </button>
              
              {importResults && (
                <div className="mt-2 text-sm">
                  <p className="font-medium text-gray-900">Import Summary:</p>
                  <ul className="list-disc pl-5 mt-1 text-gray-700">
                    <li>Added: {importResults.added}</li>
                    <li>Updated: {importResults.updated}</li>
                    <li>Unchanged: {importResults.unchanged}</li>
                    <li>Total processed: {importResults.total}</li>
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Confirmation Dialog */}
      {showConfirmDialog && (
        <Dialog
          open={showConfirmDialog}
          onClose={cancelImport}
          title="Import from AniList"
        >
          <div className="mt-2">
            <p className="text-sm text-gray-500">
              This will import your anime watch history from AniList user <strong>{anilistUsername}</strong> and synchronize it with your current watch history.
            </p>
            
            <div className="mt-3">
              <p className="text-sm text-gray-500">
                <strong>Note:</strong> This will update your existing entries with data from AniList. 
                Unrated anime entries from AniList will be imported with a default rating of 5.
              </p>
            </div>
          </div>

          <div className="mt-4 flex justify-end space-x-3">
            <button
              type="button"
              className="inline-flex justify-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              onClick={cancelImport}
            >
              Cancel
            </button>
            <button
              type="button"
              className="inline-flex justify-center px-4 py-2 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              onClick={handleConfirmAnilistImport}
            >
              Import
            </button>
          </div>
        </Dialog>
      )}
    </div>
  );
} 
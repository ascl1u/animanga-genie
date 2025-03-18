'use client';

import React, { createContext, useContext, useState, ReactNode, useCallback } from 'react';
import { AnimeData } from '@/services/recommendationService';
import { AnimeWatchHistoryItem } from '@/types/watchHistory';
import { useRecommendations as useRecommendationsHook } from '@/hooks/useRecommendations';

interface RecommendationsContextType {
  recommendations: AnimeData[];
  watchHistory: AnimeWatchHistoryItem[];
  isLoading: boolean;
  isError: boolean;
  errorMessage: string;
  isInitialized: boolean;
  hasWatchHistory: boolean;
  isWatchHistoryLoaded: boolean;
  loadAttempts: number;
  modelLoadingProgress: number;
  generateRecommendations: (customLimit?: number) => Promise<void>;
  refreshRecommendations: () => Promise<void>;
  feedback: Record<number, 'like' | 'dislike' | null>;
  addFeedback: (animeId: number, feedbackType: 'like' | 'dislike' | null) => void;
  isCollaborativeFilteringEnabled: boolean;
  similarUsers: Array<{userId: string, similarity: number}>;
  watchHistoryChanged: boolean;
}

const initialRecommendationsContext: RecommendationsContextType = {
  recommendations: [],
  watchHistory: [],
  isLoading: false,
  isError: false,
  errorMessage: '',
  isInitialized: false,
  hasWatchHistory: false,
  isWatchHistoryLoaded: false,
  loadAttempts: 0,
  modelLoadingProgress: 0,
  generateRecommendations: async () => {},
  refreshRecommendations: async () => {},
  feedback: {},
  addFeedback: () => {},
  isCollaborativeFilteringEnabled: false,
  similarUsers: [],
  watchHistoryChanged: true
};

const RecommendationsContext = createContext<RecommendationsContextType>(initialRecommendationsContext);

export function useRecommendations() {
  return useContext(RecommendationsContext);
}

export function RecommendationsProvider({ children }: { children: ReactNode }) {
  const [feedback, setFeedback] = useState<Record<number, 'like' | 'dislike' | null>>({});
  
  // Use the hook with 9 recommendations as default
  const hookResult = useRecommendationsHook({
    limit: 9, 
    autoLoad: false
  });
  
  const addFeedback = useCallback((animeId: number, feedbackType: 'like' | 'dislike' | null) => {
    setFeedback(prev => ({
      ...prev,
      [animeId]: feedbackType
    }));
    
    // Here you would also integrate with backend to store feedback
    console.log(`Feedback for anime ${animeId}: ${feedbackType}`);
  }, []);

  // Persist hook data and feedback
  const value = {
    ...hookResult,
    feedback,
    addFeedback
  };
  
  return (
    <RecommendationsContext.Provider value={value}>
      {children}
    </RecommendationsContext.Provider>
  );
} 
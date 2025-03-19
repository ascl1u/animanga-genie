export interface AnimeWatchHistoryItem {
  id: string;
  user_id: string;
  anilist_id: number;
  title: string;
  cover_image?: string;
  rating: number;
  created_at: string;
  updated_at: string;
}

export interface UpdateWatchHistoryParams {
  id: string;
  rating: number;
}

export interface WatchHistoryFormData {
  id?: string;
  anilist_id: number;
  title: string;
  cover_image?: string;
  rating: number;
} 
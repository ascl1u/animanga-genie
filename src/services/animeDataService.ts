import { createClient } from '@/utils/supabase/client';
import { AnimeDetails as BaseAnimeDetails, getAnimeDetails as getAnilistAnimeDetails } from '@/utils/anilistClient';

// Extend the base AnimeDetails to include cache timestamp
export interface AnimeDetails extends BaseAnimeDetails {
  _cachedAt?: number;
}

// Interface for anime data as stored in Supabase
export interface SupabaseAnime {
  id: number;
  anilist_id: number;
  title: string; // JSON string of title object
  rating: number;
  genres: string; // JSON string of genres array
  tags: string; // JSON string of tags array
  popularity: number;
  format: string;
  episodes: number;
  duration: number;
  status: string;
  year: number;
  description: string;
  image_url: string; // URL to cover image
  relations?: string; // JSON string of relations array
  studios?: string; // JSON string of studios array
  created_at: string;
  updated_at: string;
}

// Interface for parsed anime data from Supabase
export interface ParsedSupabaseAnime {
  id: number;
  anilist_id: number;
  title: {
    romaji: string;
    english: string;
    native: string;
  };
  rating: number;
  genres: string[];
  tags: {
    id: number;
    name: string;
    rank: number;
    category: string;
  }[];
  popularity: number;
  format: string;
  episodes: number;
  duration: number;
  status: string;
  year: number;
  description: string;
  coverImage: {
    medium: string;
    large: string;
  };
  relations?: {
    id: number;
    relationType: string;
    node: {
      id: number;
      title: {
        romaji: string;
        english: string;
      };
      type: string;
      format: string;
    };
  }[];
  studios?: {
    id: number;
    isMain: boolean;
    node: {
      id: number;
      name: string;
    };
  }[];
}

/**
 * Simple in-memory cache for anime details
 */
class AnimeCache {
  private cache: Map<number, AnimeDetails> = new Map();
  private maxSize: number;
  private expiryTime: number; // milliseconds

  constructor(maxSize: number = 1000, expiryTimeMs: number = 3600000) { // Default 1 hour cache
    this.maxSize = maxSize;
    this.expiryTime = expiryTimeMs;
  }

  get(id: number): AnimeDetails | undefined {
    const entry = this.cache.get(id);
    if (!entry) return undefined;

    // Check if entry has expired
    const now = Date.now();
    if (entry._cachedAt && (now - entry._cachedAt > this.expiryTime)) {
      this.cache.delete(id);
      return undefined;
    }

    return entry;
  }

  set(id: number, details: AnimeDetails): void {
    // Add timestamp to the cached entry
    const detailsWithTimestamp = {
      ...details,
      _cachedAt: Date.now()
    };

    // Evict oldest entry if we're at capacity
    if (this.cache.size >= this.maxSize) {
      const oldestKey = this.findOldestEntry();
      if (oldestKey) this.cache.delete(oldestKey);
    }

    this.cache.set(id, detailsWithTimestamp);
  }

  private findOldestEntry(): number | undefined {
    let oldestKey: number | undefined;
    let oldestTime = Infinity;

    for (const [key, value] of this.cache.entries()) {
      const cachedAt = value._cachedAt || 0;
      if (cachedAt < oldestTime) {
        oldestTime = cachedAt;
        oldestKey = key;
      }
    }

    return oldestKey;
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }
}

/**
 * Anime Data Service
 * 
 * This service provides anime data from Supabase with fallback to AniList API
 * when data is not available in our database.
 */
class AnimeDataService {
  private supabase = createClient();
  private cache = new AnimeCache();

  /**
   * Convert Supabase anime record to AnimeDetails format
   * @param anime Raw Supabase anime record
   */
  private parseSupabaseAnime(anime: SupabaseAnime): AnimeDetails {
    try {
      // Parse JSON strings to objects
      const title = JSON.parse(anime.title);
      const genres = JSON.parse(anime.genres);
      const tags = JSON.parse(anime.tags);
      const relations = anime.relations ? JSON.parse(anime.relations) : undefined;
      const studios = anime.studios ? JSON.parse(anime.studios) : undefined;

      return {
        id: anime.anilist_id,
        title: {
          romaji: title.romaji || '',
          english: title.english || '',
          native: title.native || ''
        },
        genres: genres,
        tags: tags,
        coverImage: {
          medium: anime.image_url,
          large: anime.image_url // Since we store large URL, use it for both
        },
        format: anime.format,
        seasonYear: anime.year,
        averageScore: anime.rating * 10, // Convert back to 0-100 scale for consistency
        popularity: anime.popularity,
        episodes: anime.episodes,
        duration: anime.duration,
        status: anime.status,
        description: anime.description,
        relations: relations ? { edges: relations } : undefined,
        studios: studios ? { edges: studios } : undefined
      };
    } catch (error) {
      console.error('[ANIME] Error parsing Supabase anime data:', error);
      throw new Error('Failed to parse anime data from Supabase');
    }
  }

  /**
   * Get anime details by Anilist ID, using Supabase first with API fallback
   * @param animeId Anilist ID of the anime
   */
  async getAnimeDetails(animeId: number): Promise<AnimeDetails | null> {
    try {
      // First check cache
      const cachedAnime = this.cache.get(animeId);
      if (cachedAnime) {
        console.log(`[ANIME] Cache hit for anime ID: ${animeId}`);
        return cachedAnime;
      }

      console.log(`[ANIME] Fetching details for anime ID: ${animeId}`);

      // Try to get from Supabase first
      const { data: animeData, error } = await this.supabase
        .from('anime')
        .select('*')
        .eq('anilist_id', animeId)
        .single();

      if (error) {
        console.log(`[ANIME] Error fetching from Supabase: ${error.message}. Falling back to AniList API.`);
      }

      if (animeData) {
        console.log(`[ANIME] Found anime in Supabase: ${JSON.parse(animeData.title).english || JSON.parse(animeData.title).romaji}`);
        const parsedAnime = this.parseSupabaseAnime(animeData as SupabaseAnime);
        
        // Cache the result
        this.cache.set(animeId, parsedAnime);
        
        return parsedAnime;
      }

      // Fallback to AniList API
      console.log(`[ANIME] Anime not found in Supabase, falling back to AniList API`);
      const apiAnime = await getAnilistAnimeDetails(animeId);
      
      if (apiAnime) {
        // Cache the API result
        this.cache.set(animeId, apiAnime);
        
        // Consider adding to Supabase in the background
        this.addAnimeToSupabase(apiAnime).catch(err => {
          console.error(`[ANIME] Failed to add anime to Supabase: ${err.message}`);
        });
      }
      
      return apiAnime;
    } catch (error) {
      console.error(`[ANIME] Error fetching anime details for ID ${animeId}:`, error);
      return null;
    }
  }

  /**
   * Get multiple anime details by their IDs
   * @param animeIds Array of anime IDs
   * @param progressCallback Optional callback for progress updates
   */
  async getMultipleAnimeDetails(
    animeIds: number[],
    progressCallback?: (completed: number, total: number) => void
  ): Promise<Map<number, AnimeDetails>> {
    const results = new Map<number, AnimeDetails>();
    let completed = 0;
    
    // Get unique IDs to avoid duplicates
    const uniqueIds = [...new Set(animeIds)];
    console.log(`[ANIME] Fetching details for ${uniqueIds.length} unique anime IDs`);
    
    // First check cache for all IDs
    const cachedResults = new Map<number, AnimeDetails>();
    const uncachedIds: number[] = [];
    
    // Check cache first to avoid unnecessary database queries
    for (const animeId of uniqueIds) {
      const cachedAnime = this.cache.get(animeId);
      if (cachedAnime) {
        cachedResults.set(animeId, cachedAnime);
      } else {
        uncachedIds.push(animeId);
      }
    }
    
    // If we found any cached results, add them immediately
    if (cachedResults.size > 0) {
      console.log(`[ANIME] Found ${cachedResults.size}/${uniqueIds.length} anime in cache`);
      
      for (const [animeId, anime] of cachedResults.entries()) {
        results.set(animeId, anime);
      }
      
      completed = cachedResults.size;
      if (progressCallback) {
        progressCallback(completed, uniqueIds.length);
      }
    }
    
    // If all anime were in cache, return early
    if (uncachedIds.length === 0) {
      return results;
    }
    
    // Fetch remaining anime from Supabase in optimized batches
    try {
      // Split into batches if the number of IDs is large to avoid query limits
      const BATCH_SIZE = 500; // Maximum number of IDs to query at once
      const batches: number[][] = [];
      
      for (let i = 0; i < uncachedIds.length; i += BATCH_SIZE) {
        batches.push(uncachedIds.slice(i, i + BATCH_SIZE));
      }
      
      console.log(`[ANIME] Splitting ${uncachedIds.length} uncached IDs into ${batches.length} batches`);
      
      // Process each batch of Supabase queries
      for (const batch of batches) {
        const { data: supabaseAnime, error } = await this.supabase
          .from('anime')
          .select('*')
          .in('anilist_id', batch);
        
        if (error) {
          console.error(`[ANIME] Error fetching bulk anime from Supabase:`, error);
        } else if (supabaseAnime && supabaseAnime.length > 0) {
          // Map the results by anime ID for quick lookup
          const animeMap = new Map<number, SupabaseAnime>();
          
          for (const anime of supabaseAnime) {
            animeMap.set((anime as SupabaseAnime).anilist_id, anime as SupabaseAnime);
          }
          
          console.log(`[ANIME] Found ${animeMap.size}/${batch.length} anime in Supabase (batch)`);
          
          // Process all the anime we found in Supabase
          for (const [anilistId, anime] of animeMap.entries()) {
            try {
              const parsedAnime = this.parseSupabaseAnime(anime);
              results.set(anilistId, parsedAnime);
              this.cache.set(anilistId, parsedAnime);
              completed++;
            } catch (error) {
              console.error(`[ANIME] Error parsing anime ${anilistId}:`, error);
            }
          }
          
          if (progressCallback) {
            progressCallback(completed, uniqueIds.length);
          }
        }
      }
    } catch (error) {
      console.error(`[ANIME] Error in bulk Supabase fetch:`, error);
    }
    
    // Determine which IDs are still missing after Supabase query
    const remainingIds = uniqueIds.filter(id => !results.has(id));
    
    if (remainingIds.length > 0) {
      console.log(`[ANIME] Still need to fetch ${remainingIds.length} anime from AniList API`);
      
      // Create a controlled queue for API requests to avoid rate limiting
      const API_BATCH_SIZE = 10; // Process 10 anime at a time
      const API_DELAY = 1000; // 1 second delay between batches
      
      // Process API requests in small batches with delays
      for (let i = 0; i < remainingIds.length; i += API_BATCH_SIZE) {
        const batchIds = remainingIds.slice(i, i + API_BATCH_SIZE);
        const batchPromises: Promise<void>[] = [];
        
        for (const animeId of batchIds) {
          batchPromises.push((async () => {
            try {
              // Use getAnimeDetails which handles individual caching and API fallback
              const animeDetails = await this.getAnimeDetails(animeId);
              if (animeDetails) {
                results.set(animeId, animeDetails);
              }
            } catch (error) {
              console.error(`[ANIME] Failed to get details for anime ID ${animeId}:`, error);
            } finally {
              completed++;
              if (progressCallback) {
                progressCallback(completed, uniqueIds.length);
              }
            }
          })());
        }
        
        // Wait for the current batch to complete
        await Promise.all(batchPromises);
        
        // Only add delay if we have more batches to process
        if (i + API_BATCH_SIZE < remainingIds.length) {
          await new Promise(resolve => setTimeout(resolve, API_DELAY));
        }
      }
    }
    
    console.log(`[ANIME] Completed fetching ${results.size}/${uniqueIds.length} anime details`);
    return results;
  }

  /**
   * Add anime to Supabase database in the background
   * @param anime Anime details to add
   */
  private async addAnimeToSupabase(anime: AnimeDetails): Promise<void> {
    try {
      console.log(`[ANIME] Adding anime to Supabase: ${anime.title.english || anime.title.romaji}`);
      
      // Format anime data for Supabase
      const formattedAnime = {
        anilist_id: anime.id,
        title: JSON.stringify({
          romaji: anime.title.romaji,
          english: anime.title.english,
          native: anime.title.native
        }),
        rating: anime.averageScore / 10, // Convert to 0-10 scale
        genres: JSON.stringify(anime.genres),
        tags: JSON.stringify(anime.tags),
        popularity: anime.popularity,
        format: anime.format,
        episodes: anime.episodes,
        duration: anime.duration,
        status: anime.status,
        year: anime.seasonYear,
        description: anime.description,
        image_url: anime.coverImage.large, // Always use large image URL
        relations: anime.relations ? JSON.stringify(anime.relations.edges) : null,
        studios: anime.studios ? JSON.stringify(anime.studios.edges) : null
      };
      
      // Insert into Supabase with upsert (update if exists)
      const { error } = await this.supabase
        .from('anime')
        .upsert(formattedAnime, {
          onConflict: 'anilist_id'
        });
      
      if (error) {
        console.error(`[ANIME] Error adding anime to Supabase:`, error);
      } else {
        console.log(`[ANIME] Successfully added anime to Supabase: ${anime.id}`);
      }
    } catch (error) {
      console.error(`[ANIME] Error in addAnimeToSupabase:`, error);
    }
  }

  /**
   * Fetch trending anime from the database
   * @param limit Number of anime to fetch
   */
  async getTrendingAnime(limit: number = 20): Promise<AnimeDetails[]> {
    try {
      const { data: animeData, error } = await this.supabase
        .from('anime')
        .select('*')
        .order('popularity', { ascending: false })
        .limit(limit);
      
      if (error) {
        console.error(`[ANIME] Error fetching trending anime:`, error);
        return [];
      }
      
      const parsed = animeData.map(anime => this.parseSupabaseAnime(anime as SupabaseAnime));
      
      // Cache results
      for (const anime of parsed) {
        this.cache.set(anime.id, anime);
      }
      
      return parsed;
    } catch (error) {
      console.error(`[ANIME] Error in getTrendingAnime:`, error);
      return [];
    }
  }

  /**
   * Search for anime by title
   * @param query Search query string
   * @param limit Maximum number of results
   */
  async searchAnime(query: string, limit: number = 10): Promise<AnimeDetails[]> {
    try {
      // We need to search the JSON title field for matches
      // This can be complex depending on Supabase setup
      // Here's a simple but limited implementation
      const { data: animeData, error } = await this.supabase
        .from('anime')
        .select('*')
        .or(`title->english.ilike.%${query}%,title->romaji.ilike.%${query}%`)
        .order('popularity', { ascending: false })
        .limit(limit);
      
      if (error) {
        console.error(`[ANIME] Error searching anime:`, error);
        return [];
      }
      
      const parsed = animeData.map(anime => this.parseSupabaseAnime(anime as SupabaseAnime));
      
      // Cache results
      for (const anime of parsed) {
        this.cache.set(anime.id, anime);
      }
      
      return parsed;
    } catch (error) {
      console.error(`[ANIME] Error in searchAnime:`, error);
      return [];
    }
  }

  /**
   * Clear the cache
   */
  clearCache(): void {
    this.cache.clear();
    console.log(`[ANIME] Cache cleared`);
  }
}

// Export a singleton instance
export const animeDataService = new AnimeDataService(); 
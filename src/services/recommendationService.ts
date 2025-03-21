import { 
  loadModel, 
  loadModelMetadata, 
  loadModelMappings,
  runModelInference,
  isModelLoaded
} from './modelService';
import { logOnnxServiceState } from './onnxModelService';
import { getUserWatchHistory } from './watchHistoryService';
import { getLocalWatchHistory } from './localStorageService';
import { AnimeWatchHistoryItem } from '@/types/watchHistory';
import { getAnimeDetails } from '@/utils/anilistClient';
import { collaborativeFilteringService, CollaborativeRecommendation } from './collaborativeFilteringService';
import { createClient } from '@/utils/supabase/client';

export interface AnimeData {
  id: number;
  title: string;
  score: number;
  averageScore?: number; // Original AniList score (out of 100)
  cover_image?: string;
  genres?: string[];
  description?: string; // Anime description from AniList
  _debugInfo?: {
    userEmbedding?: {
      dimension: number;
      sample?: number[];
    };
    negativePreferences?: {
      genres?: string[];
      tags?: string[];
      relatedAnime?: { id: number; title: string; penalty: number; reason: string }[];
    };
    collaborativeInfo?: {
      contributorCount: number;
      topContributors?: string;
    };
  };
}

export interface RecommendationResult {
  recommendations: AnimeData[];
  status: 'success' | 'loading' | 'error';
  error?: string;
  userWatchHistory?: AnimeWatchHistoryItem[]; // For debugging
  debugInfo?: {
    mappingKeys?: string[];
    watchedAnimeIds?: number[];
    mappedAnimeIds?: number[];
    animeIdMapSuccess?: boolean;
    genresUsed?: string[];
    tagsUsed?: string[];
    negativePreferences?: {
      genres?: string[];
      tags?: string[];
      relatedAnimeIds?: {id: number, weight: number}[];
    };
    userEmbedding?: {
      dimension: number;
      sample?: number[];
      method?: string;
    };
    collaborativeFiltering?: {
      used: boolean;
      similarUserCount?: number;
      blendedRecommendations?: boolean;
    };
  };
}

// User embedding calculation approach
enum UserEmbeddingMethod {
  COLLABORATIVE_FILTERING = 'collaborative_filtering', // Implemented!
  PREFERENCE_VECTOR = 'preference_vector',            // Main current approach
  DEFAULT = 'default'                                 // Fallback to a fixed index
}

/**
 * Calculate user embedding based on collaborative filtering
 * This implements matrix factorization to find similar users
 */
async function calculateCollaborativeFilteringEmbedding(
  userId: string,
  watchHistory: AnimeWatchHistoryItem[]
): Promise<{
  embeddingVector: number[];
  watchedIndices: number[];
  collaborativeRecommendations?: CollaborativeRecommendation[];
}> {
  console.log('[RECOMMENDATION] Using collaborative filtering for user', userId);
  
  try {
    // Train the collaborative filtering model if needed
    await collaborativeFilteringService.trainModel();
    
    // Get user factors from the trained model
    const userFactors = await collaborativeFilteringService.getUserFactors(userId);
    
    if (!userFactors) {
      console.log('[RECOMMENDATION] No collaborative filtering factors available for user, falling back to preference vector');
      return {
        embeddingVector: [0], // Will be replaced with preference vector
        watchedIndices: watchHistory.map(item => item.anilist_id)
      };
    }
    
    // Get collaborative recommendations
    const collaborativeRecommendations = await collaborativeFilteringService.getRecommendations(
      userId,
      watchHistory,
      20 // Get more recommendations than needed to allow for filtering
    );
    
    console.log(`[RECOMMENDATION] Generated ${collaborativeRecommendations.length} collaborative filtering recommendations`);
    
    return {
      embeddingVector: userFactors,
      watchedIndices: watchHistory.map(item => item.anilist_id),
      collaborativeRecommendations
    };
  } catch (error) {
    console.error('[RECOMMENDATION] Error in collaborative filtering:', error);
    return {
      embeddingVector: [0], // Will be replaced with preference vector
      watchedIndices: watchHistory.map(item => item.anilist_id)
    };
  }
}

/**
 * Calculate a user embedding based on watch history
 * @param watchHistory User's anime watch history
 * @param animeToIdx Mapping from anime IDs to model indices
 * @returns A vector representing the user's preferences
 */
async function calculateUserEmbedding(
  watchHistory: AnimeWatchHistoryItem[],
  animeToIdx: Record<string, number>
): Promise<{
  embeddingVector: number[];
  method: UserEmbeddingMethod;
  watchedIndices: number[];
}> {
  console.log('[RECOMMENDATION] Calculating user embedding...');
  
  if (!watchHistory || watchHistory.length === 0) {
    console.log('[RECOMMENDATION] No watch history available, using default embedding method');
    return {
      embeddingVector: [0], // Default embedding
      method: UserEmbeddingMethod.DEFAULT,
      watchedIndices: []
    };
  }
  
  // Map watched anime to indices
  const watchedIndices: number[] = [];
  for (const item of watchHistory) {
    const animeId = item.anilist_id.toString();
    // Try different formats of the ID to match mapping
    const possibleIds = [
      animeId,
      `anilist-${animeId}`,
      `anime-${animeId}`
    ];
    
    // Find a matching ID format
    for (const id of possibleIds) {
      if (animeToIdx[id] !== undefined) {
        watchedIndices.push(animeToIdx[id]);
        break;
      }
    }
  }
  
  if (watchedIndices.length === 0) {
    console.log('[RECOMMENDATION] Could not map any watched anime to indices, using default embedding');
    return {
      embeddingVector: [0], // Default embedding
      method: UserEmbeddingMethod.DEFAULT,
      watchedIndices: []
    };
  }
  
  // For now, use a preference vector approach
  // This creates a representation of the user based on their ratings and watched anime
  
  // First, normalize ratings to be between 0 and 1
  const normalizedRatings: { animeIdx: number; rating: number }[] = [];
  
  for (let i = 0; i < watchHistory.length; i++) {
    const item = watchHistory[i];
    // Only include items that we have mapped indices for
    const animeIdx = watchedIndices[i];
    if (animeIdx !== undefined) {
      // Normalize rating (1-10) to a 0-1 scale, where 5 is neutral (0.5)
      const normalizedRating = (item.rating - 1) / 9;
      normalizedRatings.push({ animeIdx, rating: normalizedRating });
    }
  }
  
  // Sort by rating (highest first)
  normalizedRatings.sort((a, b) => b.rating - a.rating);
  
  // Create a vector with weighted preferences
  // For now, this is a simple approach that we can refine later
  const preferenceVector = new Array(normalizedRatings.length * 2).fill(0);
  
  // Fill the vector with anime indices and their normalized ratings
  normalizedRatings.forEach((item, i) => {
    preferenceVector[i*2] = item.animeIdx;
    preferenceVector[i*2 + 1] = item.rating;
  });
  
  console.log(`[RECOMMENDATION] Created user embedding with ${preferenceVector.length} dimensions using ${UserEmbeddingMethod.PREFERENCE_VECTOR} method`);
  
  return {
    embeddingVector: preferenceVector,
    method: UserEmbeddingMethod.PREFERENCE_VECTOR,
    watchedIndices
  };
}

/**
 * Extract negative preferences from low-rated anime
 * @param watchHistory User's anime watch history
 * @param threshold Rating threshold below which to consider negative (typically 5/10)
 */
async function extractNegativePreferences(
  watchHistory: AnimeWatchHistoryItem[],
  threshold: number = 5
): Promise<{
  genres: Set<string>;
  tags: Set<string>;
  animeIds: number[];
  relatedAnimeIds: Map<number, number>; // Related anime IDs with their penalty weight (0-1)
}> {
  console.log(`[RECOMMENDATION] Extracting negative preferences with threshold ${threshold}...`);
  
  const negativeGenres = new Set<string>();
  const negativeTags = new Set<string>();
  const negativeAnimeIds: number[] = [];
  const relatedAnimeIds = new Map<number, number>(); // Store related anime IDs with penalty weight
  
  if (!watchHistory || watchHistory.length === 0) {
    return { genres: negativeGenres, tags: negativeTags, animeIds: negativeAnimeIds, relatedAnimeIds };
  }
  
  // Get low-rated anime
  const lowRatedAnime = watchHistory.filter(item => item.rating < threshold);
  console.log(`[RECOMMENDATION] Found ${lowRatedAnime.length} anime rated below ${threshold}/10`);
  
  if (lowRatedAnime.length === 0) {
    return { genres: negativeGenres, tags: negativeTags, animeIds: negativeAnimeIds, relatedAnimeIds };
  }
  
  // Process each low-rated anime
  const animeDetailsPromises: Promise<void>[] = [];
  
  for (const item of lowRatedAnime) {
    negativeAnimeIds.push(item.anilist_id);
    
    animeDetailsPromises.push((async () => {
      try {
        const animeDetails = await getAnimeDetails(item.anilist_id);
        
        if (animeDetails) {
          // Enhanced: Calculate negative weight with exponential penalty for lower ratings
          // For ratings from 1-5, we exponentially increase the penalty
          // Example: Rating 1 = 0.85, Rating 2 = 0.7, Rating 3 = 0.5, Rating 4 = 0.3, Rating 5 = 0.1
          const normalizedRating = item.rating / 10; // 0.1-0.5 for ratings 1-5
          const negativeWeight = Math.pow(1 - normalizedRating, 2); // Exponential penalty
          
          if (negativeWeight > 0) {
            // Add weight-filtered genres to negative set
            // Only add the most influential genres for very low ratings
            const genreLimit = Math.ceil(animeDetails.genres.length * negativeWeight);
            animeDetails.genres.slice(0, genreLimit).forEach(genre => negativeGenres.add(genre));
            
            // Add most influential tags to negative set
            const sortedTags = [...animeDetails.tags]
              .sort((a, b) => b.rank - a.rank)
              .slice(0, Math.ceil(6 * negativeWeight)); // More tags from lowest rated anime
            
            sortedTags.forEach(tag => negativeTags.add(tag.name));
            
            // Find related anime (sequels, prequels, etc.)
            await findRelatedAnime(item.anilist_id, relatedAnimeIds, negativeWeight);
            
            console.log(`[RECOMMENDATION] Added negative preferences from ${animeDetails.title.english || animeDetails.title.romaji} (${item.rating}/10): ${genreLimit} genres, ${sortedTags.length} tags, weight: ${negativeWeight.toFixed(2)}`);
          }
        }
      } catch (error) {
        console.error(`[RECOMMENDATION] Error fetching details for anime ${item.anilist_id}:`, error);
      }
    })());
  }
  
  // Wait for all anime details to be fetched
  await Promise.all(animeDetailsPromises);
  
  console.log(`[RECOMMENDATION] Extracted negative preferences: ${negativeGenres.size} genres, ${negativeTags.size} tags, ${relatedAnimeIds.size} related anime`);
  
  return {
    genres: negativeGenres,
    tags: negativeTags,
    animeIds: negativeAnimeIds,
    relatedAnimeIds
  };
}

/**
 * Find anime related to a given anime ID using pattern matching and API data
 * This helps identify sequels, prequels, and series related to low-rated anime
 * @param animeId The source anime ID
 * @param relatedAnimeIds Map to store related anime IDs with penalty weights
 * @param penaltyWeight The penalty weight to apply to related anime
 */
async function findRelatedAnime(
  animeId: number,
  relatedAnimeIds: Map<number, number>,
  penaltyWeight: number
): Promise<void> {
  try {
    console.log(`[RECOMMENDATION] Finding related anime for ${animeId}...`);
    
    // First, try to get relation data directly from AniList API
    const animeDetails = await getAnimeDetails(animeId);
    
    if (animeDetails?.relations?.edges && animeDetails.relations.edges.length > 0) {
      console.log(`[RECOMMENDATION] Found ${animeDetails.relations.edges.length} related anime from AniList API for ${animeId}`);
      
      // Process direct relations from AniList
      for (const relation of animeDetails.relations.edges) {
        if (relation.node && relation.node.id && relation.node.type === 'ANIME') {
          const relatedAnimeId = relation.node.id;
          // Skip if it's the same anime or already in our map
          if (relatedAnimeId === animeId || relatedAnimeIds.has(relatedAnimeId)) {
            continue;
          }
          
          let relationPenalty = penaltyWeight;
          
          // Adjust penalty based on relation type
          switch (relation.relationType) {
            case 'SEQUEL':
            case 'PREQUEL': 
              // Direct sequels/prequels get a higher penalty (90% of original)
              relationPenalty = penaltyWeight * 0.9;
              break;
            case 'SIDE_STORY':
            case 'SPIN_OFF':
            case 'ALTERNATIVE':
              // Side stories get a moderate penalty (70% of original)
              relationPenalty = penaltyWeight * 0.7;
              break;
            default:
              // Other relations get a lower penalty (50% of original)
              relationPenalty = penaltyWeight * 0.5;
          }
          
          // Add to our map
          relatedAnimeIds.set(relatedAnimeId, relationPenalty);
          console.log(`[RECOMMENDATION] Found related anime from API: ${relation.node.title?.english || relation.node.title?.romaji} (ID: ${relatedAnimeId}) relation: ${relation.relationType} with penalty ${relationPenalty.toFixed(2)}`);
        }
      }
      
      // If we found relations from the API, we can return
      if (relatedAnimeIds.size > 0) {
        return;
      }
    }
    
    // Fallback to title-based matching if API relation data is not available
    console.log(`[RECOMMENDATION] No relation data from API for ${animeId}, falling back to title matching`);
    
    // Use Supabase for title matching
    const supabase = createClient();
    
    // Fetch the source anime title for pattern matching
    const { data: sourceAnime, error: sourceError } = await supabase
      .from('anime')
      .select('anilist_id, title')
      .eq('anilist_id', animeId)
      .single();
    
    if (sourceError || !sourceAnime) {
      console.log(`[RECOMMENDATION] Could not find source anime ${animeId} for relation detection`);
      return;
    }
    
    // Extract series name for pattern matching
    const sourceTitle = sourceAnime.title?.english || sourceAnime.title?.romaji || '';
    if (!sourceTitle) return;
    
    // Remove season numbers, "season", "part" from title to get base series name
    const baseSeriesName = sourceTitle
      .replace(/\s+(season|series|part)\s+\d+/i, '')
      .replace(/\s+\d+(st|nd|rd|th)\s+season/i, '')
      .replace(/\s+S\d+/i, '')
      .replace(/\s+II|III|IV|V|VI/i, '')
      .replace(/\s+\d+$/i, '')
      .trim();
    
    if (baseSeriesName.length < 4) return; // Too short to be meaningful
    
    // Find potential related anime with similar titles
    const { data: relatedAnime, error: relatedError } = await supabase
      .from('anime')
      .select('anilist_id, title')
      .neq('anilist_id', animeId)
      .order('popularity', { ascending: false })
      .limit(50); // Limit to avoid excessive processing
    
    if (relatedError || !relatedAnime || relatedAnime.length === 0) {
      console.log(`[RECOMMENDATION] No potential related anime found for ${animeId}`);
      return;
    }
    
    // Calculate title similarity and detect related anime
    for (const anime of relatedAnime) {
      const relatedTitle = anime.title?.english || anime.title?.romaji || '';
      if (!relatedTitle) continue;
      
      // Skip if already in our map
      if (relatedAnimeIds.has(anime.anilist_id)) {
        continue;
      }
      
      // Check if titles are related (either contains the other or they share significant words)
      const isTitleRelated = 
        // Direct series relationship
        relatedTitle.includes(baseSeriesName) || 
        baseSeriesName.includes(relatedTitle) ||
        // Same franchise different naming
        (getSignificantWords(baseSeriesName).some(word => 
          relatedTitle.toLowerCase().includes(word.toLowerCase())) &&
         baseSeriesName.length > 8);
      
      if (isTitleRelated) {
        // Apply slightly reduced penalty to related anime (70% of original penalty)
        const relatedPenalty = penaltyWeight * 0.7;
        relatedAnimeIds.set(anime.anilist_id, relatedPenalty);
        console.log(`[RECOMMENDATION] Found related anime by title: ${relatedTitle} (ID: ${anime.anilist_id}) with penalty ${relatedPenalty.toFixed(2)}`);
      }
    }
  } catch (error) {
    console.error(`[RECOMMENDATION] Error finding related anime for ${animeId}:`, error);
  }
}

/**
 * Extract significant words from a title for matching
 * @param title The anime title
 * @returns Array of significant words
 */
function getSignificantWords(title: string): string[] {
  // Remove common words that don't help with matching
  const commonWords = ['the', 'a', 'an', 'of', 'in', 'on', 'at', 'and', 'or', 'to'];
  
  return title.split(/\s+/)
    .filter(word => 
      word.length > 3 && // Only words longer than 3 characters
      !commonWords.includes(word.toLowerCase()) && // Not common words
      !/^\d+$/.test(word) // Not just numbers
    );
}

/**
 * Interface for anime data from Supabase
 */
interface SupabaseAnimeData {
  anilist_id: number;
  title?: Record<string, string>; // JSON object with title variants
  rating?: number;
  genres?: string[];
  tags?: unknown[]; // Array of tags
  popularity?: number;
  format?: string;
  episodes?: number;
  year?: number;
  description?: string;
  image_url?: string;
}

/**
 * Fetch anime data from the Supabase database
 * This allows us to consider all anime in the database for recommendations
 * @returns Promise with the fetched anime data
 */
async function fetchAnimeFromSupabase(limit: number = 5000): Promise<{
  animeIndices: number[];
  animeMapping: Record<number, number>;
  success: boolean;
  animeData?: SupabaseAnimeData[]; // Use the specific interface instead of any[]
}> {
  const supabase = createClient();
  const animeIndices: number[] = [];
  const animeMapping: Record<number, number> = {};
  
  try {
    console.log(`[RECOMMENDATION] Fetching up to ${limit} anime from Supabase database`);
    
    const { data: animeData, error } = await supabase
      .from('anime')
      .select('anilist_id')
      .order('popularity', { ascending: false })
      .limit(limit);
    
    if (error) {
      console.error('[RECOMMENDATION] Error fetching anime from Supabase:', error);
      return { animeIndices, animeMapping, success: false };
    }
    
    console.log(`[RECOMMENDATION] Fetched ${animeData.length} anime from Supabase`);
    return { 
      animeIndices, 
      animeMapping, 
      success: true,
      animeData // Return the raw data so we can process it with mappings
    };
  } catch (error) {
    console.error('[RECOMMENDATION] Error fetching anime from Supabase:', error);
    return { animeIndices, animeMapping, success: false };
  }
}

/**
 * Get recommendations for a user based on their watch history, preferred genres, and tags
 */
export async function getRecommendations(
  userId: string,
  preferredGenres: string[] = [],
  preferredTags: string[] = [],
  limit: number = 10
): Promise<RecommendationResult> {
  try {
    logOnnxServiceState();
    
    // Prepare the debug info object
    const debugInfo: RecommendationResult['debugInfo'] = {
      watchedAnimeIds: [],
      mappedAnimeIds: [],
      animeIdMapSuccess: false,
      genresUsed: [],
      tagsUsed: [],
      negativePreferences: {
        genres: [],
        tags: []
      },
      userEmbedding: {
        dimension: 0,
        sample: [],
        method: 'unknown'
      },
      collaborativeFiltering: {
        used: false
      }
    };

    // Check if the model is loaded
    if (!isModelLoaded()) {
      console.log('[RECOMMENDATION] Model not loaded, loading now...');
      await loadModel();
    }

    // Check for user authentication status
    const supabase = createClient();
    const { data: { user } } = await supabase.auth.getUser();
    
    // Log authentication status and userId for debugging
    console.log(`[RECOMMENDATION] User authentication check: user present=${!!user}, userId passed=${userId}`);
    
    // Get user watch history
    let userWatchHistory: AnimeWatchHistoryItem[] | undefined;
    
    if (user) {
      // Authenticated user - get from database
      console.log('[RECOMMENDATION] User is authenticated, getting watch history from database');
      userWatchHistory = await getUserWatchHistory();
      console.log(`[RECOMMENDATION] Fetched ${userWatchHistory?.length || 0} items from database for user ${user.id}`);
    } else {
      // Non-authenticated user - get from localStorage
      console.log('[RECOMMENDATION] User is not authenticated, getting watch history from localStorage');
      userWatchHistory = getLocalWatchHistory();
      console.log(`[RECOMMENDATION] Fetched ${userWatchHistory?.length || 0} items from localStorage`);
    }
    
    console.log(`[RECOMMENDATION] User authentication status: ${user ? 'Authenticated' : 'Unauthenticated'}`);
    console.log(`[RECOMMENDATION] Watch history source: ${user ? 'Database' : 'LocalStorage'}`);
    console.log(`[RECOMMENDATION] Watch history count: ${userWatchHistory ? userWatchHistory.length : 0}`);
    
    // Log localStorage state for authenticated users (to check for potential leakage)
    if (user && typeof window !== 'undefined') {
      const localWatchHistory = localStorage.getItem('animanga-genie-watch-history');
      const localRecommendations = localStorage.getItem('animanga-genie-recommendations');
      console.log(`[RECOMMENDATION] For authenticated user, localStorage state: watchHistory=${!!localWatchHistory}, recommendations=${!!localRecommendations}`);
    }
    
    if (userWatchHistory && userWatchHistory.length > 0) {
      console.log(`[RECOMMENDATION] First watch history item: anilist_id=${userWatchHistory[0].anilist_id}, title=${userWatchHistory[0].title}`);
    }
    
    if (!userWatchHistory || userWatchHistory.length === 0) {
      console.log('[RECOMMENDATION] No watch history found, falling back to preferred genres');
      // Fallback implementation for users with no watch history would go here
      return {
        recommendations: [],
        status: 'error',
        error: 'No watch history found and genre-based recommendations not implemented yet',
        debugInfo
      };
    }
    
    console.log(`[RECOMMENDATION] Found ${userWatchHistory.length} items in watch history`);
    debugInfo.watchedAnimeIds = userWatchHistory.map(item => item.anilist_id);
    
    // Extract negative preferences (low-rated anime)
    const negativePreferences = await extractNegativePreferences(userWatchHistory);
    
    // Update debug info with negative preferences
    debugInfo.negativePreferences = {
      genres: Array.from(negativePreferences.genres),
      tags: Array.from(negativePreferences.tags)
    };
    
    // Add related anime IDs to debug info if available
    if (negativePreferences.relatedAnimeIds.size > 0) {
      debugInfo.negativePreferences.relatedAnimeIds = Array.from(negativePreferences.relatedAnimeIds.entries())
        .map(([id, weight]) => ({ id, weight }));
    }

    // Check if model is already loaded to avoid reloading
    const modelIsLoaded = isModelLoaded();
    console.log(`[RECOMMENDATION] Model already loaded: ${modelIsLoaded}`);
    
    // Load model and mappings - only load the model if not already loaded
    try {
      // Try loading mappings first to check if files exist before loading the model
      const mappingsData = await loadModelMappings();
      debugInfo.mappingKeys = Object.keys(mappingsData);
      
      // Load model metadata which includes required model parameters
      await loadModelMetadata();
      
      // Only load the model if it's not already loaded
      if (!modelIsLoaded) {
        await loadModel();
      }
      
      console.log('[RECOMMENDATION] Got model, mappings and metadata');
      
      // Check if the mappings have the expected structure
      if (!mappingsData) {
        throw new Error('Mappings data is missing');
      }
      
      // Log the structure of mappings for debugging
      const mappingKeys = Object.keys(mappingsData);
      console.log('[RECOMMENDATION] Mappings keys:', mappingKeys);
      
      // Handle different mapping structures
      const idxToAnime = mappingsData.idx_to_anime || {};
      const animeToIdx = mappingsData.anime_to_idx || {};
      const genreToIdx = mappingsData.genre_to_idx || {};
      const tagToIdx = mappingsData.tag_to_idx || {};
      
      if (!animeToIdx || Object.keys(animeToIdx).length === 0) {
        console.error('[RECOMMENDATION] anime_to_idx mapping is missing or empty');
      } else {
        console.log('[RECOMMENDATION] anime_to_idx has', Object.keys(animeToIdx).length, 'entries');
        
        // Log a few sample entries to debug mapping format
        const animeIdSamples = Object.keys(animeToIdx).slice(0, 5);
        console.log('[RECOMMENDATION] Sample anime IDs in mapping:', animeIdSamples);
      }
      
      // Try collaborative filtering approach first
      let userEmbedding;
      let collaborativeRecommendations: CollaborativeRecommendation[] | undefined;
      
      if (userWatchHistory.length >= 5) {
        console.log('[RECOMMENDATION] Trying collaborative filtering approach');
        
        // Calculate user embedding using collaborative filtering
        const collaborativeResult = await calculateCollaborativeFilteringEmbedding(
          userId,
          userWatchHistory
        );
        
        // Store collaborative recommendations if available
        collaborativeRecommendations = collaborativeResult.collaborativeRecommendations;
        
        // If we got collaborative recommendations, use those
        if (collaborativeRecommendations && collaborativeRecommendations.length > 0) {
          console.log('[RECOMMENDATION] Using collaborative filtering recommendations');
          
          // Set flag in debug info
          debugInfo.collaborativeFiltering = {
            used: true,
            similarUserCount: collaborativeRecommendations.reduce(
              (count, rec) => count + rec.contributors.length, 0
            ) / collaborativeRecommendations.length,
            blendedRecommendations: false
          };
          
          // Use collaborative embedding
          userEmbedding = {
            embeddingVector: collaborativeResult.embeddingVector,
            method: UserEmbeddingMethod.COLLABORATIVE_FILTERING,
            watchedIndices: collaborativeResult.watchedIndices
          };
          
          // Save user embedding for debugging
          debugInfo.userEmbedding = {
            dimension: userEmbedding.embeddingVector.length,
            sample: userEmbedding.embeddingVector.slice(0, 10), // Just show a sample
            method: UserEmbeddingMethod.COLLABORATIVE_FILTERING
          };
        } else {
          // Fall back to preference vector approach
          console.log('[RECOMMENDATION] Collaborative filtering did not return recommendations, falling back to preference vector');
          userEmbedding = await calculateUserEmbedding(
            userWatchHistory,
            animeToIdx
          );
          
          debugInfo.collaborativeFiltering = {
            used: false,
            blendedRecommendations: false
          };
        }
      } else {
        // Not enough watch history for collaborative filtering
        console.log('[RECOMMENDATION] Not enough watch history for collaborative filtering, using preference vector');
        userEmbedding = await calculateUserEmbedding(
          userWatchHistory,
          animeToIdx
        );
        
        debugInfo.collaborativeFiltering = {
          used: false,
          blendedRecommendations: false
        };
      }
      
      // Save user embedding for debugging if not already set
      if (!debugInfo.userEmbedding) {
        debugInfo.userEmbedding = {
          dimension: userEmbedding.embeddingVector.length,
          sample: userEmbedding.embeddingVector.slice(0, 10), // Just show a sample
          method: userEmbedding.method
        };
      }
      
      // Use a default user index of 0 since we don't have user mappings
      // This will be a fallback if we can't use our embedding approach
      const userIndex = 0;
      
      // Get the total number of anime in the model
      const totalAnimeCount = Object.keys(idxToAnime).length;
      console.log(`[RECOMMENDATION] Total anime in model: ${totalAnimeCount}`);
      
      // Fetch all anime from the Supabase database
      console.log(`[RECOMMENDATION] Fetching all anime from Supabase database`);
      
      // Get all anime from the database
      let candidateAnimeIndices: number[] = [];
      const animeIdToAnilistId: Record<number, number> = {};
      
      // Fetch anime data from Supabase
      const { success, animeData } = await fetchAnimeFromSupabase();
      
      if (success && animeData && animeData.length > 0) {
        console.log(`[RECOMMENDATION] Successfully retrieved ${animeData.length} anime from Supabase`);
        
        // Find the model indices for each of the anime IDs from Supabase
        for (const anime of animeData) {
          const anilistId = anime.anilist_id;
          const possibleIds = [
            anilistId.toString(),
            `anilist-${anilistId}`,
            `anime-${anilistId}`
          ];
          
          // Try different formats of the ID to match the mapping
          for (const id of possibleIds) {
            const animeIdx = animeToIdx[id];
            if (animeIdx !== undefined) {
              candidateAnimeIndices.push(animeIdx);
              animeIdToAnilistId[animeIdx] = anilistId;
              break;
            }
          }
        }
        
        console.log(`[RECOMMENDATION] Mapped ${candidateAnimeIndices.length} out of ${animeData.length} Supabase anime to model indices`);

        // If we didn't find any matches, fall back to the original approach
        if (candidateAnimeIndices.length === 0) {
          console.log('[RECOMMENDATION] Could not map any Supabase anime IDs to model indices, using fallback');
          fallbackToModelIndices();
        }
      } else {
        // Fall back to the original logic if Supabase query failed
        console.log('[RECOMMENDATION] Supabase query failed or returned no results, using fallback');
        fallbackToModelIndices();
      }
      
      // Define a function to use the fallback approach with model indices
      function fallbackToModelIndices() {
        const maxCandidates = Math.min(100, totalAnimeCount);
        
        // Generate a diverse set of candidate indices
        const allAnimeIndices = Object.keys(idxToAnime).map(key => parseInt(key));
        
        if (allAnimeIndices.length > 0) {
          // If we have a proper mapping, use a diverse sampling approach
          const step = Math.max(1, Math.floor(allAnimeIndices.length / maxCandidates));
          
          // Take evenly spaced samples from the full range
          for (let i = 0; i < allAnimeIndices.length && candidateAnimeIndices.length < maxCandidates; i += step) {
            candidateAnimeIndices.push(allAnimeIndices[i]);
          }
          
          // If we still need more, add some random ones
          if (candidateAnimeIndices.length < maxCandidates) {
            const remainingCount = maxCandidates - candidateAnimeIndices.length;
            
            // Shuffle the array using Fisher-Yates algorithm and take what we need
            const unusedIndices = [...allAnimeIndices].filter(idx => !candidateAnimeIndices.includes(idx));
            for (let i = unusedIndices.length - 1; i > 0; i--) {
              const j = Math.floor(Math.random() * (i + 1));
              [unusedIndices[i], unusedIndices[j]] = [unusedIndices[j], unusedIndices[i]];
            }
            
            candidateAnimeIndices = [...candidateAnimeIndices, ...unusedIndices.slice(0, remainingCount)];
          }
        } else {
          // Fallback to sequential indices if mapping failed
          candidateAnimeIndices = Array.from({ length: maxCandidates }, (_, i) => i);
          console.log('[RECOMMENDATION] Using sequential indices as fallback');
        }
      }
            
      console.log(`[RECOMMENDATION] Using ${candidateAnimeIndices.length} candidate indices for recommendations`);
      
      // Get anime genres and tags from AniList API if we have watch history
      let genres: string[] = [];
      let tags: string[] = [];
      
      if (userWatchHistory.length === 0) {
        // No watch history, use a simplified approach with default preferences
        console.log('[RECOMMENDATION] No watch history, using simplified approach with default preferences');
        genres = preferredGenres;
        tags = preferredTags;
        
        // Return a basic recommendation result
        return {
          recommendations: [],
          status: 'error',
          error: 'Not enough watch history to generate personalized recommendations',
          debugInfo
        };
      }
      
      // Continue with regular recommendation flow using watch history
      console.log('[RECOMMENDATION] Using user watch history for candidate selection');
      
      // Use only highly-rated anime (>= 6/10) for positive preferences
      const highRatedAnime = userWatchHistory.filter(item => item.rating >= 6);
      console.log(`[RECOMMENDATION] Using ${highRatedAnime.length} highly-rated anime to gather positive preferences`);
      
      // Collect genres and tags from multiple anime in watch history
      const allGenres: Set<string> = new Set();
      const allTags: Set<string> = new Set();
      const animeDetailsPromises: Promise<void>[] = [];
      
      // Process each watched anime in parallel
      for (const watchItem of highRatedAnime) {
        animeDetailsPromises.push((async () => {
          try {
            const animeDetails = await getAnimeDetails(watchItem.anilist_id);
            
            if (animeDetails) {
              // Calculate weight based on user rating (higher rating = more influence)
              // Scale is 1-10, with 5 being neutral. Ratings below 5 won't contribute positively.
              const ratingWeight = Math.max(0, (watchItem.rating - 5) / 5); // 0 to 1 scale
              
              if (ratingWeight > 0) {
                // Add genres from this anime to our set with weight consideration
                animeDetails.genres.forEach(genre => allGenres.add(genre));
                
                // Add top tags from this anime to our set (prioritize by rank and user rating)
                const sortedTags = [...animeDetails.tags]
                  .sort((a, b) => b.rank - a.rank)
                  .slice(0, Math.ceil(10 * ratingWeight)); // More tags from highly rated anime
                
                sortedTags.forEach(tag => allTags.add(tag.name));
                
                console.log(`[RECOMMENDATION] Added weighted preferences from ${animeDetails.title.english || animeDetails.title.romaji} (${watchItem.rating}/10): ${animeDetails.genres.length} genres, ${sortedTags.length} tags, weight: ${ratingWeight.toFixed(2)}`);
              } else {
                console.log(`[RECOMMENDATION] Skipping preferences from neutral anime: ${animeDetails.title.english || animeDetails.title.romaji} (${watchItem.rating}/10)`);
              }
            }
          } catch (error) {
            console.error(`[RECOMMENDATION] Error fetching details for anime ${watchItem.anilist_id}:`, error);
          }
        })());
      }
      
      // Wait for all anime details to be fetched
      await Promise.all(animeDetailsPromises);
      
      // Convert Sets to arrays
      genres = Array.from(allGenres);
      tags = Array.from(allTags);
      
      if (genres.length > 0 || tags.length > 0) {
        console.log('[RECOMMENDATION] Collected preferences from watch history successfully');
        console.log('[RECOMMENDATION] Combined genres:', genres);
        console.log('[RECOMMENDATION] Combined tags:', tags.slice(0, 10), '...');
        
        debugInfo.genresUsed = genres;
        debugInfo.tagsUsed = tags.slice(0, 10);
      } else {
        // Fallback if we couldn't get any genres or tags
        console.log('[RECOMMENDATION] Could not fetch anime details, using fallback genres and tags');
        genres = preferredGenres;
        tags = preferredTags;
      }
      
      // Map watched anime indices and save for debugging
      debugInfo.mappedAnimeIds = userEmbedding.watchedIndices;
      debugInfo.animeIdMapSuccess = debugInfo.mappedAnimeIds.length > 0;
      
      // Get ratings for all candidate anime
      console.log(`[RECOMMENDATION] Running model inference on ${candidateAnimeIndices.length} candidates`);
      
      // Transform genres and tags to indices
      const genreIndices: number[] = [];
      for (const genre of genres) {
        const genreIdx = genreToIdx[genre];
        if (genreIdx !== undefined) {
          genreIndices.push(genreIdx);
        }
      }
      
      // Handle tags - we might have too many so we need to filter
      const tagIndices: number[] = [];
      for (const tag of tags) {
        const tagIdx = tagToIdx[tag];
        if (tagIdx !== undefined) {
          tagIndices.push(tagIdx);
        }
      }
      
      // Get ratings for candidate anime
      const candidateRatings = await runModelInference(
        userIndex, 
        candidateAnimeIndices, 
        genreIndices,
        tagIndices
      );
      
      // Collect results and create full records with Anilist ID and title
      interface CandidateResult {
        animeIdx: number;
        anilistId: number;
        title: string;
        score: number;
        averageScore?: number;
        cover_image?: string;
        genres?: string[];
      }
      
      const candidateResults: CandidateResult[] = [];
      
      for (let i = 0; i < candidateAnimeIndices.length; i++) {
        const animeIdx = candidateAnimeIndices[i];
        const rating = candidateRatings[i];
        
        if (rating > 0) { // Skip negative ratings
          // Get the anime ID using our direct mapping if available, otherwise fall back to the model mapping
          let numericId: number;
          
          if (animeIdToAnilistId[animeIdx] !== undefined) {
            numericId = animeIdToAnilistId[animeIdx];
          } else {
            // Get the anime ID from the idx using the model mapping
            const animeId = idxToAnime[animeIdx.toString()];
            
            if (!animeId) continue; // Skip if no mapping
            
            // Extract the numeric ID from the string (might be prefixed)
            numericId = parseInt(animeId.replace(/^(anilist-|anime-)?/, ''));
          }
          
          // Check if this is in the negative preferences list
          const isInNegativePreferences = negativePreferences.animeIds.includes(numericId);
          
          // Check if this is related to a negative preference anime
          const relatedPenalty = negativePreferences.relatedAnimeIds.get(numericId) || 0;
          
          // Apply an appropriate penalty based on:
          // 1. Direct negative preference: 50% penalty (existing)
          // 2. Related to negative anime: variable penalty based on relationship strength
          let adjustedRating = rating;
          if (isInNegativePreferences) {
            adjustedRating *= 0.5; // Original 50% penalty for direct matches
          } else if (relatedPenalty > 0) {
            adjustedRating *= (1 - relatedPenalty); // Variable penalty for related anime
          }
          
          // Check if this anime is in the watched list to avoid recommending it
          const isWatched = debugInfo.watchedAnimeIds.includes(numericId);
          
          // Only include unwatched anime
          if (!isWatched) {
            candidateResults.push({
              animeIdx,
              anilistId: numericId,
              title: `Anime ${numericId}`, // Placeholder, will be replaced with API data
              score: adjustedRating,
              averageScore: undefined,
              cover_image: undefined,
              genres: undefined
            });
          }
        }
      }
      
      // Sort by score (highest first)
      candidateResults.sort((a, b) => b.score - a.score);
      
      // Log the top results for debugging
      console.log(`[RECOMMENDATION] Top ${Math.min(10, candidateResults.length)} candidates before enrichment:`);
      candidateResults.slice(0, 10).forEach((result, index) => {
        console.log(`  ${index + 1}. Anime ID: ${result.anilistId}, Score: ${result.score.toFixed(4)}`);
      });
      
      // Limit to requested number of recommendations
      const topResults = candidateResults.slice(0, limit);
      
      // Fetch details for each anime to enrich the recommendations
      const enrichedResults: AnimeData[] = [];
      const enrichmentPromises: Promise<void>[] = [];
      
      for (const result of topResults) {
        enrichmentPromises.push((async () => {
          try {
            const details = await getAnimeDetails(result.anilistId);
            
            if (details) {
              // Add any related anime info for debugging
              const relatedAnimeDebugInfo: { id: number; title: string; penalty: number; reason: string }[] = [];
              
              // If this anime had a penalty applied, note it
              if (negativePreferences.animeIds.includes(result.anilistId)) {
                relatedAnimeDebugInfo.push({
                  id: result.anilistId,
                  title: details.title.english || details.title.romaji,
                  penalty: 0.5, // Direct match penalty
                  reason: "Direct low rating"
                });
              } else if (negativePreferences.relatedAnimeIds.has(result.anilistId)) {
                relatedAnimeDebugInfo.push({
                  id: result.anilistId,
                  title: details.title.english || details.title.romaji,
                  penalty: negativePreferences.relatedAnimeIds.get(result.anilistId) || 0,
                  reason: "Related to low-rated anime"
                });
              }
              
              // Create the enriched result with debug info
              enrichedResults.push({
                id: result.anilistId,
                title: details.title.english || details.title.romaji || `Anime ${result.anilistId}`,
                score: result.score,
                averageScore: details.averageScore,
                cover_image: details.coverImage.large || details.coverImage.medium,
                genres: details.genres,
                description: details.description,
                _debugInfo: {
                  userEmbedding: {
                    dimension: userEmbedding.embeddingVector.length,
                    sample: userEmbedding.embeddingVector.slice(0, 10)
                  },
                  negativePreferences: {
                    genres: Array.from(negativePreferences.genres),
                    tags: Array.from(negativePreferences.tags),
                    relatedAnime: relatedAnimeDebugInfo
                  }
                }
              });
            } else {
              // If we couldn't get details, still include the basic info
              enrichedResults.push({
                id: result.anilistId,
                title: `Anime ${result.anilistId}`,
                score: result.score,
                _debugInfo: {
                  userEmbedding: {
                    dimension: userEmbedding.embeddingVector.length,
                    sample: userEmbedding.embeddingVector.slice(0, 10)
                  },
                  negativePreferences: {
                    genres: Array.from(negativePreferences.genres),
                    tags: Array.from(negativePreferences.tags)
                  }
                }
              });
            }
          } catch (error) {
            console.error(`[RECOMMENDATION] Error fetching details for anime ${result.anilistId}:`, error);
            
            // Even if we have an error, still add a basic entry
            enrichedResults.push({
              id: result.anilistId,
              title: `Anime ${result.anilistId}`,
              score: result.score,
              _debugInfo: {
                userEmbedding: {
                  dimension: userEmbedding.embeddingVector.length,
                  sample: userEmbedding.embeddingVector.slice(0, 10)
                },
                negativePreferences: {
                  genres: Array.from(negativePreferences.genres),
                  tags: Array.from(negativePreferences.tags)
                }
              }
            });
          }
        })());
      }
      
      // Wait for all enrichment to complete
      await Promise.all(enrichmentPromises);
      
      // Sort again in case the order was lost
      enrichedResults.sort((a, b) => b.score - a.score);
      
      return {
        recommendations: enrichedResults,
        status: 'success',
        userWatchHistory,
        debugInfo
      };
    } catch (error) {
      console.error('[RECOMMENDATION] Error while getting recommendations:', error);
      return {
        recommendations: [],
        status: 'error',
        error: error instanceof Error ? error.message : String(error)
      };
    }
  } catch (error) {
    console.error('[RECOMMENDATION] Uncaught error in getRecommendations:', error);
    return {
      recommendations: [],
      status: 'error',
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

/**
 * Preload the recommendation system
 * This can be called early in the application lifecycle to warm up the model
 */
export async function preloadRecommendationSystem(): Promise<void> {
  try {
    console.log('[RECOMMENDATION] Starting preloadRecommendationSystem...');
    logOnnxServiceState();
    
    // Skip loading if model is already loaded
    if (isModelLoaded()) {
      console.log('[RECOMMENDATION] Model already loaded, skipping preload');
      return;
    }
    
    // Load model and mappings in parallel
    await Promise.all([
      loadModel().then(() => console.log('[RECOMMENDATION] Model loaded')),
      loadModelMetadata().then(() => console.log('[RECOMMENDATION] Metadata loaded')),
      loadModelMappings().then(() => console.log('[RECOMMENDATION] Mappings loaded'))
    ]);
    
    console.log('[RECOMMENDATION] Recommendation system preloaded successfully');
    logOnnxServiceState();
  } catch (error) {
    console.error('[RECOMMENDATION] Error preloading recommendation system:', error);
    throw error;
  }
} 
import { 
  loadModel, 
  loadModelMetadata, 
  loadModelMappings,
  runModelInference,
  isModelLoaded
} from './modelService';
import { logOnnxServiceState } from './onnxModelService';
import { getUserWatchHistory } from './watchHistoryService';
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
}> {
  console.log(`[RECOMMENDATION] Extracting negative preferences with threshold ${threshold}...`);
  
  const negativeGenres = new Set<string>();
  const negativeTags = new Set<string>();
  const negativeAnimeIds: number[] = [];
  
  if (!watchHistory || watchHistory.length === 0) {
    return { genres: negativeGenres, tags: negativeTags, animeIds: negativeAnimeIds };
  }
  
  // Get low-rated anime
  const lowRatedAnime = watchHistory.filter(item => item.rating < threshold);
  console.log(`[RECOMMENDATION] Found ${lowRatedAnime.length} anime rated below ${threshold}/10`);
  
  if (lowRatedAnime.length === 0) {
    return { genres: negativeGenres, tags: negativeTags, animeIds: negativeAnimeIds };
  }
  
  // Process each low-rated anime
  const animeDetailsPromises: Promise<void>[] = [];
  
  for (const item of lowRatedAnime) {
    negativeAnimeIds.push(item.anilist_id);
    
    animeDetailsPromises.push((async () => {
      try {
        const animeDetails = await getAnimeDetails(item.anilist_id);
        
        if (animeDetails) {
          // Calculate negative weight based on how low the rating is
          // Scale is 1-10, with 5 being neutral. Lower ratings have more negative weight.
          const negativeWeight = (threshold - item.rating) / threshold; // 0 to 1 scale
          
          if (negativeWeight > 0) {
            // Add weight-filtered genres to negative set
            // Only add the most influential genres for very low ratings
            const genreLimit = Math.ceil(animeDetails.genres.length * negativeWeight);
            animeDetails.genres.slice(0, genreLimit).forEach(genre => negativeGenres.add(genre));
            
            // Add most influential tags to negative set
            const sortedTags = [...animeDetails.tags]
              .sort((a, b) => b.rank - a.rank)
              .slice(0, Math.ceil(5 * negativeWeight)); // More tags from lowest rated anime
            
            sortedTags.forEach(tag => negativeTags.add(tag.name));
            
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
  
  console.log(`[RECOMMENDATION] Extracted negative preferences: ${negativeGenres.size} genres, ${negativeTags.size} tags`);
  
  return {
    genres: negativeGenres,
    tags: negativeTags,
    animeIds: negativeAnimeIds
  };
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
 * Generate recommendations for a user based on their watch history
 * @param userId The ID of the user to generate recommendations for
 * @param preferredGenres List of preferred genres (fallback)
 * @param preferredTags List of preferred tags (fallback)
 * @param limit Maximum number of recommendations to return
 */
export async function getRecommendations(
  userId: string,
  preferredGenres: string[] = [],
  preferredTags: string[] = [],
  limit: number = 10
): Promise<RecommendationResult> {
  try {
    console.log('[RECOMMENDATION] Starting getRecommendations...');
    logOnnxServiceState();
    
    // Fetch user's watch history
    console.log('[RECOMMENDATION] Fetching user watch history...');
    const userWatchHistory = await getUserWatchHistory();
    console.log(`[RECOMMENDATION] Found ${userWatchHistory.length} items in user watch history`);
    
    // Log watch history details early for debugging
    if (userWatchHistory.length > 0) {
      console.log('[RECOMMENDATION] Watch history details:', 
        userWatchHistory.map(item => ({
          anilist_id: item.anilist_id,
          title: item.title,
          rating: item.rating
        }))
      );
    }
    
    // Debug info to return with result
    const debugInfo: {
      watchedAnimeIds: number[];
      mappedAnimeIds: number[];
      animeIdMapSuccess: boolean;
      mappingKeys?: string[];
      genresUsed?: string[];
      tagsUsed?: string[];
      negativePreferences?: {
        genres?: string[];
        tags?: string[];
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
    } = {
      watchedAnimeIds: userWatchHistory.map(item => item.anilist_id),
      mappedAnimeIds: [] as number[],
      animeIdMapSuccess: false
    };
    
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
      
      // Extract negative preferences from low-rated anime
      const negativePreferences = await extractNegativePreferences(userWatchHistory);
      
      // Save negative preferences for debugging
      debugInfo.negativePreferences = {
        genres: Array.from(negativePreferences.genres),
        tags: Array.from(negativePreferences.tags)
      };
      
      // If we have collaborative recommendations, process and return them directly
      if (collaborativeRecommendations && collaborativeRecommendations.length > 0) {
        console.log('[RECOMMENDATION] Processing collaborative filtering recommendations');
        
        // Filter out anime that the user has already watched
        const watchedAnimeIds = new Set(userWatchHistory.map(item => item.anilist_id));
        const filteredRecommendations = collaborativeRecommendations.filter(
          rec => !watchedAnimeIds.has(rec.animeId)
        );
        
        // Limit to the requested number
        const topRecommendations = filteredRecommendations.slice(0, limit);
        
        // Fetch details for each anime
        const enrichedResults: AnimeData[] = [];
        const enrichmentPromises: Promise<void>[] = [];
        
        for (const recommendation of topRecommendations) {
          enrichmentPromises.push((async () => {
            try {
              const details = await getAnimeDetails(recommendation.animeId);
              
              if (details) {
                // Create string listing top contributors
                const contributorInfo = recommendation.contributors
                  .slice(0, 3)
                  .map(c => `User rated ${c.rating}/10 (similarity: ${(c.weight * 100).toFixed(0)}%)`)
                  .join(', ');
                
                enrichedResults.push({
                  id: recommendation.animeId,
                  title: details.title.english || details.title.romaji || `Anime ${recommendation.animeId}`,
                  score: recommendation.score / 10, // Convert to 0-1 scale
                  averageScore: details.averageScore,
                  cover_image: details.coverImage?.large || details.coverImage?.medium,
                  genres: details.genres,
                  description: details.description,
                  _debugInfo: {
                    userEmbedding: {
                      dimension: userEmbedding.embeddingVector.length,
                      sample: userEmbedding.embeddingVector.slice(0, 10)
                    },
                    negativePreferences: {
                      genres: Array.from(negativePreferences.genres),
                      tags: Array.from(negativePreferences.tags)
                    },
                    collaborativeInfo: {
                      contributorCount: recommendation.contributors.length,
                      topContributors: contributorInfo
                    }
                  }
                });
              } else {
                // If we couldn't get details, still include the basic info
                enrichedResults.push({
                  id: recommendation.animeId,
                  title: `Anime ${recommendation.animeId}`,
                  score: recommendation.score / 10, // Convert to 0-1 scale
                  _debugInfo: {
                    userEmbedding: {
                      dimension: userEmbedding.embeddingVector.length,
                      sample: userEmbedding.embeddingVector.slice(0, 10)
                    },
                    negativePreferences: {
                      genres: Array.from(negativePreferences.genres),
                      tags: Array.from(negativePreferences.tags)
                    },
                    collaborativeInfo: {
                      contributorCount: recommendation.contributors.length
                    }
                  }
                });
              }
            } catch (error) {
              console.error(`[RECOMMENDATION] Error fetching details for anime ${recommendation.animeId}:`, error);
              
              // Even if we have an error, still add a basic entry
              enrichedResults.push({
                id: recommendation.animeId,
                title: `Anime ${recommendation.animeId}`,
                score: recommendation.score / 10, // Convert to 0-1 scale
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
      }
      
      // Continue with existing model-based recommendation logic...
      
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
          
          // Apply a penalty to anime in negative preferences
          const adjustedRating = isInNegativePreferences ? rating * 0.5 : rating;
          
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
              enrichedResults.push({
                id: result.anilistId,
                title: details.title.english || details.title.romaji || `Anime ${result.anilistId}`,
                score: result.score,
                averageScore: details.averageScore,
                cover_image: details.coverImage?.large || details.coverImage?.medium,
                genres: details.genres,
                description: details.description,
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
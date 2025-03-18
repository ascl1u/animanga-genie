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

export interface AnimeData {
  id: number;
  title: string;
  score: number;
  cover_image?: string;
  genres?: string[];
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
  };
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
      const metadata = await loadModelMetadata();
      
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
      
      // Use a default user index of 0 since we don't have user mappings
      const userIndex = 0;
      
      // Create candidate anime indices from watch history if available
      // MODIFIED: Limit to 10 candidates for testing
      const maxCandidates = 10; // CHANGED: Reduced from 100 to 10 for testing
      let candidateAnimeIndices = Array.from({ length: maxCandidates }, (_, i) => i); // Default fallback
      
      // Get anime genres and tags from AniList API if we have watch history
      let genres: string[] = [];
      let tags: string[] = [];
      
      if (userWatchHistory.length > 0) {
        console.log('[RECOMMENDATION] Using user watch history for candidate selection');
        
        // Attempt to fetch anime details for the first anime in watch history
        const watchItem = userWatchHistory[0]; // Start with the first item
        try {
          const animeDetails = await getAnimeDetails(watchItem.anilist_id);
          
          if (animeDetails) {
            genres = animeDetails.genres || [];
            tags = animeDetails.tags.map(tag => tag.name) || [];
            
            console.log('[RECOMMENDATION] Fetched anime details successfully');
            console.log('[RECOMMENDATION] Genres:', genres);
            console.log('[RECOMMENDATION] Tags:', tags.slice(0, 10), '...');
            
            debugInfo.genresUsed = genres;
            debugInfo.tagsUsed = tags.slice(0, 10);
          } else {
            console.log('[RECOMMENDATION] Could not fetch anime details, using fallback genres and tags');
            genres = preferredGenres;
            tags = preferredTags;
          }
        } catch (error) {
          console.error('[RECOMMENDATION] Error fetching anime details:', error);
          genres = preferredGenres;
          tags = preferredTags;
        }
        
        // Get anime indices from the animeToIdx mapping
        // Map anilist_id to model indices where possible
        const watchHistoryIndices = userWatchHistory
          .map(item => {
            // Try different formats of the ID to match mapping
            const animeId = item.anilist_id.toString();
            const possibleIds = [
              animeId,                         // try as-is
              `anilist-${animeId}`,            // try with prefix
              item.anilist_id                  // try as number
            ];
            
            // Try to find a match using any format
            let modelIdx = null;
            for (const id of possibleIds) {
              const stringId = String(id);
              if (animeToIdx[stringId] !== undefined) {
                modelIdx = animeToIdx[stringId];
                console.log(`[RECOMMENDATION] Found mapping for anime ID: ${animeId} using format: ${stringId} → ${modelIdx}`);
                debugInfo.mappedAnimeIds.push(item.anilist_id);
                return { modelIdx, anilist_id: item.anilist_id, rating: item.rating };
              }
            }
            
            console.log(`[RECOMMENDATION] Could not find mapping for anime ID: ${animeId}`);
            return null;
          })
          .filter(item => item !== null) as { modelIdx: number, anilist_id: number, rating: number }[];
        
        console.log('[RECOMMENDATION] Mapped watch history to model indices:', watchHistoryIndices);
        debugInfo.animeIdMapSuccess = watchHistoryIndices.length > 0;
        
        if (watchHistoryIndices.length > 0) {
          // Get recommendation candidates (anime not in watch history)
          // We'll take the model's anime indices and exclude ones the user has already watched
          const watchedIndices = new Set(watchHistoryIndices.map(item => item.modelIdx));
          candidateAnimeIndices = Object.keys(idxToAnime)
            .map(key => parseInt(key))
            .filter(idx => !watchedIndices.has(idx))
            .slice(0, maxCandidates); // CHANGED: Limit to maxCandidates for testing
        } else {
          console.log('[RECOMMENDATION] No watch history items could be mapped to model indices, using default candidates');
        }
      } else {
        genres = preferredGenres;
        tags = preferredTags;
      }
      
      // Map genres and tags to indices the model understands
      const genreIndices = genres
        .map(genre => {
          // Try multiple variations of the genre name to increase chances of mapping
          const variations = [
            genre.toLowerCase(),
            genre.toUpperCase(),
            genre,
            genre.toLowerCase().replace(/[-\s]/g, ''),
            genre.toLowerCase().trim()
          ];
          
          let idx = null;
          for (const variation of variations) {
            if (genreToIdx[variation] !== undefined) {
              idx = genreToIdx[variation];
              console.log(`[RECOMMENDATION] Successfully mapped genre "${genre}" using format "${variation}" → ${idx}`);
              break;
            }
          }
          
          if (idx === null) {
            // Log available genres for debugging
            if (genres.indexOf(genre) === 0) { // Only do this once to avoid spam
              console.log('[RECOMMENDATION] Available genres in model:', Object.keys(genreToIdx).slice(0, 20));
            }
            console.log(`[RECOMMENDATION] Could not map genre: "${genre}"`);
          }
          
          return idx;
        })
        .filter(idx => idx !== null)
        .slice(0, metadata.max_genres);
      
      // If we don't have enough mapped genres, add some default ones
      if (genreIndices.length === 0) {
        console.log('[RECOMMENDATION] No genres could be mapped, using fallback genre indices');
        
        // Try to use some common genres that are likely to be in the model
        const commonGenres = ['action', 'adventure', 'comedy', 'drama', 'fantasy', 'romance', 'sci-fi'];
        let foundGenres = false;
        
        for (const genre of commonGenres) {
          const idx = genreToIdx[genre];
          if (idx !== undefined) {
            genreIndices.push(idx);
            console.log(`[RECOMMENDATION] Added common genre "${genre}" → ${idx}`);
            foundGenres = true;
            if (genreIndices.length >= metadata.max_genres) break;
          }
        }
        
        // If still no genres found, fall back to default indices
        if (!foundGenres) {
          for (let i = 0; i < Math.min(preferredGenres.length, metadata.max_genres); i++) {
            genreIndices.push(i % metadata.n_genres);
          }
        }
      }
      
      // Map tags to indices
      const tagIndices = tags
        .map(tag => {
          // Try multiple variations of the tag name to increase chances of mapping
          const variations = [
            tag.toLowerCase(),
            tag.toUpperCase(),
            tag,
            tag.toLowerCase().replace(/[-\s]/g, ''),
            tag.toLowerCase().trim()
          ];
          
          let idx = null;
          for (const variation of variations) {
            if (tagToIdx[variation] !== undefined) {
              idx = tagToIdx[variation];
              console.log(`[RECOMMENDATION] Successfully mapped tag "${tag}" using format "${variation}" → ${idx}`);
              break;
            }
          }
          
          if (idx === null) {
            // Log available tags for debugging, but only for the first tag to avoid spam
            if (tags.indexOf(tag) === 0) {
              console.log('[RECOMMENDATION] Available tags in model (sample):', Object.keys(tagToIdx).slice(0, 20));
            }
            console.log(`[RECOMMENDATION] Could not map tag: "${tag}"`);
          }
          
          return idx;
        })
        .filter(idx => idx !== null)
        .slice(0, metadata.max_tags);
      
      // If we don't have enough mapped tags, add some default ones
      if (tagIndices.length === 0) {
        console.log('[RECOMMENDATION] No tags could be mapped, using fallback tag indices');
        
        // Try to use some common tags that are likely to be in the model
        const commonTags = ['shounen', 'action', 'magic', 'fantasy', 'adventure', 'drama', 'comedy'];
        let foundTags = false;
        
        for (const tag of commonTags) {
          const idx = tagToIdx[tag];
          if (idx !== undefined) {
            tagIndices.push(idx);
            console.log(`[RECOMMENDATION] Added common tag "${tag}" → ${idx}`);
            foundTags = true;
            if (tagIndices.length >= metadata.max_tags) break;
          }
        }
        
        // If still no tags found, fall back to default indices
        if (!foundTags) {
          for (let i = 0; i < Math.min(preferredTags.length, metadata.max_tags); i++) {
            tagIndices.push(i % metadata.n_tags);
          }
        }
      }
      
      console.log('[RECOMMENDATION] Running inference with:', {
        userIndex,
        candidateCount: candidateAnimeIndices.length,
        genreIndices,
        tagIndices
      });
      
      // MODIFIED: Use batch inference instead of individual calls
      // Run model inference through our adapter
      console.log('[RECOMMENDATION] Starting batch inference for', candidateAnimeIndices.length, 'candidates');
      const scores = await runModelInference(
        userIndex,
        candidateAnimeIndices,
        genreIndices as number[],
        tagIndices as number[]
      );
      
      console.log('[RECOMMENDATION] Got scores, creating recommendations');
      console.log(`[RECOMMENDATION] Score sample (first 5): ${scores.slice(0, 5).join(', ')}`);
      
      // Create array of [index, score] pairs
      const indexedScores = candidateAnimeIndices.map((animeIndex, i) => [animeIndex, scores[i]]);
      
      // Sort by score in descending order
      indexedScores.sort((a, b) => b[1] - a[1]);
      
      // Take top N results
      const topN = indexedScores.slice(0, limit);
      
      // Map back to anime IDs and create result objects
      const recommendations = await Promise.all(topN.map(async ([index, score]) => {
        const animeId = idxToAnime[index.toString()] || `unknown-${index}`;
        // Convert to number if possible
        const numericId = parseInt(animeId, 10) || index;
        
        // Attempt to get more details from AniList for each recommendation
        let title = `Anime ${animeId}`;
        let coverImage = undefined;
        let animeGenres = undefined;
        
        // First check if it matches something in watch history
        const matchingHistoryItem = userWatchHistory.find(
          item => item.anilist_id.toString() === animeId || item.anilist_id === numericId
        );
        
        if (matchingHistoryItem) {
          title = matchingHistoryItem.title;
          coverImage = matchingHistoryItem.cover_image;
        } else {
          // Try fetching from AniList
          try {
            const details = await getAnimeDetails(numericId);
            if (details) {
              title = details.title.english || details.title.romaji;
              coverImage = details.coverImage.medium;
              animeGenres = details.genres;
            }
          } catch (error) {
            console.log(`[RECOMMENDATION] Could not fetch details for anime ${numericId}:`, error);
          }
        }
        
        return {
          id: numericId,
          title,
          cover_image: coverImage,
          score: score as number,
          genres: animeGenres
        };
      }));
      
      console.log('[RECOMMENDATION] Successfully created recommendations');
      
      return {
        recommendations,
        status: 'success',
        userWatchHistory, // Include watch history for debugging
        debugInfo
      };
    } catch (error) {
      console.error('[RECOMMENDATION] Error during model loading or inference:', error);
      return {
        recommendations: [],
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error during model operation',
        userWatchHistory,
        debugInfo
      };
    }
  } catch (error) {
    console.error('[RECOMMENDATION] Error generating recommendations:', error);
    return {
      recommendations: [],
      status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error'
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
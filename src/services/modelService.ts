import { onnxModelService } from './onnxModelService';
import { MODEL_MAPPINGS_PATH } from '@/constants/modelPaths';

/**
 * Service to load and cache the ONNX model
 */
export interface ModelMetadata {
  n_users: number;
  n_anime: number;
  n_genres: number;
  n_tags: number;
  user_embedding_dim: number;
  anime_embedding_dim: number;
  genre_embedding_dim: number;
  tag_embedding_dim: number;
  dense_layers: number[];
  max_genres: number;
  max_tags: number;
  model_type: string;
}

export interface ModelMappings {
  user_id_to_index?: Record<string, number>;
  anime_id_to_index?: Record<string, number>;
  index_to_anime_id?: Record<string, number>;
  genre_to_index?: Record<string, number>;
  tag_to_index?: Record<string, number>;
  // Add the actual structure from our mapping file
  idx_to_anime?: Record<string, string>;
  anime_to_idx?: Record<string, number>;
  idx_to_genre?: Record<string, string>;
  genre_to_idx?: Record<string, number>;
  idx_to_tag?: Record<string, string>;
  tag_to_idx?: Record<string, number>;
}

// Cache for mappings since they're not part of the ONNX service
const mappingsCache: {
  mappings: ModelMappings | null;
  loading: boolean;
  error: Error | null;
} = {
  mappings: null,
  loading: false,
  error: null,
};

function tracelog(message: string): void {
  console.log(`[MODEL-SERVICE] ${message}`);
}

/**
 * Load the model metadata from the ONNX service
 */
export async function loadModelMetadata(): Promise<ModelMetadata> {
  tracelog("Loading model metadata - Start");
  tracelog(`Current modelLoaded state: ${onnxModelService.isModelLoaded()}`);
  
  // Ensure model is initialized
  tracelog("Initializing model...");
  await onnxModelService.initModel();
  tracelog(`After init call - modelLoaded: ${onnxModelService.isModelLoaded()}`);
  
  // Get metadata from ONNX service
  tracelog("Getting metadata from ONNX service");
  const metadata = onnxModelService.getMetadata();
  tracelog(`Metadata received: ${metadata ? 'YES' : 'NO'}`);
  
  if (!metadata) {
    tracelog("METADATA IS NULL - WILL THROW ERROR");
    
    // Check initialization errors
    const errors = onnxModelService.getInitializationErrors();
    tracelog(`Initialization errors: ${errors.length > 0 ? errors.join(', ') : 'none'}`);
    
    throw new Error('Failed to load model metadata from ONNX service');
  }
  
  tracelog("Metadata loaded successfully");
  return metadata;
}

/**
 * Load the model mappings
 */
export async function loadModelMappings(): Promise<ModelMappings> {
  if (mappingsCache.mappings) {
    return mappingsCache.mappings;
  }

  if (mappingsCache.loading) {
    // Wait until mappings are loaded
    return new Promise((resolve, reject) => {
      const checkInterval = setInterval(() => {
        if (mappingsCache.mappings) {
          clearInterval(checkInterval);
          resolve(mappingsCache.mappings);
        }
        if (mappingsCache.error) {
          clearInterval(checkInterval);
          reject(mappingsCache.error);
        }
      }, 100);
    });
  }

  // Set loading state
  mappingsCache.loading = true;
  mappingsCache.error = null;

  try {
    // Using centralized path constant
    console.log(`[MODEL-SERVICE] Loading mappings from ${MODEL_MAPPINGS_PATH}`);
    const response = await fetch(MODEL_MAPPINGS_PATH);
    if (!response.ok) {
      throw new Error(`Failed to load model mappings: ${response.statusText}`);
    }
    const mappings = await response.json();
    mappingsCache.mappings = mappings;
    return mappings;
  } catch (error) {
    console.error('Error loading model mappings:', error);
    mappingsCache.error = error as Error;
    throw error;
  } finally {
    mappingsCache.loading = false;
  }
}

/**
 * Load the ONNX model
 * This is a compatibility function that just calls initModel on the ONNX service
 */
export async function loadModel(): Promise<Record<string, unknown>> {
  // Initialize ONNX model
  const success = await onnxModelService.initModel();
  
  if (!success) {
    const errors = onnxModelService.getInitializationErrors();
    throw new Error(`Failed to load ONNX model: ${errors.join(', ')}`);
  }
  
  // Return a dummy model object for compatibility
  return {};
}

/**
 * Check if the model is loaded
 */
export function isModelLoaded(): boolean {
  return onnxModelService.isModelLoaded();
}

/**
 * Get the current model loading status
 */
export function getModelStatus(): {
  isLoaded: boolean;
  isLoading: boolean;
  error: Error | null;
} {
  const errors = onnxModelService.getInitializationErrors();
  return {
    isLoaded: onnxModelService.isModelLoaded(),
    isLoading: false, // We don't have direct access to loading state
    error: errors.length > 0 ? new Error(errors[0]) : null,
  };
}

/**
 * Preload the model and all required files
 */
export async function preloadModel(): Promise<void> {
  try {
    // Load everything in parallel
    await Promise.all([
      onnxModelService.initModel(),
      loadModelMappings(),
    ]);
    console.log('Model and metadata preloaded successfully');
  } catch (error) {
    console.error('Error preloading model:', error);
    throw error;
  }
}

/**
 * Run inference with the model - adapter for the ONNX model service
 * This provides compatibility with the older interface while supporting enhanced features
 */
export async function runModelInference(
  userIdx: number,
  animeIndices: number[],
  genreIndices: number[],
  tagIndices: number[],
  studioIndices: number[] = [],
  studioWeights: number[] = [],
  relationIndices: number[] = [],
  relationWeights: number[] = []
): Promise<number[]> {
  if (!onnxModelService.isModelLoaded()) {
    await onnxModelService.initModel();
  }
  
  console.log(`[DEBUG] Starting inference for ${animeIndices.length} anime indices`);
  
  // Run inference for each anime index sequentially instead of with Promise.all
  const ratings: number[] = [];
  
  for (let i = 0; i < animeIndices.length; i++) {
    const animeIdx = animeIndices[i];
    console.log(`[DEBUG] Starting inference for anime index ${animeIdx} (${i+1}/${animeIndices.length})`);
    
    try {
      const rating = await onnxModelService.runInference(
        userIdx,
        animeIdx,
        genreIndices,
        tagIndices,
        studioIndices,
        studioWeights,
        relationIndices,
        relationWeights
      );
      ratings.push(rating);
      console.log(`[DEBUG] Completed inference for anime index ${animeIdx}: rating = ${rating}`);
    } catch (error) {
      console.error(`[DEBUG] Error during inference for anime index ${animeIdx}:`, error);
      // Push a default low rating to maintain array length
      ratings.push(-1);
    }
  }
  
  console.log(`[DEBUG] Completed all inferences, got ${ratings.length} ratings`);
  return ratings;
}
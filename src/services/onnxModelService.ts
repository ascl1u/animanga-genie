// @ts-expect-error - ONNX runtime types issue
import * as ort from 'onnxruntime-web';
import {
  MODEL_FILE_PATH,
  MODEL_METADATA_PATH,
  WASM_PATH,
  WASM_SIMD_PATH,
  WASM_THREADED_PATH
} from '@/constants/modelPaths';

// Define the model metadata interface
interface AnimeRecommenderMetadata {
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
  rating_normalization: {
    mean: number;
    std: number;
  };
  input_names: string[];
  output_names: string[];
}

// Class for managing ONNX model loading and inference
export class OnnxModelService {
  private static instance: OnnxModelService;
  private session: ort.InferenceSession | null = null;
  private metadata: AnimeRecommenderMetadata | null = null;
  private initErrors: string[] = [];
  private modelLoaded = false;
  private modelLoading = false;
  private modelPath = MODEL_FILE_PATH;
  private metadataPath = MODEL_METADATA_PATH;

  private constructor() {}

  // Singleton pattern to ensure only one instance
  public static getInstance(): OnnxModelService {
    if (!OnnxModelService.instance) {
      OnnxModelService.instance = new OnnxModelService();
    }
    return OnnxModelService.instance;
  }

  // Get initialization errors
  public getInitializationErrors(): string[] {
    return [...this.initErrors];
  }

  // Clear initialization errors
  public clearInitializationErrors(): void {
    this.initErrors = [];
  }

  // Debug method to log available files
  private async checkFileExists(url: string): Promise<boolean> {
    try {
      const response = await fetch(url, { method: 'HEAD' });
      console.log(`File check for ${url}: ${response.status === 200 ? 'exists' : 'not found'}`);
      return response.status === 200;
    } catch (error) {
      console.error(`Error checking file at ${url}:`, error);
      this.initErrors.push(`File check error for ${url}: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  // Reset the service state for retrying
  public resetState(): void {
    this.modelLoaded = false;
    this.modelLoading = false;
    this.session = null;
    this.clearInitializationErrors();
  }

  // Initialize model with path and options
  public async initModel(forceReload = false): Promise<boolean> {
    // If already loaded and not forcing reload, return current state
    if ((this.modelLoaded || this.modelLoading) && !forceReload) {
      return this.modelLoaded;
    }

    // If forcing reload, reset state
    if (forceReload) {
      this.resetState();
    }

    try {
      this.modelLoading = true;
      this.clearInitializationErrors();
      console.log('Starting ONNX model initialization...');
      
      // Verify essential files exist before proceeding
      console.log('Checking if model files exist...');
      
      const modelExists = await this.checkFileExists(this.modelPath);
      const metadataExists = await this.checkFileExists(this.metadataPath);
      const wasmExists = await this.checkFileExists(WASM_PATH);
      const wasmSimdExists = await this.checkFileExists(WASM_SIMD_PATH);
      const wasmThreadedExists = await this.checkFileExists(WASM_THREADED_PATH);
      
      // Add debug logging for file paths
      console.log(`[DEBUG] Model path being checked: ${this.modelPath}`);
      console.log(`[DEBUG] Metadata path being checked: ${this.metadataPath}`);
      
      if (!modelExists || !metadataExists) {
        const error = `Model files not found. Model: ${modelExists}, Metadata: ${metadataExists}`;
        this.initErrors.push(error);
        throw new Error(error);
      }
      
      if (!wasmExists || !wasmSimdExists || !wasmThreadedExists) {
        console.warn(`Some WASM files are missing. Basic: ${wasmExists}, SIMD: ${wasmSimdExists}, Threaded: ${wasmThreadedExists}`);
        this.initErrors.push(`Some WASM files are missing: Basic: ${wasmExists}, SIMD: ${wasmSimdExists}, Threaded: ${wasmThreadedExists}`);
      }
      
      // Load model metadata
      console.log('Loading model metadata...');
      let metadataResponse;
      try {
        metadataResponse = await fetch(this.metadataPath);
        if (!metadataResponse.ok) {
          const error = `Failed to fetch metadata: ${metadataResponse.status} ${metadataResponse.statusText}`;
          this.initErrors.push(error);
          throw new Error(error);
        }
      } catch (error) {
        const errorMsg = `Fetch error for metadata: ${error instanceof Error ? error.message : String(error)}`;
        this.initErrors.push(errorMsg);
        throw new Error(errorMsg);
      }
      
      try {
        this.metadata = await metadataResponse.json() as AnimeRecommenderMetadata;
        console.log('Model metadata loaded:', this.metadata);
      } catch (error) {
        const errorMsg = `Failed to parse metadata JSON: ${error instanceof Error ? error.message : String(error)}`;
        this.initErrors.push(errorMsg);
        throw new Error(errorMsg);
      }
      
      // Configure ONNX runtime
      console.log('Configuring ONNX runtime...');
      try {
        // Ensure ort.env.wasm exists
        if (!ort.env || !ort.env.wasm) {
          const error = 'ONNX Runtime Web environment not properly initialized. ort.env.wasm is undefined.';
          this.initErrors.push(error);
          throw new Error(error);
        }

        ort.env.wasm.wasmPaths = {
          'ort-wasm.wasm': WASM_PATH,
          'ort-wasm-simd.wasm': WASM_SIMD_PATH,
          'ort-wasm-threaded.wasm': WASM_THREADED_PATH,
        };
        console.log('WASM paths configured:', ort.env.wasm.wasmPaths);
      } catch (error) {
        const errorMsg = `WASM configuration failed: ${error instanceof Error ? error.message : String(error)}`;
        this.initErrors.push(errorMsg);
        throw new Error(errorMsg);
      }
      
      // Create session options
      console.log('Creating session options...');
      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      };
      
      // Log ONNX environment
      try {
        console.log('ONNX runtime version:', ort.env.versions);
        console.log('Execution providers:', ort.env.wasm);
      } catch (error) {
        console.warn('Could not log ONNX environment:', error);
      }
      
      // Load the model
      console.log('Loading ONNX model from:', this.modelPath);
      try {
        // Add a timeout to prevent hanging
        const modelPromise = ort.InferenceSession.create(this.modelPath, sessionOptions);
        
        // Create a timeout promise
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error('Model loading timed out after 30 seconds')), 30000);
        });
        
        // Race the promises
        this.session = await Promise.race([modelPromise, timeoutPromise]);
        console.log('ONNX model session created successfully');
      } catch (error) {
        console.error('Failed to create ONNX session:', error);
        if (error instanceof Error) {
          if (error.message.includes('CORS')) {
            console.error('CORS issue detected. Make sure proper CORS headers are set.');
            this.initErrors.push('CORS issue detected: ' + error.message);
          } else if (error.message.includes('timeout')) {
            this.initErrors.push('Model loading timed out: ' + error.message);
          } else {
            this.initErrors.push('Failed to create ONNX session: ' + error.message);
          }
        }
        throw error;
      }
      
      this.modelLoaded = true;
      console.log('ONNX model loaded successfully');
      
      return true;
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      this.modelLoaded = false;
      if (!this.initErrors.length && error instanceof Error) {
        this.initErrors.push(`Failed to load ONNX model: ${error.message}`);
      }
      return false;
    } finally {
      this.modelLoading = false;
      console.log('Model loading process completed, modelLoaded =', this.modelLoaded);
    }
  }

  // Check if model is loaded
  public isModelLoaded(): boolean {
    return this.modelLoaded && this.session !== null;
  }

  // Get model metadata
  public getMetadata(): AnimeRecommenderMetadata | null {
    return this.metadata;
  }

  // Run inference with the model
  public async runInference(
    userIdx: number,
    animeIdx: number,
    genreIndices: number[],
    tagIndices: number[]
  ): Promise<number> {
    if (!this.session || !this.metadata) {
      throw new Error('Model not loaded or metadata missing');
    }

    try {
      console.log('Running inference with inputs:', {
        userIdx,
        animeIdx,
        genreIndices: genreIndices.length,
        tagIndices: tagIndices.length,
      });
      
      // Pad genre and tag indices if needed
      const paddedGenreIndices = this.padArray(genreIndices, this.metadata.max_genres, -1);
      const paddedTagIndices = this.padArray(tagIndices, this.metadata.max_tags, -1);

      // Create input tensors
      console.log('Creating input tensors...');
      const feeds: Record<string, ort.Tensor> = {
        user_idx: new ort.Tensor('int64', [BigInt(userIdx)], [1]),
        anime_idx: new ort.Tensor('int64', [BigInt(animeIdx)], [1]),
        genre_indices: new ort.Tensor(
          'int64',
          paddedGenreIndices.map(idx => BigInt(idx)),
          [1, this.metadata.max_genres]
        ),
        tag_indices: new ort.Tensor(
          'int64',
          paddedTagIndices.map(idx => BigInt(idx)),
          [1, this.metadata.max_tags]
        ),
      };

      // Run inference
      console.log('Running model inference...');
      const results = await this.session.run(feeds);
      console.log('Inference completed, processing results...');
      
      // Get output tensor
      const outputTensor = results[this.metadata.output_names[0]];
      
      // Convert to original rating scale
      const normalizedRating = outputTensor.data[0] as number;
      const rating = normalizedRating * this.metadata.rating_normalization.std + 
                    this.metadata.rating_normalization.mean;
      
      console.log('Prediction result:', rating);
      return rating;
    } catch (error) {
      console.error('ONNX inference error:', error);
      throw error;
    }
  }

  // Utility to pad arrays to a fixed length
  private padArray(array: number[], length: number, padValue: number): number[] {
    if (array.length >= length) {
      return array.slice(0, length);
    }
    return [...array, ...Array(length - array.length).fill(padValue)];
  }

  /**
   * Get a JSON representation of the current model state for debugging
   */
  public debugState(): string {
    return JSON.stringify({
      modelLoaded: this.modelLoaded,
      modelLoading: this.modelLoading,
      modelPath: this.modelPath,
      metadataPath: this.metadataPath,
      hasSession: !!this.session,
      hasMetadata: !!this.metadata,
      initErrors: this.initErrors,
      metadata: this.metadata,
    });
  }
}

// Export a singleton instance
export const onnxModelService = OnnxModelService.getInstance();

// Then add this function outside the class
export function logOnnxServiceState(): void {
  console.log(`[ONNX-SERVICE] Current State: ${onnxModelService.debugState()}`);
} 
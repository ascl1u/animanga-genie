/**
 * Anime Recommender - Client-side recommendation engine
 * 
 * This class provides methods for loading the anime recommendation model
 * and generating recommendations using TensorFlow.js
 */
class AnimeRecommender {
  /**
   * Initialize the recommender
   * @param {string} modelPath - Path to the model directory (default: '/models/anime_recommender')
   */
  constructor(modelPath = '/models/anime_recommender') {
    this.modelPath = modelPath;
    this.model = null;
    this.metadata = null;
    this.mappings = null;
    this.animeLookup = null;
    this.tf = null;
    this.isLoaded = false;
  }

  /**
   * Load the model and associated data
   * @returns {Promise<boolean>} - Whether loading was successful
   */
  async load() {
    try {
      // Import TensorFlow.js
      const tf = await import('@tensorflow/tfjs');
      this.tf = tf;
      
      // Load model and data in parallel
      const [model, metadataResponse, mappingsResponse, lookupResponse] = await Promise.all([
        tf.loadGraphModel(`${this.modelPath}/model.json`),
        fetch(`${this.modelPath}/model_metadata.json`),
        fetch(`${this.modelPath}/model_mappings.json`),
        fetch(`${this.modelPath}/anime_lookup.json`).catch(() => ({ json: () => ({}) }))
      ]);
      
      this.model = model;
      this.metadata = await metadataResponse.json();
      this.mappings = await mappingsResponse.json();
      this.animeLookup = await lookupResponse.json();
      
      this.isLoaded = true;
      console.log('Anime recommender model loaded successfully');
      return true;
    } catch (error) {
      console.error('Failed to load anime recommender model:', error);
      return false;
    }
  }

  /**
   * Get recommendations for a user
   * @param {number} userId - User ID
   * @param {number} limit - Maximum number of recommendations to return
   * @returns {Promise<Array>} Array of recommended anime with scores
   */
  async getRecommendations(userId, limit = 10) {
    if (!this.isLoaded) {
      throw new Error('Model not loaded. Call load() first.');
    }

    // Get all available anime IDs
    const animeIds = Object.keys(this.mappings.anime_to_idx || {});
    const predictions = [];
    
    // Process anime in batches to avoid memory issues
    const batchSize = 50;
    for (let i = 0; i < animeIds.length; i += batchSize) {
      const batchIds = animeIds.slice(i, i + batchSize);
      const scores = await this.predictBatch(userId, batchIds);
      
      // Add predictions to results
      for (let j = 0; j < batchIds.length; j++) {
        const animeId = batchIds[j];
        const lookup = this.animeLookup[animeId] || {};
        
        predictions.push({
          id: animeId,
          score: scores[j],
          title: lookup.title || `Anime ${animeId}`,
          genres: lookup.genres || [],
          tags: lookup.tags || []
        });
      }
    }
    
    // Sort by score (descending) and return top recommendations
    return predictions
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  /**
   * Predict scores for a batch of anime
   * @param {number} userId - User ID
   * @param {Array<string>} animeIds - Array of anime IDs
   * @returns {Promise<Array<number>>} Predicted scores
   */
  async predictBatch(userId, animeIds) {
    // Convert user ID to model index
    const userIdx = 0; // Default to first user for demo purposes
    
    // Convert anime IDs to model indices
    const animeIndices = animeIds.map(id => 
      parseInt(this.mappings.anime_to_idx[id] || 0)
    );
    
    // Create input tensors
    const batchSize = animeIds.length;
    const userTensor = this.tf.tensor1d(new Array(batchSize).fill(userIdx), 'int32');
    const animeTensor = this.tf.tensor1d(animeIndices, 'int32');
    
    // Create empty genre and tag tensors
    const genreTensor = this.tf.tensor2d(
      new Array(batchSize).fill(new Array(10).fill(-1)),
      [batchSize, 10],
      'int32'
    );
    
    const tagTensor = this.tf.tensor2d(
      new Array(batchSize).fill(new Array(20).fill(-1)),
      [batchSize, 20],
      'int32'
    );
    
    // Make prediction
    const result = await this.model.predict({
      user_idx: userTensor,
      anime_idx: animeTensor,
      genre_indices: genreTensor,
      tag_indices: tagTensor
    });
    
    // Get prediction values
    const scores = await result.data();
    
    // Clean up tensors
    this.tf.dispose([userTensor, animeTensor, genreTensor, tagTensor, result]);
    
    return Array.from(scores);
  }
}

export default AnimeRecommender;
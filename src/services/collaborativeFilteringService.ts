import { createClient } from '@/utils/supabase/client';
import { AnimeWatchHistoryItem } from '@/types/watchHistory';

/**
 * Interface for user-anime rating pairs
 */
interface RatingData {
  userId: string;
  animeId: number;
  rating: number;
}

/**
 * Interface for user similarity data
 */
interface UserSimilarity {
  userId: string;
  similarity: number;
}

/**
 * Interface for collaborative filtering recommendations
 */
export interface CollaborativeRecommendation {
  animeId: number;
  score: number;
  contributors: {
    userId: string;
    rating: number;
    weight: number;
  }[];
}

/**
 * Latent Factor Model using Singular Value Decomposition (SVD)
 * This implements a simplified version of SVD for matrix factorization
 */
class SVDModel {
  private userFactors: Map<string, number[]> = new Map();
  private itemFactors: Map<number, number[]> = new Map();
  private latentDimensions: number;
  private userBias: Map<string, number> = new Map();
  private itemBias: Map<number, number> = new Map();
  private globalMean: number = 0;

  constructor(latentDimensions: number = 20) {
    this.latentDimensions = latentDimensions;
  }

  /**
   * Train the SVD model using user ratings
   */
  public async train(ratings: RatingData[]): Promise<void> {
    if (ratings.length === 0) {
      console.log('[COLLABORATIVE] No ratings data available for training');
      return;
    }

    // Calculate global mean
    const sum = ratings.reduce((acc, r) => acc + r.rating, 0);
    this.globalMean = sum / ratings.length;
    console.log(`[COLLABORATIVE] Global mean rating: ${this.globalMean.toFixed(2)}`);

    // Collect unique users and items
    const uniqueUsers = new Set<string>();
    const uniqueItems = new Set<number>();
    
    ratings.forEach(r => {
      uniqueUsers.add(r.userId);
      uniqueItems.add(r.animeId);
    });

    console.log(`[COLLABORATIVE] Training model with ${uniqueUsers.size} users and ${uniqueItems.size} items`);

    // Calculate user and item bias
    const userRatings = new Map<string, number[]>();
    const itemRatings = new Map<number, number[]>();

    // Group ratings by user and item
    ratings.forEach(r => {
      if (!userRatings.has(r.userId)) {
        userRatings.set(r.userId, []);
      }
      userRatings.get(r.userId)!.push(r.rating);

      if (!itemRatings.has(r.animeId)) {
        itemRatings.set(r.animeId, []);
      }
      itemRatings.get(r.animeId)!.push(r.rating);
    });

    // Calculate bias
    userRatings.forEach((ratings, userId) => {
      const mean = ratings.reduce((a, b) => a + b, 0) / ratings.length;
      this.userBias.set(userId, mean - this.globalMean);
    });

    itemRatings.forEach((ratings, itemId) => {
      const mean = ratings.reduce((a, b) => a + b, 0) / ratings.length;
      this.itemBias.set(itemId, mean - this.globalMean);
    });

    // Create user-item matrix
    const userItemMatrix = new Map<string, Map<number, number>>();
    
    ratings.forEach(r => {
      if (!userItemMatrix.has(r.userId)) {
        userItemMatrix.set(r.userId, new Map<number, number>());
      }
      userItemMatrix.get(r.userId)!.set(r.animeId, r.rating);
    });

    // Simple stochastic gradient descent implementation of SVD
    // Initialize factors with small random values
    uniqueUsers.forEach(userId => {
      this.userFactors.set(userId, Array.from(
        { length: this.latentDimensions }, 
        () => (Math.random() - 0.5) * 0.1
      ));
    });

    uniqueItems.forEach(itemId => {
      this.itemFactors.set(itemId, Array.from(
        { length: this.latentDimensions }, 
        () => (Math.random() - 0.5) * 0.1
      ));
    });

    // SGD parameters
    const iterations = 50;
    const learningRate = 0.005;
    const regularization = 0.02;
    
    // Training loop
    for (let iter = 0; iter < iterations; iter++) {
      let totalError = 0;
      
      for (const rating of ratings) {
        const userId = rating.userId;
        const itemId = rating.animeId;
        const actualRating = rating.rating;
        
        // Get factors
        const userFactor = this.userFactors.get(userId)!;
        const itemFactor = this.itemFactors.get(itemId)!;
        const userBias = this.userBias.get(userId) || 0;
        const itemBias = this.itemBias.get(itemId) || 0;
        
        // Predict rating using dot product + bias
        let predictedRating = this.globalMean + userBias + itemBias;
        for (let f = 0; f < this.latentDimensions; f++) {
          predictedRating += userFactor[f] * itemFactor[f];
        }
        
        // Calculate error
        const error = actualRating - predictedRating;
        totalError += error * error;
        
        // Update bias
        this.userBias.set(userId, userBias + learningRate * (error - regularization * userBias));
        this.itemBias.set(itemId, itemBias + learningRate * (error - regularization * itemBias));
        
        // Update latent factors
        for (let f = 0; f < this.latentDimensions; f++) {
          const userOld = userFactor[f];
          const itemOld = itemFactor[f];
          
          userFactor[f] += learningRate * (error * itemOld - regularization * userOld);
          itemFactor[f] += learningRate * (error * userOld - regularization * itemOld);
        }
      }
      
      const rmse = Math.sqrt(totalError / ratings.length);
      if (iter % 10 === 0 || iter === iterations - 1) {
        console.log(`[COLLABORATIVE] Iteration ${iter+1}/${iterations}, RMSE: ${rmse.toFixed(4)}`);
      }
    }

    console.log('[COLLABORATIVE] Model training completed');
  }

  /**
   * Predict rating for a user-item pair
   */
  public predict(userId: string, itemId: number): number {
    const userFactor = this.userFactors.get(userId);
    const itemFactor = this.itemFactors.get(itemId);
    
    if (!userFactor || !itemFactor) {
      // If we don't have factors for this user or item, return global mean
      return this.globalMean;
    }
    
    const userBias = this.userBias.get(userId) || 0;
    const itemBias = this.itemBias.get(itemId) || 0;
    
    // Calculate prediction using dot product
    let prediction = this.globalMean + userBias + itemBias;
    for (let f = 0; f < this.latentDimensions; f++) {
      prediction += userFactor[f] * itemFactor[f];
    }
    
    // Clip to rating range (1-10)
    return Math.max(1, Math.min(10, prediction));
  }

  /**
   * Get similar users for a given user
   */
  public getSimilarUsers(userId: string, limit: number = 10): UserSimilarity[] {
    const userFactor = this.userFactors.get(userId);
    if (!userFactor) {
      return [];
    }
    
    const similarities: UserSimilarity[] = [];
    
    // Calculate cosine similarity with all other users
    this.userFactors.forEach((factors, otherUserId) => {
      if (otherUserId === userId) return;
      
      // Calculate dot product
      let dotProduct = 0;
      let userMagnitude = 0;
      let otherMagnitude = 0;
      
      for (let f = 0; f < this.latentDimensions; f++) {
        dotProduct += userFactor[f] * factors[f];
        userMagnitude += userFactor[f] * userFactor[f];
        otherMagnitude += factors[f] * factors[f];
      }
      
      userMagnitude = Math.sqrt(userMagnitude);
      otherMagnitude = Math.sqrt(otherMagnitude);
      
      // Cosine similarity
      const similarity = dotProduct / (userMagnitude * otherMagnitude + 1e-8);
      
      similarities.push({ userId: otherUserId, similarity });
    });
    
    // Sort by similarity (descending) and take top N
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  }

  /**
   * Get latent factors for a user
   */
  public getUserFactors(userId: string): number[] | undefined {
    return this.userFactors.get(userId);
  }

  /**
   * Get latent factors for an item
   */
  public getItemFactors(itemId: number): number[] | undefined {
    return this.itemFactors.get(itemId);
  }

  /**
   * Get all users in the model
   */
  public getUsers(): string[] {
    return Array.from(this.userFactors.keys());
  }

  /**
   * Get all items in the model
   */
  public getItems(): number[] {
    return Array.from(this.itemFactors.keys());
  }
}

// Singleton collaborative filtering service
class CollaborativeFilteringService {
  private static instance: CollaborativeFilteringService;
  private supabase = createClient();
  private model: SVDModel | null = null;
  private isModelTrained = false;
  private isTraining = false;
  private lastTrainingTime = 0;
  private cachedRatings: RatingData[] = [];
  private trainingPromise: Promise<void> | null = null;
  
  private constructor() {}
  
  public static getInstance(): CollaborativeFilteringService {
    if (!CollaborativeFilteringService.instance) {
      CollaborativeFilteringService.instance = new CollaborativeFilteringService();
    }
    return CollaborativeFilteringService.instance;
  }
  
  /**
   * Fetch watch history data for all users from the database
   */
  private async fetchAllWatchHistoryData(): Promise<RatingData[]> {
    try {
      console.log('[COLLABORATIVE] Fetching all watch history data from database');
      
      const { data, error } = await this.supabase
        .from('anime_watch_history')
        .select('user_id, anilist_id, rating')
        .gt('rating', 0); // Only include items with ratings
      
      if (error) {
        console.error('[COLLABORATIVE] Error fetching watch history:', error);
        return [];
      }
      
      // Transform to the format we need
      const ratings: RatingData[] = data.map(item => ({
        userId: item.user_id,
        animeId: item.anilist_id,
        rating: item.rating
      }));
      
      console.log(`[COLLABORATIVE] Fetched ${ratings.length} ratings from ${new Set(ratings.map(r => r.userId)).size} users`);
      
      return ratings;
    } catch (error) {
      console.error('[COLLABORATIVE] Error fetching watch history data:', error);
      return [];
    }
  }
  
  /**
   * Train the collaborative filtering model
   */
  public async trainModel(): Promise<void> {
    // If already training, return the existing promise
    if (this.isTraining && this.trainingPromise) {
      return this.trainingPromise;
    }
    
    // If model was trained recently (within last hour), skip training
    const currentTime = Date.now();
    if (this.isModelTrained && currentTime - this.lastTrainingTime < 60 * 60 * 1000) {
      console.log('[COLLABORATIVE] Using recently trained model');
      return;
    }
    
    this.isTraining = true;
    this.trainingPromise = (async () => {
      try {
        // Fetch ratings data
        this.cachedRatings = await this.fetchAllWatchHistoryData();
        
        if (this.cachedRatings.length === 0) {
          console.log('[COLLABORATIVE] No ratings data available, skipping training');
          return;
        }
        
        // Initialize model
        this.model = new SVDModel();
        
        // Train model
        await this.model.train(this.cachedRatings);
        
        this.isModelTrained = true;
        this.lastTrainingTime = Date.now();
        console.log('[COLLABORATIVE] Model training completed successfully');
      } catch (error) {
        console.error('[COLLABORATIVE] Error training model:', error);
        this.isModelTrained = false;
      } finally {
        this.isTraining = false;
        this.trainingPromise = null;
      }
    })();
    
    return this.trainingPromise;
  }
  
  /**
   * Check if the model is trained
   */
  public isReady(): boolean {
    return this.isModelTrained && this.model !== null;
  }
  
  /**
   * Get similar users for a given user
   */
  public async getSimilarUsers(userId: string, limit: number = 10): Promise<UserSimilarity[]> {
    await this.ensureModelTrained();
    
    if (!this.model) {
      return [];
    }
    
    return this.model.getSimilarUsers(userId, limit);
  }
  
  /**
   * Generate recommendations for a user using collaborative filtering
   */
  public async getRecommendations(
    userId: string, 
    userWatchHistory: AnimeWatchHistoryItem[], 
    limit: number = 10
  ): Promise<CollaborativeRecommendation[]> {
    await this.ensureModelTrained();
    
    if (!this.model) {
      return [];
    }
    
    // Get already watched anime IDs to exclude them from recommendations
    const watchedAnimeIds = new Set(userWatchHistory.map(item => item.anilist_id));
    
    // Get similar users
    const similarUsers = this.model.getSimilarUsers(userId, 20);
    
    if (similarUsers.length === 0) {
      console.log('[COLLABORATIVE] No similar users found for user', userId);
      return [];
    }
    
    // Log similar users for debugging
    console.log('[COLLABORATIVE] Similar users for', userId, ':', 
      similarUsers.slice(0, 5).map(u => `${u.userId.slice(0, 6)}(${u.similarity.toFixed(2)})`).join(', '));
    
    // Get all available items
    const allItems = this.model.getItems();
    
    // For each item the user hasn't rated, predict the rating
    const predictions: CollaborativeRecommendation[] = [];
    
    for (const itemId of allItems) {
      // Skip already watched items
      if (watchedAnimeIds.has(itemId)) {
        continue;
      }
      
      // Get similar users who have rated this item
      const contributors = similarUsers
        .map(user => {
          const userRating = this.cachedRatings.find(
            r => r.userId === user.userId && r.animeId === itemId
          );
          
          if (userRating) {
            return {
              userId: user.userId,
              rating: userRating.rating,
              weight: user.similarity
            };
          }
          return null;
        })
        .filter((item): item is NonNullable<typeof item> => item !== null);
      
      // If no similar users have rated this item, skip it
      if (contributors.length === 0) {
        continue;
      }
      
      // Predict rating
      const predictedRating = this.model.predict(userId, itemId);
      
      // Add to predictions
      predictions.push({
        animeId: itemId,
        score: predictedRating,
        contributors
      });
    }
    
    // Sort by predicted rating (descending) and take top N
    return predictions
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }
  
  /**
   * Ensure the model is trained
   */
  private async ensureModelTrained(): Promise<void> {
    if (!this.isModelTrained) {
      await this.trainModel();
    }
  }
  
  /**
   * Get user latent factors
   */
  public async getUserFactors(userId: string): Promise<number[] | undefined> {
    await this.ensureModelTrained();
    return this.model?.getUserFactors(userId);
  }
}

// Export singleton instance
export const collaborativeFilteringService = CollaborativeFilteringService.getInstance(); 
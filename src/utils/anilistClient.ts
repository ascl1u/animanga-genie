import { ApolloClient, InMemoryCache, HttpLink, gql } from '@apollo/client';

// Create an Apollo Client instance with proper cache configuration
export const anilistClient = new ApolloClient({
  link: new HttpLink({
    uri: '/api/anilist',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    }
  }),
  cache: new InMemoryCache({
    typePolicies: {
      Page: {
        // Tell Apollo this type doesn't have a key field and shouldn't be normalized
        keyFields: false,
        // Fields that return non-normalized objects
        fields: {
          media: {
            // Don't merge arrays of media, replace them
            merge: false
          }
        }
      }
    }
  }),
});

// GraphQL query for searching anime using gql tag
export const SEARCH_ANIME_QUERY = gql`
  query SearchAnime($search: String) {
    Page(page: 1, perPage: 10) {
      media(search: $search, type: ANIME, sort: POPULARITY_DESC) {
        id
        title {
          romaji
          english
          native
        }
        coverImage {
          medium
        }
        format
        seasonYear
        averageScore
      }
    }
  }
`;

// GraphQL query for fetching detailed anime information
export const GET_ANIME_DETAILS_QUERY = gql`
  query GetAnimeDetails($id: Int) {
    Media(id: $id, type: ANIME) {
      id
      title {
        romaji
        english
        native
      }
      genres
      tags {
        id
        name
        rank
        category
      }
      coverImage {
        medium
        large
      }
      format
      seasonYear
      averageScore
      popularity
      episodes
      duration
      status
      description
      relations {
        edges {
          id
          relationType
          node {
            id
            title {
              romaji
              english
            }
            type
            format
          }
        }
      }
      studios {
        edges {
          id
          isMain
          node {
            id
            name
          }
        }
      }
    }
  }
`;

// GraphQL query for fetching user's anime list from AniList
export const GET_USER_ANIME_LIST_QUERY = gql`
  query GetUserAnimeList($username: String) {
    MediaListCollection(userName: $username, type: ANIME) {
      lists {
        name
        status
        entries {
          id
          status
          score
          media {
            id
            title {
              romaji
              english
              native
            }
            coverImage {
              medium
              large
            }
          }
        }
      }
    }
  }
`;

// Type for anime search result
export type AnimeSearchResult = {
  id: number;
  title: {
    romaji: string;
    english: string;
    native: string;
  };
  coverImage: {
    medium: string;
  };
  format: string;
  seasonYear: number;
  averageScore: number;
};

// Type for detailed anime information including genres and tags
export interface AnimeDetails extends AnimeSearchResult {
  genres: string[];
  tags: {
    id: number;
    name: string;
    rank: number;
    category: string;
  }[];
  coverImage: {
    medium: string;
    large: string;
  };
  popularity: number;
  episodes: number;
  duration: number;
  status: string;
  description: string;
  relations?: {
    edges: {
      id: number;
      relationType: string; // PREQUEL, SEQUEL, SIDE_STORY, etc.
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
  };
  studios?: {
    edges: {
      id: number;
      isMain: boolean;
      node: {
        id: number;
        name: string;
      };
    }[];
  };
}

// Type for user anime list from AniList
export interface AnilistUserAnimeEntry {
  id: number;
  status: string;
  score: number;
  media: {
    id: number;
    title: {
      romaji: string;
      english: string;
      native: string;
    };
    coverImage: {
      medium: string;
      large: string;
    };
  };
}

export interface AnilistUserAnimeList {
  name: string;
  status: string;
  entries: AnilistUserAnimeEntry[];
}

// Helper function to handle rate limited requests with retries
interface GraphQLError {
  message?: string;
  extensions?: {
    code?: string;
    retryAfter?: number;
  };
}

interface ApolloError {
  graphQLErrors?: GraphQLError[];
  message?: string;
}

// API request queue to control concurrency and respect rate limits
class ApiRequestQueue {
  private queue: Array<() => Promise<unknown>> = [];
  private running = 0;
  private maxConcurrent = 1;
  private waitBetweenRequests = 1000; // 1 second between requests
  private retryDelay = 60000; // Default 60 seconds delay after a rate limit
  private isProcessing = false;
  
  constructor(maxConcurrent = 1, waitBetweenRequests = 1000) {
    this.maxConcurrent = maxConcurrent;
    this.waitBetweenRequests = waitBetweenRequests;
  }
  
  // Add a request to the queue
  public add<T>(requestFn: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      // Wrap the request in a function that resolves/rejects the returned promise
      this.queue.push(async () => {
        try {
          // Wait between requests to avoid hitting rate limits
          await new Promise(r => setTimeout(r, this.waitBetweenRequests));
          const result = await requestFn();
          resolve(result);
          return result;
        } catch (error) {
          // Handle rate limiting specifically
          const apolloError = error as ApolloError;
          
          // Check if this is a rate limit error
          if (
            apolloError.graphQLErrors?.some((e: GraphQLError) => 
              e.extensions?.code === 'RATE_LIMITED' || 
              (e.message && (
                e.message.includes('rate limit') || 
                e.message.includes('too many requests')
              ))
            ) ||
            apolloError.message?.includes('429')
          ) {
            // Get retry delay from error if available, otherwise use default
            const retryAfter = apolloError.graphQLErrors?.[0]?.extensions?.retryAfter || 60;
            // Use longer delay for the whole queue when rate limited
            this.waitBetweenRequests = Math.max(this.waitBetweenRequests, retryAfter * 1000);
            console.log(`[ANILIST] Rate limit reached. Increasing delay to ${this.waitBetweenRequests/1000}s between requests.`);
            
            // Re-add this request to the end of the queue
            return this.add(requestFn).then(resolve).catch(reject);
          }
          
          reject(error);
          return null;
        }
      });
      
      // Start processing the queue if it's not already running
      if (!this.isProcessing) {
        this.processQueue();
      }
    });
  }
  
  // Process the queue with controlled concurrency
  private async processQueue() {
    if (this.isProcessing) return;
    this.isProcessing = true;
    
    while (this.queue.length > 0) {
      if (this.running < this.maxConcurrent) {
        const request = this.queue.shift();
        if (request) {
          this.running++;
          
          // Execute the request and decrease running count when done
          request().finally(() => {
            this.running--;
          });
        }
      }
      
      // Small delay to avoid tight loops
      await new Promise(r => setTimeout(r, 100));
    }
    
    this.isProcessing = false;
  }
  
  // Reset the queue delays to default values
  public resetDelays() {
    this.waitBetweenRequests = 1000;
  }
}

// Create a singleton request queue for AniList
const anilistRequestQueue = new ApiRequestQueue(1, 1000);

const handleRateLimitedRequest = async <T>(
  requestFn: () => Promise<T>
): Promise<T> => {
  // Add the request to our controlled queue instead of executing directly
  return anilistRequestQueue.add(requestFn);
};

// Function to search for anime
export const searchAnime = async (searchText: string): Promise<AnimeSearchResult[]> => {
  try {
    const result = await handleRateLimitedRequest(async () => {
      const { data } = await anilistClient.query({
        query: SEARCH_ANIME_QUERY,
        variables: { search: searchText },
      });
      return data.Page.media;
    });
    
    return result;
  } catch (error) {
    console.error('Error searching anime:', error);
    return [];
  }
};

// Function to get detailed anime information including genres and tags
export const getAnimeDetails = async (animeId: number): Promise<AnimeDetails | null> => {
  try {
    console.log(`[ANILIST] Fetching details for anime ID: ${animeId}`);
    
    const result = await handleRateLimitedRequest(async () => {
      const { data } = await anilistClient.query({
        query: GET_ANIME_DETAILS_QUERY,
        variables: { id: animeId },
      });
      return data.Media;
    });
    
    console.log(`[ANILIST] Received details for anime: ${result.title.english || result.title.romaji}`);
    console.log(`[ANILIST] Genres: ${result.genres.join(', ')}`);
    console.log(`[ANILIST] Tags count: ${result.tags.length}`);
    
    return result;
  } catch (error) {
    console.error(`[ANILIST] Error fetching anime details for ID ${animeId}:`, error);
    return null;
  }
};

// Function to get user's anime list from AniList
export const getUserAnimeList = async (username: string): Promise<AnilistUserAnimeList[]> => {
  try {
    console.log(`[ANILIST] Fetching anime list for user: ${username}`);
    
    const result = await handleRateLimitedRequest(async () => {
      const { data } = await anilistClient.query({
        query: GET_USER_ANIME_LIST_QUERY,
        variables: { username },
      });
      return data.MediaListCollection.lists || [];
    });
    
    console.log(`[ANILIST] Received anime list for user: ${username}`);
    return result;
  } catch (error) {
    console.error(`[ANILIST] Error fetching anime list for user ${username}:`, error);
    return [];
  }
};

// Batch processing function for anime details
export const getAnimeDetailsInBatches = async (
  animeIds: number[],
  progressCallback?: (completed: number, total: number) => void
): Promise<Map<number, AnimeDetails>> => {
  const results = new Map<number, AnimeDetails>();
  let completed = 0;
  
  // Process each anime ID one at a time through the queue
  for (const animeId of animeIds) {
    try {
      const animeDetails = await getAnimeDetails(animeId);
      if (animeDetails) {
        results.set(animeId, animeDetails);
      }
      
      completed++;
      if (progressCallback) {
        progressCallback(completed, animeIds.length);
      }
    } catch (error) {
      console.error(`[ANILIST] Failed to get details for anime ID ${animeId}:`, error);
      completed++;
      if (progressCallback) {
        progressCallback(completed, animeIds.length);
      }
    }
  }
  
  return results;
}; 
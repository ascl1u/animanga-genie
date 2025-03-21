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
    };
  };
}

export interface AnilistUserAnimeList {
  name: string;
  status: string;
  entries: AnilistUserAnimeEntry[];
}

// Function to search for anime
export const searchAnime = async (searchText: string): Promise<AnimeSearchResult[]> => {
  try {
    const { data } = await anilistClient.query({
      query: SEARCH_ANIME_QUERY,
      variables: { search: searchText },
    });
    
    return data.Page.media;
  } catch (error) {
    console.error('Error searching anime:', error);
    return [];
  }
};

// Function to get detailed anime information including genres and tags
export const getAnimeDetails = async (animeId: number): Promise<AnimeDetails | null> => {
  try {
    console.log(`[ANILIST] Fetching details for anime ID: ${animeId}`);
    const { data } = await anilistClient.query({
      query: GET_ANIME_DETAILS_QUERY,
      variables: { id: animeId },
    });
    
    console.log(`[ANILIST] Received details for anime: ${data.Media.title.english || data.Media.title.romaji}`);
    console.log(`[ANILIST] Genres: ${data.Media.genres.join(', ')}`);
    console.log(`[ANILIST] Tags count: ${data.Media.tags.length}`);
    
    return data.Media;
  } catch (error) {
    console.error(`[ANILIST] Error fetching anime details for ID ${animeId}:`, error);
    return null;
  }
};

// Function to get user's anime list from AniList
export const getUserAnimeList = async (username: string): Promise<AnilistUserAnimeList[]> => {
  try {
    console.log(`[ANILIST] Fetching anime list for user: ${username}`);
    const { data } = await anilistClient.query({
      query: GET_USER_ANIME_LIST_QUERY,
      variables: { username },
    });
    
    console.log(`[ANILIST] Received anime list for user: ${username}`);
    return data.MediaListCollection.lists || [];
  } catch (error) {
    console.error(`[ANILIST] Error fetching anime list for user ${username}:`, error);
    return [];
  }
}; 
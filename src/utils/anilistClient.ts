import { ApolloClient, InMemoryCache, HttpLink, gql } from '@apollo/client';

// Create an Apollo Client instance
export const anilistClient = new ApolloClient({
  link: new HttpLink({
    uri: '/api/anilist',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    }
  }),
  cache: new InMemoryCache(),
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
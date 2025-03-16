import { ApolloClient, InMemoryCache, HttpLink, gql } from '@apollo/client';

// Create an Apollo Client instance
export const anilistClient = new ApolloClient({
  link: new HttpLink({
    uri: 'https://graphql.anilist.co',
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

// Function to search for anime
export const searchAnime = async (searchText: string): Promise<AnimeSearchResult[]> => {
  try {
    const response = await anilistClient.query({
      query: SEARCH_ANIME_QUERY,
      variables: {
        search: searchText
      }
    });
    
    return response.data?.Page?.media || [];
  } catch (error) {
    console.error('Error searching anime:', error);
    return [];
  }
}; 
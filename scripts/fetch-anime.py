#!/usr/bin/env python3
"""
Fetch Anime Data Script

This script fetches anime data from the AniList GraphQL API and stores it locally
in JSON format for use in training our recommendation model.
"""

import json
import time
import os
import argparse
from typing import List, Dict, Any, Optional
import requests


class AnimeDataFetcher:
    """
    A class for fetching anime data from the AniList GraphQL API.
    
    This fetcher collects anime data including titles, genres, tags, and ratings
    for use in our recommendation system training.
    """
    
    def __init__(self, output_dir: str = "data", page_size: int = 100):
        """
        Initialize the AnimeDataFetcher.
        
        Args:
            output_dir: Directory to store output JSON files
            page_size: Number of anime to fetch per page/request
        """
        self.api_url = "https://graphql.anilist.co"
        self.page_size = page_size
        self.output_dir = output_dir
        self.anime_data = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def fetch_anime_page(self, page: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a single page of anime data from AniList.
        
        Args:
            page: Page number to fetch
            
        Returns:
            Dictionary containing the API response or None if request failed
        """
        query = """
        query ($page: Int, $perPage: Int) {
            Page(page: $page, perPage: $perPage) {
                pageInfo {
                    hasNextPage
                    total
                }
                media(type: ANIME, sort: POPULARITY_DESC) {
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
                    averageScore
                    popularity
                    format
                    episodes
                    duration
                    status
                    seasonYear
                    description
                    coverImage {
                        medium
                        large
                    }
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
        }
        """
        
        variables = {
            "page": page,
            "perPage": self.page_size
        }
        
        try:
            response = requests.post(
                self.api_url,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 429:
                print(f"Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                return self.fetch_anime_page(page)
                
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            time.sleep(5)  # Wait a bit before retrying or continuing
            return None

    def process_anime_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process and transform raw API response into our desired format.
        
        Args:
            data: Raw API response data
            
        Returns:
            List of processed anime entries
        """
        processed_anime = []
        
        for anime in data["data"]["Page"]["media"]:
            # Only process anime that have sufficient data
            if not anime["genres"] or anime["averageScore"] is None:
                continue
                
            processed_entry = {
                "anilist_id": anime["id"],
                "title": {
                    "romaji": anime["title"]["romaji"],
                    "english": anime["title"]["english"],
                    "native": anime["title"]["native"]
                },
                "genres": anime["genres"],
                "tags": anime["tags"],  # Keep full tag objects with rank and category
                "rating": anime["averageScore"] / 10.0,  # Convert to 0-10 scale
                "popularity": anime["popularity"],
                "format": anime["format"],
                "episodes": anime["episodes"],
                "duration": anime["duration"],
                "status": anime["status"],
                "year": anime["seasonYear"],
                "description": anime["description"],
                "image_url": anime["coverImage"]["medium"] or "",
                "relations": anime["relations"]["edges"] if anime.get("relations") else [],
                "studios": anime["studios"]["edges"] if anime.get("studios") else []
            }
            processed_anime.append(processed_entry)
            
        return processed_anime

    def fetch_all_anime(self, max_pages: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch and process multiple pages of anime data.
        
        Args:
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of all processed anime data
        """
        page = 1
        has_next_page = True
        total_collected = 0
        empty_pages_in_a_row = 0  # Track consecutive empty pages
        
        while has_next_page and page <= max_pages:
            print(f"Fetching page {page}...")
            response = self.fetch_anime_page(page)
            
            if not response:
                print(f"Failed to fetch page {page}, trying next page")
                page += 1
                continue
                
            try:
                page_info = response["data"]["Page"]["pageInfo"]
                anime_page = self.process_anime_data(response)
                
                # Check if the page yielded no anime
                if len(anime_page) == 0:
                    empty_pages_in_a_row += 1
                    print(f"Page {page} yielded 0 anime. Empty pages in a row: {empty_pages_in_a_row}")
                    
                    # Early stop if we have 5 empty pages in a row
                    if empty_pages_in_a_row >= 5:
                        print(f"Detected {empty_pages_in_a_row} empty pages in a row. Early stopping to avoid wasting API calls.")
                        break
                else:
                    # Reset counter when we find anime
                    empty_pages_in_a_row = 0
                
                self.anime_data.extend(anime_page)
                total_collected += len(anime_page)
                
                print(f"Collected {len(anime_page)} anime from page {page}")
                print(f"Total collected: {total_collected}")
                
                has_next_page = page_info["hasNextPage"]
                page += 1
                
                # Rate limiting - be nice to the API
                time.sleep(1)
            except (KeyError, TypeError) as e:
                print(f"Error processing page {page}: {e}")
                page += 1
        
        if empty_pages_in_a_row >= 5:
            print("Script ended early due to lack of new anime data.")
            if total_collected > 0:
                self.save_to_json() # Make sure to save what we've got
                print(f"Saved {total_collected} anime entries despite early termination.")
        elif page > max_pages:
            print(f"Reached maximum page limit of {max_pages}.")
        else:
            print("No more pages to fetch.")
            
        return self.anime_data
    
    def save_to_json(self, filename: str = "anime_catalog.json") -> None:
        """
        Save collected anime data to a JSON file.
        
        Args:
            filename: Name of the output file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.anime_data, f, ensure_ascii=False, indent=2)
            
        print(f"Saved {len(self.anime_data)} anime entries to {output_path}")


def main():
    """Execute the anime data fetching process."""
    parser = argparse.ArgumentParser(description="Fetch anime data from AniList API")
    parser.add_argument("--output-dir", default="data", help="Directory to store output files")
    parser.add_argument("--page-size", type=int, default=100, help="Number of anime per page")
    parser.add_argument("--max-pages", type=int, default=500, help="Maximum number of pages to fetch")
    
    args = parser.parse_args()
    
    fetcher = AnimeDataFetcher(output_dir=args.output_dir, page_size=args.page_size)
    fetcher.fetch_all_anime(max_pages=args.max_pages)
    fetcher.save_to_json()
    
    print("Anime data fetching completed!")


if __name__ == "__main__":
    main() 
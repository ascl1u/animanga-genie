#!/usr/bin/env python3
"""
Fetch User Data Script

This script fetches user preference data from the AniList GraphQL API, including
watched anime lists and ratings to use for training our recommendation model.
"""

import json
import time
import os
import argparse
import random
from typing import List, Dict, Any, Optional, Set
import requests


class UserDataFetcher:
    """
    A class for fetching user preference data from the AniList GraphQL API.
    
    This fetcher collects user anime lists with ratings, allowing us to build
    a user-item interaction dataset for collaborative filtering.
    """
    
    def __init__(self, 
                 output_dir: str = "data", 
                 min_list_size: int = 30, 
                 max_users: int = 1000,
                 include_usernames: bool = False):
        """
        Initialize the UserDataFetcher.
        
        Args:
            output_dir: Directory to store output JSON files
            min_list_size: Minimum number of rated anime for a user to be included
            max_users: Maximum number of users to collect data for
            include_usernames: Whether to include usernames in the output (False for anonymity)
        """
        self.api_url = "https://graphql.anilist.co"
        self.output_dir = output_dir
        self.min_list_size = min_list_size
        self.max_users = max_users
        self.include_usernames = include_usernames
        self.user_data = []
        self.collected_user_ids = set()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load anime catalog if available (to validate anime IDs)
        self.anime_catalog = self._load_anime_catalog()
        self.valid_anime_ids = set(item["anilist_id"] for item in self.anime_catalog) if self.anime_catalog else set()
    
    def _load_anime_catalog(self) -> List[Dict[str, Any]]:
        """
        Load the anime catalog from the JSON file if it exists.
        
        Returns:
            List of anime or empty list if file doesn't exist
        """
        anime_catalog_path = os.path.join(self.output_dir, "anime_catalog.json")
        if os.path.exists(anime_catalog_path):
            try:
                with open(anime_catalog_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Error loading anime catalog - file may be corrupted")
                return []
        else:
            print("Anime catalog not found. Will collect all anime ratings.")
            return []

    def _fetch_random_users(self, count: int = 100) -> List[int]:
        """
        Fetch random active users from AniList.
        
        Args:
            count: Number of user IDs to fetch
            
        Returns:
            List of user IDs
        """
        # This approach uses the fact that AniList IDs are typically within a certain range
        # A more sophisticated approach would scrape user IDs from recent activity
        # But this simple approach works for our demo purposes
        
        # AniList user IDs range approximately from 1 to 6,000,000
        # We'll use a narrower range to target more active users
        min_id = 100000
        max_id = 1000000
        
        # Generate a set of random user IDs
        user_ids = set()
        while len(user_ids) < count:
            user_id = random.randint(min_id, max_id)
            user_ids.add(user_id)
            
        return list(user_ids)

    def fetch_user_list(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a user's anime list with ratings.
        
        Args:
            user_id: AniList user ID
            
        Returns:
            Dictionary with user data or None if request failed
        """
        query = """
        query ($userId: Int) {
            User(id: $userId) {
                id
                name
                statistics {
                    anime {
                        count
                    }
                }
            }
            MediaListCollection(userId: $userId, type: ANIME) {
                lists {
                    entries {
                        mediaId
                        score
                        status
                        media {
                            title {
                                romaji
                                english
                            }
                            genres
                            tags {
                                name
                            }
                        }
                    }
                }
            }
        }
        """
        
        variables = {
            "userId": user_id
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
                return self.fetch_user_list(user_id)
                
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching user {user_id}: {e}")
            time.sleep(3)  # Wait a bit before retrying or continuing
            return None

    def process_user_data(self, user_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process and transform raw API response into our desired format.
        
        Args:
            user_id: The user's ID
            data: Raw API response data
            
        Returns:
            Dictionary with processed user data or None if user should be skipped
        """
        if "errors" in data or not data.get("data") or not data["data"].get("User"):
            return None  # User not found or other API error
            
        user_data = data["data"]["User"]
        lists_data = data["data"]["MediaListCollection"]["lists"]
        
        # Extract all anime entries across all user's lists
        all_entries = []
        for list_obj in lists_data:
            all_entries.extend(list_obj["entries"])
        
        # Filter for entries with scores
        rated_entries = [entry for entry in all_entries if entry["score"] > 0]
        
        # Check if user has enough rated anime
        if len(rated_entries) < self.min_list_size:
            return None
        
        # Process each rated anime
        processed_ratings = []
        for entry in rated_entries:
            media_id = entry["mediaId"]
            
            # Skip if not in our anime catalog (optional)
            if self.valid_anime_ids and media_id not in self.valid_anime_ids:
                continue
                
            rating_data = {
                "anilist_id": media_id,
                "rating": entry["score"] / 10.0,  # Normalize to 0-1 scale
                "status": entry["status"]
            }
            
            # Include title and genre data if available
            if entry.get("media"):
                media = entry["media"]
                rating_data["title"] = media["title"]["english"] or media["title"]["romaji"]
                rating_data["genres"] = media["genres"]
                rating_data["tags"] = [tag["name"] for tag in media["tags"] if tag["name"]]
                
            processed_ratings.append(rating_data)
        
        # Skip if not enough valid ratings after filtering
        if len(processed_ratings) < self.min_list_size:
            return None
            
        # Create the user record
        processed_user = {
            "user_id": f"user_{user_id}",  # Anonymize the ID
            "ratings_count": len(processed_ratings),
            "ratings": processed_ratings
        }
        
        # Optionally include the username
        if self.include_usernames and user_data.get("name"):
            processed_user["username"] = user_data["name"]
            
        return processed_user

    def fetch_users_data(self) -> List[Dict[str, Any]]:
        """
        Fetch and process data for multiple users.
        
        Returns:
            List of processed user data dictionaries
        """
        # Start by getting a pool of random user IDs to try
        initial_user_pool = self._fetch_random_users(count=self.max_users * 3)  # Fetch extra to account for filtering
        
        total_fetched = 0
        processed_count = 0
        
        # Process users until we reach max_users or run out of candidates
        for user_id in initial_user_pool:
            if processed_count >= self.max_users:
                break
                
            if user_id in self.collected_user_ids:
                continue
                
            print(f"Fetching data for user {user_id}...")
            user_response = self.fetch_user_list(user_id)
            total_fetched += 1
            
            if not user_response:
                print(f"Failed to fetch user {user_id}")
                continue
                
            processed_user = self.process_user_data(user_id, user_response)
            
            if processed_user:
                self.user_data.append(processed_user)
                self.collected_user_ids.add(user_id)
                processed_count += 1
                print(f"Collected user {processed_count}/{self.max_users} - {len(processed_user['ratings'])} ratings")
            else:
                print(f"Skipped user {user_id} (insufficient data)")
                
            # Rate limiting - be nice to the API
            time.sleep(1)
            
            # Progress report every 10 users
            if total_fetched % 10 == 0:
                print(f"Progress: {processed_count}/{self.max_users} users collected " +
                      f"({total_fetched} total attempts)")
                
        print(f"Completed user data collection: {len(self.user_data)} users")
        return self.user_data
    
    def save_to_json(self, filename: str = "user_ratings.json") -> None:
        """
        Save collected user data to a JSON file.
        
        Args:
            filename: Name of the output file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.user_data, f, ensure_ascii=False, indent=2)
            
        print(f"Saved data for {len(self.user_data)} users to {output_path}")


def main():
    """Execute the user data fetching process."""
    parser = argparse.ArgumentParser(description="Fetch user data from AniList API")
    parser.add_argument("--output-dir", default="data", help="Directory to store output files")
    parser.add_argument("--min-list-size", type=int, default=30, 
                        help="Minimum number of rated anime for a user to be included")
    parser.add_argument("--max-users", type=int, default=100, 
                        help="Maximum number of users to collect data for")
    parser.add_argument("--include-usernames", action="store_true", 
                        help="Include usernames in the output (default: False for anonymity)")
    
    args = parser.parse_args()
    
    fetcher = UserDataFetcher(
        output_dir=args.output_dir,
        min_list_size=args.min_list_size,
        max_users=args.max_users,
        include_usernames=args.include_usernames
    )
    
    fetcher.fetch_users_data()
    fetcher.save_to_json()
    
    print("User data fetching completed!")


if __name__ == "__main__":
    main() 
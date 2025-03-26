#!/usr/bin/env python3
"""
Load Anime Data to Supabase

This script loads the collected anime data from our JSON files into the Supabase database
for use in the web application. It reads from the locally stored data created by the
fetch-anime.py script.
"""

import json
import os
import time
import argparse
from typing import List, Dict, Any, Optional
import requests
import dotenv


class SupabaseLoader:
    """
    A class for loading anime data into Supabase.
    
    This loader takes the local JSON data and inserts it into the Supabase database
    using the REST API for database operations.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 batch_size: int = 50):
        """
        Initialize the SupabaseLoader.
        
        Args:
            data_dir: Directory where JSON data is stored
            batch_size: Number of records to insert in a single batch
        """
        # Load environment variables from .env.local
        dotenv.load_dotenv(".env.local")
        
        self.supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        self.supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing Supabase credentials. Ensure NEXT_PUBLIC_SUPABASE_URL and "
                            "NEXT_PUBLIC_SUPABASE_ANON_KEY are set in .env.local")
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.rest_url = f"{self.supabase_url}/rest/v1"
        self.headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
    
    def load_anime_data(self) -> List[Dict[str, Any]]:
        """
        Load anime data from the JSON file.
        
        Returns:
            List of anime data dictionaries
        """
        anime_file = os.path.join(self.data_dir, "anime_catalog.json")
        
        if not os.path.exists(anime_file):
            raise FileNotFoundError(f"Anime data file not found at {anime_file}. "
                                   "Run fetch-anime.py first to collect the data.")
        
        with open(anime_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def format_anime_for_supabase(self, anime: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format anime data for insertion into Supabase.
        
        Args:
            anime: Raw anime data dictionary
            
        Returns:
            Formatted anime data for Supabase
        """
        return {
            "anilist_id": anime["anilist_id"],
            "title": json.dumps(anime["title"]),
            "rating": anime["rating"],
            "genres": json.dumps(anime["genres"]),
            "tags": json.dumps(anime["tags"]),
            "popularity": anime.get("popularity"),
            "format": anime.get("format"),
            "episodes": anime.get("episodes"),
            "duration": anime.get("duration"),
            "status": anime.get("status"),
            "year": anime.get("year"),
            "description": anime.get("description"),
            "image_url": anime.get("image_url", ""),
            "relations": json.dumps(anime.get("relations", [])),
            "studios": json.dumps(anime.get("studios", [])),
            "created_at": "now()",
            "updated_at": "now()"
        }
    
    def insert_anime_batch(self, anime_batch: List[Dict[str, Any]]) -> bool:
        """
        Insert a batch of anime records into Supabase.
        
        Args:
            anime_batch: List of formatted anime dictionaries
            
        Returns:
            Success status (True if successful)
        """
        try:
            response = requests.post(
                f"{self.rest_url}/anime",
                headers=self.headers,
                json=anime_batch
            )
            
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"Error inserting batch: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return False
    
    def upsert_anime_batch(self, anime_batch: List[Dict[str, Any]]) -> bool:
        """
        Upsert (update or insert) a batch of anime records into Supabase.
        
        Args:
            anime_batch: List of formatted anime dictionaries
            
        Returns:
            Success status (True if successful)
        """
        try:
            # Use upsert with on_conflict=anilist_id to update existing records
            headers = self.headers.copy()
            headers["Prefer"] = "resolution=merge-duplicates"
            
            response = requests.post(
                f"{self.rest_url}/anime",
                headers=headers,
                json=anime_batch
            )
            
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"Error upserting batch: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return False
    
    def load_to_supabase(self, upsert: bool = True) -> None:
        """
        Load all anime data into Supabase in batches.
        
        Args:
            upsert: Whether to use upsert (update if exists) instead of insert
        """
        anime_data = self.load_anime_data()
        print(f"Loaded {len(anime_data)} anime entries from JSON file")
        
        total_anime = len(anime_data)
        processed = 0
        success_count = 0
        
        # Process anime in batches
        for i in range(0, total_anime, self.batch_size):
            batch = anime_data[i:i + self.batch_size]
            formatted_batch = [self.format_anime_for_supabase(anime) for anime in batch]
            
            print(f"Processing batch {i//self.batch_size + 1}/{(total_anime + self.batch_size - 1)//self.batch_size}...")
            
            if upsert:
                success = self.upsert_anime_batch(formatted_batch)
                operation = "upserted"
            else:
                success = self.insert_anime_batch(formatted_batch)
                operation = "inserted"
                
            if success:
                success_count += len(batch)
                print(f"Successfully {operation} {len(batch)} anime records")
            else:
                print(f"Failed to {operation} batch starting at index {i}")
                
            processed += len(batch)
            print(f"Progress: {processed}/{total_anime} ({processed/total_anime*100:.1f}%)")
            
            # Avoid rate limiting
            time.sleep(1)
        
        print(f"Data loading complete. {success_count}/{total_anime} anime records {operation} successfully.")


def main():
    """Execute the anime data loading process."""
    parser = argparse.ArgumentParser(description="Load anime data to Supabase")
    parser.add_argument("--data-dir", default="data", help="Directory where JSON data is stored")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of records to insert in a single batch")
    parser.add_argument("--upsert", action="store_true", help="Use upsert instead of insert (update if exists)")
    
    args = parser.parse_args()
    
    loader = SupabaseLoader(data_dir=args.data_dir, batch_size=args.batch_size)
    loader.load_to_supabase(upsert=args.upsert)
    
    print("Anime data loading to Supabase completed!")


if __name__ == "__main__":
    main() 
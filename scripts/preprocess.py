#!/usr/bin/env python3
"""
Enhanced Anime Data Preprocessing Script

This script processes anime catalog and user ratings data to create datasets 
suitable for training our recommendation model. It handles anime relationships, 
studio information, and enhanced tag processing.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
import random


class AnimeDataPreprocessor:
    """
    Preprocessor for anime recommendation data with enhanced features.
    
    This class handles the preprocessing of anime catalog data and user ratings,
    including anime relationships, studio information, and tag importance.
    """
    
    # Define relationship type weights
    RELATIONSHIP_WEIGHTS = {
        "SEQUEL": 1.0,     # Highest weight for direct sequels
        "PREQUEL": 0.9,    # High weight for prequels
        "SIDE_STORY": 0.7, # Medium weight for side stories
        "PARENT": 0.8,     # High weight for parent stories
        "SPIN_OFF": 0.6,   # Medium weight for spin-offs
        "ALTERNATIVE": 0.5, # Medium-low weight for alternatives
        "CHARACTER": 0.4,  # Low weight for character appearances
        "SUMMARY": 0.3,    # Low weight for summaries
        "COMPILATION": 0.3, # Low weight for compilations
        "CONTAINS": 0.3,   # Low weight for contained material
        "OTHER": 0.2       # Lowest weight for other relationships
    }
    
    def __init__(
        self,
        input_dir: str = "data",
        output_dir: str = "data/processed",
        min_ratings_per_user: int = 10,
        min_ratings_per_anime: int = 5,
        max_tags: int = 100,
        max_genres: int = 20,
        max_studios: int = 10,
        max_relations: int = 20,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize the anime data preprocessor.
        
        Args:
            input_dir: Directory containing input data files
            output_dir: Directory to store processed output files
            min_ratings_per_user: Minimum ratings required for a user to be included
            min_ratings_per_anime: Minimum ratings required for an anime to be included
            max_tags: Maximum number of tags to include in the model
            max_genres: Maximum number of genres to include in the model
            max_studios: Maximum number of studios to include in the model
            max_relations: Maximum number of related anime to include per anime
            validation_split: Proportion of data for validation
            test_split: Proportion of data for testing
            random_seed: Random seed for reproducibility
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.min_ratings_per_user = min_ratings_per_user
        self.min_ratings_per_anime = min_ratings_per_anime
        self.max_tags = max_tags
        self.max_genres = max_genres
        self.max_studios = max_studios
        self.max_relations = max_relations
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data containers
        self.anime_data = {}
        self.user_ratings = {}
        self.ratings_df = None
        
        # Initialize mappings
        self.user_id_map = {}  # map user_id -> index
        self.anime_id_map = {}  # map anime_id -> index
        self.genre_map = {}  # map genre -> index
        self.tag_map = {}  # map tag -> index
        self.studio_map = {}  # map studio_id -> index
        
        # Initialize processed data
        self.anime_metadata = {}
        self.train_ratings = None
        self.val_ratings = None
        self.test_ratings = None
    
    def load_data(self) -> None:
        """Load anime catalog and user ratings data from input files."""
        print("Loading data...")
        
        # Load anime catalog
        anime_catalog_path = os.path.join(self.input_dir, "anime_catalog.json")
        with open(anime_catalog_path, 'r', encoding='utf-8') as f:
            anime_catalog = json.load(f)
        
        # Process anime catalog
        for anime in tqdm(anime_catalog, desc="Processing anime catalog"):
            anime_id = str(anime["anilist_id"])
            self.anime_data[anime_id] = anime
        
        print(f"Loaded {len(self.anime_data)} anime entries")
        
        # Load user ratings
        user_ratings_path = os.path.join(self.input_dir, "user_ratings.json")
        with open(user_ratings_path, 'r', encoding='utf-8') as f:
            user_ratings_data = json.load(f)
        
        # Process user ratings
        for user_data in tqdm(user_ratings_data, desc="Processing user ratings"):
            user_id = user_data["user_id"]
            ratings = user_data["ratings"]
            
            if len(ratings) >= self.min_ratings_per_user:
                self.user_ratings[user_id] = ratings
        
        print(f"Loaded ratings from {len(self.user_ratings)} users")

    def create_rating_dataframe(self) -> None:
        """Create a dataframe of user ratings with filtering."""
        print("Creating ratings dataframe...")
        
        # Collect all ratings
        ratings_data = []
        for user_id, ratings in tqdm(self.user_ratings.items(), desc="Collecting ratings"):
            for rating_entry in ratings:
                anime_id = str(rating_entry["anilist_id"])
                # Skip anime not in our catalog
                if anime_id not in self.anime_data:
                    continue
                
                ratings_data.append({
                    "user_id": user_id,
                    "anime_id": anime_id,
                    "rating": rating_entry["rating"] * 10,  # Scale ratings to 0-10
                    "status": rating_entry["status"]
                })
        
        # Create initial dataframe
        df = pd.DataFrame(ratings_data)
        print(f"Initial ratings count: {len(df)}")
        
        # Count ratings per anime
        anime_counts = df["anime_id"].value_counts()
        valid_anime = anime_counts[anime_counts >= self.min_ratings_per_anime].index
        df = df[df["anime_id"].isin(valid_anime)]
        
        # Recount ratings per user after anime filtering
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        df = df[df["user_id"].isin(valid_users)]
        
        print(f"Filtered ratings count: {len(df)}")
        print(f"Unique users: {df['user_id'].nunique()}")
        print(f"Unique anime: {df['anime_id'].nunique()}")
        
        self.ratings_df = df

    def create_mappings(self) -> None:
        """Create index mappings for users, anime, genres, tags, and studios."""
        print("Creating feature mappings...")
        
        # Create user ID mapping
        for idx, user_id in enumerate(sorted(self.ratings_df["user_id"].unique())):
            self.user_id_map[user_id] = idx
        
        # Create anime ID mapping (only include anime with ratings)
        rated_anime_ids = set(self.ratings_df["anime_id"].unique())
        for idx, anime_id in enumerate(sorted(rated_anime_ids)):
            self.anime_id_map[anime_id] = idx
        
        # Create genre mapping
        genre_counter = Counter()
        for anime_id, anime in self.anime_data.items():
            if anime_id in rated_anime_ids:
                genres = anime.get("genres", [])
                genre_counter.update(genres)
        
        # Add padding entry for unknown genres
        self.genre_map["<pad>"] = 0
        for idx, (genre, _) in enumerate(genre_counter.most_common(), start=1):
            if len(self.genre_map) >= self.max_genres + 1:
                break
            self.genre_map[genre] = idx
        
        # Create tag mapping
        tag_counter = Counter()
        for anime_id, anime in self.anime_data.items():
            if anime_id in rated_anime_ids:
                tags = [tag["name"] for tag in anime.get("tags", [])]
                tag_counter.update(tags)
        
        # Add padding entry for unknown tags
        self.tag_map["<pad>"] = 0
        for idx, (tag, _) in enumerate(tag_counter.most_common(self.max_tags), start=1):
            self.tag_map[tag] = idx
        
        # Create studio mapping
        studio_counter = Counter()
        for anime_id, anime in self.anime_data.items():
            if anime_id in rated_anime_ids:
                for studio in anime.get("studios", []):
                    studio_id = str(studio["node"]["id"])
                    studio_name = studio["node"]["name"]
                    studio_counter.update([studio_id])
        
        # Add padding entry for unknown studios
        self.studio_map["<pad>"] = 0
        for idx, (studio_id, _) in enumerate(studio_counter.most_common(self.max_studios), start=1):
            self.studio_map[studio_id] = idx
        
        print(f"Created mappings for {len(self.user_id_map)} users, {len(self.anime_id_map)} anime, "
              f"{len(self.genre_map)-1} genres, {len(self.tag_map)-1} tags, {len(self.studio_map)-1} studios")

    def process_anime_metadata(self) -> None:
        """Process anime metadata with enhanced features."""
        print("Processing anime metadata...")
        
        # Get set of all anime IDs in our filtered dataset
        anime_ids = set(self.anime_id_map.keys())
        
        # Create relationship mapping for each anime
        anime_relations = defaultdict(list)
        
        # First pass: collect all relationships
        for anime_id, anime in tqdm(self.anime_data.items(), desc="Collecting relationships"):
            if anime_id not in anime_ids:
                continue
            
            for relation in anime.get("relations", []):
                relation_type = relation.get("relationType", "OTHER")
                related_anime_id = str(relation["node"]["id"])
                
                # Skip if the related anime is not in our filtered dataset
                if related_anime_id not in anime_ids:
                    continue
                
                # Store relationship with weight
                weight = self.RELATIONSHIP_WEIGHTS.get(relation_type, 0.2)
                anime_relations[anime_id].append((related_anime_id, relation_type, weight))
        
        # Process metadata for each anime
        for anime_id, anime in tqdm(self.anime_data.items(), desc="Processing anime metadata"):
            if anime_id not in anime_ids:
                continue
            
            # Get anime index
            anime_idx = self.anime_id_map[anime_id]
            
            # Process genres
            genre_indices = []
            for genre in anime.get("genres", []):
                if genre in self.genre_map:
                    genre_indices.append(self.genre_map[genre])
            
            # Process tags with ranks
            tags = anime.get("tags", [])
            tag_indices = []
            tag_weights = []
            
            # Sort tags by rank (lower rank = more important)
            sorted_tags = sorted(tags, key=lambda x: x.get("rank", 100))
            
            for tag in sorted_tags:
                tag_name = tag["name"]
                if tag_name in self.tag_map:
                    tag_idx = self.tag_map[tag_name]
                    tag_indices.append(tag_idx)
                    
                    # Calculate tag weight based on rank (inverse relationship)
                    # Rank 1 = highest weight (1.0), higher ranks = lower weights
                    rank = tag.get("rank", 50)
                    weight = max(0.1, 1.0 - (rank / 100))
                    tag_weights.append(weight)
            
            # Process studios
            studio_indices = []
            studio_weights = []
            
            for studio in anime.get("studios", []):
                studio_id = str(studio["node"]["id"])
                if studio_id in self.studio_map:
                    studio_idx = self.studio_map[studio_id]
                    studio_indices.append(studio_idx)
                    
                    # Assign higher weight to main studios
                    is_main = studio.get("isMain", False)
                    weight = 1.0 if is_main else 0.5
                    studio_weights.append(weight)
            
            # Process relationships
            relation_indices = []
            relation_weights = []
            
            # Sort relations by weight (higher weight = more important)
            sorted_relations = sorted(
                anime_relations.get(anime_id, []),
                key=lambda x: x[2],
                reverse=True
            )[:self.max_relations]
            
            for related_id, relation_type, weight in sorted_relations:
                if related_id in self.anime_id_map:
                    relation_idx = self.anime_id_map[related_id]
                    relation_indices.append(relation_idx)
                    relation_weights.append(weight)
            
            # Get additional metadata
            popularity = anime.get("popularity", 0)
            average_score = anime.get("rating", 0)
            
            # Store processed metadata
            self.anime_metadata[anime_id] = {
                "anime_idx": anime_idx,
                "genre_indices": genre_indices,
                "tag_indices": tag_indices,
                "tag_weights": tag_weights,
                "studio_indices": studio_indices,
                "studio_weights": studio_weights,
                "relation_indices": relation_indices,
                "relation_weights": relation_weights,
                "popularity_score": popularity,
                "average_score": average_score
            }

    def split_ratings(self) -> None:
        """Split ratings into train, validation, and test sets."""
        print("Splitting ratings data...")
        
        # Add user and anime indices to the dataframe
        self.ratings_df["user_idx"] = self.ratings_df["user_id"].map(self.user_id_map)
        self.ratings_df["anime_idx"] = self.ratings_df["anime_id"].map(self.anime_id_map)
        
        # Group by user for stratified splitting
        user_groups = self.ratings_df.groupby("user_id")
        
        train_data = []
        val_data = []
        test_data = []
        
        # For each user, split their ratings
        for user_id, group in tqdm(user_groups, desc="Splitting user ratings"):
            # Shuffle the user's ratings
            shuffled = group.sample(frac=1, random_state=self.random_seed)
            
            # Calculate split points
            n_ratings = len(shuffled)
            n_test = max(1, int(n_ratings * self.test_split))
            n_val = max(1, int(n_ratings * self.validation_split))
            n_train = n_ratings - n_test - n_val
            
            # Split the data
            train_data.append(shuffled.iloc[:n_train])
            val_data.append(shuffled.iloc[n_train:n_train+n_val])
            test_data.append(shuffled.iloc[n_train+n_val:])
        
        # Combine and shuffle the splits
        self.train_ratings = pd.concat(train_data).sample(frac=1, random_state=self.random_seed)
        self.val_ratings = pd.concat(val_data).sample(frac=1, random_state=self.random_seed)
        self.test_ratings = pd.concat(test_data).sample(frac=1, random_state=self.random_seed)
        
        print(f"Training set: {len(self.train_ratings)} ratings")
        print(f"Validation set: {len(self.val_ratings)} ratings")
        print(f"Test set: {len(self.test_ratings)} ratings")

    def save_outputs(self) -> None:
        """Save processed data to output files."""
        print("Saving processed data...")
        
        # Create model configuration
        model_config = {
            "n_users": len(self.user_id_map),
            "n_anime": len(self.anime_id_map),
            "n_genres": len(self.genre_map) - 1,  # Subtract padding entry
            "n_tags": len(self.tag_map) - 1,      # Subtract padding entry
            "n_studios": len(self.studio_map) - 1,  # Subtract padding entry
            "max_genres": self.max_genres,
            "max_tags": self.max_tags,
            "max_studios": self.max_studios,
            "max_relations": self.max_relations
        }
        
        # Save model configuration
        with open(os.path.join(self.output_dir, "model_config.json"), 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save all mappings
        mappings = {
            "user_id_map": self.user_id_map,
            "anime_id_map": self.anime_id_map,
            "genre_map": self.genre_map,
            "tag_map": self.tag_map,
            "studio_map": self.studio_map
        }
        
        with open(os.path.join(self.output_dir, "mappings.pkl"), 'wb') as f:
            pickle.dump(mappings, f)
        
        # Save processed anime metadata
        with open(os.path.join(self.output_dir, "anime_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(self.anime_metadata, f)
        
        # Save rating splits as CSV files
        columns_to_save = ["user_idx", "anime_idx", "anime_id", "rating"]
        self.train_ratings[columns_to_save].to_csv(
            os.path.join(self.output_dir, "train_ratings.csv"), index=False)
        self.val_ratings[columns_to_save].to_csv(
            os.path.join(self.output_dir, "val_ratings.csv"), index=False)
        self.test_ratings[columns_to_save].to_csv(
            os.path.join(self.output_dir, "test_ratings.csv"), index=False)
        
        print("All processed data saved successfully")

    def run(self) -> None:
        """Run the complete preprocessing pipeline."""
        self.load_data()
        self.create_rating_dataframe()
        self.create_mappings()
        self.process_anime_metadata()
        self.split_ratings()
        self.save_outputs()
        print("Preprocessing completed!")


def main() -> None:
    """Parse command line arguments and run the preprocessor."""
    parser = argparse.ArgumentParser(description="Preprocess anime data for recommendation model")
    
    parser.add_argument("--input-dir", default="data",
                        help="Directory containing input data files")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Directory to store processed output files")
    parser.add_argument("--min-ratings-per-user", type=int, default=10,
                        help="Minimum ratings required for a user to be included")
    parser.add_argument("--min-ratings-per-anime", type=int, default=5,
                        help="Minimum ratings required for an anime to be included")
    parser.add_argument("--max-tags", type=int, default=100,
                        help="Maximum number of tags to include")
    parser.add_argument("--max-genres", type=int, default=20,
                        help="Maximum number of genres to include")
    parser.add_argument("--max-studios", type=int, default=10,
                        help="Maximum number of studios to include")
    parser.add_argument("--max-relations", type=int, default=20,
                        help="Maximum number of related anime to include per anime")
    parser.add_argument("--validation-split", type=float, default=0.1,
                        help="Proportion of data for validation")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Proportion of data for testing")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    preprocessor = AnimeDataPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_ratings_per_user=args.min_ratings_per_user,
        min_ratings_per_anime=args.min_ratings_per_anime,
        max_tags=args.max_tags,
        max_genres=args.max_genres,
        max_studios=args.max_studios,
        max_relations=args.max_relations,
        validation_split=args.validation_split,
        test_split=args.test_split,
        random_seed=args.random_seed
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main() 
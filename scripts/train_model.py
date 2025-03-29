#!/usr/bin/env python3
"""
Improved Anime Recommendation Model Training with PyTorch

This script trains an enhanced neural network recommendation model using PyTorch,
converted from an original TensorFlow implementation with improvements.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

### Dataset Class
class AnimeDataset(Dataset):
    """PyTorch dataset for anime recommendation data with enhanced features."""
    
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        anime_metadata: Dict[str, Dict],
        max_genres: int = 10,
        max_tags: int = 20,
        max_studios: int = 10,
        max_relations: int = 20
    ):
        """
        Initialize the dataset with ratings and metadata.

        Args:
            ratings_df (pd.DataFrame): DataFrame containing user ratings.
            anime_metadata (dict): Metadata for anime including genre, tag, studio and relation indices.
            max_genres (int): Maximum number of genres per anime (for padding).
            max_tags (int): Maximum number of tags per anime (for padding).
            max_studios (int): Maximum number of studios per anime (for padding).
            max_relations (int): Maximum number of related anime per anime (for padding).
        """
        self.user_indices = ratings_df['user_idx'].values
        self.anime_indices = ratings_df['anime_idx'].values
        self.anime_ids = ratings_df['anime_id'].values.astype(str)
        self.ratings = ratings_df['rating'].values
        self.anime_metadata = anime_metadata
        self.max_genres = max_genres
        self.max_tags = max_tags
        self.max_studios = max_studios
        self.max_relations = max_relations
        self.metadata_cache = {}
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.ratings)
    
    def get_anime_metadata(self, anime_id: str) -> Tuple[np.ndarray, ...]:
        """
        Retrieve or compute padded feature indices for an anime.

        Args:
            anime_id (str): ID of the anime.

        Returns:
            Tuple of padded feature arrays: genre_indices, tag_indices, studio_indices, relation_indices
        """
        if anime_id in self.metadata_cache:
            return self.metadata_cache[anime_id]
        
        metadata = self.anime_metadata.get(anime_id, {})
        
        # Process genres
        genre_indices = metadata.get('genre_indices', [])[:self.max_genres]
        genre_indices = genre_indices + [0] * (self.max_genres - len(genre_indices))
        
        # Process tags
        tag_indices = metadata.get('tag_indices', [])[:self.max_tags]
        tag_indices = tag_indices + [0] * (self.max_tags - len(tag_indices))
        
        # Process studios
        studio_indices = metadata.get('studio_indices', [])[:self.max_studios]
        studio_indices = studio_indices + [0] * (self.max_studios - len(studio_indices))
        
        # Process studio weights (or use defaults if not available)
        studio_weights = metadata.get('studio_weights', [1.0] * len(studio_indices))[:self.max_studios]
        studio_weights = studio_weights + [0.0] * (self.max_studios - len(studio_weights))
        
        # Process relations
        relation_indices = metadata.get('relation_indices', [])[:self.max_relations]
        relation_indices = relation_indices + [0] * (self.max_relations - len(relation_indices))
        
        # Process relation weights (or use defaults if not available)
        relation_weights = metadata.get('relation_weights', [1.0] * len(relation_indices))[:self.max_relations]
        relation_weights = relation_weights + [0.0] * (self.max_relations - len(relation_weights))
        
        result = (
            np.array(genre_indices), 
            np.array(tag_indices),
            np.array(studio_indices),
            np.array(studio_weights),
            np.array(relation_indices),
            np.array(relation_weights)
        )
        
        self.metadata_cache[anime_id] = result
        return result
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple of tensors: user_idx, anime_idx, genre_indices, tag_indices, 
                             studio_indices, studio_weights, relation_indices, relation_weights, rating
        """
        user_idx = self.user_indices[idx]
        anime_idx = self.anime_indices[idx]
        anime_id = self.anime_ids[idx]
        rating = self.ratings[idx]
        
        genre_indices, tag_indices, studio_indices, studio_weights, relation_indices, relation_weights = self.get_anime_metadata(anime_id)
        
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(anime_idx, dtype=torch.long),
            torch.tensor(genre_indices, dtype=torch.long),
            torch.tensor(tag_indices, dtype=torch.long),
            torch.tensor(studio_indices, dtype=torch.long),
            torch.tensor(studio_weights, dtype=torch.float),
            torch.tensor(relation_indices, dtype=torch.long),
            torch.tensor(relation_weights, dtype=torch.float),
            torch.tensor(rating, dtype=torch.float)
        )

### Model Class
class ImprovedAnimeRecommenderModel(nn.Module):
    """Enhanced PyTorch model for anime recommendations with personalized user-specific attention."""
    
    def __init__(
        self, 
        n_users: int, 
        n_anime: int, 
        n_genres: int, 
        n_tags: int,
        n_studios: int,
        embedding_dim_users: int = 128,  # Increased from 64
        embedding_dim_anime: int = 256,  # Increased from 128
        embedding_dim_genres: int = 64,  # Increased from 32
        embedding_dim_tags: int = 64,    # Increased from 32
        embedding_dim_studios: int = 32, # Increased from 16
        embedding_dim_relations: int = 64, # Increased from 32
        dropout_rate: float = 0.2        # Reduced from 0.4
    ):
        """
        Initialize the recommendation model with enhanced personalization capabilities.

        Args:
            n_users (int): Number of unique users.
            n_anime (int): Number of unique anime.
            n_genres (int): Number of unique genres.
            n_tags (int): Number of unique tags.
            n_studios (int): Number of unique studios.
            embedding_dim_users (int): Dimension of user embeddings (increased).
            embedding_dim_anime (int): Dimension of anime embeddings (increased).
            embedding_dim_genres (int): Dimension of genre embeddings (increased).
            embedding_dim_tags (int): Dimension of tag embeddings (increased).
            embedding_dim_studios (int): Dimension of studio embeddings (increased).
            embedding_dim_relations (int): Dimension of relation embeddings (increased).
            dropout_rate (float): Dropout rate for regularization (reduced).
        """
        super(ImprovedAnimeRecommenderModel, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim_users)
        self.anime_embedding = nn.Embedding(n_anime, embedding_dim_anime)
        self.genre_embedding = nn.Embedding(n_genres + 1, embedding_dim_genres, padding_idx=0)
        self.tag_embedding = nn.Embedding(n_tags + 1, embedding_dim_tags, padding_idx=0)
        self.studio_embedding = nn.Embedding(n_studios + 1, embedding_dim_studios, padding_idx=0)
        self.relation_embedding = nn.Embedding(n_anime, embedding_dim_relations)
        
        # User embedding projection layers for personalized cross-attention
        self.user_to_genre_query = nn.Linear(embedding_dim_users, embedding_dim_genres)
        self.user_to_tag_query = nn.Linear(embedding_dim_users, embedding_dim_tags)
        self.user_to_studio_query = nn.Linear(embedding_dim_users, embedding_dim_studios)
        self.user_to_relation_query = nn.Linear(embedding_dim_users, embedding_dim_relations)
        
        # Enhanced multi-head attention layers with more heads for better personalization
        self.genre_attention = nn.MultiheadAttention(embed_dim=embedding_dim_genres, num_heads=8)
        self.tag_attention = nn.MultiheadAttention(embed_dim=embedding_dim_tags, num_heads=8)
        self.studio_attention = nn.MultiheadAttention(embed_dim=embedding_dim_studios, num_heads=4)
        self.relation_attention = nn.MultiheadAttention(embed_dim=embedding_dim_relations, num_heads=8)
        
        # Calculate total dimension after all features are concatenated
        total_dim = (
            embedding_dim_users +      # User embedding
            embedding_dim_anime +      # Anime embedding
            embedding_dim_genres +     # Genre attention output 
            embedding_dim_tags +       # Tag attention output
            embedding_dim_studios +    # Studio attention output
            embedding_dim_relations    # Relation attention output
        )
        
        # Feature fusion layer to better integrate user preferences with anime features
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # MLP layers with reduced regularization and improved architecture
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 512),  # Increased from 256
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 1)
        )
        
        # Initialization with improved variance scaling for better differentiation
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with improved scaling for better personalization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)  # Slightly increased std
    
    def forward(
        self, 
        user_idx, 
        anime_idx, 
        genre_indices, 
        tag_indices, 
        studio_indices, 
        studio_weights, 
        relation_indices, 
        relation_weights
    ):
        """
        Forward pass of the model with enhanced personalization through cross-attention.

        Args:
            user_idx (torch.Tensor): User indices.
            anime_idx (torch.Tensor): Anime indices.
            genre_indices (torch.Tensor): Genre indices for each anime.
            tag_indices (torch.Tensor): Tag indices for each anime.
            studio_indices (torch.Tensor): Studio indices for each anime.
            studio_weights (torch.Tensor): Weights for each studio.
            relation_indices (torch.Tensor): Related anime indices.
            relation_weights (torch.Tensor): Weights for each relationship.

        Returns:
            torch.Tensor: Predicted ratings.
        """
        user_embed = self.user_embedding(user_idx)  # (batch_size, user_dim)
        anime_embed = self.anime_embedding(anime_idx)  # (batch_size, anime_dim)
        
        # Apply L2 normalization with dampening factor for improved gradient flow
        user_embed = nn.functional.normalize(user_embed, p=2, dim=1) * 1.414
        anime_embed = nn.functional.normalize(anime_embed, p=2, dim=1) * 1.414
        
        # Project user embeddings to create personalized attention queries
        genre_query = self.user_to_genre_query(user_embed)  # (batch_size, genre_dim)
        genre_query = nn.functional.normalize(genre_query, p=2, dim=1) * 1.414
        genre_query = genre_query.unsqueeze(0)  # (1, batch_size, genre_dim)
        
        tag_query = self.user_to_tag_query(user_embed)  # (batch_size, tag_dim)
        tag_query = nn.functional.normalize(tag_query, p=2, dim=1) * 1.414
        tag_query = tag_query.unsqueeze(0)  # (1, batch_size, tag_dim)
        
        studio_query = self.user_to_studio_query(user_embed)  # (batch_size, studio_dim)
        studio_query = nn.functional.normalize(studio_query, p=2, dim=1) * 1.414
        studio_query = studio_query.unsqueeze(0)  # (1, batch_size, studio_dim)
        
        relation_query = self.user_to_relation_query(user_embed)  # (batch_size, relation_dim)
        relation_query = nn.functional.normalize(relation_query, p=2, dim=1) * 1.414
        relation_query = relation_query.unsqueeze(0)  # (1, batch_size, relation_dim)
        
        # Process genres with user-specific cross-attention
        genre_embeds = self.genre_embedding(genre_indices)  # (batch_size, max_genres, genre_dim)
        genre_mask = (genre_indices == 0)  # (batch_size, max_genres)
        
        # Handle empty genre lists by ensuring at least one valid entry
        all_masked = genre_mask.all(dim=1, keepdim=True)
        fixed_genre_mask = genre_mask.clone()
        # Set the first position to False for rows where all are masked
        fixed_genre_mask[:, 0] = fixed_genre_mask[:, 0] & ~all_masked.squeeze(1)
        
        genre_embeds = genre_embeds.permute(1, 0, 2)  # (max_genres, batch_size, genre_dim)
        # Use user embedding as query for personalized attention, capturing need_weights=True
        genre_attended, genre_attn_weights = self.genre_attention(
            genre_query,  # User-specific query
            genre_embeds,  # Keys from anime genres
            genre_embeds,  # Values from anime genres
            key_padding_mask=fixed_genre_mask,
            need_weights=True
        )
        # Apply weighted average based on attention weights
        genre_attended = genre_attended.mean(dim=0)  # (batch_size, genre_dim)
        genre_attended = nn.functional.normalize(genre_attended, p=2, dim=1) * 1.414
        
        # Process tags with user-specific cross-attention
        tag_embeds = self.tag_embedding(tag_indices)  # (batch_size, max_tags, tag_dim)
        tag_mask = (tag_indices == 0)  # (batch_size, max_tags)
        
        # Handle empty tag lists
        all_masked_tags = tag_mask.all(dim=1, keepdim=True)
        fixed_tag_mask = tag_mask.clone()
        fixed_tag_mask[:, 0] = fixed_tag_mask[:, 0] & ~all_masked_tags.squeeze(1)
            
        tag_embeds = tag_embeds.permute(1, 0, 2)  # (max_tags, batch_size, tag_dim)
        tag_attended, tag_attn_weights = self.tag_attention(
            tag_query,  # User-specific query
            tag_embeds,  # Keys from anime tags
            tag_embeds,  # Values from anime tags
            key_padding_mask=fixed_tag_mask,
            need_weights=True
        )
        tag_attended = tag_attended.mean(dim=0)  # (batch_size, tag_dim)
        tag_attended = nn.functional.normalize(tag_attended, p=2, dim=1) * 1.414
        
        # Process studios with user-specific cross-attention
        studio_embeds = self.studio_embedding(studio_indices)  # (batch_size, max_studios, studio_dim)
        studio_mask = (studio_indices == 0)  # (batch_size, max_studios)
        
        # Apply studio weights to embeddings
        studio_weights = studio_weights.unsqueeze(-1)  # (batch_size, max_studios, 1)
        studio_embeds = studio_embeds * studio_weights  # (batch_size, max_studios, studio_dim)
        
        # Handle empty studio lists
        all_masked_studios = studio_mask.all(dim=1, keepdim=True)
        fixed_studio_mask = studio_mask.clone()
        fixed_studio_mask[:, 0] = fixed_studio_mask[:, 0] & ~all_masked_studios.squeeze(1)
        
        studio_embeds = studio_embeds.permute(1, 0, 2)  # (max_studios, batch_size, studio_dim)
        studio_attended, studio_attn_weights = self.studio_attention(
            studio_query,  # User-specific query
            studio_embeds,  # Keys from anime studios
            studio_embeds,  # Values from anime studios
            key_padding_mask=fixed_studio_mask,
            need_weights=True
        )
        studio_attended = studio_attended.mean(dim=0)  # (batch_size, studio_dim)
        studio_attended = nn.functional.normalize(studio_attended, p=2, dim=1) * 1.414
        
        # Process related anime with user-specific cross-attention
        relation_embeds = self.relation_embedding(relation_indices)  # (batch_size, max_relations, relation_dim)
        relation_mask = (relation_indices == 0)  # (batch_size, max_relations)
        
        # Apply relation weights to embeddings
        relation_weights = relation_weights.unsqueeze(-1)  # (batch_size, max_relations, 1)
        relation_embeds = relation_embeds * relation_weights  # (batch_size, max_relations, relation_dim)
        
        # Handle empty relation lists
        all_masked_relations = relation_mask.all(dim=1, keepdim=True)
        fixed_relation_mask = relation_mask.clone()
        fixed_relation_mask[:, 0] = fixed_relation_mask[:, 0] & ~all_masked_relations.squeeze(1)
        
        relation_embeds = relation_embeds.permute(1, 0, 2)  # (max_relations, batch_size, relation_dim)
        relation_attended, relation_attn_weights = self.relation_attention(
            relation_query,  # User-specific query
            relation_embeds,  # Keys from related anime
            relation_embeds,  # Values from related anime
            key_padding_mask=fixed_relation_mask,
            need_weights=True
        )
        relation_attended = relation_attended.mean(dim=0)  # (batch_size, relation_dim)
        relation_attended = nn.functional.normalize(relation_attended, p=2, dim=1) * 1.414
        
        # Concatenate all features
        concatenated = torch.cat([
            user_embed, 
            anime_embed, 
            genre_attended, 
            tag_attended,
            studio_attended,
            relation_attended
        ], dim=1)
        
        # Apply feature fusion for better integration of user preferences
        fused_features = self.feature_fusion(concatenated)
        
        # Feed through MLP
        output = self.mlp(fused_features)  # (batch_size, 1)
        
        # Replace NaN values with zeros as a safeguard
        output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        
        return output.squeeze(1)  # (batch_size,)

### Trainer Class
class ModelTrainer:
    """Manages training of the enhanced anime recommendation model in PyTorch."""
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        output_dir: str = "data/model/pytorch",
        batch_size: int = 128,  # Reduced from 256/512
        epochs: int = 100,      # Increased from 50
        learning_rate: float = 0.001, # Increased from 0.0005/0.0003
        weight_decay: float = 1e-5,
        early_stopping_patience: int = 20, # Increased from 10/15
        gradient_clip_val: float = 1.0,
        use_robust_norm: bool = True,  # Use robust normalization
        k_values: list = [5, 10, 20],  # For precision@k, recall@k metrics
    ):
        """
        Initialize the trainer with enhanced training parameters.

        Args:
            data_dir (str): Directory containing processed data.
            output_dir (str): Directory to save the trained model.
            batch_size (int): Batch size for training (reduced).
            epochs (int): Number of training epochs (increased).
            learning_rate (float): Learning rate for the optimizer (increased).
            weight_decay (float): Weight decay for regularization.
            early_stopping_patience (int): Patience for early stopping (increased).
            gradient_clip_val (float): Maximum gradient norm for gradient clipping.
            use_robust_norm (bool): Whether to use robust normalization with median/IQR.
            k_values (list): K values for precision@k, recall@k metrics.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_val = gradient_clip_val
        self.use_robust_norm = use_robust_norm
        self.k_values = k_values
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.anime_metadata = None
        self.model_config = None
        
        # Normalization parameters
        self.rating_mean = None
        self.rating_std = None
        self.rating_median = None
        self.rating_iqr = None
    
    def load_data(self):
        """Load processed data with robust normalization and create DataLoaders."""
        print("Loading processed data...")
        
        train_df = pd.read_csv(os.path.join(self.data_dir, "train_ratings.csv"))
        val_df = pd.read_csv(os.path.join(self.data_dir, "val_ratings.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test_ratings.csv"))
        
        # Standard normalization parameters
        self.rating_mean = train_df['rating'].mean()
        self.rating_std = train_df['rating'].std()
        
        # Robust normalization parameters
        self.rating_median = train_df['rating'].median()
        q1 = train_df['rating'].quantile(0.25)
        q3 = train_df['rating'].quantile(0.75)
        self.rating_iqr = q3 - q1
        
        # Apply normalization based on chosen method
        if self.use_robust_norm and self.rating_iqr > 0:
            print(f"Using robust normalization - Median: {self.rating_median:.4f}, IQR: {self.rating_iqr:.4f}")
            train_df['rating'] = (train_df['rating'] - self.rating_median) / self.rating_iqr
            val_df['rating'] = (val_df['rating'] - self.rating_median) / self.rating_iqr
            test_df['rating'] = (test_df['rating'] - self.rating_median) / self.rating_iqr
        elif self.rating_std > 0:
            print(f"Using standard normalization - Mean: {self.rating_mean:.4f}, Std: {self.rating_std:.4f}")
            train_df['rating'] = (train_df['rating'] - self.rating_mean) / self.rating_std
            val_df['rating'] = (val_df['rating'] - self.rating_mean) / self.rating_std
            test_df['rating'] = (test_df['rating'] - self.rating_mean) / self.rating_std
        
        print(f"Loaded {len(train_df)} training samples")
        print(f"Loaded {len(val_df)} validation samples")
        print(f"Loaded {len(test_df)} test samples")
        
        # Create dictionaries for user and anime lookup for recommendation metrics
        self.user_anime_dict = {}
        for user_idx, anime_idx, rating in zip(train_df['user_idx'], train_df['anime_idx'], train_df['rating']):
            if user_idx not in self.user_anime_dict:
                self.user_anime_dict[user_idx] = {}
            self.user_anime_dict[user_idx][anime_idx] = rating
            
        # Load model config and anime metadata
        with open(os.path.join(self.data_dir, "model_config.json"), 'r') as f:
            self.model_config = json.load(f)
        
        with open(os.path.join(self.data_dir, "anime_metadata.json"), 'r', encoding='utf-8') as f:
            self.anime_metadata = json.load(f)
        
        # Initialize dataset with enhanced features
        max_genres = self.model_config.get('max_genres', 10)
        max_tags = self.model_config.get('max_tags', 20)
        max_studios = self.model_config.get('max_studios', 10)
        max_relations = self.model_config.get('max_relations', 20)
        
        train_dataset = AnimeDataset(
            train_df, 
            self.anime_metadata,
            max_genres=max_genres,
            max_tags=max_tags,
            max_studios=max_studios,
            max_relations=max_relations
        )
        val_dataset = AnimeDataset(
            val_df, 
            self.anime_metadata,
            max_genres=max_genres,
            max_tags=max_tags,
            max_studios=max_studios,
            max_relations=max_relations
        )
        test_dataset = AnimeDataset(
            test_df, 
            self.anime_metadata,
            max_genres=max_genres,
            max_tags=max_tags,
            max_studios=max_studios,
            max_relations=max_relations
        )
        
        # Create dataloaders with smaller batch size for better personalization
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    def create_model(self):
        """Create the enhanced model with improved optimizer and learning rate scheduler."""
        print("Creating enhanced personalization model...")
        self.model = ImprovedAnimeRecommenderModel(
            n_users=self.model_config['n_users'],
            n_anime=self.model_config['n_anime'],
            n_genres=self.model_config['n_genres'],
            n_tags=self.model_config['n_tags'],
            n_studios=self.model_config['n_studios']
        )
        
        # Custom weight initialization is now handled in the model itself
        
        # Use AdamW with improved learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Use a mixed loss function: MAE loss for accuracy and cosine similarity for personalization
        self.criterion = nn.L1Loss()
        
        # Use OneCycleLR scheduler for better convergence
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.epochs * len(self.train_loader),
            pct_start=0.1,  # Warm-up period
            div_factor=10,  # Initial lr is max_lr/10
            final_div_factor=100  # Min lr is max_lr/1000
        )
    
    def calculate_recommendation_metrics(self, user_indices, anime_indices, scores):
        """
        Calculate precision@k and recall@k for different k values.
        
        Args:
            user_indices (torch.Tensor): User indices.
            anime_indices (torch.Tensor): Anime indices.
            scores (torch.Tensor): Predicted scores/ratings.
            
        Returns:
            dict: Dictionary containing precision@k and recall@k metrics.
        """
        metrics = {}
        for k in self.k_values:
            total_precision = 0.0
            total_recall = 0.0
            user_count = 0
            
            # Group predictions by user
            unique_users = set(user_indices.cpu().numpy())
            for user in unique_users:
                user_mask = user_indices == user
                user_scores = scores[user_mask].cpu().numpy()
                user_anime = anime_indices[user_mask].cpu().numpy()
                
                # Sort by predicted scores to get top-k recommendations
                top_indices = np.argsort(-user_scores)[:k]
                top_anime = user_anime[top_indices]
                
                # Count relevant items (anime with positive normalized ratings)
                if user in self.user_anime_dict:
                    liked_anime = {a for a, r in self.user_anime_dict[user].items() if r > 0}
                    
                    # Calculate precision and recall
                    if len(top_anime) > 0:
                        hits = len(set(top_anime) & liked_anime)
                        precision = hits / len(top_anime)
                        
                        if len(liked_anime) > 0:
                            recall = hits / len(liked_anime)
                        else:
                            recall = 0.0
                        
                        total_precision += precision
                        total_recall += recall
                        user_count += 1
            
            # Calculate average metrics
            if user_count > 0:
                metrics[f'precision@{k}'] = total_precision / user_count
                metrics[f'recall@{k}'] = total_recall / user_count
            else:
                metrics[f'precision@{k}'] = 0.0
                metrics[f'recall@{k}'] = 0.0
                
        return metrics
    
    def train(self):
        """Train the model with enhanced early stopping and learning rate scheduling."""
        print("Training enhanced personalization model...")
        self.model.to(self.device)
        
        # Create output directory at the beginning of training
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize training history with additional metrics
        history = {
            'train_loss': [], 
            'val_loss': [], 
            'val_mae': [], 
            'val_mse': []
        }
        
        # Add recommendation metrics to history
        for k in self.k_values:
            history[f'val_precision@{k}'] = []
            history[f'val_recall@{k}'] = []
        
        best_val_loss = float('inf')
        best_val_ndcg = 0.0
        patience_counter = 0
        
        # Enable mixed precision training for faster computation
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            # Training loop with progress bar
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                # Unpack batch with enhanced features
                user_idx, anime_idx, genre_indices, tag_indices, studio_indices, studio_weights, \
                relation_indices, relation_weights, rating = [b.to(self.device) for b in batch]
                
                self.optimizer.zero_grad()
                
                # Use mixed precision training if available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        # Forward pass with enhanced features
                        output = self.model(
                            user_idx, 
                            anime_idx, 
                            genre_indices, 
                            tag_indices, 
                            studio_indices, 
                            studio_weights,
                            relation_indices, 
                            relation_weights
                        )
                        # Calculate loss
                        loss = self.criterion(output, rating)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Forward pass with enhanced features (without mixed precision)
                    output = self.model(
                        user_idx, 
                        anime_idx, 
                        genre_indices, 
                        tag_indices, 
                        studio_indices, 
                        studio_weights,
                        relation_indices, 
                        relation_weights
                    )
                    
                    # Calculate loss
                    loss = self.criterion(output, rating)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.optimizer.step()
                
                # Update learning rate with OneCycleLR per batch
                self.lr_scheduler.step()
                
                # Accumulate loss
                train_loss += loss.item() * len(rating)
            
            # Calculate average training loss
            train_loss /= len(self.train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_loader, calculate_rec_metrics=True)
            
            # Record validation metrics
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_mse'].append(val_metrics['mse'])
            
            # Record recommendation metrics
            for k in self.k_values:
                history[f'val_precision@{k}'].append(val_metrics[f'precision@{k}'])
                history[f'val_recall@{k}'].append(val_metrics[f'recall@{k}'])
            
            # Print epoch results with detailed metrics
            print(f"Epoch {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
            print(f"  Val P@10: {val_metrics['precision@10']:.4f}, Val R@10: {val_metrics['recall@10']:.4f}")
            
            # Check for NaN values
            if np.isnan(train_loss) or np.isnan(val_metrics['loss']):
                print("NaN values detected, stopping training")
                break
            
            # Save model if validation loss improves
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
                print("  New best model saved!")
            else:
                patience_counter += 1
                print(f"  No improvement, patience: {patience_counter}/{self.early_stopping_patience}")
                if patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model after training
        if os.path.exists(os.path.join(self.output_dir, 'best_model.pth')):
            self.model.load_state_dict(torch.load(os.path.join(self.output_dir, 'best_model.pth')))
        
        return history
    
    def evaluate(self, loader, calculate_rec_metrics=False):
        """
        Evaluate the model on a given dataset with enhanced metrics.

        Args:
            loader (DataLoader): DataLoader for evaluation.
            calculate_rec_metrics (bool): Whether to calculate recommendation metrics.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        all_user_indices = []
        all_anime_indices = []
        all_scores = []
        all_ratings = []
        
        with torch.no_grad():
            for batch in loader:
                # Unpack batch with enhanced features
                user_idx, anime_idx, genre_indices, tag_indices, studio_indices, studio_weights, \
                relation_indices, relation_weights, rating = [b.to(self.device) for b in batch]
                
                # Forward pass with enhanced features
                output = self.model(
                    user_idx, 
                    anime_idx, 
                    genre_indices, 
                    tag_indices, 
                    studio_indices, 
                    studio_weights,
                    relation_indices, 
                    relation_weights
                )
                
                # Calculate loss on normalized values (as trained)
                loss = self.criterion(output, rating)
                total_loss += loss.item() * len(rating)
                
                # Denormalize predictions and ratings for more interpretable metrics
                if self.use_robust_norm and self.rating_iqr is not None:
                    denorm_output = output * self.rating_iqr + self.rating_median
                    denorm_rating = rating * self.rating_iqr + self.rating_median
                elif self.rating_std is not None and self.rating_mean is not None:
                    denorm_output = output * self.rating_std + self.rating_mean
                    denorm_rating = rating * self.rating_std + self.rating_mean
                else:
                    denorm_output = output
                    denorm_rating = rating
                
                # Calculate standard metrics
                total_mae += torch.abs(denorm_output - denorm_rating).sum().item()
                total_mse += ((denorm_output - denorm_rating) ** 2).sum().item()
                
                # Store data for recommendation metrics
                if calculate_rec_metrics:
                    all_user_indices.append(user_idx)
                    all_anime_indices.append(anime_idx)
                    all_scores.append(output)
                    all_ratings.append(rating)
        
        # Calculate average metrics
        metrics = {
            'loss': total_loss / len(loader.dataset),
            'mae': total_mae / len(loader.dataset),
            'mse': total_mse / len(loader.dataset)
        }
        
        # Calculate recommendation metrics if requested
        if calculate_rec_metrics:
            # Combine all batches
            all_user_indices = torch.cat(all_user_indices)
            all_anime_indices = torch.cat(all_anime_indices)
            all_scores = torch.cat(all_scores)
            all_ratings = torch.cat(all_ratings)
            
            # Calculate recommendation metrics
            rec_metrics = self.calculate_recommendation_metrics(
                all_user_indices, all_anime_indices, all_scores
            )
            metrics.update(rec_metrics)
        
        return metrics
    
    def save_model(self):
        """Save the trained model and associated metadata."""
        print("Saving enhanced personalization model and assets...")
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pth'))
        
        # Load and process all mappings
        with open(os.path.join(self.data_dir, "mappings.pkl"), 'rb') as f:
            mappings = pickle.load(f)
        
        # Create reverse mappings for all available maps
        reverse_mappings = {
            'idx_to_anime': {str(v): str(k) for k, v in mappings['anime_id_map'].items()},
            'anime_to_idx': {str(k): v for k, v in mappings['anime_id_map'].items()},
            'idx_to_genre': {str(v): k for k, v in mappings['genre_map'].items()},
            'genre_to_idx': {str(k): v for k, v in mappings['genre_map'].items()},
            'idx_to_tag': {str(v): k for k, v in mappings['tag_map'].items()},
            'tag_to_idx': {str(k): v for k, v in mappings['tag_map'].items()},
            'idx_to_studio': {str(v): k for k, v in mappings.get('studio_map', {}).items()},
            'studio_to_idx': {str(k): v for k, v in mappings.get('studio_map', {}).items()},
        }
        
        with open(os.path.join(self.output_dir, 'model_mappings.json'), 'w') as f:
            json.dump(reverse_mappings, f, indent=2)
        
        # Save enhanced model metadata
        model_metadata = {
            'n_users': self.model_config['n_users'],
            'n_anime': self.model_config['n_anime'],
            'n_genres': self.model_config['n_genres'],
            'n_tags': self.model_config['n_tags'],
            'n_studios': self.model_config.get('n_studios', 0),
            'user_embedding_dim': 128,
            'anime_embedding_dim': 256,
            'genre_embedding_dim': 64,
            'tag_embedding_dim': 64,
            'studio_embedding_dim': 32,
            'relation_embedding_dim': 64,
            'dense_layers': [512, 256, 128, 64],
            'max_genres': self.model_config.get('max_genres', 10),
            'max_tags': self.model_config.get('max_tags', 20),
            'max_studios': self.model_config.get('max_studios', 10),
            'max_relations': self.model_config.get('max_relations', 20),
            'model_type': 'pytorch',
            'rating_normalization': {
                'method': 'robust' if self.use_robust_norm else 'standard',
                'mean': float(self.rating_mean) if self.rating_mean is not None else None,
                'std': float(self.rating_std) if self.rating_std is not None else None,
                'median': float(self.rating_median) if self.rating_median is not None else None,
                'iqr': float(self.rating_iqr) if self.rating_iqr is not None else None
            },
            'features': ['users', 'anime', 'genres', 'tags', 'studios', 'relationships'],
            'personalization': 'enhanced_cross_attention'
        }
        
        with open(os.path.join(self.output_dir, 'model_metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=2)

    def run(self):
        """Execute the full training pipeline for the enhanced personalization model."""
        self.load_data()
        self.create_model()
        history = self.train()
        
        # Evaluate on test set with recommendation metrics
        test_metrics = self.evaluate(self.test_loader, calculate_rec_metrics=True)
        
        # Print test metrics with detailed format
        print("\nTest Results:")
        print(f"  Loss (MAE): {test_metrics['loss']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        
        # Print recommendation metrics
        for k in self.k_values:
            print(f"  Precision@{k}: {test_metrics[f'precision@{k}']:.4f}")
            print(f"  Recall@{k}: {test_metrics[f'recall@{k}']:.4f}")
        
        self.save_model()
        
        # Save detailed training history and test metrics
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump({
                'history': {k: [float(v) for v in vals] for k, vals in history.items()},
                'test_metrics': {k: float(v) for k, v in test_metrics.items()}
            }, f, indent=2)
        
        print("\nEnhanced personalization model training completed successfully!")

### Main Function
def main():
    """Parse arguments and run the trainer with enhanced personalization features."""
    parser = argparse.ArgumentParser(description="Train enhanced anime recommendation model with personalized attention")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed data")
    parser.add_argument("--output-dir", default="data/model/pytorch", help="Directory to save trained model")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training (smaller for better personalization)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (increased)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer (increased)")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularization factor (reduced)")
    parser.add_argument("--early-stopping-patience", type=int, default=20, help="Patience for early stopping (increased)")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--use-robust-norm", action="store_true", default=True, help="Use robust normalization (median/IQR)")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 20], help="K values for precision@k and recall@k")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Print training parameters
    print("\n=== Enhanced Personalization Model Training ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Normalization: {'Robust (median/IQR)' if args.use_robust_norm else 'Standard (mean/std)'}")
    print(f"Recommendation metrics @ k values: {args.k_values}")
    
    # Create trainer with enhanced parameters
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_val=args.gradient_clip_val,
        use_robust_norm=args.use_robust_norm,
        k_values=args.k_values
    )
    
    trainer.run()
    
    print("\nImproved personalization model training completed!")

if __name__ == "__main__":
    main()
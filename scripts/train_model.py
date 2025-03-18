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

### Dataset Class
class AnimeDataset(Dataset):
    """PyTorch dataset for anime recommendation data."""
    
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        anime_metadata: Dict[str, Dict],
        max_genres: int = 10,
        max_tags: int = 20
    ):
        """
        Initialize the dataset with ratings and metadata.

        Args:
            ratings_df (pd.DataFrame): DataFrame containing user ratings.
            anime_metadata (dict): Metadata for anime including genre and tag indices.
            max_genres (int): Maximum number of genres per anime (for padding).
            max_tags (int): Maximum number of tags per anime (for padding).
        """
        self.user_indices = ratings_df['user_idx'].values
        self.anime_indices = ratings_df['anime_idx'].values
        self.anime_ids = ratings_df['anime_id'].values.astype(str)
        self.ratings = ratings_df['rating'].values
        self.anime_metadata = anime_metadata
        self.max_genres = max_genres
        self.max_tags = max_tags
        self.genre_tag_cache = {}
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.ratings)
    
    def get_genre_tag_indices(self, anime_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve or compute padded genre and tag indices for an anime.

        Args:
            anime_id (str): ID of the anime.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Padded genre and tag indices.
        """
        if anime_id in self.genre_tag_cache:
            return self.genre_tag_cache[anime_id]
        
        metadata = self.anime_metadata.get(anime_id, {})
        genre_indices = metadata.get('genre_indices', [])[:self.max_genres]
        genre_indices = genre_indices + [0] * (self.max_genres - len(genre_indices))
        
        tag_indices = metadata.get('tag_indices', [])[:self.max_tags]
        tag_indices = tag_indices + [0] * (self.max_tags - len(tag_indices))
        
        self.genre_tag_cache[anime_id] = (np.array(genre_indices), np.array(tag_indices))
        return self.genre_tag_cache[anime_id]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, ...]: User index, anime index, genre indices, tag indices, and rating.
        """
        user_idx = self.user_indices[idx]
        anime_idx = self.anime_indices[idx]
        anime_id = self.anime_ids[idx]
        rating = self.ratings[idx]
        
        genre_indices, tag_indices = self.get_genre_tag_indices(anime_id)
        
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(anime_idx, dtype=torch.long),
            torch.tensor(genre_indices, dtype=torch.long),
            torch.tensor(tag_indices, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
        )

### Model Class
class ImprovedAnimeRecommenderModel(nn.Module):
    """Enhanced PyTorch model for anime recommendations."""
    
    def __init__(self, n_users: int, n_anime: int, n_genres: int, n_tags: int):
        """
        Initialize the recommendation model.

        Args:
            n_users (int): Number of unique users.
            n_anime (int): Number of unique anime.
            n_genres (int): Number of unique genres.
            n_tags (int): Number of unique tags.
        """
        super(ImprovedAnimeRecommenderModel, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, 64)
        self.anime_embedding = nn.Embedding(n_anime, 128)
        self.genre_embedding = nn.Embedding(n_genres + 1, 32, padding_idx=0)
        self.tag_embedding = nn.Embedding(n_tags + 1, 32, padding_idx=0)
        
        # Attention layers
        self.genre_attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        self.tag_attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(64 + 128 + 32 + 32, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )
    
    def forward(self, user_idx, anime_idx, genre_indices, tag_indices):
        """
        Forward pass of the model.

        Args:
            user_idx (torch.Tensor): User indices.
            anime_idx (torch.Tensor): Anime indices.
            genre_indices (torch.Tensor): Genre indices for each anime.
            tag_indices (torch.Tensor): Tag indices for each anime.

        Returns:
            torch.Tensor: Predicted ratings.
        """
        user_embed = self.user_embedding(user_idx)  # (batch_size, 64)
        anime_embed = self.anime_embedding(anime_idx)  # (batch_size, 128)
        
        # Apply normalization for stability
        user_embed = nn.functional.normalize(user_embed, p=2, dim=1)
        anime_embed = nn.functional.normalize(anime_embed, p=2, dim=1)
        
        genre_embeds = self.genre_embedding(genre_indices)  # (batch_size, max_genres, 32)
        genre_mask = (genre_indices == 0)  # (batch_size, max_genres)
        
        # Handle empty genre lists by ensuring at least one valid entry
        # Instead of using if condition, use tensor operations
        all_masked = genre_mask.all(dim=1, keepdim=True)
        fixed_genre_mask = genre_mask.clone()
        # Set the first position to False for rows where all are masked
        fixed_genre_mask[:, 0] = fixed_genre_mask[:, 0] & ~all_masked.squeeze(1)
        
        genre_embeds = genre_embeds.permute(1, 0, 2)  # (max_genres, batch_size, 32)
        genre_attended, _ = self.genre_attention(
            genre_embeds, 
            genre_embeds, 
            genre_embeds, 
            key_padding_mask=fixed_genre_mask,
            need_weights=False
        )
        genre_attended = genre_attended.mean(dim=0)  # (batch_size, 32)
        genre_attended = nn.functional.normalize(genre_attended, p=2, dim=1)
        
        tag_embeds = self.tag_embedding(tag_indices)  # (batch_size, max_tags, 32)
        tag_mask = (tag_indices == 0)  # (batch_size, max_tags)
        
        # Handle empty tag lists by ensuring at least one valid entry
        # Using tensor operations instead of if condition
        all_masked_tags = tag_mask.all(dim=1, keepdim=True)
        fixed_tag_mask = tag_mask.clone()
        # Set the first position to False for rows where all are masked
        fixed_tag_mask[:, 0] = fixed_tag_mask[:, 0] & ~all_masked_tags.squeeze(1)
            
        tag_embeds = tag_embeds.permute(1, 0, 2)  # (max_tags, batch_size, 32)
        tag_attended, _ = self.tag_attention(
            tag_embeds, 
            tag_embeds, 
            tag_embeds, 
            key_padding_mask=fixed_tag_mask,
            need_weights=False
        )
        tag_attended = tag_attended.mean(dim=0)  # (batch_size, 32)
        tag_attended = nn.functional.normalize(tag_attended, p=2, dim=1)
        
        concatenated = torch.cat([user_embed, anime_embed, genre_attended, tag_attended], dim=1)  # (batch_size, 256)
        
        # Add a small epsilon to prevent potential NaN values
        output = self.mlp(concatenated)  # (batch_size, 1)
        
        # Replace NaN values with zeros as a safeguard
        output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        
        return output.squeeze(1)  # (batch_size,)

### Trainer Class
class ModelTrainer:
    """Manages training of the anime recommendation model in PyTorch."""
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        output_dir: str = "data/model/pytorch",
        batch_size: int = 256,
        epochs: int = 50,
        learning_rate: float = 0.0005,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 10,
        gradient_clip_val: float = 1.0
    ):
        """
        Initialize the trainer with training parameters.

        Args:
            data_dir (str): Directory containing processed data.
            output_dir (str): Directory to save the trained model.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for regularization.
            early_stopping_patience (int): Patience for early stopping.
            gradient_clip_val (float): Maximum gradient norm for gradient clipping.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_val = gradient_clip_val
        
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
        self.rating_mean = None
        self.rating_std = None
    
    def load_data(self):
        """Load processed data and create DataLoaders."""
        print("Loading processed data...")
        
        train_df = pd.read_csv(os.path.join(self.data_dir, "train_ratings.csv"))
        val_df = pd.read_csv(os.path.join(self.data_dir, "val_ratings.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test_ratings.csv"))
        
        self.rating_mean = train_df['rating'].mean()
        self.rating_std = train_df['rating'].std()
        print(f"Rating mean: {self.rating_mean:.4f}, std: {self.rating_std:.4f}")
        
        if self.rating_std > 0:
            train_df['rating'] = (train_df['rating'] - self.rating_mean) / self.rating_std
            val_df['rating'] = (val_df['rating'] - self.rating_mean) / self.rating_std
            test_df['rating'] = (test_df['rating'] - self.rating_mean) / self.rating_std
        
        print(f"Loaded {len(train_df)} training samples")
        print(f"Loaded {len(val_df)} validation samples")
        print(f"Loaded {len(test_df)} test samples")
        
        with open(os.path.join(self.data_dir, "model_config.json"), 'r') as f:
            self.model_config = json.load(f)
        
        with open(os.path.join(self.data_dir, "anime_metadata.json"), 'r', encoding='utf-8') as f:
            self.anime_metadata = json.load(f)
        
        train_dataset = AnimeDataset(train_df, self.anime_metadata)
        val_dataset = AnimeDataset(val_df, self.anime_metadata)
        test_dataset = AnimeDataset(test_df, self.anime_metadata)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def create_model(self):
        """Create the model, optimizer, loss function, and scheduler."""
        print("Creating model...")
        self.model = ImprovedAnimeRecommenderModel(
            n_users=self.model_config['n_users'],
            n_anime=self.model_config['n_anime'],
            n_genres=self.model_config['n_genres'],
            n_tags=self.model_config['n_tags']
        )
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
        
        self.model.apply(init_weights)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = nn.L1Loss()
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train(self):
        """Train the model with early stopping and learning rate scheduling."""
        print("Training model...")
        self.model.to(self.device)
        
        # Create output directory at the beginning of training
        os.makedirs(self.output_dir, exist_ok=True)
        
        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_mse': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                user_idx, anime_idx, genre_indices, tag_indices, rating = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                output = self.model(user_idx, anime_idx, genre_indices, tag_indices)
                loss = self.criterion(output, rating)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                train_loss += loss.item() * len(rating)
            train_loss /= len(self.train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            val_loss, val_mae, val_mse = self.evaluate(self.val_loader)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_mse'].append(val_mse)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val MSE: {val_mse:.4f}")
            
            if np.isnan(train_loss) or np.isnan(val_loss):
                print("NaN values detected, stopping training")
                break
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered")
                    break
            
            self.lr_scheduler.step(val_loss)
        
        if os.path.exists(os.path.join(self.output_dir, 'best_model.pth')):
            self.model.load_state_dict(torch.load(os.path.join(self.output_dir, 'best_model.pth')))
        
        return history
    
    def evaluate(self, loader):
        """
        Evaluate the model on a given dataset.

        Args:
            loader (DataLoader): DataLoader for evaluation.

        Returns:
            Tuple[float, float, float]: Loss (MAE), MAE, and MSE.
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        with torch.no_grad():
            for batch in loader:
                user_idx, anime_idx, genre_indices, tag_indices, rating = [b.to(self.device) for b in batch]
                output = self.model(user_idx, anime_idx, genre_indices, tag_indices)
                
                # Calculate loss on normalized values (as trained)
                loss = self.criterion(output, rating)
                total_loss += loss.item() * len(rating)
                
                # Denormalize predictions and ratings for more interpretable metrics
                if self.rating_std is not None and self.rating_mean is not None:
                    denorm_output = output * self.rating_std + self.rating_mean
                    denorm_rating = rating * self.rating_std + self.rating_mean
                else:
                    denorm_output = output
                    denorm_rating = rating
                
                total_mae += torch.abs(denorm_output - denorm_rating).sum().item()
                total_mse += ((denorm_output - denorm_rating) ** 2).sum().item()
                
        total_loss /= len(loader.dataset)
        total_mae /= len(loader.dataset)
        total_mse /= len(loader.dataset)
        return total_loss, total_mae, total_mse
    
    def save_model(self):
        """Save the trained model and associated metadata."""
        print("Saving model and assets...")
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pth'))
        
        with open(os.path.join(self.data_dir, "mappings.pkl"), 'rb') as f:
            mappings = pickle.load(f)
        
        reverse_mappings = {
            'idx_to_anime': {str(v): str(k) for k, v in mappings['anime_id_map'].items()},
            'anime_to_idx': {str(k): v for k, v in mappings['anime_id_map'].items()},
            'idx_to_genre': {str(v): k for k, v in mappings['genre_map'].items()},
            'genre_to_idx': {str(k): v for k, v in mappings['genre_map'].items()},
            'idx_to_tag': {str(v): k for k, v in mappings['tag_map'].items()},
            'tag_to_idx': {str(k): v for k, v in mappings['tag_map'].items()},
        }
        
        with open(os.path.join(self.output_dir, 'model_mappings.json'), 'w') as f:
            json.dump(reverse_mappings, f, indent=2)
        
        model_metadata = {
            'n_users': self.model_config['n_users'],
            'n_anime': self.model_config['n_anime'],
            'n_genres': self.model_config['n_genres'],
            'n_tags': self.model_config['n_tags'],
            'user_embedding_dim': 64,
            'anime_embedding_dim': 128,
            'genre_embedding_dim': 32,
            'tag_embedding_dim': 32,
            'dense_layers': [256, 128, 64, 32],
            'max_genres': 10,
            'max_tags': 20,
            'model_type': 'pytorch',
            'rating_normalization': {
                'mean': float(self.rating_mean) if self.rating_mean is not None else None,
                'std': float(self.rating_std) if self.rating_std is not None else None
            }
        }
        
        with open(os.path.join(self.output_dir, 'model_metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=2)
            
        # Save model in ONNX format for web deployment
        try:
            # Create example inputs for ONNX export
            dummy_user = torch.zeros(1, dtype=torch.long).to(self.device)
            dummy_anime = torch.zeros(1, dtype=torch.long).to(self.device)
            dummy_genres = torch.zeros((1, 10), dtype=torch.long).to(self.device)
            dummy_tags = torch.zeros((1, 20), dtype=torch.long).to(self.device)
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Create a separate export-friendly model for ONNX
            class ONNXExportModel(torch.nn.Module):
                def __init__(self, original_model):
                    super(ONNXExportModel, self).__init__()
                    self.original_model = original_model
                
                def forward(self, user_idx, anime_idx, genre_indices, tag_indices):
                    with torch.no_grad():
                        return self.original_model(user_idx, anime_idx, genre_indices, tag_indices)
            
            export_model = ONNXExportModel(self.model)
            
            # Export the model to ONNX format
            onnx_path = os.path.join(self.output_dir, 'model.onnx')
            torch.onnx.export(
                export_model,
                (dummy_user, dummy_anime, dummy_genres, dummy_tags),
                onnx_path,
                export_params=True,
                opset_version=13,  # Updated from 12 to 13 to support unflatten operator
                do_constant_folding=True,
                input_names=['user_idx', 'anime_idx', 'genre_indices', 'tag_indices'],
                output_names=['rating'],
                dynamic_axes={
                    'user_idx': {0: 'batch_size'},
                    'anime_idx': {0: 'batch_size'},
                    'genre_indices': {0: 'batch_size'},
                    'tag_indices': {0: 'batch_size'},
                    'rating': {0: 'batch_size'}
                }
            )
            
            # Try to optimize the exported ONNX model if onnx package is available
            try:
                import onnx
                from onnxsim import simplify
                
                # Load the model
                onnx_model = onnx.load(onnx_path)
                
                # Check that the model is well-formed
                onnx.checker.check_model(onnx_model)
                
                # Simplify the model
                simplified_model, check = simplify(onnx_model)
                if check:
                    onnx.save(simplified_model, onnx_path)
                    print(f"Simplified ONNX model saved to {onnx_path}")
                else:
                    print("Failed to simplify ONNX model")
            except ImportError:
                print("ONNX simplification packages not available, skipping optimization")
            
            print(f"Model successfully exported to ONNX format at {onnx_path}")
        except Exception as e:
            print(f"Error exporting model to ONNX format: {e}")
    
    def run(self):
        """Execute the full training pipeline."""
        self.load_data()
        self.create_model()
        history = self.train()
        test_loss, test_mae, test_mse = self.evaluate(self.test_loader)
        print(f"Test Loss (MAE): {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        self.save_model()
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump({
                'history': {k: [float(v) for v in vals] for k, vals in history.items()},
                'test_metrics': {'loss': test_loss, 'mae': test_mae, 'mse': test_mse}
            }, f, indent=2)
        
        print("Training completed successfully!")

### Main Function
def main():
    """Parse arguments and run the trainer."""
    parser = argparse.ArgumentParser(description="Train anime recommendation model with PyTorch")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed data")
    parser.add_argument("--output-dir", default="data/model/pytorch", help="Directory to save trained model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization factor")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_val=args.gradient_clip_val
    )
    
    trainer.run()
    
    print("Model training completed!")

if __name__ == "__main__":
    main()
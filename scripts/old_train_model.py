#!/usr/bin/env python3
"""
Improved Anime Recommendation Model Training with TensorFlow

This script trains an enhanced neural network recommendation model using TensorFlow,
converting from an original PyTorch implementation with improvements.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class AnimeDataset(tf.keras.utils.Sequence):
    """TensorFlow dataset for anime recommendation data."""
    
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        anime_metadata: Dict[str, Dict],
        batch_size: int,
        max_genres: int = 10,
        max_tags: int = 20
    ):
        self.ratings_df = ratings_df
        self.anime_metadata = anime_metadata
        self.batch_size = batch_size
        self.max_genres = max_genres
        self.max_tags = max_tags
        
        self.user_indices = ratings_df['user_idx'].values
        self.anime_indices = ratings_df['anime_idx'].values
        self.anime_ids = ratings_df['anime_id'].values.astype(str)
        self.ratings = ratings_df['rating'].values
        
        self.genre_tag_cache = {}
        self.num_samples = len(ratings_df)
    
    def __len__(self) -> int:
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def get_genre_tag_indices(self, anime_id: str) -> Tuple[np.ndarray, np.ndarray]:
        if anime_id in self.genre_tag_cache:
            return self.genre_tag_cache[anime_id]
        
        metadata = self.anime_metadata.get(anime_id, {})
        genre_indices = metadata.get('genre_indices', [])[:self.max_genres]
        genre_indices = genre_indices + [0] * (self.max_genres - len(genre_indices))
        
        tag_indices = metadata.get('tag_indices', [])[:self.max_tags]
        tag_indices = tag_indices + [0] * (self.max_tags - len(tag_indices))
        
        self.genre_tag_cache[anime_id] = (np.array(genre_indices), np.array(tag_indices))
        return self.genre_tag_cache[anime_id]
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        user_batch = self.user_indices[start_idx:end_idx]
        anime_batch = self.anime_indices[start_idx:end_idx]
        anime_ids_batch = self.anime_ids[start_idx:end_idx]
        ratings_batch = self.ratings[start_idx:end_idx]
        
        genre_batch = np.zeros((len(user_batch), self.max_genres), dtype=np.int32)
        tag_batch = np.zeros((len(user_batch), self.max_tags), dtype=np.int32)
        
        for i, anime_id in enumerate(anime_ids_batch):
            genre_indices, tag_indices = self.get_genre_tag_indices(anime_id)
            genre_batch[i] = genre_indices
            tag_batch[i] = tag_indices
        
        inputs = (user_batch, anime_batch, genre_batch, tag_batch)
        return inputs, ratings_batch

class ImprovedAnimeRecommenderModel(tf.keras.Model):
    """Enhanced TensorFlow model for anime recommendations."""
    
    def __init__(
        self,
        n_users: int,
        n_anime: int,
        n_genres: int,
        n_tags: int
    ):
        super(ImprovedAnimeRecommenderModel, self).__init__()
        
        # Embedding layers
        self.user_embedding = layers.Embedding(n_users, 64)
        self.anime_embedding = layers.Embedding(n_anime, 128)
        self.genre_embedding = layers.Embedding(n_genres + 1, 32, mask_zero=True)
        self.tag_embedding = layers.Embedding(n_tags + 1, 32, mask_zero=True)
        
        # Attention layers
        self.genre_attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        self.tag_attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)
        
        # MLP layers
        self.mlp = models.Sequential([
            layers.Dense(256, activation='leaky_relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='leaky_relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='leaky_relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(32, activation='leaky_relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(1)
        ])
    
    def call(self, inputs):
        user_idx, anime_idx, genre_indices, tag_indices = inputs
        
        user_embed = self.user_embedding(user_idx)
        anime_embed = self.anime_embedding(anime_idx)
        
        genre_embeds = self.genre_embedding(genre_indices)
        genre_attended = self.genre_attention(genre_embeds, genre_embeds)
        genre_attended = tf.reduce_mean(genre_attended, axis=1)
        
        tag_embeds = self.tag_embedding(tag_indices)
        tag_attended = self.tag_attention(tag_embeds, tag_embeds)
        tag_attended = tf.reduce_mean(tag_attended, axis=1)
        
        concatenated = tf.concat([user_embed, anime_embed, genre_attended, tag_attended], axis=-1)
        output = self.mlp(concatenated)
        return output

class ModelTrainer:
    """Manages training of the anime recommendation model in TensorFlow."""
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        output_dir: str = "data/model/tensorflow",
        batch_size: int = 256,
        epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 10
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.anime_metadata = None
        self.model_config = None
    
    def load_data(self):
        print("Loading processed data...")
        
        train_df = pd.read_csv(os.path.join(self.data_dir, "train_ratings.csv"))
        val_df = pd.read_csv(os.path.join(self.data_dir, "val_ratings.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test_ratings.csv"))
        
        print(f"Loaded {len(train_df)} training samples")
        print(f"Loaded {len(val_df)} validation samples")
        print(f"Loaded {len(test_df)} test samples")
        
        with open(os.path.join(self.data_dir, "model_config.json"), 'r') as f:
            self.model_config = json.load(f)
        
        with open(os.path.join(self.data_dir, "anime_metadata.json"), 'r', encoding='utf-8') as f:
            self.anime_metadata = json.load(f)
        
        self.train_dataset = AnimeDataset(train_df, self.anime_metadata, self.batch_size)
        self.val_dataset = AnimeDataset(val_df, self.anime_metadata, self.batch_size)
        self.test_dataset = AnimeDataset(test_df, self.anime_metadata, self.batch_size)
    
    def create_model(self):
        print("Creating model...")
        
        self.model = ImprovedAnimeRecommenderModel(
            n_users=self.model_config['n_users'],
            n_anime=self.model_config['n_anime'],
            n_genres=self.model_config['n_genres'],
            n_tags=self.model_config['n_tags']
        )
        
        self.model.compile(
            optimizer=optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay),
            loss='mean_absolute_error',
            metrics=['mae', 'mse']
        )
        
        # Build the model by passing dummy inputs (forcing initialization)
        dummy_user = tf.zeros((self.batch_size,), dtype=tf.int32)
        dummy_anime = tf.zeros((self.batch_size,), dtype=tf.int32)
        dummy_genre = tf.zeros((self.batch_size, 10), dtype=tf.int32)
        dummy_tag = tf.zeros((self.batch_size, 20), dtype=tf.int32)
        self.model([dummy_user, dummy_anime, dummy_genre, dummy_tag])
        
        self.model.summary()
    
    def train(self):
        print("Training model...")
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
        
        history = self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            validation_data=self.val_dataset,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        return history.history
    
    def evaluate(self):
        print("Evaluating model on test data...")
        
        test_results = self.model.evaluate(self.test_dataset, return_dict=True)
        print(f"Test Loss (MAE): {test_results['loss']:.4f}")
        print(f"Test MAE: {test_results['mae']:.4f}")
        print(f"Test MSE: {test_results['mse']:.4f}")
        
        return test_results
    
    def save_model(self):
        print("Saving model and assets...")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save(os.path.join(self.output_dir, 'model.keras'))
        
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
            'model_type': 'tensorflow'
        }
        
        with open(os.path.join(self.output_dir, 'model_metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=2)
    
    def run(self):
        self.load_data()
        self.create_model()
        history = self.train()
        metrics = self.evaluate()
        self.save_model()
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump({
                'history': {k: [float(v) for v in vals] for k, vals in history.items()},
                'test_metrics': metrics
            }, f, indent=2)
        
        print("Training completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Train anime recommendation model with TensorFlow")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed data")
    parser.add_argument("--output-dir", default="data/model/tensorflow", help="Directory to save trained model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization factor")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience
    )
    
    trainer.run()
    
    print("Model training completed!")

if __name__ == "__main__":
    main()
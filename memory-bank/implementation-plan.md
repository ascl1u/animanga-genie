# Recommendation Model Pipeline Enhancement Plan

## Overview
This document outlines the plan to enhance our anime recommendation model by incorporating additional features from our expanded anime dataset. The goal is to improve recommendation quality by leveraging anime relationships, studio information, and other metadata while maintaining model efficiency.

## 1. Data Preprocessing Enhancement

### 1.1 Create New Preprocessing Script (`scripts/preprocess.py`)
- Input: `data/anime_catalog.json`, user ratings data
- Output: Enhanced processed datasets and mappings

#### Key Features to Add:
1. **Anime Relationships Processing**
   - Create relationship graph embeddings
   - Weight relationships by type:
     - Sequel/Prequel: Highest weight
     - Side Story/Spin-off: Medium weight
     - Other relationships: Lower weight
   - Generate relationship adjacency matrix

2. **Studio Information Processing**
   - Create studio embeddings
   - Handle multiple studios per anime
   - Normalize studio influence
   - Create studio-anime mapping matrix

3. **Enhanced Tag Processing**
   - Incorporate tag ranks and categories
   - Create weighted tag embeddings
   - Normalize tag importance

### 1.2 Data Structure Updates
```python
# Example processed anime metadata structure
{
    "anime_id": str,
    "relationship_indices": List[int],  # Indices of related anime
    "relationship_weights": List[float],  # Weights based on relationship type
    "studio_indices": List[int],  # Studio indices
    "studio_weights": List[float],  # Normalized studio weights
    "tag_indices": List[int],
    "tag_weights": List[float],  # Based on rank/category
    "genre_indices": List[int],
    "popularity_score": float,  # Normalized popularity
    "average_score": float  # Normalized score
}
```

## 2. Model Architecture Updates (`scripts/train_model.py`)

### 2.1 Enhanced Model Architecture
```python
class ImprovedAnimeRecommenderModel(nn.Module):
    def __init__(self):
        # Existing embeddings
        self.user_embedding = nn.Embedding(n_users, 64)
        self.anime_embedding = nn.Embedding(n_anime, 128)
        self.genre_embedding = nn.Embedding(n_genres + 1, 32)
        self.tag_embedding = nn.Embedding(n_tags + 1, 32)
        
        # New embeddings
        self.studio_embedding = nn.Embedding(n_studios + 1, 16)
        self.relationship_embedding = nn.Embedding(n_anime, 32)
        
        # Enhanced attention mechanisms
        self.genre_attention = nn.MultiheadAttention(32, 4)
        self.tag_attention = nn.MultiheadAttention(32, 4)
        self.studio_attention = nn.MultiheadAttention(16, 2)
        self.relationship_attention = nn.MultiheadAttention(32, 4)
        
        # Updated MLP for combined features
        self.mlp = nn.Sequential(
            nn.Linear(64 + 128 + 32 + 32 + 16 + 32, 256),
            # ... rest of MLP layers
        )
```

### 2.2 Training Parameter Updates
- Increase batch size for larger dataset (512 or 1024)
- Adjust learning rate schedule
- Update early stopping patience
- Modify gradient clipping values

## 3. Implementation Steps

### Phase 1: Data Preprocessing
1. Implement new preprocessing script
2. Generate enhanced datasets
3. Validate data quality and distributions
4. Create new mapping files

### Phase 2: Model Enhancement
1. Update model architecture
2. Modify training pipeline
3. Implement new loss functions
4. Add validation metrics

### Phase 3: Training and Validation
1. Train model with new features
2. Validate performance improvements
3. Fine-tune hyperparameters
4. Generate performance metrics

### Phase 4: Deployment Updates
1. Export enhanced model to ONNX
2. Update model mappings
3. Update metadata files
4. Validate inference performance

## 4. File Changes Summary

### New Files:
- `scripts/preprocess.py` (new preprocessing script)

### Modified Files:
- `scripts/train_model.py` (enhanced model architecture)
- `data/processed/*` (all processed data files)
- `public/model/anime_recommender/*` (all deployment files)

### Generated Files:
- `data/processed/train_ratings.csv`
- `data/processed/val_ratings.csv`
- `data/processed/test_ratings.csv`
- `data/processed/model_config.json`
- `data/processed/anime_metadata.json`
- `data/processed/mappings.pkl`
- `public/model/anime_recommender/model.onnx`
- `public/model/anime_recommender/model_mappings.json`
- `public/model/anime_recommender/onnx_model_metadata.json`

## 5. Execution Commands

```bash
# 1. Preprocess data with new features
python scripts/preprocess.py --input-dir data --output-dir data/processed --min-ratings 10

# 2. Train enhanced model
python scripts/train_model.py --data-dir data/processed --output-dir data/model/pytorch --batch-size 512 --epochs 50

# 3. Convert and deploy model
python scripts/convert_to_onnx.py --model-dir data/model/pytorch --output-dir public/model/anime_recommender
```

## 6. Success Metrics
- Improved recommendation accuracy (MAE, MSE)
- Better handling of series/related anime
- Maintained or improved inference speed
- Reasonable model size for web deployment

## 7. Rollback Plan
- Maintain copy of current model and mappings
- Version control for all code changes
- Automated tests for new features
- Performance comparison benchmarks

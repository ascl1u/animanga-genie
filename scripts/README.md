# Recommendation System Scripts

This directory contains scripts for building and deploying our anime recommendation system. These scripts handle data collection, model training, and integration with our Next.js application.

## Setup

Before running the scripts, install the required Python dependencies:

```bash
pip install requests python-dotenv numpy pandas tqdm tensorflow tensorflowjs
```

## Data Collection Scripts

### 1. fetch-anime.py

This script fetches anime data from the AniList GraphQL API and stores it locally in JSON format.

**Usage:**
```bash
python scripts/fetch-anime.py --output-dir data --page-size 50 --max-pages 100
```

**Arguments:**
- `--output-dir`: Directory to store output files (default: "data")
- `--page-size`: Number of anime per page/API request (default: 50)
- `--max-pages`: Maximum number of pages to fetch (default: 100)

**Output:**
- Generates `data/anime_catalog.json` containing anime metadata

### 2. fetch-user-data.py

This script fetches user preference data (watched anime and ratings) from the AniList API.

**Usage:**
```bash
python scripts/fetch-user-data.py --output-dir data --min-list-size 30 --max-users 100
```

**Arguments:**
- `--output-dir`: Directory to store output files (default: "data")
- `--min-list-size`: Minimum number of rated anime for a user to be included (default: 30)
- `--max-users`: Maximum number of users to collect data for (default: 100)
- `--include-usernames`: Include usernames in the output (optional, default: False for anonymity)

**Output:**
- Generates `data/user_ratings.json` containing user ratings and preferences

## Data Preprocessing Script

### 1. preprocess.py

This script processes the raw anime and user data into a format suitable for training the recommendation model.

**Usage:**
```bash
python scripts/preprocess.py --data-dir data --output-dir data/processed --min-ratings-per-user 10 --min-ratings-per-anime 5
```

**Arguments:**
- `--data-dir`: Directory containing raw data files (default: "data")
- `--output-dir`: Directory to store processed data (default: "data/processed")
- `--min-ratings-per-user`: Minimum ratings required for a user to be included (default: 10)
- `--min-ratings-per-anime`: Minimum ratings required for an anime to be included (default: 5)
- `--max-tags`: Maximum number of tags to include (default: 100)
- `--validation-split`: Proportion of data for validation (default: 0.1)
- `--test-split`: Proportion of data for testing (default: 0.1)
- `--random-seed`: Random seed for reproducibility (default: 42)

**Output:**
- Generates train/validation/test splits in CSV format
- Creates mappings between IDs and indices for users, anime, genres, and tags
- Saves model configuration and anime metadata

## Model Training Script

### 1. train_model.py

This script trains a neural network recommendation model based on the preprocessed data.

**Usage:**
```bash
python scripts/train_model.py --data-dir data/processed --output-dir data/model --batch-size 128 --epochs 20
```

**Arguments:**
- `--data-dir`: Directory with processed data (default: "data/processed")
- `--output-dir`: Directory to save trained model (default: "data/model")
- `--batch-size`: Batch size for training (default: 128)
- `--epochs`: Number of training epochs (default: 20)
- `--learning-rate`: Learning rate for optimizer (default: 0.001)
- `--l2-factor`: L2 regularization factor (default: 1e-5)
- `--early-stopping-patience`: Patience for early stopping (default: 3)

**Output:**
- Saves the trained model in TensorFlow format
- Generates TensorFlow.js model for web deployment
- Creates model mappings and metadata for client-side use

## Model Conversion Script

### 1. convert_model_for_web.py

This script optimizes and packages the trained model for web deployment with TensorFlow.js.

**Usage:**
```bash
python scripts/convert_model_for_web.py --model-dir data/model --output-dir public/model --quantize
```

**Arguments:**
- `--model-dir`: Directory containing the trained model (default: "data/model")
- `--output-dir`: Directory to save the converted model (default: "public/model")
- `--quantize`: Quantize the model for size reduction (optional)

**Output:**
- Optimized TensorFlow.js model files for client-side inference
- Helper JavaScript file for making recommendations
- Simplified anime lookup data for the web application

## Database Scripts

### 1. update_anime_table.sql

SQL script to update the structure of the anime table in Supabase.

**Usage:**
1. Go to the Supabase dashboard for your project
2. Navigate to the SQL Editor
3. Copy and paste the contents of this file
4. Run the query

### 2. load_anime_to_supabase.py

This script loads the collected anime data from JSON into the Supabase database.

**Usage:**
```bash
python scripts/load_anime_to_supabase.py --data-dir data --batch-size 50
```

**Arguments:**
- `--data-dir`: Directory where JSON data is stored (default: "data")
- `--batch-size`: Number of records to insert in a single batch (default: 50)

**Requirements:**
- Valid Supabase credentials in `.env.local` file
- Executed `fetch-anime.py` first to generate the source data

## Complete Workflow

1. Collect data:
   ```bash
   python scripts/fetch-anime.py
   python scripts/fetch-user-data.py --max-users 1000
   ```

2. Update database and load anime data:
   ```bash
   # Run update_anime_table.sql in Supabase dashboard
   python scripts/load_anime_to_supabase.py
   ```

3. Preprocess data and train model:
   ```bash
   python scripts/preprocess.py
   python scripts/train_model.py
   ```

4. Convert model for web use:
   ```bash
   python scripts/convert_model_for_web.py --quantize
   ```

This complete pipeline collects data, trains a neural recommendation model, and prepares it for client-side deployment with TensorFlow.js.
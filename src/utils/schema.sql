-- Users table
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email TEXT UNIQUE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Anime table
CREATE TABLE IF NOT EXISTS anime (
  id SERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  synopsis TEXT,
  genres JSONB,
  rating FLOAT
);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
  user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  preferred_genres JSONB,
  watch_history JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create a testing table for the connection test
CREATE TABLE IF NOT EXISTS testing (
  id SERIAL PRIMARY KEY,
  name TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert a sample row into the testing table
INSERT INTO testing (name) VALUES ('Test Connection');

-- Create sample data for testing
INSERT INTO users (email) 
VALUES ('test@example.com')
ON CONFLICT (email) DO NOTHING;

-- Get the inserted user's ID
DO $$
DECLARE
  test_user_id UUID;
BEGIN
  SELECT id INTO test_user_id FROM users WHERE email = 'test@example.com';
  
  -- Insert sample anime
  INSERT INTO anime (title, synopsis, genres, rating)
  VALUES 
    ('Naruto', 'A young ninja seeks to become the strongest ninja and leader of his village.', '["action", "adventure", "fantasy"]', 4.5),
    ('One Piece', 'Monkey D. Luffy and his crew search for the One Piece treasure.', '["action", "adventure", "comedy", "fantasy"]', 4.7),
    ('Attack on Titan', 'Humanity fights against giant humanoid Titans.', '["action", "drama", "fantasy", "horror"]', 4.8)
  ON CONFLICT DO NOTHING;
  
  -- Insert sample user preferences
  INSERT INTO user_preferences (user_id, preferred_genres, watch_history)
  VALUES (
    test_user_id,
    '["action", "adventure"]',
    '[
      {"anime_id": 1, "rating": 4, "watch_status": "completed", "watch_date": "2025-01-15"},
      {"anime_id": 2, "rating": 5, "watch_status": "watching", "watch_date": "2025-03-01"}
    ]'
  )
  ON CONFLICT (user_id) DO NOTHING;
END
$$; 
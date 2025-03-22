# Supabase Setup Documentation

This document provides details about the Supabase database setup for Animanga-Genie.

## Database Schema

### Tables

#### users
Contains user authentication information (managed by Supabase Auth).

#### anime
Stores anime information from external sources like AniList.

| Column       | Type                  | Description                        |
|--------------|------------------------|------------------------------------|
| id           | integer (primary key) | Auto-incrementing ID               |
| anilist_id   | integer (not null)    | ID from AniList                    |
| title        | jsonb (not null)      | Contains title in different forms  |
| rating       | double precision      | Average rating                     |
| genres       | jsonb                 | Array of genres                    |
| tags         | jsonb                 | Array of tags                      |
| popularity   | integer               | Popularity score                   |
| format       | text                  | Format (TV, Movie, etc.)           |
| episodes     | integer               | Number of episodes                 |
| year         | integer               | Release year                       |
| description  | text                  | Anime description                  |
| image_url    | text                  | URL to cover image                 |
| created_at   | timestamptz           | Creation timestamp                 |
| updated_at   | timestamptz           | Last update timestamp              |

#### anime_watch_history
Tracks which anime users have watched and their ratings.

| Column       | Type                  | Description                        |
|--------------|------------------------|------------------------------------|
| id           | uuid (primary key)    | Unique identifier                  |
| user_id      | uuid (not null)       | Reference to auth.users            |
| anilist_id   | integer (not null)    | ID from AniList                    |
| title        | text (not null)       | Anime title                        |
| cover_image  | text                  | URL to cover image                 |
| rating       | integer               | User's rating (1-10)               |
| created_at   | timestamptz           | Creation timestamp                 |
| updated_at   | timestamptz           | Last update timestamp              |

#### user_preferences
Stores user preferences including preferred genres.

| Column           | Type             | Description                        |
|------------------|------------------|------------------------------------|
| user_id          | uuid (primary key) | Reference to auth.users          |
| preferred_genres | jsonb            | Array of preferred genres          |
| watch_history    | jsonb            | Legacy/deprecated field            |
| created_at       | timestamptz      | Creation timestamp                 |
| updated_at       | timestamptz      | Last update timestamp              |

#### anime_recommendations
Stores generated recommendations for users.

| Column           | Type             | Description                        |
|------------------|------------------|------------------------------------|
| id               | uuid (primary key) | Unique identifier               |
| user_id          | uuid (not null)  | Reference to auth.users            |
| recommendations  | jsonb (not null) | Array of recommendation objects    |
| watch_history_hash | text           | Hash to track if recs need update  |
| created_at       | timestamptz      | Creation timestamp                 |
| updated_at       | timestamptz      | Last update timestamp              |

#### testing
Used for testing Supabase connection.

#### polls
Stores poll questions and their answer options.

| Column       | Type                  | Description                        |
|--------------|------------------------|------------------------------------|
| id           | integer (primary key) | Auto-incrementing ID               |
| question     | text (not null)       | The poll question                  |
| options      | jsonb (not null)      | Array of answer options            |
| created_at   | timestamptz           | Creation timestamp                 |
| active       | boolean               | Whether poll is active             |

#### votes
Tracks user votes on polls.

| Column       | Type                  | Description                        |
|--------------|------------------------|------------------------------------|
| id           | integer (primary key) | Auto-incrementing ID               |
| poll_id      | integer (not null)    | Reference to polls table           |
| user_id      | uuid (not null)       | Reference to auth.users            |
| option_index | integer (not null)    | Index of selected option           |
| voted_at     | timestamptz           | Vote timestamp                     |

## Row Level Security (RLS)

All tables have Row Level Security enabled with policies that:
- Allow users to only view/modify their own data
- Prevent access to other users' data

## Common Database Operations

### Adding to Watch History
```typescript
const { data, error } = await supabase
  .from('anime_watch_history')
  .upsert({
    user_id: user.id,
    anilist_id: 12345,
    title: "Anime Title",
    cover_image: "https://example.com/image.jpg",
    rating: 8
  })
  .select()
  .single();
```

### Getting Watch History
```typescript
const { data, error } = await supabase
  .from('anime_watch_history')
  .select('*')
  .eq('user_id', user.id);
```

### Saving Recommendations
```typescript
const { data, error } = await supabase
  .from('anime_recommendations')
  .upsert({
    user_id: user.id,
    recommendations: recommendationsArray,
    watch_history_hash: "hash-of-watch-history"
  })
  .select()
  .single();
```

### Loading Recommendations
```typescript
const { data, error } = await supabase
  .from('anime_recommendations')
  .select('*')
  .eq('user_id', user.id)
  .single();
```

## Important Notes
- The `watch_history` field in `user_preferences` is deprecated - use the `anime_watch_history` table instead.
- Recommendations are stored as JSONB arrays in the `recommendations` field.
- The `watch_history_hash` field helps determine if recommendations need to be regenerated based on watch history changes.

## Polls Database Setup

### Creating Tables

```sql
-- Create polls table
CREATE TABLE polls (
  id SERIAL PRIMARY KEY,
  question TEXT NOT NULL,
  options JSONB NOT NULL,  -- Stores answer options as a JSON array
  created_at TIMESTAMPTZ DEFAULT NOW(),
  active BOOLEAN DEFAULT TRUE
);

-- Create votes table
CREATE TABLE votes (
  id SERIAL PRIMARY KEY,
  poll_id INTEGER REFERENCES polls(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  option_index INTEGER NOT NULL,  -- Index of the chosen option in the options array
  voted_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (poll_id, user_id)  -- Ensures one vote per user per poll
);

-- Enable Row Level Security
ALTER TABLE polls ENABLE ROW LEVEL SECURITY;
ALTER TABLE votes ENABLE ROW LEVEL SECURITY;

-- Create policies for polls
CREATE POLICY "Anyone can view polls" 
ON polls FOR SELECT USING (true);

CREATE POLICY "Only administrators can insert polls" 
ON polls FOR INSERT TO authenticated 
USING (auth.jwt() -> 'app_metadata' ->> 'role' = 'admin');

CREATE POLICY "Only administrators can update polls" 
ON polls FOR UPDATE TO authenticated 
USING (auth.jwt() -> 'app_metadata' ->> 'role' = 'admin');

-- Create policies for votes
CREATE POLICY "Users can view all votes" 
ON votes FOR SELECT USING (true);

CREATE POLICY "Users can insert their own votes" 
ON votes FOR INSERT TO authenticated 
USING (auth.uid() = user_id);

CREATE POLICY "Users cannot update votes" 
ON votes FOR UPDATE USING (false);

CREATE POLICY "Users cannot delete votes" 
ON votes FOR DELETE USING (false);
```

### Common Poll Operations

#### Creating a New Poll
```typescript
const { data, error } = await supabase
  .from('polls')
  .insert({
    question: 'What is your favorite anime this season?',
    options: ['Demon Slayer', 'My Hero Academia', 'Jujutsu Kaisen']
  })
  .select()
  .single();
```

#### Fetching Active Polls
```typescript
const { data, error } = await supabase
  .from('polls')
  .select('*')
  .eq('active', true)
  .order('created_at', { ascending: false });
```

#### Submitting a Vote
```typescript
const { data, error } = await supabase
  .from('votes')
  .insert({
    poll_id: 1,
    user_id: user.id,
    option_index: 2 // Voting for the third option (index 2)
  })
  .select()
  .single();
```

#### Getting Poll Results
```typescript
// First get the poll
const { data: poll, error: pollError } = await supabase
  .from('polls')
  .select('*')
  .eq('id', pollId)
  .single();

// Then get votes for this poll
const { data: votes, error: votesError } = await supabase
  .from('votes')
  .select('option_index')
  .eq('poll_id', pollId);

// Process to count votes by option
const results = poll.options.map((option, index) => {
  const voteCount = votes.filter(vote => vote.option_index === index).length;
  return { option, votes: voteCount };
});
```

#### Checking if User Has Voted
```typescript
const { data, error } = await supabase
  .from('votes')
  .select('id')
  .eq('poll_id', pollId)
  .eq('user_id', user.id)
  .maybeSingle();

const hasVoted = !!data;
```
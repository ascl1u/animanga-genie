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
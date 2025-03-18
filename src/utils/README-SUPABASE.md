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

## Auth Triggers Setup

To ensure user data consistency between Auth and Database tables, follow these steps:

1. Go to the "SQL Editor" section in your Supabase dashboard
2. Create a new query
3. Copy and paste the contents of the `auth-triggers.sql` file into the query editor
4. Run the query to create the trigger function

This trigger will automatically:
1. Create a user record in the `users` table when a new user signs up
2. Create default user preferences in the `user_preferences` table
3. Handle both the initial user creation and email confirmation events

## Authentication Configuration

### Environment Variables

Your project requires the following environment variables for authentication:

```
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_SITE_URL=your-site-url
```

For local development, set `NEXT_PUBLIC_SITE_URL` to `http://localhost:3000`.

### Supabase Auth Settings

In your Supabase dashboard:

1. Go to "Authentication" → "URL Configuration"
2. Add `http://localhost:3000` as an allowed redirect URL for local development
3. Add your production domain for production deployments
4. Set the Site URL to match your `NEXT_PUBLIC_SITE_URL`

### Password Reset Flow

The password reset flow requires proper configuration:

1. When a user requests a password reset, a link is sent to their email
2. This link contains either a hash fragment (`#access_token=...`) or query parameters with a code
3. The reset password page (`/auth/reset-password`) handles both formats
4. Password reset links expire quickly (usually within 1 hour)

For testing the password reset flow:
1. Use a real email address that you can access
2. Click the reset link as soon as you receive it
3. If you get an "expired" error, request a new link

## Troubleshooting Auth Issues

If users are not being properly created in your database tables after signup, check:

1. The Supabase logs for any errors related to trigger execution
2. That both the `auth-triggers.sql` and `rls-policies.sql` have been executed
3. That your application's callback route is properly handling new user creation as a fallback

If password reset links are not working:
1. Check that `NEXT_PUBLIC_SITE_URL` is correctly set
2. Verify that the redirect URL is allowed in Supabase settings
3. Ensure you're clicking the link within its validity period
4. Look for error parameters in the URL when redirected (e.g., `?error=expired`)

## Hydration Issues

If you encounter hydration errors related to authentication:
1. Ensure the ClientNavigation component uses the mounting pattern to prevent hydration mismatches
2. Check that the AuthProvider correctly initializes both user and session states to null
3. Verify that client-side components are properly marked with 'use client'
4. Use useEffect for any authentication-related operations that should only run on the client

## Database Tables

The schema creates the following tables:

1. **users** - Stores user information
   - `id` (UUID, primary key, auto-generated)
   - `email` (text, unique)
   - `created_at` (timestamp with time zone, default now)

2. **anime** - Stores anime details
   - `id` (serial integer, primary key, auto-increment)
   - `title` (text)
   - `synopsis` (text)
   - `genres` (JSON array of genre names)
   - `rating` (float)

3. **user_preferences** - Stores user preferences and watch history
   - `user_id` (UUID, primary key, foreign key to users.id)
   - `preferred_genres` (JSON array of genre names)
   - `watch_history` (JSON array containing anime_id, rating, watch_status, and watch_date)
   - `created_at` (timestamp with time zone, default now)
   - `updated_at` (timestamp with time zone, default now)

4. **testing** - A simple table for testing the Supabase connection
   - `id` (serial integer, primary key)
   - `name` (text)
   - `created_at` (timestamp with time zone, default now)

## Sample Data

The schema includes sample data insertion for testing:
- A test user with email 'test@example.com'
- Three sample anime entries
- Sample user preferences with preferred genres and watch history

## Testing the Connection

After setting up the schema, you can test the connection by visiting:
- `/api/test-supabase` - Should return a success message if the connection is working properly

## Next Steps

After confirming that the Supabase integration is working correctly:
1. Implement authentication with Supabase Auth
2. Create API routes for interacting with the database
3. Build UI components to display and manipulate the data 
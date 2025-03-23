# Supabase Setup Documentation

This document provides details about the Supabase database setup for Animanga-Genie.

## Database Schema

### Tables

#### users
Contains user authentication information (managed by Supabase Auth).

| Column       | Type                  | Description                        |
|--------------|------------------------|------------------------------------|
| id           | uuid (primary key)    | User ID (same as auth.users id)    |
| email        | text                  | User's email address               |
| created_at   | timestamptz           | Creation timestamp                 |

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
| user_id      | uuid (not null)       | Reference to users.id              |
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
| user_id          | uuid (primary key) | Reference to users.id            |
| preferred_genres | jsonb            | Array of preferred genres          |
| watch_history    | jsonb            | Legacy/deprecated field            |
| created_at       | timestamptz      | Creation timestamp                 |
| updated_at       | timestamptz      | Last update timestamp              |

#### anime_recommendations
Stores generated recommendations for users.

| Column           | Type             | Description                        |
|------------------|------------------|------------------------------------|
| id               | uuid (primary key) | Unique identifier               |
| user_id          | uuid (not null)  | Reference to users.id              |
| recommendations  | jsonb (not null) | Array of recommendation objects    |
| watch_history_hash | text           | Hash to track if recs need update  |
| created_at       | timestamptz      | Creation timestamp                 |
| updated_at       | timestamptz      | Last update timestamp              |

#### audit_logs
Tracks sensitive database operations made with service role credentials.

| Column           | Type             | Description                        |
|------------------|------------------|------------------------------------|
| id               | uuid (primary key) | Unique identifier                |
| timestamp        | timestamptz      | When the action occurred           |
| action           | text             | Operation type (INSERT/UPDATE/DELETE) |
| table_name       | text             | Name of table affected             |
| user_id          | uuid             | User who performed the action      |
| record_id        | text             | ID of affected record              |
| old_data         | jsonb            | Previous record state (for updates/deletes) |
| new_data         | jsonb            | New record state (for inserts/updates) |
| client_info      | jsonb            | IP address and user agent          |

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
| user_id      | uuid (not null)       | Reference to public.users.id       |
| option_index | integer (not null)    | Index of selected option           |
| voted_at     | timestamptz           | Vote timestamp                     |

## Foreign Key Relationships

The following foreign key relationships are established:

| Table                  | Column   | References                |
|------------------------|----------|---------------------------|
| anime_recommendations  | user_id  | public.users.id           |
| anime_watch_history    | user_id  | public.users.id           |
| user_preferences       | user_id  | public.users.id           |
| votes                  | poll_id  | public.polls.id           |
| votes                  | user_id  | public.users.id           |

## Database Triggers

| Trigger Name                     | Event            | Action                                |
|----------------------------------|------------------|---------------------------------------|
| on_auth_user_created             | INSERT           | EXECUTE FUNCTION handle_new_user()    |
| on_auth_user_updated             | UPDATE           | EXECUTE FUNCTION handle_new_user()    |
| audit_users_service_role         | INSERT/UPDATE/DELETE | EXECUTE FUNCTION audit_service_role_access() |
| audit_user_preferences_service_role | INSERT/UPDATE/DELETE | EXECUTE FUNCTION audit_service_role_access() |

## Database Functions

| Function Name                | Purpose                                        |
|------------------------------|------------------------------------------------|
| handle_new_user()            | Creates user records when auth users are created |
| audit_service_role_access()  | Logs service role operations to audit_logs      |
| is_same_user(UUID)           | Safely validates if current user matches target user |

## Row Level Security (RLS)

Row Level Security is now **ENABLED** on all critical tables:
- anime_recommendations
- anime_watch_history
- user_preferences
- users

### RLS Policies

#### anime_recommendations
- "Users can insert their own recommendations" (INSERT) - Checks user_id = auth.uid()
- "Users can update their own recommendations" (UPDATE) - Checks user_id = auth.uid()
- "Users can view their own recommendations" (SELECT) - Checks user_id = auth.uid()

#### anime_watch_history
- "anime_watch_history_delete_policy" (DELETE) - Checks auth.uid() = user_id
- "anime_watch_history_insert_policy" (INSERT) - Checks auth.uid() = user_id
- "anime_watch_history_select_policy" (SELECT) - Checks auth.uid() = user_id
- "anime_watch_history_update_policy" (UPDATE) - Checks auth.uid() = user_id

#### user_preferences
- "user_preferences_delete_own" (DELETE) - Checks auth.uid() = user_id
- "user_preferences_insert_own" (INSERT) - Checks auth.uid() = user_id
- "user_preferences_read_own" (SELECT) - Checks auth.uid() = user_id
- "user_preferences_update_own" (UPDATE) - Checks auth.uid() = user_id
- "user_preferences_service_all" (ALL) - Checks auth.jwt() ->> 'role' = 'service_role'

#### users
- "users_read_own" (SELECT) - Checks auth.uid() = id
- "users_update_own" (UPDATE) - Checks auth.uid() = id
- "users_service_all" (ALL) - Checks auth.jwt() ->> 'role' = 'service_role'

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

## Security Improvements

The following security improvements have been implemented:

1. **Row Level Security Enabled**: RLS is now enabled on all critical user data tables, ensuring users can only access their own data.

2. **Audit Logging**: A new audit_logs table with triggers tracks all operations performed using service role credentials, providing traceability for administrative actions.

3. **User Validation Function**: Added a new is_same_user() function that provides safer user ID validation with proper null checks.

4. **Consistent Foreign Keys**: Updated the votes table to reference public.users.id instead of auth.users.id for consistency across the database.

5. **ON DELETE CASCADE**: Foreign key relationships now use CASCADE to ensure data integrity when users are deleted.

## Important Notes
- The `watch_history` field in `user_preferences` is deprecated - use the `anime_watch_history` table instead.
- Recommendations are stored as JSONB arrays in the `recommendations` field.
- The `watch_history_hash` field helps determine if recommendations need to be regenerated based on watch history changes.
- Service role operations are now logged for auditing purposes.

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
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
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
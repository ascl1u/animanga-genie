# Animanga-Genie Architecture Documentation

This document details the architecture of the Animanga-Genie project, including file purposes and dependencies.

## Configuration Files

### .eslintrc.json
**Purpose:** Defines linting rules and configurations for the project.
**Description:** Extends Next.js core-web-vitals and Prettier configs, with custom rules for React and TypeScript.
**Dependencies:** ESLint, Prettier

### .prettierrc
**Purpose:** Defines code formatting rules for consistent code style.
**Description:** Configures code style preferences like tab width, quotes style, and line length.
**Dependencies:** Prettier

### .eslintignore
**Purpose:** Specifies files and directories to be excluded from ESLint.
**Description:** Excludes build artifacts, dependency directories, and other non-source files.
**Dependencies:** ESLint

### .prettierignore
**Purpose:** Specifies files and directories to be excluded from Prettier formatting.
**Description:** Excludes build artifacts, dependency directories, and lock files.
**Dependencies:** Prettier

### .env.local
**Purpose:** Stores environment variables for local development.
**Description:** Contains Supabase URL, anon key, and site URL for authentication redirects.
**Dependencies:** Next.js, Supabase

### next.config.ts
**Purpose:** Configuration for Next.js.
**Description:** Defines build settings, module exports, and environment variables for the Next.js application.
**Dependencies:** Next.js

### tsconfig.json
**Purpose:** TypeScript configuration.
**Description:** Specifies TypeScript compiler options and paths.
**Dependencies:** TypeScript

### postcss.config.mjs
**Purpose:** PostCSS configuration.
**Description:** Configures CSS processing plugins.
**Dependencies:** PostCSS, Tailwind CSS

## Project Structure

The project follows the standard Next.js 15.2.2 application structure with the app router:

```
animanga-genie/
├── src/
│   ├── app/                             # Main application code using the App Router
│   │   ├── api/                         # API routes
│   │   ├── auth/                        # Auth-related pages and routes
│   │   ├── login/                       # Login page
│   │   ├── signup/                      # Signup page
│   │   ├── reset-password/              # Password reset page
│   │   ├── profile/                     # User profile pages
│   │   ├── recommendations/             # Anime recommendations pages
│   │   ├── my-anime/                    # User's anime watchlist pages
│   │   ├── search/                      # Search functionality pages
│   │   ├── error/                       # Error handling pages
│   │   ├── page.tsx                     # Home page
│   │   ├── layout.tsx                   # Root layout
│   │   └── globals.css                  # Global CSS styles
│   ├── components/                      # Reusable UI components
│   │   ├── ui/                          # Basic UI components
│   │   ├── AnimeSearch.tsx              # Anime search component
│   │   ├── ClientNavigation.tsx         # Client-side navigation component
│   │   ├── RecommendationCard.tsx       # Component to display anime recommendations
│   │   ├── SearchBar.tsx                # Search input component
│   │   ├── SimpleAuthProvider.tsx       # Authentication context provider
│   │   ├── WatchHistoryForm.tsx         # Form to add anime to watch history
│   │   ├── WatchHistoryImport.tsx       # Component to import watch history
│   │   ├── WatchHistoryList.tsx         # Component to display watch history
│   │   └── RatingEditor.tsx             # Component to edit anime ratings
│   ├── context/                         # React context providers
│   ├── constants/                       # Application constants
│   ├── hooks/                           # Custom React hooks
│   ├── services/                        # Service layer for external APIs
│   ├── types/                           # TypeScript type definitions
│   ├── utils/                           # Utility functions
│   │   ├── supabase/                    # Supabase utilities
│   │   │   ├── client.ts                # Supabase client for browser
│   │   │   ├── server.ts                # Supabase client for server components
│   │   │   └── middleware.ts            # Supabase middleware utilities
│   │   ├── anilistClient.ts             # AniList API client
│   │   ├── auth.ts                      # Authentication utilities
│   │   ├── hooks.ts                     # Utility hooks
│   │   ├── README-SUPABASE.md           # Documentation for Supabase setup
│   │   └── supabase-best-practices.md   # Supabase best practices guide
│   └── middleware.ts                    # Next.js middleware for auth protection
├── public/                              # Static assets
├── scripts/                             # Utility scripts
├── data/                                # Static data files
├── memory-bank/                         # Project documentation
└── [configuration files]
```

## Supabase Integration

Based on `src/utils/README-SUPABASE.md`, which is the source of truth for Supabase integration:

### src/utils/supabase/client.ts
**Purpose:** Initializes and exports the Supabase client for browser use.
**Description:** Creates a client instance for client-side components.
**Dependencies:** @supabase/supabase-js

### src/utils/supabase/server.ts
**Purpose:** Initializes and exports the Supabase client for server components.
**Description:** Creates a client instance with cookies for server-side rendering.
**Dependencies:** @supabase/supabase-js, @supabase/ssr

### src/utils/supabase/middleware.ts
**Purpose:** Middleware utilities for Supabase authentication.
**Description:** Handles server-side authentication checks and redirects.
**Dependencies:** @supabase/ssr, Next.js middleware

## Database Structure

The application uses the following database tables:

### users
**Purpose:** Contains user authentication information (managed by Supabase Auth).
**Description:** Stores user credentials and profile information.

### anime
**Purpose:** Stores anime information from external sources like AniList.
**Description:** Contains detailed anime metadata including:
- id (primary key)
- anilist_id (reference to AniList)
- title (in different forms as JSONB)
- rating, genres, tags
- popularity, format, episodes
- year, description, image_url
- created_at, updated_at timestamps

### anime_watch_history
**Purpose:** Tracks which anime users have watched and their ratings.
**Description:** Records user interactions with anime including:
- id (UUID primary key)
- user_id (reference to auth.users)
- anilist_id (reference to AniList)
- title, cover_image
- user's rating (1-10)
- created_at, updated_at timestamps

### user_preferences
**Purpose:** Stores user preferences including preferred genres.
**Description:** Contains:
- user_id (primary key, reference to auth.users)
- preferred_genres (JSONB array)
- watch_history (legacy/deprecated field)
- created_at, updated_at timestamps

### anime_recommendations
**Purpose:** Stores generated recommendations for users.
**Description:** Contains:
- id (UUID primary key)
- user_id (reference to auth.users)
- recommendations (JSONB array of recommendation objects)
- watch_history_hash (to track if recommendations need updating)
- created_at, updated_at timestamps

### testing
**Purpose:** Used for testing Supabase connection.
**Description:** Simple table for testing database connectivity.

## Key Components and Functionality

### Authentication
**Implementation:** Uses Supabase Auth with email/password authentication.
**Files:**
- src/utils/auth.ts
- src/components/SimpleAuthProvider.tsx
- src/app/login/page.tsx
- src/app/signup/page.tsx
- src/app/reset-password/page.tsx

### Anime Data Integration
**Implementation:** Uses AniList API to fetch anime data.
**Files:**
- src/utils/anilistClient.ts
- src/components/AnimeSearch.tsx
- src/components/SearchBar.tsx
- src/app/search/page.tsx

### Watch History Management
**Implementation:** Allows users to track anime they've watched and rate them.
**Files:**
- src/components/WatchHistoryForm.tsx
- src/components/WatchHistoryList.tsx
- src/components/WatchHistoryImport.tsx
- src/components/RatingEditor.tsx
- src/app/my-anime/page.tsx

### Recommendations
**Implementation:** Generates and displays anime recommendations based on user preferences.
**Files:**
- src/components/RecommendationCard.tsx
- src/app/recommendations/page.tsx

### Navigation
**Implementation:** Provides site navigation with authentication status awareness.
**Files:**
- src/components/ClientNavigation.tsx
- src/app/layout.tsx

## Row Level Security (RLS)

All Supabase tables have Row Level Security enabled with policies that:
- Allow users to only view/modify their own data
- Prevent access to other users' data

## Important Notes
- The `watch_history` field in `user_preferences` is deprecated - use the `anime_watch_history` table instead.
- Recommendations are stored as JSONB arrays in the `recommendations` field.
- The `watch_history_hash` field helps determine if recommendations need to be regenerated based on watch history changes.

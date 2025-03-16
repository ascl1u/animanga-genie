# AniManga Genie - Architecture Documentation

## Overview

AniManga Genie is a Next.js-based application that provides users with anime and manga recommendations based on their preferences. The application uses Supabase for authentication, database storage, and real-time features.

## Tech Stack

- **Frontend**: Next.js 14 with App Router, React, Tailwind CSS
- **Backend**: Next.js API routes, Supabase
- **Database**: PostgreSQL (via Supabase)
- **Authentication**: Supabase Auth with PKCE flow
- **External APIs**: AniList GraphQL API for anime data
- **Styling**: Tailwind CSS with custom components
- **Deployment**: Vercel (recommended)

## System Architecture

### Authentication Flow

The application uses Supabase Auth with the PKCE (Proof Key for Code Exchange) flow for secure authentication. This involves:

1. **Sign-Up Process**:
   - User enters email/password on the sign-up page
   - Credentials are sent to Supabase Auth
   - On successful authentication, user is redirected to the callback route
   - The callback route validates the session and creates user records in the application tables
   - User is then redirected to the dashboard

2. **Login Process**:
   - User enters credentials on the login page
   - Credentials are verified against Supabase Auth
   - On success, session is established and user is redirected to the dashboard

3. **Password Reset Flow**:
   - User requests a password reset from the reset-password page
   - Supabase sends a password reset link to the user's email
   - User clicks the link and is directed to the reset password form
   - The reset password page validates the token/code and allows the user to set a new password
   - On success, user is redirected to the login page

4. **Session Management**:
   - Sessions are managed through Supabase's client libraries
   - AuthProvider component maintains user and session state throughout the application
   - Navigation components adapt based on authentication state

### Database Structure

The application uses the following main tables:

1. **users**: Stores user profile information
   - Maps to auth.users but contains application-specific fields
   - Created via both auth triggers and fallback logic in the callback route

2. **user_preferences**: Stores user preferences for recommendations
   - Linked to users table via user_id
   - Contains preferred genres and watch history (legacy JSON field)

3. **anime_watch_history**: Normalized table for tracking user's anime watch history
   - Linked to users table via user_id
   - Contains details for each anime including anilist_id, title, rating, and cover image
   - Has unique constraint on user_id and anilist_id to prevent duplicates
   - Includes Row Level Security to ensure users can only access their own data

4. **anime**: Stores anime information for recommendations
   - Contains details like title, synopsis, genres, and rating

### External APIs Integration

1. **AniList GraphQL API**:
   - Provides comprehensive anime data including titles, cover images, and metadata
   - Implemented via Apollo Client for efficient GraphQL queries
   - Used for search functionality in the watch history feature
   - Supports rich anime search with images and details

### Security Implementation

1. **Row Level Security (RLS)**:
   - Implemented on all user-related tables
   - Ensures users can only access their own data
   - Service role has administrative access for maintenance

2. **Authentication Security**:
   - PKCE flow for enhanced security
   - Token-based authentication
   - Password reset with time-limited tokens
   - Environment variable protection for API keys

3. **External Resource Security**:
   - Image domain validation for cover images
   - Only allows trusted domains like anilist.co and kitsu.io
   - Configured in Next.js image configuration

## Component Architecture

### Core Components

1. **AuthProvider**: Manages authentication state throughout the application
   - Provides user and session information to all components
   - Handles synchronization between user and session states

2. **ClientNavigation**: Client-side navigation component with modern dropdown UI
   - Adapts based on authentication state
   - Prevents hydration errors using client-side mounting detection
   - Features user avatar with dropdown menu for authenticated users
   - Provides links to search, recommendations, and profile
   - Implements clean, accessible dropdown menu with icons and transitions

3. **AnimeSearch**: Reusable anime search component
   - Connects to AniList GraphQL API for real-time search
   - Features debounced input to prevent excessive API calls
   - Displays search results with cover images and metadata
   - Provides keyboard navigation and accessibility features
   - Handles image loading errors gracefully

4. **WatchHistoryForm**: Form component for adding anime to watch history
   - Integrates with AnimeSearch component for finding anime
   - Allows users to rate anime on a 1-10 scale
   - Validates data before submission
   - Handles error states and loading states

5. **WatchHistoryList**: Component for displaying user's anime watch history
   - Shows anime with cover images, titles, and ratings
   - Supports editing ratings and deleting entries
   - Updates in real-time using Supabase's real-time subscriptions
   - Implements optimistic UI updates for a responsive experience

### Services

1. **watchHistoryService**: Service for interacting with watch history data
   - Provides functions for adding, updating, retrieving, and deleting watch history entries
   - Handles authentication validation
   - Interfaces with Supabase for database operations
   - Includes error handling and logging

2. **anilistClient**: Service for interacting with AniList GraphQL API
   - Configured Apollo Client for GraphQL queries
   - Provides type-safe interfaces for anime data
   - Includes error handling and fallback mechanisms

### Pages

1. **Home Page** (`/src/app/page.tsx`): Modern landing page with feature highlights
   - Full-screen background image with gradient overlay
   - Clear value proposition and call-to-action
   - Feature cards with hover effects and backdrop blur
   - Responsive grid layout for feature presentation
   - Optimized image loading with Next.js Image component
   - Consistent typography and spacing
   - Modern UI elements with transitions and animations

2. **My Anime Page** (`/src/app/my-anime/page.tsx`): Comprehensive anime tracking interface
   - Protected route requiring authentication
   - Includes WatchHistoryForm for adding new anime
   - Displays WatchHistoryList for viewing and managing watch history
   - Responsive layout for both desktop and mobile
   - Real-time updates via Supabase subscriptions
   - Client-side navigation with authentication checks

3. **Search Page** (`/src/app/search/page.tsx`): Interface for searching anime
   - Uses the SearchBar component
   - Will display search results with details

4. **Recommendations Page** (`/src/app/recommendations/page.tsx`): Displays personalized recommendations
   - Shows anime recommendations based on user preferences
   - Includes feedback mechanisms (like/dislike)

5. **Profile Page** (`/src/app/profile/page.tsx`): User profile management
   - Displays user information
   - Allows preference management
   - Includes watch history form for tracking anime

6. **Authentication Pages**:
   - Signup (`/src/app/signup/page.tsx`): User registration
   - Login (`/src/app/login/page.tsx`): User authentication
   - Reset Password (`/src/app/reset-password/page.tsx`): Password recovery

## Data Flow

1. **Anime Search and Selection**:
   - User inputs search query in AnimeSearch component
   - Component makes debounced API calls to AniList GraphQL API
   - Results display with images and metadata
   - User selects anime which is passed to parent component

2. **Watch History Management**:
   - User adds anime via WatchHistoryForm
   - Data is sent to watchHistoryService
   - Service stores data in anime_watch_history table via Supabase
   - Real-time subscription in WatchHistoryList updates the UI
   - User can edit ratings or delete entries with immediate UI feedback

3. **Authentication Flow**:
   - Auth state managed by AuthProvider
   - Protected routes like My Anime check authentication
   - Unauthenticated users redirected to login

## Environment Configuration

The application requires the following environment variables:

```
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_SITE_URL=your-site-url (e.g., http://localhost:3000 for local development)
ANILIST_CLIENT_ID=your-anilist-client-id
ANILIST_CLIENT_SECRET=your-anilist-client-secret
``` 
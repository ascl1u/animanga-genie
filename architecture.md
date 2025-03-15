# AniManga Genie - Architecture Documentation

## Overview

AniManga Genie is a Next.js-based application that provides users with anime and manga recommendations based on their preferences. The application uses Supabase for authentication, database storage, and real-time features.

## Tech Stack

- **Frontend**: Next.js 14 with App Router, React, Tailwind CSS
- **Backend**: Next.js API routes, Supabase
- **Database**: PostgreSQL (via Supabase)
- **Authentication**: Supabase Auth with PKCE flow
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
   - Contains preferred genres and watch history

3. **anime**: Stores anime information for recommendations
   - Contains details like title, synopsis, genres, and rating

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

## Component Architecture

### Core Components

1. **AuthProvider**: Manages authentication state throughout the application
   - Provides user and session information to all components
   - Handles synchronization between user and session states

2. **ClientNavigation**: Client-side navigation component
   - Adapts based on authentication state
   - Prevents hydration errors using client-side mounting detection
   - Provides links to all main pages including search, recommendations, and feedback

3. **Layout Components**: Structured layout components for consistent UI
   - Includes headers, footers, and main content areas
   - Responsive design with Tailwind CSS

4. **SearchBar**: Reusable search component
   - Provides debounced input for efficient API calls
   - Used in search pages and other search interfaces

5. **WatchHistoryForm**: Form component for adding anime to watch history
   - Allows users to input anime titles, ratings, watch status, and dates
   - Integrates with Supabase for data storage

### Pages

1. **Home Page** (`/src/app/page.tsx`): Landing page with feature highlights
   - Provides an appealing introduction to the application
   - Links to key functionality like search and sign-up

2. **Search Page** (`/src/app/search/page.tsx`): Interface for searching anime
   - Uses the SearchBar component
   - Will display search results with details

3. **Recommendations Page** (`/src/app/recommendations/page.tsx`): Displays personalized recommendations
   - Shows anime recommendations based on user preferences
   - Includes feedback mechanisms (like/dislike)

4. **Profile Page** (`/src/app/profile/page.tsx`): User profile management
   - Displays user information
   - Allows preference management
   - Includes watch history form for tracking anime

5. **Authentication Pages**:
   - Signup (`/src/app/signup/page.tsx`): User registration
   - Login (`/src/app/login/page.tsx`): User authentication
   - Reset Password (`/src/app/reset-password/page.tsx`): Password recovery

### API Routes

1. **/api/auth**: Handles various authentication operations
   - Sign-up, login, password reset requests
   - Token validation and session management

2. **/api/auth/update-password**: Server-side password update endpoint
   - Handles authenticated password changes
   - Provides error handling and validation

3. **/api/recommendations**: Anime recommendation engine
   - Processes user preferences to generate recommendations
   - Interfaces with the database for anime information

4. **/auth/callback**: Authentication callback handler
   - Processes OAuth redirects from Supabase
   - Creates user records in application tables
   - Fallback mechanism for user creation

## Data Flow

1. User interacts with the UI components
2. Actions trigger API calls to Next.js API routes
3. API routes communicate with Supabase services
4. Data is processed and returned to the UI
5. UI updates to reflect the new state

## Environment Configuration

The application requires the following environment variables:

```
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_SITE_URL=your-site-url (e.g., http://localhost:3000 for local development)
``` 
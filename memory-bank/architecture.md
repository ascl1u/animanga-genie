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

## Project Structure

The project follows the standard Next.js 15.2.2 application structure with the app router:

```
animanga-genie/
├── src/
│   ├── app/                             # Main application code using the App Router
│   │   ├── api/                         # API routes
│   │   │   ├── test-supabase/           # Test endpoint for Supabase connection
│   │   │   └── auth/                    # Authentication API routes
│   │   │       ├── route.ts             # Main auth operations (signup, login, logout, reset)
│   │   │       └── update-password/     # Password update endpoint
│   │   ├── auth/                        # Auth-related pages and routes
│   │   │   ├── callback/                # Handle auth redirects from Supabase
│   │   │   └── reset-password/          # Password reset page
│   │   ├── login/                       # Login page
│   │   ├── signup/                      # Signup page
│   │   ├── page.tsx                     # Home page
│   │   └── layout.tsx                   # Root layout
│   ├── components/                      # Reusable UI components (to be added)
│   └── utils/                           # Utility functions
│       ├── supabaseClient.ts            # Supabase client initialization
│       ├── schema.sql                   # Database schema definition
│       └── README-SUPABASE.md           # Documentation for Supabase setup
├── public/                              # Static assets
├── memory-bank/                         # Project documentation
└── [configuration files]
```

## Supabase Integration

### src/utils/supabaseClient.ts
**Purpose:** Initializes and exports the Supabase client.
**Description:** Creates a single instance of the Supabase client using environment variables.
**Dependencies:** @supabase/supabase-js

### src/app/api/test-supabase/route.ts
**Purpose:** API route to test Supabase connection.
**Description:** Queries the testing table and returns a success or error message.
**Dependencies:** Next.js API routes, supabaseClient.ts

### src/utils/schema.sql
**Purpose:** Defines the database schema for Supabase.
**Description:** Creates tables for users, anime, user preferences, and testing. Includes sample data for testing.
**Dependencies:** Supabase PostgreSQL

### src/utils/README-SUPABASE.md
**Purpose:** Documentation for Supabase setup.
**Description:** Provides instructions for setting up the database schema and testing the connection.

## Authentication Implementation

### src/app/api/auth/route.ts
**Purpose:** API route for authentication operations.
**Description:** Handles user signup, login, logout, and password reset requests using Supabase Auth.
**Dependencies:** Next.js API routes, supabaseClient.ts

### src/app/auth/callback/route.ts
**Purpose:** Handles auth callback from Supabase.
**Description:** Processes redirects from email verification, password reset, etc., and exchanges auth code for session.
**Dependencies:** Next.js API routes, supabaseClient.ts

### src/app/api/auth/update-password/route.ts
**Purpose:** API route for updating user password.
**Description:** Handles password updates after a user has received a password reset email.
**Dependencies:** Next.js API routes, supabaseClient.ts

### src/app/auth/reset-password/page.tsx
**Purpose:** Password reset page.
**Description:** Provides a form for users to enter a new password after clicking a reset link.
**Dependencies:** React, next/navigation, supabaseClient.ts (via API)

### src/app/signup/page.tsx
**Purpose:** User registration page.
**Description:** Provides a form for new users to create an account with email and password.
**Dependencies:** React, next/link, supabaseClient.ts (via API)

### src/app/login/page.tsx
**Purpose:** User login page.
**Description:** Provides a form for users to sign in and options to reset password.
**Dependencies:** React, next/navigation, next/link, supabaseClient.ts (via API)

## Database Structure

The application uses the following database tables:

### Users Table
- `id` (UUID, primary key, auto-generated)
- `email` (text, unique)
- `created_at` (timestamp with time zone, default now)

### Anime Table
- `id` (serial integer, primary key, auto-increment)
- `title` (text)
- `synopsis` (text)
- `genres` (JSON array of genre names)
- `rating` (float)

### User Preferences Table
- `user_id` (UUID, primary key, foreign key to users.id)
- `preferred_genres` (JSON array of genre names)
- `watch_history` (JSON array containing anime_id, rating, watch_status, and watch_date)
- `created_at` (timestamp with time zone, default now)
- `updated_at` (timestamp with time zone, default now)

### Testing Table
- `id` (serial integer, primary key)
- `name` (text)
- `created_at` (timestamp with time zone, default now)

## Future Improvements

The next phase will involve:
- Implementing UI components for anime browsing and recommendations
- Adding user profile management
- Creating protected routes based on authentication state
- Building the recommendation engine

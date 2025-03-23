This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

# AniManga Genie

A Next.js application that provides personalized anime and manga recommendations based on user preferences.

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Environment Setup

Create a `.env.local` file in the root directory with the following variables:

```
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_SITE_URL=http://localhost:3000
```

For production, set `NEXT_PUBLIC_SITE_URL` to your production domain.

## Documentation

The following documentation files are available in this repository:

- **[architecture.md](./architecture.md)**: Detailed system architecture including authentication flow, database structure, and component architecture.
- **[progress.md](./progress.md)**: Implementation progress tracking, completed features, and upcoming work.
- **[src/utils/README-SUPABASE.md](./src/utils/README-SUPABASE.md)**: Instructions for setting up Supabase, including database schema, authentication, and troubleshooting.

## Features

- User authentication with Supabase Auth
- Password reset functionality
- Anime and manga database
- Personalized recommendations based on user preferences
- User profile management
- Watch history tracking

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

# Recommendation System Optimization

## Problem

The application's recommendation system was experiencing performance issues due to:

1. **Overcomplicated State Management**: Too many state variables causing unnecessary re-renders
2. **Excessive Persistence**: Storing recommendations in localStorage for all users
3. **Race Conditions**: Complex effect dependencies causing timing issues
4. **Inefficient Caching**: Overly complex watch history hash system
5. **Too Many Effect Dependencies**: Causing cascade updates and extra processing

## Solution

We've simplified the `useRecommendations` hook to make the application faster and more maintainable by:

1. Removing localStorage persistence for unauthenticated users
2. Simplifying state management to reduce re-renders
3. Eliminating race conditions through cleaner architecture
4. Generating recommendations on-demand rather than extensive caching
5. Reducing effect dependencies to prevent cascade updates

## Files Changed

- **src/hooks/useRecommendations.ts**: Main implementation of the hook
- **implementation-plan.md**: Detailed plan with steps taken
- **implementation-summary.md**: Summary of changes and benefits

## Key Improvements

### Code Size Reduction
- The hook implementation is now approximately 1/3 of its original size
- Reduced complexity through simpler control flow and fewer state variables

### Technical Improvements
- Eliminated race conditions by removing complex state tracking mechanisms
- Simplified the API while maintaining backward compatibility
- Implemented a cleaner request-based model for recommendation generation

### Performance Benefits
- Faster page loads by removing localStorage operations
- Reduced memory usage through simpler state management
- Better responsiveness during recommendation generation
- Fewer unnecessary re-renders

## How to Test

### Run the Development Server
```bash
npm run dev
```

### Test Scenarios

#### 1. Unauthenticated User Flow
1. Open the application in an incognito/private window
2. Add anime titles to your watch history
3. Generate recommendations on the Recommendations page
4. Verify they appear correctly
5. Refresh the page:
   - Watch history should persist (from localStorage)
   - Recommendations should need regeneration (not persisted)

#### 2. Authenticated User Flow
1. Log in to the application
2. Add anime titles to watch history
3. Generate recommendations
4. Refresh the page:
   - Watch history should persist
   - Recommendations should load automatically from the database

#### 3. Authentication Transitions
1. Start unauthenticated with watch history and recommendations
2. Log in and verify state handling
3. Log out and verify proper state reset

#### 4. Performance Testing
1. Monitor load times before and after implementation
2. Check memory usage in browser dev tools
3. Test with throttled network to verify improved responsiveness

## Implementation Details

The core optimization approach:

1. **State Consolidation**: Combined related states into a single status object
2. **Targeted Persistence**: Saved recommendations only for authenticated users
3. **Simplified Watch History Tracking**: Removed the hash-based change detection for unauthenticated users
4. **On-Demand Generation**: Recommendations generated when needed instead of excessively cached

## Future Improvements

After verifying the benefits of this approach, we should consider:

1. Applying similar simplifications to other complex hooks
2. Optimizing the recommendation algorithm itself
3. Improving watch history management
4. Reviewing other uses of localStorage to ensure they're necessary

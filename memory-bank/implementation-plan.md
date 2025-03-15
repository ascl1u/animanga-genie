# Implementation Plan for AI-Powered Anime Recommendation Engine

This plan outlines the development of the Minimum Viable Product (MVP) in small, testable steps, organized into phases. Each step includes a specific task and a test to confirm its success, aligning with the tech stack (Next.js, Supabase, Python, PyTorch, Vercel) and project goals.

---

## Phase 1: Project Setup and Planning (Weeks 1-2)

### 1. Initialize Next.js Project
- **Task**: Set up a new Next.js project with TypeScript support.
  - Run `npx create-next-app@latest animanga-genie` in the terminal.
  - Install ESLint and Prettier: `npm install --save-dev eslint prettier eslint-config-prettier eslint-plugin-prettier`.
  - Configure ESLint and Prettier with Airbnb TypeScript style guide settings in `.eslintrc.json` and `.prettierrc`.
- **Test**: Run `npm run dev` and visit `http://localhost:3000` to confirm the default Next.js welcome page loads without errors.

### 2. Set Up Supabase Integration
- **Task**: Create a Supabase project and connect it to Next.js.
  - Sign up at [Supabase](https://supabase.com), create a new project, and note the URL and anon key.
  - Install the Supabase client: `npm install @supabase/supabase-js`.
  - Add `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` to a `.env.local` file.
- **Test**: Create a `/pages/api/test-supabase.ts` API route that initializes the Supabase client and returns a success message. Visit `http://localhost:3000/api/test-supabase` to verify a 200 status and "Connected" response.

### 3. Design Database Schema
- **Task**: Define database tables in Supabase for users, anime, and preferences.
  - In the Supabase dashboard, create:
    - `users` table: `id` (uuid, auto-generated), `email` (text), `created_at` (timestamp, default now).
    - `anime` table: `id` (int, auto-increment), `title` (text), `synopsis` (text), `genres` (json), `rating` (float).
    - `user_preferences` table: `user_id` (uuid, foreign key to `users.id`), `preferred_genres` (json, list of genre names), `watch_history` (json, containing anime_id, rating, watch_status, and watch_date).
- **Test**: Insert one sample row into each table via the Supabase dashboard. Query each table using the Supabase client in a test API route and confirm the data is returned.

### 4. Implement Authentication
- **Task**: Add email/password authentication using Supabase.
  - Create a `utils/supabaseClient.ts` file to initialize the Supabase client with TypeScript types.
  - Use Supabase auth methods (`signUp`, `signInWithPassword`, `signOut`) in a new `/pages/api/auth.ts` route.
  - Implement password reset functionality with Supabase's password reset flow.
  - Plan for future social login integration (to be implemented after MVP).
- **Test**: Manually sign up a test user with an email and password via a temporary form at `/signup`. Log in and out, verifying the session state changes in the browser console. Test password reset flow.

---

## Phase 2: Frontend Development (Weeks 3-6)

### 1. Build Header Component
- **Task**: Create a navigation bar with authentication-related components.
  - Add `shadcn/ui` to the project: follow its setup guide to install and configure.
  - Create `/components/Header.tsx` with links to Home, Search, Recommendations (if logged in), Profile (if logged in), and Login/Signup (if not logged in) using functional components and hooks.
  - Include a user avatar with a dropdown menu (Profile, Logout) when logged in, styled with `shadcn/ui` components.
  - Use Supabase auth to dynamically update based on login status.
- **Test**: Load the app as a logged-out user to see Login/Signup links, then log in and confirm Recommendations, Profile, and avatar dropdown appear.

### 2. Build Home Page
- **Task**: Design an engaging landing page to grab attention.
  - Create `/pages/index.tsx` with a colorful anime wallpaper background and a central "Explore Anime" button using `shadcn/ui` button components.
  - Redirect to `/pages/search.tsx` on button click; optionally, display personalized content (e.g., recent recommendations) for logged-in users using Supabase auth data.
- **Test**: Load the home page as a logged-out user, click "Explore Anime," and verify redirection to the search page; log in and check for optional personalized content.

### 3. Build User Registration Page
- **Task**: Create a registration page with email and password inputs.
  - Ensure `shadcn/ui` is configured in the project.
  - Create `/pages/signup.tsx` using functional components and hooks, with `shadcn/ui` form components for email and password fields.
  - Call the Supabase auth `signUp` method on form submission and redirect to `/pages/profil.tsx` for profile setup.
- **Test**: Register a new user via the form, confirm redirection to the profile page, and check the Supabase dashboard to verify the user appears in the `users` table.

### 4. Build User Login Page
- **Task**: Create a login page for user authentication.
  - Create `/pages/login.tsx` using functional components and hooks, with `shadcn/ui` form components for email and password fields.
  - Call the Supabase auth `signIn` method on form submission and redirect to `/pages/recommendations.tsx` or `/pages/index.tsx`.
- **Test**: Log in with a test user, confirm redirection to the intended page, and verify the header updates with logged-in components.

### 5. Build User Profile Page
- **Task**: Create a profile page for managing preferred genres and watch history.
  - Create `/pages/profil.tsx` with a multi-select dropdown of genres (e.g., action, romance) fetched from the `anime` table using Supabase, styled with `shadcn/ui`.
  - Include `/components/WatchHistoryForm.tsx` for adding watch history entries (see task 7).
  - Save genre selections to a `user_preferences` table in Supabase with a `preferred_genres` column (array of strings), linked by `user_id`.
- **Test**: Log in as a test user, select three genres, submit, and verify the `preferred_genres` column in `user_preferences` reflects the choices; add a watch history entry and confirm it saves correctly.

### 6. Implement Watch History Input
- **Task**: Add a form to input watched anime with ratings and details.
  - Create `/components/WatchHistoryForm.tsx` with a search input to select anime titles (querying the `anime` table via Supabase), a rating field (1-5), watch status dropdown (e.g., watched, watching, to watch), and optional watch date, styled with `shadcn/ui`.
  - Save entries to a separate `watch_history` table in Supabase with fields: `user_id`, `anime_id`, `rating`, `watch_status`, and `watch_date`.
- **Test**: Add two anime entries for a test user, confirm they are stored correctly in the `watch_history` table with all fields populated.

### 7. Build Recommendation Display Page
- **Task**: Create a page to display recommended anime.
  - Create `/pages/recommendations.tsx` with `shadcn/ui` card components to display anime title, synopsis, genres, and rating.
  - Fetch mock data from a static JSON file initially; plan to replace with Supabase queries based on `user_preferences` and `watch_history`.
- **Test**: Load the page as a logged-in user, verify at least three mock anime cards render with all details visible, and check responsiveness.

### 8. Build Search Page with Functionality
- **Task**: Implement a search page to find anime by title.
  - Create `/pages/search.tsx` and include `/components/SearchBar.tsx` with an input field styled with `shadcn/ui`.
  - Query the `anime` table via Supabase on input change (with debouncing), displaying results below in `shadcn/ui` card components.
- **Test**: Search for "Naruto" (assuming it’s in the database), confirm matching titles appear in the results, and verify the search updates dynamically with input.

### 9. Additional Setup and Styling
- **Task**: Ensure consistency and scalability across the application.
  - Define a custom `shadcn/ui` theme with an anime-inspired aesthetic (e.g., vibrant colors, sleek typography) in the project configuration.
  - Use separate Supabase tables (`user_preferences`, `watch_history`) linked by `user_id` for better data management; ensure responsiveness with CSS or Tailwind CSS.
- **Test**: Load all pages on desktop and mobile, confirm consistent styling and functionality, and verify Supabase tables are correctly structured and populated.

## Phase 3: AI/ML Development (Weeks 7-10)

### 10. Fetch Anime Metadata
- **Task**: Populate the `anime` table with data from the MyAnimeList API.
  - Sign up for API access at [MyAnimeList](https://myanimelist.net/apiconfig) and get credentials.
  - Create `/scripts/fetch-anime.py` in Python to fetch and insert 100+ anime entries into Supabase.
  - Implement backup data sources using Kitsu/AniList APIs in case MyAnimeList API access is limited or unavailable.
- **Test**: Run the script and check the Supabase dashboard to ensure the `anime` table has at least 100 rows with valid data.

### 11. Preprocess Data for NLP
- **Task**: Clean and vectorize anime synopses for analysis.
  - In `/scripts/preprocess.py`, use Python to remove stop words and tokenize synopses from the `anime` table.
  - Generate TF-IDF vectors using scikit-learn and save them to a local file.
- **Test**: Process one synopsis and verify the output is a numerical vector of expected length.

### 12. Build Collaborative Filtering Model
- **Task**: Create a basic collaborative filtering model.
  - In `/scripts/collaborative_filtering.py`, use the Surprise library with user ratings from `watch_history`.
  - Train the model on sample data (e.g., 10 users with 5 ratings each).
  - For the MVP, train the model locally; plan for cloud-based training when scaling.
- **Test**: Predict recommendations for a test user and confirm the output includes plausible anime titles.

### 13. Integrate Content-Based Filtering
- **Task**: Add content-based filtering using anime metadata.
  - In `/scripts/content_based.py`, compute cosine similarity between anime based on genres and TF-IDF vectors.
- **Test**: Input one anime title and verify the top 5 similar anime share genres or themes.

### 14. Combine Models for Hybrid Recommendations
- **Task**: Merge collaborative and content-based recommendations.
  - In `/scripts/hybrid.py`, combine outputs using a weighted average (e.g., 60% collaborative, 40% content-based).
  - Design the model with efficient algorithms to handle scaling concerns.
  - Implement a strategy to precompute recommendations for active users periodically.
- **Test**: Generate hybrid recommendations for a test user and ensure they reflect both watch history and metadata similarities.

---

## Phase 4: Backend Integration (Weeks 11-14)

### 15. Create Recommendation API Service
- **Task**: Set up a separate API service to serve recommendations.
  - Deploy ML components as serverless functions in the cloud (AWS Lambda or similar).
  - Add `/pages/api/recommendations.ts` in Next.js to call the ML serverless functions.
  - Return a JSON list of anime IDs and titles for a given user ID.
- **Test**: Send a GET request with a test user ID and verify the response contains at least 5 valid anime entries.

### 16. Implement User Feedback Mechanism
- **Task**: Add feedback options for recommendations.
  - Create a `feedback` table in Supabase: `user_id`, `anime_id`, `liked` (boolean).
  - Add like/dislike buttons to `/pages/recommendations.tsx` and save responses via Supabase.
- **Test**: Like one recommendation and confirm the entry appears in the `feedback` table.

---

## Phase 5: Testing and Refinement (Weeks 15-18)

### 17. Conduct Unit Tests
- **Task**: Write unit tests for key functionality.
  - Use Jest to test authentication, API routes, and a mock ML model output in `/tests/`.
- **Test**: Run `npm test` and ensure all tests pass without errors.

### 18. Perform Usability Testing
- **Task**: Test the app with 3-5 users.
  - Provide access to the app and ask for feedback on UI and recommendations.
- **Test**: Collect and document feedback, identifying at least two areas for improvement.

### 19. Measure Recommendation Quality
- **Task**: Evaluate recommendation accuracy using standard metrics.
  - Implement Precision@K and Recall@K metrics to quantitatively measure recommendation performance.
  - Create a test dataset with known user preferences to evaluate against.
- **Test**: Calculate Precision@10 and Recall@10 for test users and ensure metrics exceed a minimum threshold (e.g., 0.7 precision).

### 20. Refine Based on Feedback
- **Task**: Address user-reported issues.
  - Update UI or recommendation logic as needed (e.g., tweak weights in the hybrid model).
- **Test**: Re-test with one user to confirm changes resolve the identified issues.

---

## Phase 6: Deployment and Launch (Weeks 19-22)

### 21. Deploy on Vercel and Cloud Services
- **Task**: Deploy the application and ML components.
  - Push the Next.js project to a GitHub repository and deploy to Vercel.
  - Deploy ML components to cloud services (AWS, Google Cloud, or similar).
  - Set environment variables in Vercel for Supabase credentials and ML API endpoints.
- **Test**: Visit the live URL and confirm all pages (signup, profile, recommendations) load correctly and communicate with ML services.

### 22. Monitor and Fix Issues
- **Task**: Track performance and resolve bugs post-launch.
  - Use Vercel's analytics to monitor traffic and errors.
  - Set up monitoring for ML API services.
- **Test**: Fix any reported issue (e.g., broken API route) and verify it's resolved on the live site within 24 hours.

### 23. Gather Initial User Feedback
- **Task**: Collect feedback from early users.
  - Add a simple feedback form at `/pages/feedback.tsx`.
- **Test**: Review at least 5 feedback submissions and outline one potential improvement.

---

## Reminders
- After completing each feature (e.g., "User Registration"), update `memory-bank/progress.md` with:
  - Date: [e.g., 2023-10-05]
  - Feature: [e.g., User Registration]
  - Description: [e.g., Added email/password signup with Supabase]
  - Status: Completed
- For new files (e.g., `/pages/signup.tsx`), update `memory-bank/architecture.md` with:
  - File: [e.g., `/pages/signup.tsx`]
  - Purpose: [e.g., Handles user registration UI and Supabase auth]
  - Dependencies: [e.g., `utils/supabaseClient.ts`]
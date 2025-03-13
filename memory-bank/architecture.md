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

## Project Structure

The project follows the standard Next.js 15.2.2 application structure with the app router:

```
animanga-genie/
├── src/
│   ├── app/          # Main application code using the App Router
│   ├── components/   # Reusable UI components (to be added)
│   └── utils/        # Utility functions, including Supabase client (to be added)
├── public/           # Static assets
├── memory-bank/      # Project documentation
└── [configuration files]
```

## Next Steps

The next phase will involve setting up the Supabase integration, which will add:
- utils/supabaseClient.ts
- pages/api/test-supabase.ts

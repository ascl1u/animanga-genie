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

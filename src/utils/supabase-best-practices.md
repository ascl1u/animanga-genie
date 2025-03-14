# Supabase Auth and User Data Best Practices

## User Authentication Flow

When building with Supabase Auth, understanding the relationship between Auth users and database tables is crucial. Here's the recommended approach:

### Auth Users vs. Database Users

Supabase creates two separate systems:

1. **Auth.users** - Managed by Supabase Auth service
   - Contains authentication-specific data (hashed passwords, email verification status, etc.)
   - Created when users sign up via `supabase.auth.signUp()`
   - Has its own schema (`auth`) separate from your public tables

2. **Public.users** - Your application's user table
   - Contains application-specific user data
   - Should reference the Auth user via the user's UUID
   - Must be manually created or created via triggers

### Best Practices

1. **Use Database Triggers for Consistency**
   - Set up a trigger on `auth.users` to automatically create records in your public tables
   - Ensures data consistency and removes the need for manual synchronization
   - Reduces the risk of failed user creation steps

2. **Leverage Row Level Security (RLS)**
   - Use RLS policies to restrict access to user data
   - Base policies on the authenticated user's ID
   - Example policy: `auth.uid() = user_id`

3. **Consider Alternative Designs**
   - For simpler applications, you might not need a separate `users` table at all
   - You can use the `auth.users` table directly for some use cases
   - Supabase provides functions like `auth.uid()` to access the current user's ID

4. **Handle Errors Gracefully**
   - Even with triggers, errors might occur
   - Implement error handling that provides clear feedback to users
   - Consider implementing cleanup code for partial failures

## Database Schema Design for User Data

When designing your database schema for user-related data:

1. **Foreign Key Relationships**
   - Use foreign keys to enforce referential integrity
   - Reference `id` from `auth.users` or your public `users` table
   - Consider using `ON DELETE CASCADE` for user-related data that should be deleted when a user is deleted

2. **Profile vs. Preferences Tables**
   - Consider separating user profile data (which changes infrequently) from preferences (which may change often)
   - This can help with performance and concurrency

3. **JSON vs. Relational Data**
   - Use JSON fields (like `preferred_genres` in our example) for lists and preferences that are always accessed together
   - Use relational tables for data that needs to be queried independently or has many-to-many relationships

## References

1. [Supabase Auth Documentation](https://supabase.com/docs/guides/auth)
2. [Row Level Security Guide](https://supabase.com/docs/guides/auth/row-level-security)
3. [Database Triggers in Supabase](https://supabase.com/docs/guides/database/functions)
4. [Managing User Data in Supabase](https://supabase.com/docs/guides/auth/managing-user-data) 
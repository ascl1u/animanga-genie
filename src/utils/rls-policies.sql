-- Enable Row Level Security on users table
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Enable Row Level Security on user_preferences table  
ALTER TABLE public.user_preferences ENABLE ROW LEVEL SECURITY;

-- Allow users to read their own record from users table
CREATE POLICY users_read_own ON public.users
  FOR SELECT
  USING (auth.uid() = id);

-- Allow users to update their own record in users table
CREATE POLICY users_update_own ON public.users
  FOR UPDATE
  USING (auth.uid() = id);

-- Allow users to read their own preferences
CREATE POLICY user_preferences_read_own ON public.user_preferences
  FOR SELECT  
  USING (auth.uid() = user_id);

-- Allow users to update their own preferences  
CREATE POLICY user_preferences_update_own ON public.user_preferences
  FOR UPDATE
  USING (auth.uid() = user_id);

-- Allow service role to manage all records
-- The following policies are useful for administrative operations
CREATE POLICY users_service_all ON public.users
  FOR ALL
  USING (auth.jwt() ->> 'role' = 'service_role');

CREATE POLICY user_preferences_service_all ON public.user_preferences
  FOR ALL
  USING (auth.jwt() ->> 'role' = 'service_role');

-- Grant necessary privileges to authenticated users
GRANT SELECT, UPDATE ON public.users TO authenticated;
GRANT SELECT, UPDATE ON public.user_preferences TO authenticated; 
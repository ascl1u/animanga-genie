-- Create a function to automatically create a user record when a new auth user is created
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
DECLARE
  user_exists INTEGER;
BEGIN
  -- Check if the user already exists in the public users table
  SELECT COUNT(*) INTO user_exists FROM public.users WHERE id = NEW.id;

  -- Only insert if the user doesn't exist yet
  IF user_exists = 0 THEN
    -- Insert into public users table
    INSERT INTO public.users (id, email, created_at)
    VALUES (NEW.id, NEW.email, COALESCE(NEW.created_at, NOW()));
    
    -- Insert default user preferences
    INSERT INTO public.user_preferences (user_id, preferred_genres, watch_history)
    VALUES (NEW.id, '[]'::jsonb, '[]'::jsonb);
    
    RAISE LOG 'Created new user and preferences for %', NEW.email;
  ELSE
    RAISE LOG 'User % already exists, skipping creation', NEW.email;
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create a trigger that calls the function on auth.users insert
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Create a trigger that also runs when users are confirmed (for email verification)
DROP TRIGGER IF EXISTS on_auth_user_updated ON auth.users;
CREATE TRIGGER on_auth_user_updated
  AFTER UPDATE ON auth.users
  FOR EACH ROW
  WHEN (OLD.email_confirmed_at IS NULL AND NEW.email_confirmed_at IS NOT NULL)
  EXECUTE FUNCTION public.handle_new_user();

-- Grant necessary permissions to the postgres authenticated role
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT ON public.users TO authenticated;
GRANT SELECT, INSERT ON public.user_preferences TO authenticated; 
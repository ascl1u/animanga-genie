import { createClient } from './supabase/server'
import { redirect } from 'next/navigation'

/**
 * Checks if a user is authenticated and redirects to login if not
 * Use this in Server Components that require authentication
 */
export async function requireAuth() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  
  if (!user) {
    redirect('/login')
  }
  
  return user
}

/**
 * Gets the current user without enforcing redirection
 * Use this when you want to have different UIs for authenticated/unauthenticated users
 */
export async function getCurrentUser() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  return user
} 
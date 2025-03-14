'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/components/SimpleAuthProvider';

export default function ProfilePage() {
  const { user, isLoading, isAuthenticated } = useAuth();
  const [mounted, setMounted] = useState(false);
  const router = useRouter();
  
  // Client-side only code
  useEffect(() => {
    setMounted(true);
  }, []);
  
  // Redirect to login if not authenticated after loading
  useEffect(() => {
    if (mounted && !isLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [mounted, isLoading, isAuthenticated, router]);
  
  // Show loading state
  if (isLoading || !mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Loading...</h1>
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500 mx-auto"></div>
        </div>
      </div>
    );
  }
  
  // Show profile if authenticated
  if (isAuthenticated && user) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <div className="px-4 py-5 sm:px-6 bg-indigo-600 text-white">
              <h1 className="text-2xl font-bold">Your Profile</h1>
              <p className="mt-1 text-sm">Manage your account details and preferences</p>
            </div>
            
            <div className="px-4 py-5 sm:p-6">
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-gray-900">Account Information</h3>
                  <div className="mt-2 space-y-4">
                    <div className="flex flex-col border-b pb-4">
                      <span className="text-sm text-gray-500">Email</span>
                      <span className="text-gray-900">{user.email}</span>
                    </div>
                    <div className="flex flex-col border-b pb-4">
                      <span className="text-sm text-gray-500">User ID</span>
                      <span className="text-gray-900 font-mono text-sm">{user.id}</span>
                    </div>
                    <div className="flex flex-col border-b pb-4">
                      <span className="text-sm text-gray-500">Account Created</span>
                      <span className="text-gray-900">
                        {user.created_at ? new Date(user.created_at).toLocaleDateString() : 'Unknown'}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium text-gray-900">Preferences</h3>
                  <p className="text-sm text-gray-500 mt-1">
                    You can update your preferences to customize your experience.
                  </p>
                  <div className="mt-4">
                    <button 
                      className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                      Edit Preferences
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  // Fallback - should not reach here due to redirect
  return null;
} 
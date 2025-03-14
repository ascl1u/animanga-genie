'use client';

import { useSearchParams } from 'next/navigation';
import Link from 'next/link';

export default function ErrorPage() {
  const searchParams = useSearchParams();
  const message = searchParams.get('message') || 'Sorry, something went wrong';

  return (
    <div className="flex min-h-screen flex-col items-center justify-center py-2">
      <div className="w-full max-w-md space-y-8 rounded-lg border p-6 shadow-md">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Error
          </h2>
        </div>

        <div className="rounded-md bg-red-50 p-4">
          <div className="text-sm font-medium text-red-800">{message}</div>
        </div>

        <div className="mt-4 text-center">
          <Link href="/login" className="text-indigo-600 hover:text-indigo-500">
            Return to login
          </Link>
        </div>
      </div>
    </div>
  );
} 
'use client';

export default function Footer() {
  return (
    <footer className="bg-white shadow-sm mt-auto py-4">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center">
          <div className="text-sm text-gray-500">
            © {new Date().getFullYear()} AniManga Genie
          </div>
          <div className="text-sm text-gray-500">
            Created by{' '}
            <a 
              href="https://x.com/dingusmage"
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-600 hover:text-indigo-500"
            >
              @dingusmage
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
} 
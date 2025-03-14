import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center">
          <h1 className="text-4xl font-extrabold text-gray-900 sm:text-5xl sm:tracking-tight lg:text-6xl">
            Welcome to AnimeManga Genie
          </h1>
          <p className="mt-6 max-w-2xl mx-auto text-xl text-gray-500">
            Your ultimate companion for discovering and tracking anime and manga series.
          </p>
          <div className="mt-10 flex justify-center gap-4">
            <Link
              href="/explore"
              className="px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
            >
              Explore Content
            </Link>
            <Link
              href="/signup"
              className="px-8 py-3 border border-transparent text-base font-medium rounded-md text-indigo-600 bg-white hover:bg-gray-50 border-indigo-600"
            >
              Create Account
            </Link>
          </div>
        </div>
        
        <div className="mt-20">
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">Features</h2>
          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-indigo-600 text-4xl mb-4">🔍</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Discover New Series</h3>
              <p className="text-gray-600">
                Find new anime and manga series based on your preferences and interests.
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-indigo-600 text-4xl mb-4">📝</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Track Your Progress</h3>
              <p className="text-gray-600">
                Keep track of what you&apos;re watching and reading with a simple, intuitive interface.
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-indigo-600 text-4xl mb-4">👥</div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">Connect with Fans</h3>
              <p className="text-gray-600">
                Share your thoughts and recommendations with other anime and manga enthusiasts.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

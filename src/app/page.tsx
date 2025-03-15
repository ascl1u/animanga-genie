import Link from 'next/link';
import Image from 'next/image';

export default function HomePage() {
  return (
    <div className="min-h-screen relative">
      {/* Background image with overlay */}
      <div className="absolute inset-0 z-0">
        <Image 
          src="/images/background.png" 
          alt="Anime collage background"
          fill
          className="object-cover"
          priority
          quality={90}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/90 via-indigo-800/80 to-indigo-900/85"></div>
      </div>
      
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center">
          <h1 className="text-4xl font-extrabold text-white sm:text-5xl sm:tracking-tight lg:text-6xl drop-shadow-lg">
            Welcome to AniManga Genie
          </h1>
          <p className="mt-6 max-w-2xl mx-auto text-xl text-indigo-100">
            Get personalized recommendations for anime and manga with AI
          </p>
          <div className="mt-10 flex justify-center">
            <Link
              href="/search"
              className="px-12 py-5 text-lg font-medium rounded-lg text-white bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 transform hover:scale-105 transition-all shadow-lg"
            >
              Start Your Journey Now
            </Link>
          </div>
        </div>
        
        <div className="mt-24">
          <h2 className="text-3xl font-bold text-white text-center mb-12 drop-shadow-md">Features</h2>
          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            <div className="bg-white/10 backdrop-blur-sm p-8 rounded-xl shadow-xl border border-indigo-200/20 transform hover:scale-105 transition-all">
              <div className="text-5xl mb-5 text-yellow-400">🔍</div>
              <h3 className="text-xl font-bold text-white mb-3">Discover</h3>
              <p className="text-indigo-100">
                Find stories tailored to your unique taste
              </p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm p-8 rounded-xl shadow-xl border border-indigo-200/20 transform hover:scale-105 transition-all">
              <div className="text-5xl mb-5 text-yellow-400">📝</div>
              <h3 className="text-xl font-bold text-white mb-3">Track</h3>
              <p className="text-indigo-100">
                Keep up with your favorites
              </p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm p-8 rounded-xl shadow-xl border border-indigo-200/20 transform hover:scale-105 transition-all">
              <div className="text-5xl mb-5 text-yellow-400">👥</div>
              <h3 className="text-xl font-bold text-white mb-3">Connect</h3>
              <p className="text-indigo-100">
                Share recommendations with others
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

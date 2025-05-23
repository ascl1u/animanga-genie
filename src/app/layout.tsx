import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import ClientNavigation from '@/components/ClientNavigation';
import Footer from '@/components/Footer';
import { SimpleAuthProvider } from '@/components/SimpleAuthProvider';
import { Toaster } from 'react-hot-toast';
import { ModelProvider } from '@/context/ModelContext';
import { RecommendationsProvider } from '@/context/RecommendationsContext';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'AniManga Genie',
  description: 'Your personalized anime and manga recommendation app',
  icons: {
    icon: '/favicon.ico',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased flex flex-col min-h-screen`}>
        <SimpleAuthProvider>
          <ModelProvider>
            <RecommendationsProvider>
              <ClientNavigation />
              <Toaster position="top-right" />
              <main className="flex-grow">
                {children}
              </main>
              <Footer />
            </RecommendationsProvider>
          </ModelProvider>
        </SimpleAuthProvider>
      </body>
    </html>
  );
}

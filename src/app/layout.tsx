import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import ClientNavigation from '@/components/ClientNavigation';
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
  title: 'Animanga Genie',
  description: 'Your personalized anime and manga recommendation app',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <ModelProvider>
          <RecommendationsProvider>
            <SimpleAuthProvider>
              <ClientNavigation />
              <Toaster position="top-right" />
              <main>
                {children}
              </main>
            </SimpleAuthProvider>
          </RecommendationsProvider>
        </ModelProvider>
      </body>
    </html>
  );
}

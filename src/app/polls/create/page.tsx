'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { createPoll } from '@/services/pollService';

export default function CreatePollPage() {
  const router = useRouter();
  const [question, setQuestion] = useState('');
  const [options, setOptions] = useState<string[]>(['', '']);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const addOption = () => {
    setOptions([...options, '']);
  };
  
  const removeOption = (index: number) => {
    if (options.length <= 2) {
      return; // Maintain at least 2 options
    }
    
    const newOptions = [...options];
    newOptions.splice(index, 1);
    setOptions(newOptions);
  };
  
  const updateOption = (index: number, value: string) => {
    const newOptions = [...options];
    newOptions[index] = value;
    setOptions(newOptions);
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate inputs
    if (!question.trim()) {
      setError('Poll question is required');
      return;
    }
    
    // Filter out empty options and check for at least 2 valid options
    const validOptions = options.filter((option) => option.trim() !== '');
    if (validOptions.length < 2) {
      setError('At least 2 valid options are required');
      return;
    }
    
    setIsSubmitting(true);
    setError('');
    
    try {
      await createPoll(question, validOptions);
      router.push('/polls');
    } catch (err) {
      console.error('Error creating poll:', err);
      setError('Failed to create poll. You may not have admin permissions.');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <div className="container mx-auto px-4 py-8">
      <Link 
        href="/polls"
        className="inline-block mb-6 text-blue-600 hover:text-blue-800"
      >
        ← Back to polls
      </Link>
      
      <h1 className="text-2xl font-bold mb-6">Create New Poll</h1>
      
      <div className="bg-white rounded-lg border border-gray-200 shadow-md p-6">
        <form onSubmit={handleSubmit}>
          <div className="mb-6">
            <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-1">
              Poll Question
            </label>
            <input
              type="text"
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              placeholder="e.g., What is your favorite anime genre?"
            />
          </div>
          
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Poll Options
            </label>
            
            {options.map((option, index) => (
              <div key={index} className="flex mb-3">
                <input
                  type="text"
                  value={option}
                  onChange={(e) => updateOption(index, e.target.value)}
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  placeholder={`Option ${index + 1}`}
                />
                
                {options.length > 2 && (
                  <button
                    type="button"
                    onClick={() => removeOption(index)}
                    className="ml-2 text-red-600 hover:text-red-800"
                  >
                    Remove
                  </button>
                )}
              </div>
            ))}
            
            <button
              type="button"
              onClick={addOption}
              className="mt-2 text-sm text-blue-600 hover:text-blue-800"
            >
              + Add Another Option
            </button>
          </div>
          
          {error && (
            <p className="mb-4 text-sm text-red-600">{error}</p>
          )}
          
          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
          >
            {isSubmitting ? 'Creating Poll...' : 'Create Poll'}
          </button>
        </form>
      </div>
    </div>
  );
} 
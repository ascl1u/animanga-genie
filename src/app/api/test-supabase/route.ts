import { NextResponse } from 'next/server';
import { supabase } from '@/utils/supabaseClient';

export async function GET() {
  try {
    // Test the Supabase connection
    const { data, error } = await supabase.from('testing').select('*').limit(1);
    
    if (error) {
      console.error('Supabase connection error:', error);
      return NextResponse.json(
        { 
          success: false, 
          message: 'Failed to connect to Supabase', 
          error: error.message 
        },
        { status: 500 }
      );
    }
    
    return NextResponse.json(
      { 
        success: true, 
        message: 'Successfully connected to Supabase',
        data
      },
      { status: 200 }
    );
  } catch (error) {
    console.error('Unexpected error:', error);
    return NextResponse.json(
      { 
        success: false, 
        message: 'An unexpected error occurred', 
        error: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
} 
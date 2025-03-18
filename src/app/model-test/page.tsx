'use client';

import { useEffect, useState, useCallback } from 'react';
import { onnxModelService } from '@/services/onnxModelService';

// Import the interface from onnxModelService
interface AnimeRecommenderMetadata {
  n_users: number;
  n_anime: number;
  n_genres: number;
  n_tags: number;
  user_embedding_dim: number;
  anime_embedding_dim: number;
  genre_embedding_dim: number;
  tag_embedding_dim: number;
  dense_layers: number[];
  max_genres: number;
  max_tags: number;
  model_type: string;
  rating_normalization: {
    mean: number;
    std: number;
  };
  input_names: string[];
  output_names: string[];
}

interface BrowserCompatibility {
  wasmSupported: boolean;
  simdSupported: boolean;
  threadingSupported: boolean;
  sharedArrayBufferSupported: boolean;
  issues: string[];
}

export default function ModelTestPage() {
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [metadata, setMetadata] = useState<AnimeRecommenderMetadata | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [browserInfo, setBrowserInfo] = useState<string>('');
  const [loadAttempts, setLoadAttempts] = useState(0);
  const [initErrors, setInitErrors] = useState<string[]>([]);
  const [compatibility, setCompatibility] = useState<BrowserCompatibility | null>(null);

  // Check browser compatibility for WASM features
  const checkBrowserCompatibility = useCallback(async () => {
    const issues: string[] = [];
    const result: BrowserCompatibility = {
      wasmSupported: false,
      simdSupported: false,
      threadingSupported: false,
      sharedArrayBufferSupported: false,
      issues: []
    };

    try {
      // Check for WebAssembly support
      if (typeof WebAssembly !== 'object') {
        issues.push('WebAssembly is not supported in this browser');
      } else {
        result.wasmSupported = true;
        
        // Check for SharedArrayBuffer (needed for threading)
        if (typeof SharedArrayBuffer !== 'function') {
          issues.push('SharedArrayBuffer is not supported (required for multithreading)');
        } else {
          result.sharedArrayBufferSupported = true;
        }
        
        // Try to detect SIMD support
        try {
          const simdTest = WebAssembly.validate(new Uint8Array([
            0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 
            2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
          ]));
          result.simdSupported = simdTest;
          if (!simdTest) {
            issues.push('WebAssembly SIMD is not supported');
          }
        } catch (error) {
          console.error('Error checking WebAssembly SIMD support:', error);
          issues.push('Failed to test for WebAssembly SIMD support');
        }
        
        // Check for threading support
        result.threadingSupported = result.sharedArrayBufferSupported;
        
        // Check security headers for cross-origin isolation
        // These are required for SharedArrayBuffer in modern browsers
        const isCrossOriginIsolated = (window.crossOriginIsolated === true);
        if (!isCrossOriginIsolated && result.sharedArrayBufferSupported) {
          issues.push('Cross-Origin-Isolation is not enabled (required for SharedArrayBuffer)');
        }
      }
    } catch (error) {
      issues.push(`Error checking compatibility: ${error instanceof Error ? error.message : String(error)}`);
    }
    
    result.issues = issues;
    setCompatibility(result);
    return result;
  }, []);

  // Create a memoized version of the loadModel function
  const loadModel = useCallback(async (forceReload = false) => {
    try {
      setLoading(true);
      setError(null);
      
      // Check browser compatibility first
      const compatResult = await checkBrowserCompatibility();
      if (compatResult.issues.length > 0) {
        console.warn('Browser compatibility issues detected:', compatResult.issues);
      }
      
      console.log('Initializing ONNX model...');
      const success = await onnxModelService.initModel(forceReload);
      console.log('Model initialization result:', success);
      
      // Directly check if the model is loaded from the service
      const isLoaded = onnxModelService.isModelLoaded();
      console.log('Model loaded status from service:', isLoaded);
      
      // Get any initialization errors
      const errors = onnxModelService.getInitializationErrors();
      setInitErrors(errors);
      
      setModelLoaded(isLoaded);
      
      if (isLoaded) {
        const modelMetadata = onnxModelService.getMetadata();
        console.log('Retrieved model metadata');
        setMetadata(modelMetadata);
        setError(null);
      } else {
        if (errors.length > 0) {
          setError(`Model failed to load: ${errors[0]}`);
        } else {
          setError('Model failed to load. Check console logs for details.');
        }
      }
    } catch (err) {
      console.error('Error loading model:', err);
      setError(`Failed to load model: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  }, [checkBrowserCompatibility]);

  // Function to force a refresh of model state
  const refreshModelState = useCallback(() => {
    const isLoaded = onnxModelService.isModelLoaded();
    console.log('Refreshing model state, current isLoaded:', isLoaded);
    setModelLoaded(isLoaded);
    
    if (isLoaded) {
      const modelMetadata = onnxModelService.getMetadata();
      setMetadata(modelMetadata);
      setError(null);
    }
    
    // Get any initialization errors
    const errors = onnxModelService.getInitializationErrors();
    setInitErrors(errors);
  }, []);

  // Handle retry logic
  const retryLoading = useCallback(() => {
    setLoadAttempts(prev => prev + 1);
    loadModel(true); // Force reload
  }, [loadModel]);

  useEffect(() => {
    // Capture browser information
    const browser = {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      vendor: navigator.vendor,
      language: navigator.language,
      webGLSupport: !!window.WebGLRenderingContext,
      webGL2Support: !!window.WebGL2RenderingContext,
      wasmSupport: typeof WebAssembly === 'object',
      crossOriginIsolated: window.crossOriginIsolated === true
    };
    
    setBrowserInfo(JSON.stringify(browser, null, 2));
    
    // Override console methods to capture logs
    const originalConsoleLog = console.log;
    const originalConsoleError = console.error;
    const originalConsoleWarn = console.warn;
    
    console.log = (...args) => {
      const logMessage = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
      ).join(' ');
      setLogs(prev => [...prev, `[LOG] ${logMessage}`]);
      originalConsoleLog(...args);
    };
    
    console.error = (...args) => {
      const errorMessage = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
      ).join(' ');
      setLogs(prev => [...prev, `[ERROR] ${errorMessage}`]);
      originalConsoleError(...args);
    };
    
    console.warn = (...args) => {
      const warnMessage = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
      ).join(' ');
      setLogs(prev => [...prev, `[WARN] ${warnMessage}`]);
      originalConsoleWarn(...args);
    };
    
    // Load model
    loadModel();
    
    // Setup periodic check for model state
    const stateCheckInterval = setInterval(() => {
      refreshModelState();
    }, 2000);
    
    // Cleanup
    return () => {
      console.log = originalConsoleLog;
      console.error = originalConsoleError;
      console.warn = originalConsoleWarn;
      clearInterval(stateCheckInterval);
    };
  }, [loadModel, refreshModelState]);

  const runTestPrediction = async () => {
    try {
      // Refresh model state before attempting prediction
      refreshModelState();
      
      if (!modelLoaded) {
        setError('Model not loaded yet');
        return;
      }
      
      setLoading(true);
      setPrediction(null);
      
      // Sample values for testing
      const userIdx = 0; // First user in the dataset
      const animeIdx = 1; // First anime in the dataset
      const genreIndices = [0, 1, 2]; // Some sample genres
      const tagIndices = [0, 1, 2, 3, 4]; // Some sample tags
      
      console.log('Running inference with test values');
      const rating = await onnxModelService.runInference(
        userIdx, 
        animeIdx, 
        genreIndices, 
        tagIndices
      );
      
      setPrediction(rating);
    } catch (err) {
      console.error('Error running prediction:', err);
      setError(`Failed to run prediction: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const clearLogs = () => {
    setLogs([]);
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-2xl font-bold mb-6">ONNX Model Test</h1>
      
      <div className="bg-white shadow-md rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Model Status</h2>
        <p className="mb-2">
          <span className="font-medium">Status:</span>{' '}
          {loading ? (
            <span className="text-blue-500">Loading...</span>
          ) : modelLoaded ? (
            <span className="text-green-500">Loaded successfully</span>
          ) : (
            <span className="text-red-500">Not loaded</span>
          )}
        </p>
        <p className="mb-2">
          <span className="font-medium">Load attempts:</span> {loadAttempts}
        </p>
        
        {error && (
          <div className="mt-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded">
            {error}
          </div>
        )}
        
        {initErrors.length > 0 && (
          <div className="mt-4">
            <h3 className="font-medium mb-2">Initialization Errors:</h3>
            <ul className="list-disc pl-5 text-sm text-red-600">
              {initErrors.map((err, index) => (
                <li key={index} className="mb-1">{err}</li>
              ))}
            </ul>
          </div>
        )}
        
        <div className="mt-4 flex space-x-2">
          <button
            onClick={retryLoading}
            disabled={loading}
            className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Retry Loading
          </button>
          <button
            onClick={refreshModelState}
            disabled={loading}
            className="px-4 py-2 bg-gray-500 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Refresh Status
          </button>
        </div>
      </div>
      
      <div className="bg-white shadow-md rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Browser Compatibility</h2>
        {compatibility && (
          <>
            <div className="grid grid-cols-2 gap-2 mb-4">
              <div className="flex items-center">
                <span className={`inline-block w-4 h-4 rounded-full mr-2 ${compatibility.wasmSupported ? 'bg-green-500' : 'bg-red-500'}`}></span>
                <span>WebAssembly Support</span>
              </div>
              <div className="flex items-center">
                <span className={`inline-block w-4 h-4 rounded-full mr-2 ${compatibility.simdSupported ? 'bg-green-500' : 'bg-red-500'}`}></span>
                <span>SIMD Support</span>
              </div>
              <div className="flex items-center">
                <span className={`inline-block w-4 h-4 rounded-full mr-2 ${compatibility.sharedArrayBufferSupported ? 'bg-green-500' : 'bg-red-500'}`}></span>
                <span>SharedArrayBuffer Support</span>
              </div>
              <div className="flex items-center">
                <span className={`inline-block w-4 h-4 rounded-full mr-2 ${compatibility.threadingSupported ? 'bg-green-500' : 'bg-red-500'}`}></span>
                <span>Threading Support</span>
              </div>
            </div>
            
            {compatibility.issues.length > 0 && (
              <div>
                <h3 className="font-medium mb-2 text-orange-600">Compatibility Issues:</h3>
                <ul className="list-disc pl-5 text-sm text-orange-600">
                  {compatibility.issues.map((issue, index) => (
                    <li key={index} className="mb-1">{issue}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </div>
      
      <div className="bg-white shadow-md rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Browser Information</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-auto text-sm">
          {browserInfo}
        </pre>
      </div>
      
      {modelLoaded && metadata && (
        <div className="bg-white shadow-md rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Model Metadata</h2>
          <pre className="bg-gray-100 p-4 rounded overflow-auto text-sm">
            {JSON.stringify(metadata, null, 2)}
          </pre>
        </div>
      )}
      
      <div className="bg-white shadow-md rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Test Prediction</h2>
        <button
          onClick={runTestPrediction}
          disabled={loading || !modelLoaded}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Run Test Prediction
        </button>
        
        {prediction !== null && (
          <div className="mt-4">
            <p className="font-medium">Predicted Rating: {prediction.toFixed(2)}</p>
          </div>
        )}
      </div>
      
      <div className="bg-white shadow-md rounded-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Debug Logs</h2>
          <button 
            onClick={clearLogs}
            className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
          >
            Clear Logs
          </button>
        </div>
        <pre className="bg-gray-100 p-4 rounded overflow-auto text-sm h-64">
          {logs.join('\n')}
        </pre>
      </div>
    </div>
  );
} 
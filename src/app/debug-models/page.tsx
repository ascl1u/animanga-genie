'use client';

import { useState, useEffect } from 'react';
import { useModelContext } from '@/context/ModelContext';
import { onnxModelService } from '@/services/onnxModelService';
import { MODEL_FILE_PATH, MODEL_METADATA_PATH, MODEL_MAPPINGS_PATH } from '@/constants/modelPaths';

export default function DebugModelsPage() {
  const { isModelLoaded, isModelLoading, error, loadModel } = useModelContext();
  const [logs, setLogs] = useState<string[]>([]);
  const [modelState, setModelState] = useState<string>('');
  const [fileChecks, setFileChecks] = useState<Record<string, boolean | null>>({
    model: null,
    metadata: null,
    mappings: null
  });

  // Function to add logs
  const addLog = (message: string) => {
    setLogs(prev => [...prev, `[${new Date().toISOString()}] ${message}`]);
  };

  // Function to clear logs
  const clearLogs = () => setLogs([]);

  // Check file existence
  const checkFiles = async () => {
    addLog('Checking file existence...');
    
    try {
      setFileChecks({
        model: null,
        metadata: null,
        mappings: null
      });
      
      // Check model file
      const modelResponse = await fetch(MODEL_FILE_PATH, { method: 'HEAD' });
      const modelExists = modelResponse.status === 200;
      addLog(`Model file (${MODEL_FILE_PATH}): ${modelExists ? 'EXISTS' : 'NOT FOUND'}`);
      
      // Check metadata file
      const metadataResponse = await fetch(MODEL_METADATA_PATH, { method: 'HEAD' });
      const metadataExists = metadataResponse.status === 200;
      addLog(`Metadata file (${MODEL_METADATA_PATH}): ${metadataExists ? 'EXISTS' : 'NOT FOUND'}`);
      
      // Check mappings file
      const mappingsResponse = await fetch(MODEL_MAPPINGS_PATH, { method: 'HEAD' });
      const mappingsExists = mappingsResponse.status === 200;
      addLog(`Mappings file (${MODEL_MAPPINGS_PATH}): ${mappingsExists ? 'EXISTS' : 'NOT FOUND'}`);
      
      setFileChecks({
        model: modelExists,
        metadata: metadataExists,
        mappings: mappingsExists
      });
    } catch (error) {
      addLog(`Error checking files: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  // Load model
  const handleLoadModel = async () => {
    addLog('Starting model load...');
    try {
      await loadModel();
      addLog('Model load completed');
      updateModelState();
    } catch (error) {
      addLog(`Error loading model: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  // Update model state display
  const updateModelState = () => {
    const state = onnxModelService.debugState();
    setModelState(state);
    addLog('Updated model state');
  };

  // Initialize
  useEffect(() => {
    updateModelState();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-2xl font-bold mb-6">Model Debugging</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Model Status</h2>
          <div className="mb-4">
            <div className="flex items-center space-x-2 mb-2">
              <span className="font-medium">Model Loaded:</span>
              <span className={`px-2 py-1 rounded text-sm ${
                isModelLoaded ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {isModelLoaded ? 'YES' : 'NO'}
              </span>
            </div>
            
            <div className="flex items-center space-x-2 mb-2">
              <span className="font-medium">Loading:</span>
              <span className={`px-2 py-1 rounded text-sm ${
                isModelLoading ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-800'
              }`}>
                {isModelLoading ? 'YES' : 'NO'}
              </span>
            </div>
            
            {error && (
              <div className="mt-2 p-3 bg-red-50 border border-red-200 text-red-700 rounded-md">
                <p className="font-medium">Error:</p>
                <p className="text-sm">{error}</p>
              </div>
            )}
          </div>
          
          <div className="flex space-x-2">
            <button
              onClick={handleLoadModel}
              disabled={isModelLoading}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
            >
              {isModelLoading ? 'Loading...' : 'Load Model'}
            </button>
            
            <button
              onClick={updateModelState}
              className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
            >
              Refresh State
            </button>
          </div>
        </div>
        
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">File Checks</h2>
          <button
            onClick={checkFiles}
            className="mb-4 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Check Files
          </button>
          
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="font-medium">Model File:</span>
              {fileChecks.model === null ? (
                <span className="text-gray-500">Not checked</span>
              ) : (
                <span className={`px-2 py-1 rounded text-sm ${
                  fileChecks.model ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {fileChecks.model ? 'EXISTS' : 'NOT FOUND'}
                </span>
              )}
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="font-medium">Metadata File:</span>
              {fileChecks.metadata === null ? (
                <span className="text-gray-500">Not checked</span>
              ) : (
                <span className={`px-2 py-1 rounded text-sm ${
                  fileChecks.metadata ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {fileChecks.metadata ? 'EXISTS' : 'NOT FOUND'}
                </span>
              )}
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="font-medium">Mappings File:</span>
              {fileChecks.mappings === null ? (
                <span className="text-gray-500">Not checked</span>
              ) : (
                <span className={`px-2 py-1 rounded text-sm ${
                  fileChecks.mappings ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {fileChecks.mappings ? 'EXISTS' : 'NOT FOUND'}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 gap-6">
        <div className="bg-white shadow-md rounded-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Model State Details</h2>
            <button
              onClick={updateModelState}
              className="px-3 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
            >
              Refresh
            </button>
          </div>
          
          <pre className="bg-gray-100 p-4 rounded-md overflow-auto text-sm h-48">
            {modelState ? 
              JSON.stringify(JSON.parse(modelState), null, 2) : 
              'No state available'}
          </pre>
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
          
          <pre className="bg-gray-100 p-4 rounded-md overflow-auto text-sm h-64">
            {logs.join('\n')}
          </pre>
        </div>
      </div>
    </div>
  );
} 
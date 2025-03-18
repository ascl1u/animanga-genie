'use client';

import { useState, useEffect } from 'react';
import { loadModelMappings, ModelMappings } from '@/services/modelService';
import { onnxModelService } from '@/services/onnxModelService';

export default function ModelDebugPage() {
  const [mappingsData, setMappingsData] = useState<ModelMappings | null>(null);
  const [mappingsLoading, setMappingsLoading] = useState(false);
  const [mappingsError, setMappingsError] = useState<string | null>(null);
  const [modelStateJson, setModelStateJson] = useState<string>('');
  const [logs, setLogs] = useState<string[]>([]);

  const loadData = async () => {
    setMappingsLoading(true);
    setMappingsError(null);
    try {
      // Load model mappings
      const mappings = await loadModelMappings();
      setMappingsData(mappings);
      
      // Get model state
      const stateJson = onnxModelService.debugState();
      setModelStateJson(stateJson);
      
      addLog('Loaded mappings and model state successfully');
    } catch (error) {
      console.error('Error loading data:', error);
      setMappingsError((error as Error).message);
      addLog(`Error: ${(error as Error).message}`);
    } finally {
      setMappingsLoading(false);
    }
  };

  const initModel = async () => {
    addLog('Initializing model...');
    try {
      const success = await onnxModelService.initModel(true);
      addLog(`Model initialization: ${success ? 'SUCCESS' : 'FAILED'}`);
      
      // Update model state
      const stateJson = onnxModelService.debugState();
      setModelStateJson(stateJson);
      
      // Get any errors
      const errors = onnxModelService.getInitializationErrors();
      if (errors.length > 0) {
        addLog(`Initialization errors: ${errors.join(', ')}`);
      }
    } catch (error) {
      addLog(`Error initializing model: ${(error as Error).message}`);
    }
  };

  const addLog = (message: string) => {
    setLogs(prev => [...prev, `[${new Date().toISOString()}] ${message}`]);
  };

  const clearLogs = () => {
    setLogs([]);
  };

  useEffect(() => {
    loadData();
    // Override console methods to capture logs
    const originalConsoleLog = console.log;
    const originalConsoleError = console.error;
    
    console.log = (...args) => {
      const message = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
      ).join(' ');
      setLogs(prev => [...prev, `LOG: ${message}`]);
      originalConsoleLog(...args);
    };
    
    console.error = (...args) => {
      const message = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
      ).join(' ');
      setLogs(prev => [...prev, `ERROR: ${message}`]);
      originalConsoleError(...args);
    };
    
    return () => {
      console.log = originalConsoleLog;
      console.error = originalConsoleError;
    };
  }, []);

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-2xl font-bold mb-6">Model Debug Page</h1>
      
      <div className="bg-white shadow-md rounded-lg p-6 mb-6">
        <div className="flex justify-between mb-4">
          <h2 className="text-xl font-semibold">Model State</h2>
          <button
            onClick={initModel}
            className="px-4 py-2 bg-blue-500 text-white rounded"
          >
            Reinitialize Model
          </button>
        </div>
        <pre className="bg-gray-100 p-4 rounded overflow-auto text-sm">
          {modelStateJson ? JSON.parse(modelStateJson) ? 
            JSON.stringify(JSON.parse(modelStateJson), null, 2) : 
            modelStateJson : 'Loading...'}
        </pre>
      </div>
      
      <div className="bg-white shadow-md rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Mappings Data</h2>
        {mappingsLoading ? (
          <p>Loading mappings...</p>
        ) : mappingsError ? (
          <div className="p-3 bg-red-100 border border-red-300 text-red-700 rounded">
            {mappingsError}
          </div>
        ) : (
          <div>
            <div className="mb-4">
              <h3 className="font-medium mb-2">Available Keys:</h3>
              <ul className="list-disc pl-5">
                {mappingsData && Object.keys(mappingsData).map(key => (
                  <li key={key}>{key} - {typeof mappingsData[key as keyof ModelMappings]} with {
                    typeof mappingsData[key as keyof ModelMappings] === 'object' && mappingsData[key as keyof ModelMappings] 
                      ? Object.keys(mappingsData[key as keyof ModelMappings] as object).length 
                      : 0
                  } items</li>
                ))}
              </ul>
            </div>
            
            <div className="mb-4">
              <h3 className="font-medium mb-2">First 10 Entries:</h3>
              {mappingsData && Object.keys(mappingsData).map(key => {
                const mappingKey = key as keyof ModelMappings;
                const mappingValue = mappingsData[mappingKey];
                
                return (
                  <div key={key} className="mb-4">
                    <h4 className="font-medium">{key}:</h4>
                    <pre className="bg-gray-100 p-2 rounded overflow-auto text-sm">
                      {typeof mappingValue === 'object' && mappingValue
                        ? JSON.stringify(
                            Object.fromEntries(
                              Object.entries(mappingValue as Record<string, unknown>).slice(0, 10)
                            ), null, 2)
                        : JSON.stringify(mappingValue)}
                    </pre>
                  </div>
                );
              })}
            </div>
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
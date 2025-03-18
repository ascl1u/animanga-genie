'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { 
  loadModel, 
  loadModelMappings, 
  loadModelMetadata,
  isModelLoaded as checkModelLoaded,
  ModelMetadata,
  ModelMappings
} from '@/services/modelService';
import { MODEL_FILE_PATH, MODEL_METADATA_PATH } from '@/constants/modelPaths';
import { logOnnxServiceState } from '@/services/onnxModelService';

interface ModelContextType {
  isModelLoaded: boolean;
  isModelLoading: boolean;
  error: string | null;
  metadata: ModelMetadata | null;
  mappings: ModelMappings | null;
  loadModel: () => Promise<boolean>;
  loadingProgress: number;
}

const initialModelContext: ModelContextType = {
  isModelLoaded: false,
  isModelLoading: false,
  error: null,
  metadata: null,
  mappings: null,
  loadModel: async () => false,
  loadingProgress: 0,
};

const ModelContext = createContext<ModelContextType>(initialModelContext);

export function useModelContext() {
  return useContext(ModelContext);
}

export function ModelProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState({
    isModelLoaded: false,
    isModelLoading: false,
    error: null as string | null,
    metadata: null as ModelMetadata | null,
    mappings: null as ModelMappings | null,
    loadingProgress: 0,
  });

  // Check if model is already loaded on mount
  useEffect(() => {
    const modelLoadedStatus = checkModelLoaded();
    console.log(`[MODEL-CONTEXT] Initial load check: model loaded = ${modelLoadedStatus}`);
    logOnnxServiceState();
    
    if (modelLoadedStatus) {
      // If model is loaded, also try to fetch the metadata and mappings
      Promise.all([
        loadModelMetadata().catch(err => {
          console.warn('[MODEL-CONTEXT] Error fetching metadata for loaded model:', err);
          return null;
        }), 
        loadModelMappings().catch(err => {
          console.warn('[MODEL-CONTEXT] Error fetching mappings for loaded model:', err);
          return null;
        })
      ]).then(([metadata, mappings]) => {
        setState(prev => ({
          ...prev,
          isModelLoaded: true,
          metadata: metadata || prev.metadata,
          mappings: mappings || prev.mappings,
        }));
      });
    } else {
      setState(prev => ({
        ...prev,
        isModelLoaded: modelLoadedStatus,
      }));
    }
  }, []);

  const loadModelResources = useCallback(async (): Promise<boolean> => {
    if (state.isModelLoading) {
      console.log('[MODEL-CONTEXT] Model is already loading, ignoring request');
      return false;
    }
    
    if (state.isModelLoaded && state.metadata && state.mappings) {
      console.log('[MODEL-CONTEXT] Model is already loaded with metadata and mappings, no need to reload');
      return true;
    }

    console.log(`[MODEL-CONTEXT] Starting model load from paths: model=${MODEL_FILE_PATH}, metadata=${MODEL_METADATA_PATH}`);
    setState(prev => ({ 
      ...prev, 
      isModelLoading: true, 
      error: null,
      loadingProgress: 10 
    }));

    try {
      logOnnxServiceState();
      
      // Set loading progress indicators
      setState(prev => ({ ...prev, loadingProgress: 20 }));
      
      // Load model first, as other resources depend on it
      await loadModel();
      setState(prev => ({ ...prev, loadingProgress: 50 }));
      
      // Then load mappings and metadata in parallel
      const [mappingsData, metadata] = await Promise.all([
        loadModelMappings(),
        loadModelMetadata()
      ]);
      
      setState(prev => ({ ...prev, loadingProgress: 90 }));

      console.log('[MODEL-CONTEXT] Model resources loaded successfully');
      logOnnxServiceState();
      
      setState({
        isModelLoaded: true,
        isModelLoading: false,
        error: null,
        metadata,
        mappings: mappingsData,
        loadingProgress: 100,
      });
      
      return true;
    } catch (error) {
      console.error('Failed to load model resources:', error);
      logOnnxServiceState();
      
      setState({
        isModelLoaded: false,
        isModelLoading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        metadata: null,
        mappings: null,
        loadingProgress: 0,
      });
      
      return false;
    }
  }, [state.isModelLoaded, state.isModelLoading, state.metadata, state.mappings]);

  return (
    <ModelContext.Provider
      value={{
        ...state,
        loadModel: loadModelResources,
      }}
    >
      {children}
    </ModelContext.Provider>
  );
} 
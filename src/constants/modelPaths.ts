/**
 * Centralized constants for model file paths
 * This ensures consistency across all services and components
 */

// Base paths
export const MODEL_BASE_PATH = '/models/anime_recommender';

// Model files
export const MODEL_FILE_PATH = `${MODEL_BASE_PATH}/anime_recommender.onnx`;
export const MODEL_METADATA_PATH = `${MODEL_BASE_PATH}/onnx_model_metadata.json`;
export const MODEL_MAPPINGS_PATH = `${MODEL_BASE_PATH}/model_mappings.json`;

// WASM files
export const WASM_PATH = `${MODEL_BASE_PATH}/ort-wasm.wasm`;
export const WASM_SIMD_PATH = `${MODEL_BASE_PATH}/ort-wasm-simd.wasm`;
export const WASM_THREADED_PATH = `${MODEL_BASE_PATH}/ort-wasm-threaded.wasm`; 
#!/usr/bin/env python3
"""
Convert PyTorch Anime Recommendation Model to ONNX Format

This script loads the trained PyTorch anime recommendation model and converts
it to ONNX format for inference optimization and cross-platform compatibility.
"""

import os
import json
import argparse
import torch
import numpy as np
from typing import Dict, Tuple

# Import the model architecture
from train_model import ImprovedAnimeRecommenderModel

def load_model_and_metadata(model_dir: str) -> Tuple[torch.nn.Module, Dict]:
    """
    Load the trained PyTorch model and its metadata.
    
    Args:
        model_dir (str): Directory containing the model and metadata files.
        
    Returns:
        Tuple[torch.nn.Module, Dict]: The loaded model and its metadata.
    """
    # Load model metadata
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Create model with the same architecture
    model = ImprovedAnimeRecommenderModel(
        n_users=metadata["n_users"],
        n_anime=metadata["n_anime"],
        n_genres=metadata["n_genres"],
        n_tags=metadata["n_tags"]
    )
    
    # Load the trained model weights
    model_path = os.path.join(model_dir, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set to evaluation mode
    
    return model, metadata

def convert_to_onnx(
    model: torch.nn.Module,
    metadata: Dict,
    output_path: str,
    opset_version: int = 12
) -> None:
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model (torch.nn.Module): The PyTorch model to convert.
        metadata (Dict): Model metadata containing configuration.
        output_path (str): Path to save the ONNX model.
        opset_version (int): ONNX opset version to use.
    """
    # Create dummy inputs for ONNX export
    batch_size = 1
    max_genres = metadata.get("max_genres", 10)
    max_tags = metadata.get("max_tags", 20)
    
    dummy_user_idx = torch.tensor([0], dtype=torch.int64)
    dummy_anime_idx = torch.tensor([0], dtype=torch.int64)
    dummy_genre_indices = torch.zeros((batch_size, max_genres), dtype=torch.int64)
    dummy_tag_indices = torch.zeros((batch_size, max_tags), dtype=torch.int64)
    
    # Export the model to ONNX format
    torch.onnx.export(
        model,
        (dummy_user_idx, dummy_anime_idx, dummy_genre_indices, dummy_tag_indices),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["user_idx", "anime_idx", "genre_indices", "tag_indices"],
        output_names=["rating"],
        dynamic_axes={
            "user_idx": {0: "batch_size"},
            "anime_idx": {0: "batch_size"},
            "genre_indices": {0: "batch_size"},
            "tag_indices": {0: "batch_size"},
            "rating": {0: "batch_size"}
        }
    )
    
    print(f"Model successfully converted to ONNX format and saved to {output_path}")
    
    # Save additional metadata for the ONNX model
    onnx_metadata = {
        **metadata,
        "model_type": "onnx",
        "input_names": ["user_idx", "anime_idx", "genre_indices", "tag_indices"],
        "output_names": ["rating"]
    }
    
    onnx_metadata_path = os.path.join(os.path.dirname(output_path), "onnx_model_metadata.json")
    with open(onnx_metadata_path, "w") as f:
        json.dump(onnx_metadata, f, indent=2)
    
    print(f"ONNX model metadata saved to {onnx_metadata_path}")

def verify_onnx_model(onnx_path: str) -> None:
    """
    Verify the exported ONNX model.
    
    Args:
        onnx_path (str): Path to the ONNX model file.
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Check if the model is well-formed
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is well-formed")
        
        # Check if the model can be loaded by onnxruntime
        sess = ort.InferenceSession(onnx_path)
        print("ONNX model can be loaded by onnxruntime")
        
    except ImportError:
        print("WARNING: Could not verify ONNX model. Make sure onnx and onnxruntime are installed.")
    except Exception as e:
        print(f"WARNING: ONNX model verification failed: {e}")

def try_optimize_onnx_model(onnx_path: str) -> None:
    """
    Try to optimize the ONNX model using onnxsim if available.
    
    Args:
        onnx_path (str): Path to the ONNX model file.
    """
    try:
        import onnx
        from onnxsim import simplify
        
        # Load the model
        onnx_model = onnx.load(onnx_path)
        
        # Simplify the model
        optimized_model, check = simplify(onnx_model)
        
        if check:
            # Save the optimized model
            onnx.save(optimized_model, onnx_path)
            print(f"ONNX model optimized successfully")
        else:
            print("ONNX model simplification check failed")
    
    except ImportError:
        print("INFO: onnxsim not available for model optimization. You can install it with 'pip install onnxsim'.")
    except Exception as e:
        print(f"WARNING: ONNX model optimization failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch anime recommendation model to ONNX format")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="data/model/pytorch",
        help="Directory containing the PyTorch model and metadata"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/model/onnx",
        help="Directory to save the ONNX model"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=12,
        help="ONNX opset version to use for export"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Try to optimize the ONNX model after conversion"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and metadata
    print(f"Loading PyTorch model from {args.model_dir}")
    model, metadata = load_model_and_metadata(args.model_dir)
    
    # Set output path
    onnx_path = os.path.join(args.output_dir, "anime_recommender.onnx")
    
    # Convert model to ONNX
    print(f"Converting model to ONNX format (opset version {args.opset_version})")
    convert_to_onnx(model, metadata, onnx_path, args.opset_version)
    
    # Verify the ONNX model
    print("Verifying ONNX model...")
    verify_onnx_model(onnx_path)
    
    # Optimize if requested
    if args.optimize:
        print("Trying to optimize ONNX model...")
        try_optimize_onnx_model(onnx_path)
    
    print("\nConversion complete!")
    print(f"ONNX model saved to: {onnx_path}")
    print(f"Metadata saved to: {os.path.join(args.output_dir, 'onnx_model_metadata.json')}")
    print("\nYou can now use the ONNX model for inference using ONNX Runtime or other compatible frameworks.")

if __name__ == "__main__":
    main()
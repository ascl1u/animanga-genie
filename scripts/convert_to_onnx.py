#!/usr/bin/env python3
"""
Convert PyTorch Anime Recommendation Model to ONNX Format

This script loads the trained PyTorch anime recommendation model and converts
it to ONNX format for inference optimization and cross-platform compatibility.
Supports enhanced features including studios and relationships.
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
        n_tags=metadata["n_tags"],
        n_studios=metadata["n_studios"],
        embedding_dim_users=metadata.get("user_embedding_dim", 64),
        embedding_dim_anime=metadata.get("anime_embedding_dim", 128),
        embedding_dim_genres=metadata.get("genre_embedding_dim", 32),
        embedding_dim_tags=metadata.get("tag_embedding_dim", 32),
        embedding_dim_studios=metadata.get("studio_embedding_dim", 16),
        embedding_dim_relations=metadata.get("relation_embedding_dim", 32)
    )
    
    # Load the trained model weights
    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "best_model.pth")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set to evaluation mode
    
    return model, metadata

class ONNXExportModel(torch.nn.Module):
    """Wrapper model for ONNX export to ensure proper handling of inputs."""
    
    def __init__(self, original_model: torch.nn.Module):
        super(ONNXExportModel, self).__init__()
        self.original_model = original_model
    
    def forward(
        self,
        user_idx,
        anime_idx,
        genre_indices,
        tag_indices,
        studio_indices,
        studio_weights,
        relation_indices,
        relation_weights
    ):
        with torch.no_grad():
            return self.original_model(
                user_idx,
                anime_idx,
                genre_indices,
                tag_indices,
                studio_indices,
                studio_weights,
                relation_indices,
                relation_weights
            )

def convert_to_onnx(
    model: torch.nn.Module,
    metadata: Dict,
    output_path: str,
    opset_version: int = 15
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
    max_studios = metadata.get("max_studios", 10)
    max_relations = metadata.get("max_relations", 20)
    
    # Create export model wrapper
    export_model = ONNXExportModel(model)
    
    # Create dummy inputs
    dummy_inputs = (
        torch.zeros(batch_size, dtype=torch.int64),  # user_idx
        torch.zeros(batch_size, dtype=torch.int64),  # anime_idx
        torch.zeros((batch_size, max_genres), dtype=torch.int64),  # genre_indices
        torch.zeros((batch_size, max_tags), dtype=torch.int64),  # tag_indices
        torch.zeros((batch_size, max_studios), dtype=torch.int64),  # studio_indices
        torch.zeros((batch_size, max_studios), dtype=torch.float32),  # studio_weights
        torch.zeros((batch_size, max_relations), dtype=torch.int64),  # relation_indices
        torch.zeros((batch_size, max_relations), dtype=torch.float32)  # relation_weights
    )
    
    # Export the model to ONNX format
    torch.onnx.export(
        export_model,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=[
            "user_idx",
            "anime_idx",
            "genre_indices",
            "tag_indices",
            "studio_indices",
            "studio_weights",
            "relation_indices",
            "relation_weights"
        ],
        output_names=["rating"],
        dynamic_axes={
            "user_idx": {0: "batch_size"},
            "anime_idx": {0: "batch_size"},
            "genre_indices": {0: "batch_size"},
            "tag_indices": {0: "batch_size"},
            "studio_indices": {0: "batch_size"},
            "studio_weights": {0: "batch_size"},
            "relation_indices": {0: "batch_size"},
            "relation_weights": {0: "batch_size"},
            "rating": {0: "batch_size"}
        }
    )
    
    print(f"Model successfully converted to ONNX format and saved to {output_path}")
    
    # Save additional metadata for the ONNX model
    onnx_metadata = {
        **metadata,
        "model_type": "onnx",
        "opset_version": opset_version,
        "input_names": [
            "user_idx",
            "anime_idx",
            "genre_indices",
            "tag_indices",
            "studio_indices",
            "studio_weights",
            "relation_indices",
            "relation_weights"
        ],
        "output_names": ["rating"],
        "input_shapes": {
            "user_idx": [-1],
            "anime_idx": [-1],
            "genre_indices": [-1, max_genres],
            "tag_indices": [-1, max_tags],
            "studio_indices": [-1, max_studios],
            "studio_weights": [-1, max_studios],
            "relation_indices": [-1, max_relations],
            "relation_weights": [-1, max_relations]
        }
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
        
        # Load metadata to get input shapes
        metadata_path = os.path.join(os.path.dirname(onnx_path), "onnx_model_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Create sample inputs for inference test
        batch_size = 2
        input_data = {
            "user_idx": np.zeros(batch_size, dtype=np.int64),
            "anime_idx": np.zeros(batch_size, dtype=np.int64),
            "genre_indices": np.zeros((batch_size, metadata["max_genres"]), dtype=np.int64),
            "tag_indices": np.zeros((batch_size, metadata["max_tags"]), dtype=np.int64),
            "studio_indices": np.zeros((batch_size, metadata["max_studios"]), dtype=np.int64),
            "studio_weights": np.zeros((batch_size, metadata["max_studios"]), dtype=np.float32),
            "relation_indices": np.zeros((batch_size, metadata["max_relations"]), dtype=np.int64),
            "relation_weights": np.zeros((batch_size, metadata["max_relations"]), dtype=np.float32)
        }
        
        # Test inference with onnxruntime
        sess = ort.InferenceSession(onnx_path)
        outputs = sess.run(None, input_data)
        
        if outputs[0].shape == (batch_size,):
            print("ONNX model inference test passed")
        else:
            print(f"WARNING: Unexpected output shape: {outputs[0].shape}, expected: ({batch_size},)")
        
        print("ONNX model can be loaded and run by onnxruntime")
        
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
        
        # Load metadata to get input shapes
        metadata_path = os.path.join(os.path.dirname(onnx_path), "onnx_model_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Load the model
        onnx_model = onnx.load(onnx_path)
        
        # Define input shapes for optimization
        input_shapes = {
            "user_idx": [1],
            "anime_idx": [1],
            "genre_indices": [1, metadata["max_genres"]],
            "tag_indices": [1, metadata["max_tags"]],
            "studio_indices": [1, metadata["max_studios"]],
            "studio_weights": [1, metadata["max_studios"]],
            "relation_indices": [1, metadata["max_relations"]],
            "relation_weights": [1, metadata["max_relations"]]
        }
        
        # Simplify the model
        optimized_model, check = simplify(
            onnx_model,
            input_shapes=input_shapes,
            dynamic_input_shape=True
        )
        
        if check:
            # Save the optimized model
            onnx.save(optimized_model, onnx_path)
            print("ONNX model optimized successfully")
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
        default=15,
        help="ONNX opset version to use for export"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Try to optimize the ONNX model after conversion"
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        help="Skip model verification step"
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
    if not args.skip_verify:
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
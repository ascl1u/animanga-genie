#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    # Paths for the input model and output directory
    input_model_path = "data/model/tensorflow/model.keras"
    output_dir = "data/model/tfjs/"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the conversion command.
    # Use "--input_format keras_saved_model" for the native Keras model format.
    cmd = [
        "tensorflowjs_converter",
        "--input_format", "keras_saved_model",
        input_model_path,
        output_dir
    ]
    
    print("Converting model to TensorFlow.js format...")
    print("Running command:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("Conversion successful! The TF.js model is saved at:", output_dir)
    except subprocess.CalledProcessError as error:
        print("Conversion failed with error:", error)
        sys.exit(1)

if __name__ == "__main__":
    main()

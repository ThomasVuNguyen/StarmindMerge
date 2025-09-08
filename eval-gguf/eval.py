#!/usr/bin/env python3
"""
Simplified evaluation script - just specify the model GGUF file(s) and everything else is automatic.

Usage:
    python eval.py model.gguf                    # Evaluate single model on MMLU
    python eval.py model1.gguf model2.gguf       # Compare multiple models
    python eval.py model.gguf --tasks all        # Run all tasks
    python eval.py model.gguf --plot             # Evaluate and plot results
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Simple model evaluation - specify GGUF file(s) or run all models in gguf/ folder")
    parser.add_argument("models", nargs="*", help="GGUF model file(s) to evaluate (default: all models in gguf/ folder)")
    parser.add_argument("--tasks", default="all", help="Tasks to run (default: all)")
    parser.add_argument("--plot", action="store_true", help="Generate plot after evaluation")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    # If no models specified, find all GGUF files in gguf/ folder
    if not args.models:
        import glob
        gguf_files = glob.glob("gguf/*.gguf")
        if not gguf_files:
            print("Error: No GGUF files found in gguf/ folder and no models specified")
            print("Please either:")
            print("  1. Place GGUF files in the gguf/ folder, or")
            print("  2. Specify model files: python eval.py model1.gguf model2.gguf")
            sys.exit(1)
        args.models = gguf_files
        if not args.quiet:
            print(f"Found {len(gguf_files)} model(s) in gguf/ folder: {', '.join(gguf_files)}")
    
    # Check if llama.cpp exists
    llama_path = "./llama.cpp"
    if not os.path.exists(llama_path):
        print(f"Error: llama.cpp not found at {llama_path}")
        print("Please ensure llama.cpp is in the current directory")
        sys.exit(1)
    
    # Check if models exist
    for model in args.models:
        if not os.path.exists(model):
            print(f"Error: Model file not found: {model}")
            sys.exit(1)
    
    # Build evaluation command
    cmd = [
        "python3", "evaluate.py",
        *args.models,
        "--llama_path", llama_path,
        "--tasks", args.tasks
    ]
    
    if args.quiet:
        cmd.append("--quiet")
    
    print(f"Running evaluation: {' '.join(cmd)}")
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Evaluation completed successfully!")
        print("Results saved to results/ directory")
        
        # Generate plot if requested
        if args.plot:
            print("\nGenerating plot...")
            plot_cmd = [
                "python3", "plot.py",
                *args.models,
                "--tasks", args.tasks
            ]
            if args.quiet:
                plot_cmd.append("--quiet")
            
            subprocess.run(plot_cmd, check=True)
            print("✅ Plot generated successfully!")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()

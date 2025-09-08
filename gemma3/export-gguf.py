#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma3 270M GGUF Conversion Script
Convert ThomasTheMaker/gemma-3-270m-tulu-3-sft-personas-instruction-following to GGUF format
"""

from unsloth import FastLanguageModel
import os

def create_gguf_directory():
    """Create gguf directory if it doesn't exist"""
    os.makedirs("gguf", exist_ok=True)
    print("Created gguf/ directory")

def load_model():
    """Load the fine-tuned Gemma3 270M model"""
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="ThomasTheMaker/gemma-3-270m-tulu-3-sft-personas-instruction-following",
        max_seq_length=2048,
        load_in_4bit=False,
    )
    print("Model loaded successfully!")
    return model, tokenizer

def convert_to_gguf(model, tokenizer, quantization_type="Q8_0"):
    """Convert model to GGUF format"""
    # Create the main gguf directory if it doesn't exist
    os.makedirs("gguf", exist_ok=True)
    
    # Create a temporary directory for the model, then convert to GGUF
    temp_model_dir = f"temp_model_{quantization_type.lower()}"
    final_filename = f"gguf/gemma-3-270m-tulu-3-sft-personas-instruction-following-{quantization_type.lower()}.gguf"
    
    print(f"Converting to GGUF format with {quantization_type} quantization...")
    print(f"Target file: {final_filename}")
    
    try:
        # First save the model in standard format to create proper directory structure
        print("Saving model in standard format...")
        model.save_pretrained(temp_model_dir)
        tokenizer.save_pretrained(temp_model_dir)
        
        # Now convert the saved model to GGUF
        print("Converting to GGUF...")
        model.save_pretrained_gguf(
            temp_model_dir,
            tokenizer,
            quantization_type=quantization_type,
        )
        
        # Move the GGUF file to the final location
        import glob
        import shutil
        
        # Look for GGUF files both in the temp directory and current directory
        gguf_files = glob.glob(f"{temp_model_dir}/*.gguf") + glob.glob(f"{temp_model_dir}.*.gguf")
        
        if gguf_files:
            shutil.move(gguf_files[0], final_filename)
            print(f"GGUF file moved to: {final_filename}")
        else:
            print(f"Warning: No GGUF file found in {temp_model_dir}")
            # List what files were created for debugging
            if os.path.exists(temp_model_dir):
                files = os.listdir(temp_model_dir)
                print(f"Files in {temp_model_dir}: {files}")
            # Also check current directory for gguf files with temp_model prefix
            current_dir_gguf = glob.glob(f"{temp_model_dir}*.gguf")
            if current_dir_gguf:
                print(f"Found GGUF files in current directory: {current_dir_gguf}")
                shutil.move(current_dir_gguf[0], final_filename)
                print(f"GGUF file moved to: {final_filename}")
                # Clean up any remaining temp gguf files
                for f in current_dir_gguf[1:]:
                    os.remove(f)
        
        # Clean up temporary directory
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        # Clean up on error
        import shutil
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir, ignore_errors=True)
        raise
    
    return final_filename

def main():
    """Main conversion function"""
    # Create gguf directory
    create_gguf_directory()
    
    # Load model
    model, tokenizer = load_model()
    
    # Available quantization types (only those supported by unsloth)
    quantization_types = [
        "F32",      # Full 32-bit precision (largest file)
        "F16",      # Full 16-bit precision 
        "BF16",     # Brain Float 16-bit precision
        "Q8_0",     # 8-bit quantization (smallest supported)
    ]
    
    print("\nAvailable quantization types (ordered by quality/size):")
    for i, qtype in enumerate(quantization_types, 1):
        print(f"{i:2d}. {qtype}")
    print(f"{len(quantization_types) + 1:2d}. All types")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\nSelect quantization type (1-{len(quantization_types) + 1}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(quantization_types):
                selected_type = quantization_types[choice_num - 1]
                convert_to_gguf(model, tokenizer, selected_type)
                break
            elif choice_num == len(quantization_types) + 1:
                for qtype in quantization_types:
                    convert_to_gguf(model, tokenizer, qtype)
                break
            else:
                print(f"Invalid choice. Please select 1-{len(quantization_types) + 1}.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nConversion cancelled.")
            return
        except Exception as e:
            print(f"Error: {e}")
            return
    
    print("\nGGUF conversion completed successfully!")
    print("Files are available in the gguf/ directory")

if __name__ == "__main__":
    main()

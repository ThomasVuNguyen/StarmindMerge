from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import os

def reformat_dataset():
    """
    Load the dataset, reformat it to separate input/output/system columns,
    and upload to Hugging Face Hub
    """
    
    # Variables
    dataset_name = "allenai/tulu-3-sft-personas-math"
    repo_id = "ThomasTheMaker/tulu-3-sft-personas-math"
    local_path = "./tulu-3-sft-personas-math"
    default_system_message = "You are a helpful AI assistant."
    
    print("🔄 Loading dataset...")
    # Load the original dataset
    dataset = load_dataset(dataset_name)
    
    print(f"📊 Original dataset size: {len(dataset['train'])} samples")
    
    def extract_columns(example):
        """
        Extract input, output, and system from messages format
        """
        messages = example["messages"]
        
        # Initialize variables
        input_text = ""
        output_text = ""
        system_text = ""
        
        # Process each message
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                input_text = content
            elif role == "assistant":
                output_text = content
            elif role == "system":
                system_text = content
        
        # If no system message, use a default
        if not system_text:
            system_text = default_system_message
        
        return {
            "input": input_text,
            "output": output_text,
            "system": system_text
        }
    
    print("🔄 Reformatting dataset...")
    # Apply the transformation
    reformatted_dataset = dataset.map(extract_columns, remove_columns=["messages"])
    
    # Show a sample of the reformatted data
    print("\n📝 Sample of reformatted data:")
    sample = reformatted_dataset["train"][0]
    print(f"System: {sample['system']}")
    print(f"Input: {sample['input'][:100]}...")
    print(f"Output: {sample['output'][:100]}...")
    
    # Create the new dataset
    new_dataset = Dataset.from_dict({
        "input": [item["input"] for item in reformatted_dataset["train"]],
        "output": [item["output"] for item in reformatted_dataset["train"]],
        "system": [item["system"] for item in reformatted_dataset["train"]]
    })
    
    print(f"✅ Reformatted dataset created with {len(new_dataset)} samples")
    
    # Upload to Hugging Face Hub
    print(f"🚀 Uploading to {repo_id}...")
    
    try:
        # Push to hub
        new_dataset.push_to_hub(
            repo_id=repo_id,
            private=False,  # Set to True if you want a private dataset
            token=os.getenv("HF_TOKEN")  # Make sure to set your HF token
        )
        print(f"✅ Successfully uploaded to {repo_id}")
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        print("💡 Make sure to set your HF_TOKEN environment variable")
        print("   export HF_TOKEN=your_token_here")
        
        # Save locally as backup
        new_dataset.save_to_disk(local_path)
        print(f"💾 Dataset saved locally to {local_path}")

if __name__ == "__main__":
    reformat_dataset()

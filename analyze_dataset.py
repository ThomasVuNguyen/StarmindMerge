#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes the dataset to verify structure and content for fine-tuning.
"""

from datasets import load_dataset
import json
from collections import Counter
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_NAME = "allenai/tulu-3-sft-personas-instruction-following"

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_dataset_structure(ds):
    """Analyze the basic structure of the dataset."""
    print("=" * 60)
    print("DATASET STRUCTURE ANALYSIS")
    print("=" * 60)
    
    print(f"Dataset: {DATASET_NAME}")
    print(f"Dataset splits: {list(ds.keys())}")
    
    for split_name, split_data in ds.items():
        print(f"\n{split_name.upper()} split:")
        print(f"  Number of examples: {len(split_data)}")
        print(f"  Features: {list(split_data.features.keys())}")
        
        # Show feature types
        for feature_name, feature_type in split_data.features.items():
            print(f"    {feature_name}: {feature_type}")

def analyze_messages_structure(ds, split_name="train", num_samples=5):
    """Analyze the structure of the messages column."""
    print(f"\n" + "=" * 60)
    print(f"MESSAGES STRUCTURE ANALYSIS ({split_name.upper()} split)")
    print("=" * 60)
    
    split_data = ds[split_name]
    
    # Analyze message roles
    all_roles = []
    message_lengths = []
    conversation_lengths = []
    
    for i, example in enumerate(split_data):
        if i >= 1000:  # Limit analysis to first 1000 examples for performance
            break
            
        messages = example['messages']
        conversation_lengths.append(len(messages))
        
        for message in messages:
            all_roles.append(message['role'])
            message_lengths.append(len(message['content']))
    
    # Role statistics
    role_counts = Counter(all_roles)
    print("Role distribution:")
    for role, count in role_counts.most_common():
        percentage = (count / len(all_roles)) * 100
        print(f"  {role}: {count} ({percentage:.1f}%)")
    
    # Message length statistics
    avg_msg_length = sum(message_lengths) / len(message_lengths)
    print(f"\nMessage length statistics:")
    print(f"  Average message length: {avg_msg_length:.1f} characters")
    print(f"  Min message length: {min(message_lengths)}")
    print(f"  Max message length: {max(message_lengths)}")
    
    # Conversation length statistics
    avg_conv_length = sum(conversation_lengths) / len(conversation_lengths)
    print(f"\nConversation length statistics:")
    print(f"  Average messages per conversation: {avg_conv_length:.1f}")
    print(f"  Min messages per conversation: {min(conversation_lengths)}")
    print(f"  Max messages per conversation: {max(conversation_lengths)}")
    
    # Show conversation length distribution
    conv_length_counts = Counter(conversation_lengths)
    print(f"\nConversation length distribution:")
    for length, count in sorted(conv_length_counts.items())[:10]:  # Show top 10
        percentage = (count / len(conversation_lengths)) * 100
        print(f"  {length} messages: {count} conversations ({percentage:.1f}%)")

def show_sample_conversations(ds, split_name="train", num_samples=3):
    """Show sample conversations from the dataset."""
    print(f"\n" + "=" * 60)
    print(f"SAMPLE CONVERSATIONS ({split_name.upper()} split)")
    print("=" * 60)
    
    split_data = ds[split_name]
    
    # Get random samples
    sample_indices = random.sample(range(len(split_data)), min(num_samples, len(split_data)))
    
    for i, idx in enumerate(sample_indices, 1):
        example = split_data[idx]
        messages = example['messages']
        
        print(f"\nSample {i} (Index {idx}):")
        print("-" * 40)
        
        for j, message in enumerate(messages):
            role = message['role']
            content = message['content']
            
            # Truncate very long content
            if len(content) > 200:
                content_display = content[:200] + "..."
            else:
                content_display = content
                
            print(f"Message {j+1} ({role}):")
            print(f"  {content_display}")
            print()

def analyze_content_patterns(ds, split_name="train"):
    """Analyze patterns in the content."""
    print(f"\n" + "=" * 60)
    print(f"CONTENT PATTERNS ANALYSIS ({split_name.upper()} split)")
    print("=" * 60)
    
    split_data = ds[split_name]
    
    # Analyze first 500 examples for performance
    sample_size = min(500, len(split_data))
    
    # Track conversation patterns
    conversation_patterns = []
    system_message_count = 0
    
    for i in range(sample_size):
        example = split_data[i]
        messages = example['messages']
        
        # Create pattern string
        pattern = " -> ".join([msg['role'] for msg in messages])
        conversation_patterns.append(pattern)
        
        # Count system messages
        if any(msg['role'] == 'system' for msg in messages):
            system_message_count += 1
    
    # Show most common patterns
    pattern_counts = Counter(conversation_patterns)
    print("Most common conversation patterns:")
    for pattern, count in pattern_counts.most_common(10):
        percentage = (count / sample_size) * 100
        print(f"  {pattern}: {count} ({percentage:.1f}%)")
    
    print(f"\nSystem message statistics:")
    system_percentage = (system_message_count / sample_size) * 100
    print(f"  Conversations with system messages: {system_message_count}/{sample_size} ({system_percentage:.1f}%)")

def test_tokenization_compatibility(ds, split_name="train"):
    """Test if the dataset is compatible with our tokenization function."""
    print(f"\n" + "=" * 60)
    print("TOKENIZATION COMPATIBILITY TEST")
    print("=" * 60)
    
    from transformers import AutoTokenizer
    from trl import setup_chat_format
    
    # Load tokenizer with our chat template
    MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
    CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.chat_template = CHAT_TEMPLATE
    
    split_data = ds[split_name]
    
    # Test tokenization on first few examples
    success_count = 0
    error_count = 0
    
    for i in range(min(10, len(split_data))):
        try:
            example = split_data[i]
            messages = example['messages']
            
            # Test our tokenization function
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
            
            success_count += 1
            
            if i < 2:  # Show first 2 examples
                print(f"\nExample {i+1} tokenization:")
                print(f"  Input messages: {len(messages)} messages")
                print(f"  Formatted length: {len(formatted_text)} characters")
                print(f"  Token count: {len(token_ids)} tokens")
                print(f"  Sample formatted text: {repr(formatted_text[:100])}...")
                
        except Exception as e:
            error_count += 1
            print(f"Error in example {i}: {e}")
    
    print(f"\nTokenization test results:")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Success rate: {(success_count/(success_count+error_count)*100):.1f}%")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("Loading dataset...")
    
    try:
        ds = load_dataset(DATASET_NAME)
        
        # Run all analyses
        analyze_dataset_structure(ds)
        analyze_messages_structure(ds)
        show_sample_conversations(ds)
        analyze_content_patterns(ds)
        test_tokenization_compatibility(ds)
        
        print("\n" + "=" * 60)
        print("DATASET ANALYSIS COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

if __name__ == "__main__":
    main()

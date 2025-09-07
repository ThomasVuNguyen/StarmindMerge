#!/usr/bin/env python3
"""
Token Analysis Script
Analyzes the tokenizer and model to verify special tokens and chat template formatting.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_special_tokens(tokenizer):
    """Analyze and display special tokens from the tokenizer."""
    print("=" * 60)
    print("SPECIAL TOKENS ANALYSIS")
    print("=" * 60)
    
    special_tokens = {
        "BOS Token": tokenizer.bos_token,
        "EOS Token": tokenizer.eos_token,
        "UNK Token": tokenizer.unk_token,
        "PAD Token": tokenizer.pad_token,
        "SEP Token": tokenizer.sep_token,
        "CLS Token": tokenizer.cls_token,
        "MASK Token": tokenizer.mask_token,
    }
    
    for name, token in special_tokens.items():
        if token is not None:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"{name:12}: '{token}' (ID: {token_id})")
        else:
            print(f"{name:12}: None")
    
    print(f"\nVocabulary Size: {len(tokenizer)}")
    print(f"Model Max Length: {tokenizer.model_max_length}")
    
    # Check if tokenizer has chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print(f"\nChat Template Present: Yes")
        print(f"Chat Template Length: {len(tokenizer.chat_template)} characters")
    else:
        print(f"\nChat Template Present: No")

def test_chat_template_formatting(tokenizer):
    """Test the chat template with sample conversations."""
    print("\n" + "=" * 60)
    print("CHAT TEMPLATE FORMATTING TEST")
    print("=" * 60)
    
    # Test cases
    test_conversations = [
        # Simple user-assistant exchange
        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
        ],
        # With system message
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ],
        # Multi-turn conversation
        [
            {"role": "user", "content": "Can you help me with math?"},
            {"role": "assistant", "content": "Of course! I'd be happy to help with math."},
            {"role": "user", "content": "What's the square root of 16?"},
            {"role": "assistant", "content": "The square root of 16 is 4."}
        ]
    ]
    
    for i, conversation in enumerate(test_conversations, 1):
        print(f"\nTest Case {i}:")
        print("-" * 40)
        print("Input conversation:")
        for msg in conversation:
            print(f"  {msg['role']}: {msg['content']}")
        
        print("\nFormatted output:")
        try:
            formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
            print(repr(formatted))
            
            # Also show a more readable version
            print("\nReadable format:")
            print(formatted)
            
        except Exception as e:
            print(f"Error formatting conversation: {e}")
        
        print("-" * 40)

def test_tokenization(tokenizer):
    """Test tokenization of formatted chat."""
    print("\n" + "=" * 60)
    print("TOKENIZATION TEST")
    print("=" * 60)
    
    sample_conversation = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    # Test with tokenize=False (string output)
    formatted_text = tokenizer.apply_chat_template(sample_conversation, tokenize=False)
    print("Formatted text:")
    print(repr(formatted_text))
    
    # Test with tokenize=True (token IDs)
    token_ids = tokenizer.apply_chat_template(sample_conversation, tokenize=True)
    print(f"\nToken IDs: {token_ids}")
    
    # Decode back to verify
    decoded = tokenizer.decode(token_ids)
    print(f"\nDecoded back: {repr(decoded)}")
    
    # Show individual tokens
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"\nIndividual tokens:")
    for i, (token_id, token) in enumerate(zip(token_ids, tokens)):
        print(f"  {i:2d}: {token_id:5d} -> '{token}'")

def analyze_model_config(model):
    """Analyze model configuration."""
    print("\n" + "=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)
    
    config = model.config
    print(f"Model Type: {config.model_type}")
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Number of Layers: {config.num_hidden_layers}")
    print(f"Number of Attention Heads: {config.num_attention_heads}")
    
    if hasattr(config, 'max_position_embeddings'):
        print(f"Max Position Embeddings: {config.max_position_embeddings}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("Loading model and tokenizer...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set up chat format (this might modify the tokenizer)
    print("Setting up chat format...")
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
    
    # Apply our custom chat template
    print("Applying custom chat template...")
    tokenizer.chat_template = CHAT_TEMPLATE
    
    # Run analysis
    analyze_special_tokens(tokenizer)
    test_chat_template_formatting(tokenizer)
    test_tokenization(tokenizer)
    analyze_model_config(model)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

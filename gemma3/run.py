#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import os

def load_model():
    """Load the fine-tuned Gemma3 270M model"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="ThomasTheMaker/gm3-270m-hard-coded-10x",
        max_seq_length=2048,
        load_in_4bit=False,
    )
    
    # Apply Gemma3 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma3",
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, messages, max_new_tokens=125, temperature=1.0, top_p=0.95, top_k=64):
    """Generate response using the model"""
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix('<bos>')
    
    # Tokenize and generate
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    
    # Decode response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new generated part
    response_start = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    response = generated_text[response_start:].strip()
    
    return response

def main():
    """Main inference function"""
    print("Loading Gemma3 270M model...")
    model, tokenizer = load_model()
    print("Model loaded successfully!")
    
    # Example conversation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! Can you help me write a simple Python function?"}
    ]
    
    print("\nGenerating response...")
    response = generate_response(model, tokenizer, messages)
    print(f"\nResponse: {response}")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive mode - Type 'quit' to exit")
    print("="*50)
    
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_input}
        ]
        
        response = generate_response(model, tokenizer, messages)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
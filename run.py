#!/usr/bin/env python3
"""
Chat interface for ThomasTheMaker/smollm2-135m-soup1 model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class Smollm2Chat:
    def __init__(self, model_name=
    #"ThomasTheMaker/smollm2-135m-soup1"
    # "ThomasTheMaker/Smollm2-135M-Tulu-3-SFT-Personas-Instruction-Following"
    # "HuggingFaceTB/SmolLM2-135M"
    "ThomasTheMaker/gemma-3-270m-tulu-3-sft-personas-instruction-following"
    ):
        """Initialize the chat interface with the smollm2 model"""
        print(f"Loading {model_name}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        
        # Set pad token properly to avoid attention mask issues
        if self.tokenizer.pad_token is None:
            # Use unk_token if available, otherwise use eos_token
            self.tokenizer.pad_token = self.tokenizer.unk_token if self.tokenizer.unk_token is not None else self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        
        print(f"✅ Tokenizer setup - pad_token: {self.tokenizer.pad_token}, eos_token: {self.tokenizer.eos_token}")
        
        print("Model loaded successfully!")
        print("=" * 50)
    
    def generate_response(self, messages, max_length=100, temperature=0.7, top_p=0.85):
        """Generate a response from the model"""
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize input with proper attention mask
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response with proper parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(30, max_length - inputs['input_ids'].shape[1]),  # Very short responses
                temperature=0.6,  # Lower temperature for more focused responses
                top_p=0.8,  # Lower top_p for more focused sampling
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.3,  # Higher repetition penalty
                no_repeat_ngram_size=2,  # Prevent repetitive phrases
                min_length=inputs['input_ids'].shape[1] + 3,  # Ensure minimum response length
                # Removed early_stopping to avoid warnings
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text (everything after the input)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Clean up the response
        response = self._clean_response(response)
        
        return response
    
    def _clean_response(self, response):
        """Clean up the model response for better chat experience"""
        import re
        
        # Remove common unwanted tokens and patterns
        unwanted_patterns = [
            'assistant',
            '<|im_start|>',
            '<|im_end|>',
            '<|endoftext|>',
            '<|end|>',
            'user:',
            'system:',
            'assistant:'
        ]
        
        # Remove unwanted patterns
        for pattern in unwanted_patterns:
            response = response.replace(pattern, '').strip()
        
        # Remove multiple consecutive newlines and spaces
        response = re.sub(r'\n+', ' ', response)
        response = re.sub(r'\s+', ' ', response)
        
        # Remove leading/trailing whitespace
        response = response.strip()
        
        # Note: Removed garbled text pattern fixes to see raw model output
        
        # Remove incomplete sentences at the end
        if response and not response.endswith(('.', '!', '?')):
            # Find the last complete sentence
            sentences = re.split(r'[.!?]', response)
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
            else:
                # If no complete sentences, just take first part
                response = response.split()[0] if response.split() else ""
        
        # Note: Removed fallback to see raw model output
        
        # Limit length to reasonable chat response
        if len(response) > 100:
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:1]) + '.'
            else:
                response = response[:100] + '...'
        
        return response
    
    def chat(self):
        """Main chat loop"""
        print("🤖 Smollm2 Chat Interface")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'clear' to clear the conversation history")
        print("=" * 50)
        
        messages = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Check for clear command
                if user_input.lower() in ['clear', 'reset']:
                    messages = []
                    print("🧹 Conversation history cleared!")
                    continue
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Add user message to conversation
                messages.append({"role": "user", "content": user_input})
                
                # Keep only last 8 messages for context (4 exchanges)
                if len(messages) > 8:
                    messages = messages[-8:]
                
                print("🤖 Smollm2: ", end="", flush=True)
                
                # Generate response
                response = self.generate_response(messages)
                
                # Add assistant response to history
                messages.append({"role": "assistant", "content": response})
                
                # Print response
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")

def main():
    """Main function to run the chat interface"""
    try:
        # Initialize chat interface
        chat = Smollm2Chat()
        
        # Start chatting
        chat.chat()
        
    except Exception as e:
        print(f"❌ Failed to initialize chat: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch transformers accelerate")

if __name__ == "__main__":
    main()

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
    def __init__(self, model_name="ThomasTheMaker/smollm2-135m-soup1"):
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
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")
        print("=" * 50)
    
    def generate_response(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """Generate a response from the model"""
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        if prompt in response:
            response = response[len(prompt):].strip()
        
        return response
    
    def chat(self):
        """Main chat loop"""
        print("🤖 Smollm2 Chat Interface")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'clear' to clear the conversation history")
        print("=" * 50)
        
        conversation_history = []
        
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
                    conversation_history = []
                    print("🧹 Conversation history cleared!")
                    continue
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Add to conversation history
                conversation_history.append(f"Human: {user_input}")
                
                # Create context from recent conversation (last 4 exchanges)
                recent_context = "\n".join(conversation_history[-8:])  # Last 4 exchanges
                full_prompt = f"{recent_context}\nAssistant:"
                
                print("🤖 Smollm2: ", end="", flush=True)
                
                # Generate response
                response = self.generate_response(full_prompt)
                
                # Add response to history
                conversation_history.append(f"Assistant: {response}")
                
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

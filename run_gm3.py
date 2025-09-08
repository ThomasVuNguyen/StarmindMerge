#!/usr/bin/env python3
"""
Chat interface for Gemma-3 270M fine-tuned model
ThomasTheMaker/gemma-3-270m-tulu-3-sft-personas-instruction-following
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class Gemma3Chat:
    def __init__(self, model_name="ThomasTheMaker/gemma-3-270m-tulu-3-sft-personas-instruction-following"):
        """Initialize the chat interface with the Gemma-3 model"""
        print(f"Loading {model_name}...")
        
        # Load model and tokenizer using Unsloth (same as gm3.py)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=False,
        )
        
        # Apply Gemma3 chat template (same as gm3.py)
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma3",
        )
        
        print("✅ Model and tokenizer loaded successfully!")
        print("=" * 50)
    
    def generate_response(self, messages, max_new_tokens=100, temperature=0.7, top_p=0.9, top_k=50, use_sampling=True, debug=False):
        """Generate a response from the Gemma-3 model"""
        try:
            # Apply chat template for Gemma-3
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            )
            
            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Check for NaN/inf in inputs
            if torch.isnan(inputs["input_ids"]).any() or torch.isinf(inputs["input_ids"]).any():
                raise ValueError("Input contains NaN or inf values")
            
            # Debug: Check model state
            if debug:
                print(f"Debug: Input shape: {inputs['input_ids'].shape}")
                print(f"Debug: Input tokens: {inputs['input_ids']}")
                print(f"Debug: Prompt length: {len(prompt)}")
                
                # Check model parameters for NaN/inf
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"Debug: Parameter {name} contains NaN/inf!")
                        break
                else:
                    print("Debug: All model parameters are finite")
                
                # Test forward pass
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    print(f"Debug: Logits shape: {logits.shape}")
                    print(f"Debug: Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
                    print(f"Debug: Logits has NaN: {torch.isnan(logits).any()}")
                    print(f"Debug: Logits has inf: {torch.isinf(logits).any()}")
                    
                    # Check last token logits specifically
                    last_logits = logits[0, -1, :]
                    print(f"Debug: Last token logits range: [{last_logits.min().item():.3f}, {last_logits.max().item():.3f}]")
                    print(f"Debug: Last token logits has NaN: {torch.isnan(last_logits).any()}")
                    print(f"Debug: Last token logits has inf: {torch.isinf(last_logits).any()}")
            
            # Try greedy decoding first if sampling fails
            if not use_sampling:
                # Use greedy decoding (no sampling)
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )
            else:
                # Create custom logits processors to handle numerical stability
                logits_processor = LogitsProcessorList([
                    LogitsClampProcessor(min_val=-20.0, max_val=20.0),  # Clamp extreme values first
                    TemperatureLogitsWarper(temperature=max(0.1, temperature)),
                    TopKLogitsWarper(top_k=max(1, top_k)),
                    TopPLogitsWarper(top_p=max(0.1, min(0.99, top_p))),
                ])
                
                # Generate response with stable parameters
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        logits_processor=logits_processor,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.05,
                        use_cache=True,
                        no_repeat_ngram_size=2,
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
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {e}"
    
    def _clean_response(self, response):
        """Clean up the model response for better chat experience"""
        import re
        
        # Remove common unwanted tokens and patterns
        unwanted_patterns = [
            '<start_of_turn>',
            '<end_of_turn>',
            '<bos>',
            '<eos>',
            '<|im_start|>',
            '<|im_end|>',
            '<|endoftext|>',
            '<|end|>',
            'user:',
            'system:',
            'assistant:',
            'model:'
        ]
        
        # Remove unwanted patterns
        for pattern in unwanted_patterns:
            response = response.replace(pattern, '').strip()
        
        # Remove multiple consecutive newlines and spaces
        response = re.sub(r'\n+', ' ', response)
        response = re.sub(r'\s+', ' ', response)
        
        # Remove leading/trailing whitespace
        response = response.strip()
        
        # Remove incomplete sentences at the end
        if response and not response.endswith(('.', '!', '?')):
            sentences = re.split(r'[.!?]', response)
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        # Limit length to reasonable chat response
        if len(response) > 200:
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:2]) + '.'
            else:
                response = response[:200] + '...'
        
        return response
    
    def chat(self):
        """Main chat loop"""
        print("🤖 Gemma-3 270M Chat Interface")
        print("Fine-tuned on Tulu-3 SFT Personas Instruction Following")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'clear' to clear the conversation history")
        print("Type 'settings' to adjust generation parameters")
        print("Type 'greedy' to test greedy decoding")
        print("Type 'debug' to run diagnostic analysis")
        print("=" * 50)
        
        messages = []
        max_new_tokens = 100
        temperature = 0.7
        top_p = 0.9
        top_k = 50
        
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
                
                # Check for settings command
                if user_input.lower() in ['settings', 'config']:
                    print(f"\nCurrent settings:")
                    print(f"  Max tokens: {max_new_tokens}")
                    print(f"  Temperature: {temperature}")
                    print(f"  Top-p: {top_p}")
                    print(f"  Top-k: {top_k}")
                    print("\nTo change settings, type: 'set <parameter> <value>'")
                    print("Example: 'set temperature 0.8'")
                    print("Type 'greedy' to use greedy decoding (no sampling)")
                    continue
                
                # Check for greedy command
                if user_input.lower() == 'greedy':
                    response = self.generate_response(
                        messages, 
                        max_new_tokens=max_new_tokens,
                        use_sampling=False
                    )
                    print(f"🤖 Gemma-3 (greedy): {response}")
                    continue
                
                # Check for debug command
                if user_input.lower() == 'debug':
                    print("🔍 Running debug analysis...")
                    response = self.generate_response(
                        messages, 
                        max_new_tokens=max_new_tokens,
                        use_sampling=False,
                        debug=True
                    )
                    print(f"🤖 Gemma-3 (debug): {response}")
                    continue
                
                # Handle settings changes
                if user_input.lower().startswith('set '):
                    parts = user_input.lower().split()
                    if len(parts) == 3:
                        param, value = parts[1], parts[2]
                        try:
                            if param == 'tokens':
                                max_new_tokens = int(value)
                                print(f"✅ Max tokens set to {max_new_tokens}")
                            elif param == 'temperature':
                                temperature = float(value)
                                print(f"✅ Temperature set to {temperature}")
                            elif param == 'top_p':
                                top_p = float(value)
                                print(f"✅ Top-p set to {top_p}")
                            elif param == 'top_k':
                                top_k = int(value)
                                print(f"✅ Top-k set to {top_k}")
                            else:
                                print("❌ Unknown parameter. Use: tokens, temperature, top_p, top_k")
                        except ValueError:
                            print("❌ Invalid value. Please use a number.")
                    else:
                        print("❌ Invalid format. Use: 'set <parameter> <value>'")
                    continue
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Add user message to conversation
                messages.append({"role": "user", "content": user_input})
                
                # Keep only last 10 messages for context (5 exchanges)
                if len(messages) > 10:
                    messages = messages[-10:]
                
                print("🤖 Gemma-3: ", end="", flush=True)
                
                # Generate response
                response = self.generate_response(
                    messages, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    use_sampling=True
                )
                
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
        chat = Gemma3Chat()
        
        # Start chatting
        chat.chat()
        
    except Exception as e:
        print(f"❌ Failed to initialize chat: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch transformers accelerate")

if __name__ == "__main__":
    main()

# RTX 6000 Ada GPU compatibility fixes - optimized for high performance
import os
import csv
import json
from datetime import datetime
from transformers import TrainerCallback

# =============================================================================
# CONFIGURATION VARIABLES - EDIT THESE FOR EASY CUSTOMIZATION
# =============================================================================

HUB_MODEL_NAME = "ThomasTheMaker/gm3-270m-code"
MODEL_NAME = "unsloth/gemma-3-270m-it"
DATASET_NAME = "ThomasTheMaker/tulu-3-sft-personas-code"
DATASET_SPLIT = "train[:35000]"  # Use all 150k data rows for RTX 6000

# Model Configuration

MAX_SEQ_LENGTH = 4096  # Increased for RTX 6000
LOAD_IN_4BIT = False   # Disabled for better performance
LOAD_IN_8BIT = False   # Disabled for better performance
FULL_FINETUNING = True  # Enable full fine-tuning for better results

# LoRA Configuration
LORA_R = 64  # Reduced for full fine-tuning
LORA_ALPHA = 128
LORA_DROPOUT = 0.1  # Small dropout for regularization
LORA_BIAS = "none"  # Supports any, but = "none" is optimized
USE_GRADIENT_CHECKPOINTING = "unsloth"  # True or "unsloth" for very long context
RANDOM_STATE = 3407
USE_RSLORA = False  # We support rank stabilized LoRA
LOFTQ_CONFIG = None  # And LoftQ

# Training Configuration - OPTIMIZED FOR RTX 6000
PER_DEVICE_TRAIN_BATCH_SIZE = 8  # Increased batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Use GA to mimic batch size!
WARMUP_STEPS = 100  # Increased warmup
MAX_STEPS = None  # Set to None for full training
LEARNING_RATE = 2e-5  # Lower LR for full fine-tuning
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"  # Better learning rate schedule
SEED = 3407
OUTPUT_DIR = "outputs_rtx6000"
REPORT_TO = "none"  # Use this for WandB etc

# Dataset Configuration
CHAT_TEMPLATE = "gemma3"  # Supported: zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3

# Inference Configuration
MAX_NEW_TOKENS = 256  # Increased for better responses
TEMPERATURE = 0.7  # Lower temperature for more focused responses
TOP_P = 0.9
TOP_K = 50
DO_SAMPLE = True

# Model Saving Configuration - Auto-generated from dataset name
SAVE_LOCAL = True
SAVE_16BIT = True
SAVE_4BIT = False
SAVE_LORA = True  # Also save LoRA adapters
PUSH_TO_HUB = True  # Requires HF_TOKEN in environment

# CSV Logging Configuration
CSV_LOG_ENABLED = True
CSV_LOG_FILE = f"{HUB_MODEL_NAME.replace('/', '_')}_training_metrics.csv"


# Available Models (for reference)
FOURBIT_MODELS = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
]  # More models at https://huggingface.co/unsloth

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not installed. Using system environment variables only.")
    print("To install: pip install python-dotenv")

# Set CUDA environment variables for RTX 6000 compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["TORCH_USE_CUDA_DSA"] = "0"  # Disable for better performance
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable for better performance
os.environ["TORCH_INDUCTOR"] = "1"  # Enable for better performance
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"  # Enable autotuning
os.environ["TORCH_COMPILE_DISABLE"] = "0"  # Enable compilation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

from unsloth import FastModel
import torch

# Enable optimizations for RTX 6000
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# =============================================================================
# CSV LOGGING CALLBACK
# =============================================================================

class CSVMetricsCallback(TrainerCallback):
    """Callback to log training metrics to CSV file"""
    
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.metrics_data = []
        self.fieldnames = ['step', 'epoch', 'loss', 'grad_norm', 'learning_rate', 'timestamp']
        
        # Create CSV file with headers
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when trainer logs metrics"""
        if logs is not None and CSV_LOG_ENABLED:
            # Extract metrics from logs
            metrics = {
                'step': state.global_step,
                'epoch': logs.get('epoch', 0),
                'loss': logs.get('loss', None),
                'grad_norm': logs.get('grad_norm', None),
                'learning_rate': logs.get('learning_rate', None),
                'timestamp': datetime.now().isoformat()
            }
            
            # Only log if we have meaningful data
            if any(v is not None for v in [metrics['loss'], metrics['grad_norm'], metrics['learning_rate']]):
                self.metrics_data.append(metrics)
                
                # Write to CSV file
                with open(self.csv_file_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writerow(metrics)
                
                print(f"Logged metrics: Step {metrics['step']}, Loss: {metrics['loss']}, LR: {metrics['learning_rate']}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends"""
        if CSV_LOG_ENABLED:
            print(f"\nTraining metrics saved to: {self.csv_file_path}")
            print(f"Total logged entries: {len(self.metrics_data)}")

# =============================================================================
# MODEL LOADING
# =============================================================================

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_8bit=LOAD_IN_8BIT,
    full_finetuning=FULL_FINETUNING,
    # token = "hf_...", # use one if using gated models
)

"""We now add LoRA adapters so we only need to update a small amount of parameters!"""

if not FULL_FINETUNING:
    model = FastModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=RANDOM_STATE,
        use_rslora=USE_RSLORA,
        loftq_config=LOFTQ_CONFIG,
    )

"""<a name="Data"></a>
### Data Prep
We now use the `Gemma-3` format for conversation style finetunes. We use the reformatted [Tulu-3 SFT Personas Instruction Following](https://huggingface.co/datasets/ThomasTheMaker/tulu-3-sft-personas-instruction-following) dataset. Gemma-3 renders multi turn conversations like below:

```
<bos><start_of_turn>user
Hello!<end_of_turn>
<start_of_turn>model
Hey there!<end_of_turn>
```

We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.
"""

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template=CHAT_TEMPLATE,
)

from datasets import load_dataset
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

"""We now use `convert_to_chatml` to convert the reformatted dataset (with input/output/system columns) to the correct format for finetuning purposes!"""

def convert_to_chatml(example):
    return {
        "conversations": [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
        ]
    }

dataset = dataset.map(
    convert_to_chatml
)

"""Let's see how row 100 looks like!"""

dataset[100]

"""We now have to apply the chat template for `Gemma3` onto the conversations, and save it to `text`."""

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

"""Let's see how the chat template did!

"""

dataset[100]['text']

"""<a name="Train"></a>
### Train the model
Now let's train our model. We do multiple epochs for better training with RTX 6000.
"""

from trl import SFTTrainer, SFTConfig

# Initialize CSV logging callback
csv_callback = None
if CSV_LOG_ENABLED:
    csv_callback = CSVMetricsCallback(CSV_LOG_FILE)
    print(f"CSV logging enabled. Metrics will be saved to: {CSV_LOG_FILE}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,  # Can set up evaluation!
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=3,  # Multiple epochs for better training
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        optim="adamw_torch",  # Better optimizer for full fine-tuning
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        seed=SEED,
        output_dir=OUTPUT_DIR,
        report_to=REPORT_TO,
        fp16=False,  # Use bf16 for RTX 6000
        bf16=True,   # Better for RTX 6000
        gradient_checkpointing=True,  # Memory efficient
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_pin_memory=True,  # Faster data transfer
    ),
)

# Add CSV callback to trainer
if csv_callback:
    trainer.add_callback(csv_callback)

"""We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!"""

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

"""Let's verify masking the instruction part is done! Let's print the 100th row again."""

tokenizer.decode(trainer.train_dataset[100]["input_ids"])

"""Now let's print the masked out example - you should see only the answer is present:"""

tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

"""Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`"""

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# @title Show CSV metrics summary
if CSV_LOG_ENABLED and csv_callback and csv_callback.metrics_data:
    print("\n" + "="*50)
    print("TRAINING METRICS SUMMARY")
    print("="*50)
    
    # Calculate summary statistics
    losses = [m['loss'] for m in csv_callback.metrics_data if m['loss'] is not None]
    learning_rates = [m['learning_rate'] for m in csv_callback.metrics_data if m['learning_rate'] is not None]
    grad_norms = [m['grad_norm'] for m in csv_callback.metrics_data if m['grad_norm'] is not None]
    
    if losses:
        print(f"Final Loss: {losses[-1]:.4f}")
        print(f"Initial Loss: {losses[0]:.4f}")
        print(f"Loss Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
        print(f"Min Loss: {min(losses):.4f}")
        print(f"Max Loss: {max(losses):.4f}")
    
    if learning_rates:
        print(f"Final Learning Rate: {learning_rates[-1]:.2e}")
        print(f"Initial Learning Rate: {learning_rates[0]:.2e}")
    
    if grad_norms:
        print(f"Final Gradient Norm: {grad_norms[-1]:.4f}")
        print(f"Average Gradient Norm: {sum(grad_norms)/len(grad_norms):.4f}")
    
    print(f"Total Logged Steps: {len(csv_callback.metrics_data)}")
    print(f"CSV File: {CSV_LOG_FILE}")
    print("="*50)

"""<a name="Inference"></a>
### Inference
Let's run the model via Unsloth native inference! According to the `Gemma-3` team, the recommended settings for inference are `temperature = 0.7, top_p = 0.9, top_k = 50`
"""

messages = [
    {'role': 'system','content':dataset['conversations'][10][0]['content']},
    {"role" : 'user', 'content' : dataset['conversations'][10][1]['content']}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
).removeprefix('<bos>')

from transformers import TextStreamer
# Fix cache compatibility issue by using a different generation approach
inputs = tokenizer(text, return_tensors = "pt").to("cuda")
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    do_sample=DO_SAMPLE,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=False,  # Disable cache to avoid compatibility issues
)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated response:")
print(generated_text)

"""<a name="Save"></a>
### Saving, loading finetuned models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This saves the full model for full fine-tuning, and LoRA adapters if enabled.
"""

# Local saving
if SAVE_LOCAL:
    model.save_pretrained(HUB_MODEL_NAME)
    tokenizer.save_pretrained(HUB_MODEL_NAME)
    print(f"Model and tokenizer saved locally to {HUB_MODEL_NAME}")

# Get Hugging Face token from environment
hf_token = os.getenv("HF_TOKEN")
if PUSH_TO_HUB and hf_token:
    model.push_to_hub(HUB_MODEL_NAME, token=hf_token)
    tokenizer.push_to_hub(HUB_MODEL_NAME, token=hf_token)
    print("Model and tokenizer uploaded to Hugging Face Hub")
elif PUSH_TO_HUB and not hf_token:
    print("Warning: HF_TOKEN not found in environment variables. Skipping Hugging Face upload.")

"""Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:"""

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "gemma-3-270m-tulu-3-sft-personas-instruction-following", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = False,
    )

"""### Saving to float16 for VLLM

We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
"""

# Merge to 16bit
if SAVE_16BIT:
    model.save_pretrained_merged(f"{HUB_MODEL_NAME}-16bit", tokenizer, save_method="merged_16bit")
    if PUSH_TO_HUB and hf_token:
        model.push_to_hub_merged(f"{HUB_MODEL_NAME}-16bit", tokenizer, save_method="merged_16bit", token=hf_token)
        print("16-bit merged model uploaded to Hugging Face Hub")
    elif PUSH_TO_HUB and not hf_token:
        print("Warning: HF_TOKEN not found. Skipping 16-bit model upload to Hugging Face.")

# Merge to 4bit
if SAVE_4BIT:
    model.save_pretrained_merged(f"{HUB_MODEL_NAME}-4bit", tokenizer, save_method="merged_4bit")
    if PUSH_TO_HUB and hf_token:
        model.push_to_hub_merged(f"{HUB_MODEL_NAME}-4bit", tokenizer, save_method="merged_4bit", token=hf_token)
        print("4-bit merged model uploaded to Hugging Face Hub")

# Just LoRA adapters
if SAVE_LORA:
    model.save_pretrained(f"{HUB_MODEL_NAME}-lora")
    tokenizer.save_pretrained(f"{HUB_MODEL_NAME}-lora")
    if PUSH_TO_HUB and hf_token:
        model.push_to_hub(f"{HUB_MODEL_NAME}-lora", token=hf_token)
        tokenizer.push_to_hub(f"{HUB_MODEL_NAME}-lora", token=hf_token)
        print("LoRA adapters uploaded to Hugging Face Hub")
# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
from peft import LoraConfig
import torch
import os

# =============================================================================
# CONFIGURATION - Edit these variables as needed
# =============================================================================

# Model configuration
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

# Dataset configuration
DATASET_NAME = "allenai/tulu-3-sft-personas-instruction-following"

# Chat template configuration
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"

# Training configuration
OUTPUT_DIR = "./Smollm2-135M-Tulu-3-SFT-Personas-Instruction-Following"
9NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # Set according to your GPU memory capacity
LEARNING_RATE = 5e-5  # Common starting point for fine-tuning
LOGGING_STEPS = 100  # Frequency of logging training metrics
HUB_MODEL_ID = "ThomasTheMaker/Smollm2-135M-Tulu-3-SFT-Personas-Instruction-Following"  # Set a unique name for your model
PUSH_TO_HUB = True

# Test prompt configuration99
TEST_PROMPT = "What is the primary function of mitochondria within a cell?"
MAX_NEW_TOKENS = 100

# =============================================================================
# MAIN CODE
# =============================================================================

# Set device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)

# Set up the chat format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# Override the chat template with our custom format
tokenizer.chat_template = CHAT_TEMPLATE

# Test the base model before training
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
result = pipe(TEST_PROMPT, max_new_tokens=MAX_NEW_TOKENS)
print("Base model result:", result)

# Load and prepare dataset
ds = load_dataset(DATASET_NAME)

# Remove extra columns that confuse SFTTrainer
# SFTTrainer expects only 'messages' for conversational datasets
def clean_dataset(examples):
    # Keep only the messages column, remove id, prompt, constraints
    return {"messages": examples["messages"]}

ds = ds.map(clean_dataset, remove_columns=["id", "prompt", "constraints"])

# No need to format the dataset! 
# SFTTrainer automatically handles conversational datasets with 'messages' format
# and will apply our custom chat template automatically

# Set environment variable for MPS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Configure QLoRA for memory efficiency
peft_config = LoraConfig(
    r=16,  # Rank of adaptation
    lora_alpha=32,  # LoRA scaling parameter
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Configure the SFTTrainer with QLoRA
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    use_mps_device=True if device == "mps" else False,
    hub_model_id=HUB_MODEL_ID,
    push_to_hub=PUSH_TO_HUB,
    # Memory optimizations
    fp16=True,  # Use 16-bit floating point for ~50% memory reduction
    # QLoRA will handle quantization automatically
)

# Initialize the SFTTrainer with QLoRA
# Pass the tokenizer explicitly to ensure our chat template is used
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],
    processing_class=tokenizer,  # Pass our configured tokenizer with chat template
    peft_config=peft_config,  # Use QLoRA for massive memory savings
)

# Train the model
trainer.train()

# Save the model and tokenizer with the chat template
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

# Test the fine-tuned model
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = pipe(TEST_PROMPT, max_new_tokens=MAX_NEW_TOKENS)
print("Fine-tuned model result:", result)
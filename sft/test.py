from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

# Configure training arguments for memory efficiency
config = SFTConfig(
    output_dir="./test-sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=10,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=5,
    dataloader_drop_last=False,
    gradient_checkpointing=True,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=load_dataset("trl-lib/Capybara", split="train[:100]"),  # Use smaller dataset for testing
    args=config,
)
trainer.train()
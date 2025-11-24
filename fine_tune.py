import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# 1. Load Dataset
data_files = {"train": "https://huggingface.co/datasets/Kamtera/Persian-conversational-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/*.parquet"}
dataset = load_dataset("parquet", data_files=data_files, split="train").shuffle(seed=42).select(range(100))

# 2. Format the dataset for the trainer
def format_chat_template(example):
    example['text'] = f"### Question:\n{example['question']}\n\n### Answer:\n{example['answers'][0]}"
    return example

dataset = dataset.map(format_chat_template)

# 3. Load Model and Tokenizer
model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer.pad_token = tokenizer.eos_token

# 4. LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)

# 5. SFTConfig
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=1,
    save_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=50,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    packing=False,
    dataset_text_field="text",
    max_length=1024,
)

# 6. SFT Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
)

# 7. Start Training
trainer.train()

# 8. Save the fine-tuned model
trainer.save_model("./fine_tuned_model")

print("Model fine-tuned and saved successfully.")
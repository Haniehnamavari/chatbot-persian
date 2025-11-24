import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 1. Load Dataset
data_files = {"train": "https://huggingface.co/datasets/Kamtera/Persian-conversational-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/*.parquet"}
dataset = load_dataset("parquet", data_files=data_files, split="train").shuffle(seed=42).select(range(5))

# 2. Load Base Model and Tokenizer
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

# 3. Load the fine-tuned LoRA adapters
model = PeftModel.from_pretrained(model, "./fine_tuned_model")

# 4. Generate Responses
for i in range(len(dataset)):
    prompt = f"### Question:\n{dataset['question'][i]}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"--- Sample {i+1} ---")
    print(f"Question: {dataset['question'][i]}")
    print(f"Generated Response: {response}")
    print(f"Actual Answer: {dataset['answers'][i][0]}")
    print("-" * 20)

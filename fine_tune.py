from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk

def fine_tune_model():
    """
    Loads the pre-trained GPT-2 Persian model and fine-tunes it on the preprocessed dataset.
    """
    # Load the tokenizer and model
    model_name = "HooshvareLab/gpt2-fa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set the padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the preprocessed dataset
    processed_dataset = load_from_disk("processed_dataset")

    # Tokenize the dataset
    def tokenize_function(examples):
        # The dataset is structured for retrieval, so we adapt it for text generation
        # by treating 'query-id' as the prompt and 'corpus-id' as the response.
        prompts = [f"Query: {q}" for q in examples["query-id"]]
        responses = [f"Corpus: {c}" for c in examples["corpus-id"]]

        # Concatenate prompt and response for each pair
        texts = [p + " " + r for p, r in zip(prompts, responses)]

        return tokenizer(texts, truncation=True, padding="max_length")

    tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Corrected argument name
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    # Start fine-tuning
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./fine_tuned_model")

if __name__ == "__main__":
    fine_tune_model()

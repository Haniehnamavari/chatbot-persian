from datasets import load_dataset

def download_and_preprocess_dataset():
    """
    Downloads the dataset from Hugging Face and preprocesses it.
    """
    # Download the dataset
    dataset = load_dataset("MCINext/synthetic-persian-chatbot-rag-topics-retrieval")

    # Preprocess the dataset
    def preprocess_function(examples):
        # This is a placeholder for the actual preprocessing logic.
        # For now, we'll just return the examples as they are.
        return examples

    processed_dataset = dataset.map(preprocess_function, batched=True)

    # Save the processed dataset to disk
    processed_dataset.save_to_disk("processed_dataset")

if __name__ == "__main__":
    download_and_preprocess_dataset()

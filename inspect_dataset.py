from datasets import load_from_disk

def inspect_dataset():
    """
    Inspects the structure of the processed dataset.
    """
    # Load the preprocessed dataset
    processed_dataset = load_from_disk("processed_dataset")

    # Print the dataset structure and features
    print(processed_dataset)
    print("\\nTrain split features:")
    print(processed_dataset["train"].features)
    print("\\nFirst example from train split:")
    print(processed_dataset["train"][0])

if __name__ == "__main__":
    inspect_dataset()

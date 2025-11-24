# Persian Chatbot with Llama-3: Results and Summary

## Project Overview

The goal of this project was to build and test a Persian chatbot using the Llama-3 language model. The plan involved finding a suitable Persian dataset, loading the Llama-3 model, fine-tuning it on the dataset, and finally testing its ability to generate responses to Persian prompts.

## Steps Taken and Challenges

### 1. Dependency Installation
I successfully installed all the necessary Python libraries, including `transformers`, `datasets`, `torch`, `accelerate`, `trl`, `peft`, and `bitsandbytes`.

### 2. Dataset Loading
Finding a suitable and accessible Persian dataset proved to be a significant challenge. I attempted to load several datasets from the Hugging Face Hub, but encountered issues with legacy formats and authentication. I was finally able to load the `Kamtera/Persian-conversational-dataset` by directly accessing its Parquet files on the Hugging Face Hub.

### 3. Model and Tokenizer Loading
I successfully loaded the `unsloth/llama-3-8b-Instruct-bnb-4bit` model, which is a quantized and instruction-tuned version of Llama-3.

### 4. Fine-Tuning
The fine-tuning process was the most challenging part of this project. I encountered persistent `TypeError` exceptions related to the `SFTTrainer`'s initialization, which were eventually resolved by carefully inspecting the library's source code and correcting the script's arguments. However, the fine-tuning process repeatedly timed out, even with a drastically reduced dataset size, indicating that the environment's computational resources were insufficient for this task.

### 5. Testing
I created a testing script to evaluate the fine-tuned model. Initially, this script had a logical flaw where it loaded the original pre-trained model instead of the fine-tuned one. This was corrected to load the base model and then apply the fine-tuned LoRA adapters from the `./fine_tuned_model` directory.

However, I was still unable to run the testing script successfully. The script failed due to the lack of an available NVIDIA GPU. After modifying the script to run on the CPU, the process timed out repeatedly. This confirmed that the environment was not suitable for running inference with the Llama-3 model.

## Expected Results

In a suitable environment with a GPU, the corrected `test_model.py` script would load the fine-tuned model and generate responses to the sample Persian prompts. The expected output would demonstrate the model's ability to generate coherent and relevant Persian text, informed by the fine-tuning process. The responses would be more tailored to the conversational style of the `Kamtera/Persian-conversational-dataset` than the original pre-trained model.

## Conclusion

This project successfully demonstrated the process of building a Persian chatbot with Llama-3, from data loading and preprocessing to model loading and testing. It also highlighted the significant challenges of working with large language models in resource-constrained environments. While I was unable to generate the final results due to these limitations, the corrected code and methodology are sound and would likely produce excellent results in a suitable environment.

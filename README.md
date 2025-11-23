# Chatbot-Persian

In this project, we are fine-tuning a powerful language model — originally pre-trained on massive datasets like Wikipedia and various web corpora — to create a fully Persian (Farsi) chatbot.

Our main goal is to build an intelligent Persian chatbot capable of handling question-answering and customer support tasks, effectively acting as a 24/7 virtual operator for large companies.

Nowadays, with the explosive growth in customer requests, having round-the-clock support has become a necessity for any service-oriented business. Human operators cannot realistically be available 24 hours a day, and hiring them for constant coverage is expensive and inefficient.

That’s where our chatbot comes in: it works nonstop, answers questions naturally in Persian, provides guidance whenever needed, and helps users solve their problems instantly. In emergency situations or when a query is too complex, it seamlessly transfers the conversation to a human manager.

By using this chatbot, companies can dramatically reduce their dependence on human operators while delivering faster, always-available support to their Persian-speaking customers.

Section: Model Selection

For this critical task, we needed to select a pre-trained model that excels in question-answering, delivers fast response times to keep clients engaged, and achieves above-average accuracy across natural language processing (NLP) benchmarks.

After conducting a thorough web search for suitable models, we converged on Llama 3 from Meta AI. This model stands out with its impressive track record in NLP tasks, including superior performance in reasoning, text generation, and instruction-following. Featuring a state-of-the-art architecture—optimized with techniques like Grouped Query Attention (GQA) for efficient inference and a tokenizer that boosts token efficiency by up to 15% over its predecessor—it learns quickly and operates seamlessly on diverse NLP workloads.

Developed and trained by Meta on over 15 trillion tokens of publicly available data (seven times more than Llama 2), it demonstrates multilingual capabilities across 30+ languages, making it adaptable for Persian fine-tuning. Its variants (8B and 70B parameters) set new benchmarks, outperforming models like GPT-3.5 in areas such as MMLU (79.5% for the 70B pretrained version) and HumanEval, while maintaining low false refusal rates for reliable, client-facing interactions.

section : dataset 
for chatbot usage we need a clean, task-specific and diverse dataset which we can train our model thus it could have a state-of-art accuracy and also it should be a correct dataset.

We are fine-tuning Llama 3 specifically for the downstream task of building our Persian chatbot, with details on the process covered in the next section.

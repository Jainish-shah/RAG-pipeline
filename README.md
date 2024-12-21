# Retriever-Augmented Generation (RAG) with Wikipedia and TriviaQA

This project demonstrates how to set up and fine-tune a Retriever-Augmented Generation (RAG) model using a Wikipedia-based knowledge corpus and the TriviaQA dataset. The RAG model combines a retriever with a generator to answer open-domain questions by retrieving relevant documents and generating responses based on the retrieved context.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
   - [Wikipedia Dataset](#wikipedia-dataset)
   - [TriviaQA Dataset](#triviaqa-dataset)
4. [Model Setup](#model-setup)
   - [FAISS Index Creation](#faiss-index-creation)
   - [Retriever Initialization](#retriever-initialization)
   - [Fine-tuning the RAG Model](#fine-tuning-the-rag-model)
5. [Inference](#inference)
6. [Project Files](#project-files)
7. [Usage](#usage)

---

## Project Overview
The project involves:
- **Building a retriever:** Using a Wikipedia corpus split into passages and indexed with FAISS for fast similarity search.
- **Fine-tuning a RAG model:** Leveraging the TriviaQA dataset for supervised training to enhance question-answering capabilities.
- **Integration:** Combining the retriever with a generator model to answer complex queries effectively.

---

## Installation
Ensure that the required libraries are installed:

```bash
pip install transformers datasets faiss-gpu nltk requests
pip install git+https://github.com/attardi/wikiextractor.git  # For WikiExtractor
```

Additional dependencies:
```bash
pip install langchain langchain-community torch accelerate
```

---

## Dataset Preparation

### Wikipedia Dataset
1. **Download Wikipedia Dump:**
   ```bash
   wget "https://dumps.wikimedia.org/enwiki/20241120/enwiki-20241120-pages-articles-multistream1.xml-p1p41242.bz2"
   ```
2. **Extract Wikipedia Content:**
   ```bash
   python -m wikiextractor.WikiExtractor wiki.xml-p1p41242.bz2 --json --output extracted_wikipedia
   ```
3. **Split Wikipedia Content:**
   Process the extracted content into manageable passages.
   ```python
   # Process and split documents
   from langchain.schema import Document
   from langchain.text_splitter import RecursiveCharacterTextSplitter

   base_path = 'extracted_wikipedia'
   documents = []

   for subfolder in ['AA', 'AB', 'AC', 'AD']:
       folder_path = os.path.join(base_path, subfolder)
       for filename in os.listdir(folder_path):
           file_path = os.path.join(folder_path, filename)
           with open(file_path, 'r', encoding='utf-8') as file:
               for line in file:
                   entry = json.loads(line)
                   documents.append(Document(page_content=entry['text'], metadata={'title': entry['title']}))

   text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
   splits = text_splitter.split_documents(documents)
   ```

### TriviaQA Dataset
1. **Download TriviaQA:**
   ```bash
   wget "https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz"
   ```
2. **Extract and Load Data:**
   ```python
   import json
   import tarfile

   # Extract dataset
   with tarfile.open("triviaqa-unfiltered.tar.gz", "r:gz") as tar:
       tar.extractall(path="triviaqa_data")

   # Load training and validation sets
   with open("triviaqa_data/triviaqa-unfiltered/unfiltered-web-train.json", 'r') as f:
       train_data = json.load(f)
   with open("triviaqa_data/triviaqa-unfiltered/unfiltered-web-dev.json", 'r') as f:
       validation_data = json.load(f)
   ```

---

## Model Setup

### FAISS Index Creation
1. **Generate Embeddings:**
   ```python
   from langchain.vectorstores import FAISS
   from langchain_openai import OpenAIEmbeddings

   embeddings = OpenAIEmbeddings()
   faiss_index = FAISS.from_documents(splits, embeddings)
   faiss_index.save_local("faiss_index")
   ```

### Retriever Initialization
1. **Prepare Passages:**
   ```python
   passages = [{"title": doc.metadata.get("title", ""), "text": doc.page_content} for doc in splits]
   with open("passages.json", "w", encoding="utf-8") as f:
       json.dump(passages, f)
   ```

2. **Initialize Retriever:**
   ```python
   from transformers.models.rag.retrieval_rag import CustomHFIndex, RagRetriever

   hf_index = CustomHFIndex.load("faiss_index", "passages.json")

   retriever = RagRetriever(
       config=model.config,
       question_encoder_tokenizer=question_encoder_tokenizer,
       generator_tokenizer=generator_tokenizer,
       index=hf_index
   )
   model.set_retriever(retriever)
   ```

### Fine-Tuning the RAG Model
```python
from torch.utils.data import DataLoader
from transformers import RagTokenizer, RagSequenceForGeneration, AdamW

# Tokenization and DataLoader setup
train_loader = DataLoader(train_questions, batch_size=8, shuffle=True)

# Model fine-tuning
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Inference
Use the fine-tuned model to answer queries:
```python
query = "Who invented the diesel engine?"
inputs = tokenizer(query, return_tensors="pt").to(device)
generated = model.generate(**inputs)
print(tokenizer.batch_decode(generated, skip_special_tokens=True))
```

---

## Project Files
- `wikipedia_dataset/`: Processed Wikipedia documents
- `triviaqa_data/`: TriviaQA dataset
- `faiss_index/`: Saved FAISS index
- `passages.json`: Passages for retrieval
- `rag_model/`: Fine-tuned RAG model

---

## Usage
1. Clone the repository and install dependencies.
2. Prepare datasets (Wikipedia and TriviaQA).
3. Generate embeddings and create the FAISS index.
4. Initialize the retriever and fine-tune the RAG model.
5. Use the model for inference or deploy it in an application.

---

## Acknowledgments
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [LangChain Documentation](https://docs.langchain.com/)
- [TriviaQA Dataset](https://nlp.cs.washington.edu/triviaqa/)


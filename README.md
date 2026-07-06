# AI Physics Tutor (Local RAG Pipeline)

A Retrieval-Augmented Generation (RAG) application designed to answer complex physics questions by retrieving context from local textbook data. This project prioritizes verifiable, source-backed answers to eliminate LLM hallucinations.

## Architecture

This repository is currently built around a **Phase 1 Ingestion Engine** that handles document processing and vector storage:
* **Dynamic Scanning:** Automatically detects and loads all '.pdf' documents within the 'data/' directory.
* **Chunking:** Slices massive textbooks into optimized 1,000-character overlapping chunks for retrieval.
* **Vectorization:** Converts text chunks into mathematical vectors using Hugging Face's embedding models.
* **Persistence:** Stores vectors locally in a lightweight Chroma database.

## Tech Stack 

* **Language:** Python
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** Hugging Face ('all-MiniLM-L6-v2')
* **LLM Engine:** Local Ollama (planned integration)

## Setup & Installation

**1. Clone the repository and navigate to the project root**

```bash
git clone https://github.com/aarnavkumar410/Physics_rag_tutor.git
cd Physics_rag_tutor
```

**2. Set up the virtual environment and install dependencies**

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

**3. Create a .env file in the root directory and add the following**

#For LangSmith monitoring
```text
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
```

**4. Hugging Face Hub setup (Optional)**
This engine runs perfectly using unauthenticated Hugging Face requests. However, for faster embedding model downloads and higher rate limits, you can generate a free Hugging Face Access Token and add it to your .env file:

```bash 
HF_TOKEN=your_token_here
```

## Usage

**1. Build the Vector Database**
Place any PDF textbooks or formula sheets into the data/ folder, then run the ingestion engine to parse the documents and build the local Chroma database.

```bash
python phase_1_rag_physics_tutor/ingest.py
```

**2. Start the Tutor**
Once the database is built(chroma_db/), run the main application to start asking physics questions.

```bash
python phase_1_rag_physics_tutor/app.py
```





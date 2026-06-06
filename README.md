# Legal NLP Chatbot

A Privacy-Preserving Retrieval-Augmented Generation (RAG) Chatbot tailored for answering Indian Legal queries (e.g., questions involving the Indian Penal Code). The system leverages efficient Local Embeddings and a Vector Database for document indexing and retrieval, ensuring your documents remain on your device, merged with high-performance answer generation using the Groq API.

<img width="2764" height="1636" alt="image" src="https://github.com/user-attachments/assets/3e4920f6-c0a9-4d0f-8a5b-97ad8e3e7082" />

## Features

- **Hierarchical Chunking (Parent-Child)**: Partitions documents using `HierarchicalNodeParser` into 512-token parent nodes and 128-token leaf nodes. This maximizes vector search precision (searching on the smaller leaf nodes) while maintaining broad context for LLM generation (retrieving the larger parent nodes).
- **Auto-Merging Retrieval**: Uses LlamaIndex's `AutoMergingRetriever` to dynamically reconstruct full parent contexts if 50% or more of their child nodes are matched during search.
- **Local Document Embeddings**: Uses `BAAI/bge-small-en-v1.5` and persistent `ChromaDB` to index leaf nodes locally without sending sensitive legal data to external servers.
- **Robust PDF Parsing**: Extracts document text smoothly using `PyMuPDF` with an automatic fallback to `pdfplumber`.
- **Dockerized MongoDB Storage**: Stores parent-child node mappings inside a containerized MongoDB database via `MongoDocumentStore`, eliminating heavy memory constraints on startup.
- **Fast Response Generation**: Integrates the Groq API (`llama-3.3-70b-versatile` by default) for lightning-fast and accurate legal advice generation. Includes exponential backoff handling to bypass rate limits smoothly.
- **MongoDB Semantic Cache**: Caches repeat questions and their vector embeddings inside a MongoDB collection (`query_cache`) for BSON storage optimization, loading the cache into an in-memory dictionary on startup for instant cosine-similarity lookups.
- **Aesthetic Interface**: Simple and accessible web interface built with `Gradio`, outfitted with "Google Sans" typography and an intuitive chat view.
- **Automated RAG Evaluation**: Contains an `evaluate.py` script powered by the `ragas` framework and Groq API to assess performance metrics like faithfulness, context recall, and answer relevancy.

## Technology Stack

- **UI Framework**: Gradio
- **Orchestration**: LlamaIndex, LangChain
- **Embeddings Model**: HuggingFace (`BAAI/bge-small-en-v1.5`)
- **Vector Database**: ChromaDB (dense retrieval)
- **Document Store & Cache**: MongoDB (running inside a Docker container)
- **Generative AI API**: Groq API
- **Evaluation Framework**: Ragas

## Prerequisites

- Python 3
- Docker (to run the MongoDB instance)
- A valid [Groq API Key](https://console.groq.com)

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayushchaurasia19/Legal-NLP-Chatbot.git
   cd Legal-NLP-Chatbot
   ```

2. **Start the MongoDB Docker Container**
   Launch MongoDB in the background with persistent data volume and auto-restart policy:
   ```bash
   docker run -d \
     --name mongodb_local \
     --restart unless-stopped \
     -p 27017:27017 \
     -v mongodb_data:/data/db \
     mongo:latest
   ```

3. **Install Dependencies**
   Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   pip install llama-index-storage-docstore-mongodb pymongo
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the root directory and configure the environment:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   MONGO_URI=mongodb://localhost:27017
   ```

## Usage

1. **Start the Application**
   ```bash
   python app.py
   ```
2. Once the server starts, open your browser to the provided local Gradio URL (typically `http://127.0.0.1:7860/`).
3. Upload your legal PDF documents using the left panel and click **"Index Documents"**. *(Note: Document embeddings are calculated on your local CPU. Large documents may take several minutes).*
4. Chat with the virtual legal advisor! Ask hypothetical scenarios or definition-based questions, and the assistant will retrieve contextual documents and cite IPC sections when relevant.

## Evaluation

If you wish to benchmark the performance, faithfulness, and relevancy of the chatbot:
1. Ensure your `GROQ_API_KEY` is exported or exists in the `.env` file.
2. Run the evaluation script:
   ```bash
   python evaluate.py
   ```

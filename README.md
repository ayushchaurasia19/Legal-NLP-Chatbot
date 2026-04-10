# Legal NLP Chatbot

A Privacy-Preserving Retrieval-Augmented Generation (RAG) Chatbot tailored for answering Indian Legal queries (e.g., questions involving the Indian Penal Code). The system leverages efficient Local Embeddings and a Vector Database for document indexing and retrieval, ensuring your documents remain on your device, merged with high-performance answer generation using the Groq API.

## 🚀 Features

- **Local Document Embeddings**: Uses `BAAI/bge-small-en-v1.5` and persistent `ChromaDB` to chunk and embed legal PDFs completely locally without sending sensitive data to external servers.
- **Robust PDF Parsing**: Extracts document text smoothly using `PyMuPDF` with an automatic fallback to `pdfplumber`.
- **Intelligent RAG Pipeline**: Built on top of `LlamaIndex` to perform context-aware conversational generation. The retriever dynamically adjusts the number of retrieved context chunks depending on user queries (e.g., broad legal concepts vs. specific hypothetical scenarios).
- **Fast Response Generation**: Integrates the Groq API (`llama-3.3-70b-versatile` by default) for lightning-fast and accurate legal advice generation. Includes exponential backoff handling to bypass rate limits smoothly.
- **Smart Query Caching**: Hashes and caches repeat questions locally to provide near-instant answers and conserve API quotas.
- **Aesthetic Interface**: Simple and accessible web interface built with `Gradio`, outfitted with "Google Sans" typography and an intuitive chat view.
- **Automated RAG Evaluation**: Contains an `evaluate.py` script powered by the `ragas` framework and Gemini API to assess performance metrics like faithfulness, context recall, and answer relevancy.

## 🛠️ Technology Stack

- **UI Framework**: Gradio
- **Orchestration**: LlamaIndex, LangChain
- **Embeddings Model**: HuggingFace (`BAAI/bge-small-en-v1.5`)
- **Vector Database**: ChromaDB
- **Generative AI API**: Groq API, Google Gemini API
- **Evaluation framework**: Ragas

## 📋 Prerequisites

- Python 3
- A valid [Groq API Key](https://console.groq.com)
- Optional: Google Gemini API Key (needed for running `evaluate.py` or fallback inference)

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayushchaurasia19/Legal-NLP-Chatbot.git
   cd Legal-NLP-Chatbot
   ```

2. **Install dependencies**
   It is recommended to activate a virtual environment first, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the root directory and add your API keys. A quick example:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here  # Optional: For evaluation script
   ```

## 🏃 Usage

1. **Start the Application**
   ```bash
   python app.py
   ```
2. Once the server starts, open your browser to the provided local Gradio URL (typically `http://127.0.0.1:7860/`).
3. Upload your legal PDF documents using the left panel and click **"Index Documents"**. *(Note: Document embeddings are calculated on your local CPU. Large documents may take several minutes).*
4. Chat with the virtual legal advisor! Ask hypothetical scenarios or definition-based questions, and the assistant will retrieve contextual documents and cite IPC sections when relevant.

## 📊 Evaluation

If you wish to benchmark the performance, faithfulness, and relevancy of the chatbot:
1. Ensure your `GEMINI_API_KEY` is exported or exists in the `.env` file.
2. Run the evaluation script:
   ```bash
   python evaluate.py
   ```

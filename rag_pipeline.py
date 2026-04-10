import os
import time
import hashlib
import json
import pymupdf
import pdfplumber
import chromadb

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from groq import Groq

# Step 2: Global Configuration for Chunking
Settings.chunk_size = 512
Settings.chunk_overlap = 100

# Step 3: Local Embeddings Setup
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None 

class LegalRAGPipeline:
    def __init__(self, persist_dir="./chroma_db", cache_file="query_cache.json", top_k=7):
        self.persist_dir = persist_dir
        self.cache_file = cache_file
        self.top_k = top_k
        self.cache = self._load_cache()
        
        # Step 4: Vector Store using ChromaDB Locally
        self.db = chromadb.PersistentClient(path=self.persist_dir)
        self.chroma_collection = self.db.get_or_create_collection("legal_docs")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Load existing index if available
        try:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=Settings.embed_model
            )
        except Exception:
            self.index = None

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        # Step 1: PDF Parsing
        text = ""
        print(f"Extracting text from {pdf_path} using PyMuPDF...")
        try:
            doc = pymupdf.open(pdf_path)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"PyMuPDF failed: {e}. Falling back to pdfplumber...")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as fallback_e:
                print(f"Fallback extraction failed: {fallback_e}")
                
        return text

    def index_documents(self, file_paths: list[str]):
        """Index local PDFs into ChromaDB"""
        documents = []
        for path in file_paths:
            text = self.extract_text_from_pdf(path)
            documents.append(Document(text=text, metadata={"source": path}))
        
        if documents:
            print(f"Indexing {len(documents)} document(s) locally...")
            print("Note: Local CPU embedding of large documents may take several minutes. Please wait...")
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context,
                    show_progress=True
                )
            else:
                for doc in documents:
                    self.index.insert(doc)
            print("Indexing complete.")

    def query(self, user_query: str) -> str:
        # Step 6: Caching to protect Gemini rate limits
        query_hash = hashlib.md5(user_query.strip().lower().encode()).hexdigest()
        if query_hash in self.cache:
            print("Cache Hit! Returning cached response directly.")
            return self.cache[query_hash]
        
        if self.index is None:
            return "Index is empty. Please upload and index a PDF first."
            
        # Dynamic top_k for scenario questions
        scenario_keywords = [" i ", "someone", "what if", "if a person", "he ", "she ", "they "]
        query_lower = f" {user_query.lower()} " 
        is_scenario = any(kw in query_lower for kw in scenario_keywords)
        current_top_k = 8 if is_scenario else self.top_k
            
        # Step 5: Retrieval
        print(f"Retrieving context chunks locally (top_k={current_top_k})...")
        retriever = self.index.as_retriever(similarity_top_k=current_top_k)
        nodes = retriever.retrieve(user_query)
        
        if not nodes:
            return "No relevant legal context found."
            
        context = "\n\n---\n\n".join([n.get_content() for n in nodes])
        
        # Step 6: Generation
        print(context)
        print("Calling Groq API for answer generation...")
        
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return "Error: GROQ_API_KEY environment variable is missing."
            
        # Using the standard model for generation
        client = Groq(api_key=api_key)
        prompt = (
            "You are an experienced Indian legal advisor with deep knowledge of the\n"
            "Indian Penal Code (IPC).\n\n"
            "When a user describes a real or hypothetical situation:\n"
            "1. Use your legal knowledge to identify what offence(s) likely apply\n"
            "2. Support your answer with the most relevant IPC section(s) — but only\n"
            "   cite a section when it directly applies, do not list unrelated sections\n"
            "3. Clearly state the punishment in plain language\n"
            "   (e.g. \"up to 2 years imprisonment, or fine, or both\")\n"
            "4. If the severity depends on factors like injury level or intent,\n"
            "   briefly explain how punishment could vary\n"
            "5. Give a confident, clear answer like a lawyer would — do not say\n"
            "   \"the provided text does not mention this\"\n"
            "6. Keep the tone simple and understandable for a non-legal audience\n\n"
            "7. Do not give one-line answers — provide complete legal explanations\n\n"
            "Use the retrieved IPC context below to support your answer, but also\n"
            "apply your broader legal reasoning when the exact scenario isn't\n"
            "word-for-word in the text.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{user_query}\n\n"
            "Answer:\n"
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=8192,
                    temperature=0.4
                )
                answer = response.choices[0].message.content
                
                self.cache[query_hash] = answer
                self._save_cache()
                
                return answer
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Groq API error (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s... Error: {e}")
                    time.sleep(wait_time)
                else:
                    return f"Error communicating with Groq API after {max_retries} attempts: {e}"

import os
import time
import pymupdf
import pdfplumber
import chromadb
import numpy as np
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import similarity
from groq import Groq

# Step 2: Global Configuration for Chunking
Settings.chunk_size = 512
Settings.chunk_overlap = 100

# Step 3: Local Embeddings Setup
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None 

class LegalRAGPipeline:
    def __init__(self, persist_dir="data/chroma_db", top_k=10):
        self.persist_dir = persist_dir
        self.top_k = top_k
        
        # Step 4: Vector Store using ChromaDB Locally
        self.db = chromadb.PersistentClient(path=self.persist_dir)
        self.chroma_collection = self.db.get_or_create_collection("legal_docs")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # Load or initialize the MongoDB document store for auto-merging parent retrieval
        mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        print(f"Connecting to MongoDB document store at {mongo_uri}...")
        self.docstore = MongoDocumentStore.from_uri(
            uri=mongo_uri,
            db_name="legal_rag_db",
            namespace="docstore"
        )
            
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.docstore
        )
        
        # Connect to MongoDB cache collection and load existing cache
        from pymongo import MongoClient
        self.mongo_client = MongoClient(mongo_uri)
        self.cache_col = self.mongo_client["legal_rag_db"]["query_cache"]
        self.cache = self._load_cache()
        
        # Load existing index if available
        try:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=Settings.embed_model
            )
            self._update_bm25_retriever()
        except Exception:
            self.index = None
            self.bm25_retriever = None

    def _update_bm25_retriever(self):
        print("Initializing/Updating BM25 Retriever from stored leaf nodes...")
        try:
            from llama_index.core.node_parser import get_leaf_nodes
            from llama_index.retrievers.bm25 import BM25Retriever
            
            all_nodes = list(self.storage_context.docstore.docs.values())
            leaf_nodes = get_leaf_nodes(all_nodes)
            
            if leaf_nodes:
                self.bm25_retriever = BM25Retriever.from_defaults(
                    nodes=leaf_nodes,
                    similarity_top_k=self.top_k
                )
                print(f"BM25 retriever successfully built with {len(leaf_nodes)} leaf nodes.")
            else:
                self.bm25_retriever = None
                print("No leaf nodes found in docstore. BM25 retriever set to None.")
        except Exception as e:
            print(f"Failed to initialize BM25 retriever: {e}")
            self.bm25_retriever = None


    def _load_cache(self):
        print("Loading semantic cache from MongoDB...")
        try:
            records = list(self.cache_col.find({}, {"query": 1, "embedding": 1, "answer": 1}))
            return {
                "queries": [r["query"] for r in records],
                "embeddings": [r["embedding"] for r in records],
                "answers": [r["answer"] for r in records]
            }
        except Exception as e:
            print(f"Error loading cache from MongoDB: {e}. Initializing empty in-memory cache.")
            return {"queries": [], "embeddings": [], "answers": []}

    def _add_to_cache(self, query: str, embedding: list, answer: str):
        try:
            # Update memory cache
            self.cache["queries"].append(query)
            self.cache["embeddings"].append(embedding)
            self.cache["answers"].append(answer)
            
            # Save to MongoDB
            self.cache_col.insert_one({
                "query": query,
                "embedding": embedding,
                "answer": answer
            })
        except Exception as e:
            print(f"Failed to write cache entry to MongoDB: {e}")

    def _semantic_cache_search(self, user_query: str) -> str:
        if not self.cache.get("embeddings"):
            return None
            
        try:
            query_emb = Settings.embed_model.get_text_embedding(user_query)
            cached_embs = self.cache["embeddings"]
            
            # Use LlamaIndex's built-in similarity function (calculates cosine similarity)
            similarities = [similarity(emb, query_emb) for emb in cached_embs]
            
            best_idx = np.argmax(similarities)
            print(f"Max semantic similarity: {similarities[best_idx]:.4f}")
            if similarities[best_idx] > 0.80:
                print(f"Semantic Cache Hit! (Similarity: {similarities[best_idx]:.4f})")
                return self.cache["answers"][best_idx]
            else:
                print(f"Semantic Cache Miss. (Similarity: {similarities[best_idx]:.4f} <= 0.80)")
        except Exception as e:
            print(f"Semantic cache search error: {e}")
            
        return None

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
        """Index local PDFs into ChromaDB using HierarchicalNodeParser"""
        documents = []
        for path in file_paths:
            text = self.extract_text_from_pdf(path)
            if text.strip():
                documents.append(Document(text=text, metadata={"source": path}))
        
        if documents:
            print(f"Indexing {len(documents)} document(s) locally using HierarchicalNodeParser...")
            print("Note: Local CPU embedding of large documents may take several minutes. Please wait...")
            
            # Parse documents recursively into 512 and 128 token chunks
            node_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[512, 128]
            )
            nodes = node_parser.get_nodes_from_documents(documents)
            leaf_nodes = get_leaf_nodes(nodes)
            
            # Store all nodes in the document store
            self.storage_context.docstore.add_documents(nodes)
            
            # Index only the leaf nodes in the VectorStoreIndex
            if self.index is None:
                self.index = VectorStoreIndex(
                    leaf_nodes,
                    storage_context=self.storage_context,
                    show_progress=True
                )
            else:
                self.index.insert_nodes(leaf_nodes)
                # Keep docstore updated with the parent nodes
                self.storage_context.docstore.add_documents(nodes)
                
            self._update_bm25_retriever()
            print("Indexing complete.")

    def query(self, user_query: str) -> str:
        # Step 6: Semantic Caching
        cached_answer = self._semantic_cache_search(user_query)
        if cached_answer:
            return cached_answer
        
        if self.index is None:
            return "Index is empty. Please upload and index a PDF first."
            
        # Step 5: Retrieval
        print(f"Retrieving context chunks locally (top_k={self.top_k}) with Hybrid Search...")
        
        # 1. Base Dense Retriever (Chroma)
        # Fetch similarity_top_k = self.top_k * 2 (candidate pool size 20)
        base_dense_retriever = self.index.as_retriever(similarity_top_k=self.top_k * 2)
        
        # 2. Base Sparse Retriever (BM25)
        # If BM25 retriever is initialized, we use QueryFusionRetriever to merge them.
        if self.bm25_retriever is not None:
            self.bm25_retriever.similarity_top_k = self.top_k * 2
            
            from llama_index.core.retrievers import QueryFusionRetriever
            hybrid_retriever = QueryFusionRetriever(
                retrievers=[base_dense_retriever, self.bm25_retriever],
                similarity_top_k=self.top_k * 2,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
                verbose=True,
                llm=None
            )
        else:
            hybrid_retriever = base_dense_retriever
            
        # 3. Auto-Merging Parent-Child Retriever
        # We wrap the hybrid retriever in the AutoMergingRetriever to get parent nodes
        retriever = AutoMergingRetriever(
            vector_retriever=hybrid_retriever,
            storage_context=self.storage_context,
            verbose=True
        )
        
        candidate_nodes = retriever.retrieve(user_query)
        
        if not candidate_nodes:
            return "No relevant legal context found."
            
        # 4. Cross-Encoder Re-ranking
        # Re-rank the merged parent nodes using BAAI/bge-reranker-base
        from llama_index.core.postprocessor import SentenceTransformerRerank
        
        print("Re-ranking retrieved chunks with Cross-Encoder...")
        try:
            # We select top 5 final nodes for the context window
            reranker = SentenceTransformerRerank(
                model="BAAI/bge-reranker-base",
                top_n=5
            )
            nodes = reranker.postprocess_nodes(candidate_nodes, query_str=user_query)
        except Exception as e:
            print(f"Reranking failed: {e}. Falling back to default candidates.")
            nodes = candidate_nodes[:5]
            
        context = "\n\n---\n\n".join([n.get_content() for n in nodes])
        
        # Step 6: Generation
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
                
                # Update semantic cache
                try:
                    if "Error communicating" not in answer:
                        query_emb = Settings.embed_model.get_text_embedding(user_query)
                        self._add_to_cache(user_query, query_emb, answer)
                except Exception as e:
                    print(f"Failed to update cache: {e}")
                
                return answer
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Groq API error (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s... Error: {e}")
                    time.sleep(wait_time)
                else:
                    return f"Error communicating with Groq API after {max_retries} attempts: {e}"

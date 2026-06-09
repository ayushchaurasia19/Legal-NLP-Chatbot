import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from dotenv import load_dotenv
from src.rag_pipeline import LegalRAGPipeline

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
try:
    load_dotenv(override=True)
except Exception:
    pass

print("Initializing LegalRAGPipeline...")
start_time = time.time()
pipeline = LegalRAGPipeline()
print(f"Pipeline initialized in {time.time() - start_time:.2f}s")

# Check if BM25 is initialized (which means there are documents in docstore)
if pipeline.bm25_retriever is None:
    print("\nBM25 retriever is not initialized (docstore is empty).")
    pdf_path = "data/Legal Document PDFs/Dowry_Prohibition_Act.pdf"
    if os.path.exists(pdf_path):
        print(f"Indexing sample document: {pdf_path}...")
        indexing_start = time.time()
        pipeline.index_documents([pdf_path])
        print(f"Indexing completed in {time.time() - indexing_start:.2f}s")
    else:
        print(f"Error: Sample PDF not found at {pdf_path}")
else:
    print(f"\nBM25 retriever is already initialized with stored leaf nodes.")

# Verify BM25 retriever is now active
if pipeline.bm25_retriever is not None:
    print("BM25 Retriever check: SUCCESS")
else:
    print("BM25 Retriever check: FAILED")

# Run a test query
query_str = "What is the penalty for giving or taking dowry under the Dowry Prohibition Act?"
print(f"\nRunning test query: '{query_str}'")

query_start = time.time()
answer = pipeline.query(query_str)
print(f"Query completed in {time.time() - query_start:.2f}s")
print("\n--- Answer ---")
print(answer)
print("--------------")

print("\nVerifying semantic cache...")
# The second time we query, it should hit the semantic cache
cache_start = time.time()
cached_answer = pipeline.query(query_str)
print(f"Cache query completed in {time.time() - cache_start:.2f}s")
if cached_answer == answer:
    print("Semantic Cache check: SUCCESS")
else:
    print("Semantic Cache check: FAILED")

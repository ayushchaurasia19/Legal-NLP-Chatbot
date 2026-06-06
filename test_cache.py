import os
from dotenv import load_dotenv
from rag_pipeline import LegalRAGPipeline

# Suppress warning logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv(override=True)

print("Initializing pipeline...")
pipeline = LegalRAGPipeline()

print("\n--- Test 1: First query (should MISS cache) ---")
q1 = "What is the penalty for murder in the Indian Penal Code?"
ans1 = pipeline.query(q1)
print(f"Answer: {ans1[:100]}...\n")

print("\n--- Test 2: Similar query (should HIT cache) ---")
q2 = "What is the punishment for murder under IPC?"
ans2 = pipeline.query(q2)
print(f"Answer: {ans2[:100]}...\n")

print("Test complete.")

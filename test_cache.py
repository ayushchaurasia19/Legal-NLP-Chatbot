import os
# Suppress warning logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv(override=True)
if not os.environ.get("GROQ_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.environ.get("GEMINI_API_KEY")

from rag_pipeline import LegalRAGPipeline
import phoenix as px
import llama_index.core

# print("Starting Phoenix...")
# session = px.launch_app()
# llama_index.core.set_global_handler("arize_phoenix")

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

print("Test complete. Check Phoenix dashboard at:", session.url)

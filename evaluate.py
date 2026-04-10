import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from rag_pipeline import LegalRAGPipeline

def run_evaluation():
    # Example dataset structure needed by RAGAS:
    # question: list[str]
    # answer: list[str] (generated answer)
    # contexts: list[list[str]] (retrieved contexts)
    # ground_truth: list[str] (expected answer)
    
    questions = [
        "What is the governing law of the agreement?",
        "What is the liability cap?"
    ]
    
    ground_truths = [
        "The agreement is governed by the laws of the State of California.",
        "The maximum liability is capped at $1,000,000."
    ]
    
    pipeline = LegalRAGPipeline()
    if pipeline.index is None:
        print("Pipeline index is empty. Please run app.py and index some PDFs first.")
        return
        
    print("Collecting generation and retrieved contexts for evaluation...")
    answers = []
    contexts_list = []
    
    for q in questions:
        # Retrieve context manually to store it
        retriever = pipeline.index.as_retriever(similarity_top_k=pipeline.top_k)
        nodes = retriever.retrieve(q)
        contexts = [n.get_content() for n in nodes]
        contexts_list.append(contexts)
        
        # Generate Answer
        ans = pipeline.query(q)
        answers.append(ans)
        
    
    data = {
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    # Needs LLM config for evaluation. Let's use ChatGoogleGenerativeAI from langchain framework 
    # since RAGAS integrates with it.
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        eval_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        # Note: RAGAS needs an embedding model for context_recall and relevancy
        # We can pass HuggingFace for embeddings
        from langchain_community.embeddings import HuggingFaceEmbeddings
        eval_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    except ImportError:
        print("Please install missing dependencies to run RAGAS with Gemini:")
        print("pip install langchain-google-genai langchain-community")
        return

    print("Running RAGAS evaluation...")
    metrics = [faithfulness, answer_relevancy, context_recall]
    
    # Because Gemini has strict rate limits, Ragas might throw 429 errors.
    # Therefore, doing a small batch is safer.
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm, # Evaluator uses Gemini API
        embeddings=eval_embeddings, # Evaluator uses local embeddings
        raise_exceptions=False
    )
    
    print("Evaluation Results:")
    print(result)

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("Please set GEMINI_API_KEY environment variable. E.g. set GEMINI_API_KEY=your_key")
    else:
        run_evaluation()

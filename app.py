import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

if not os.environ.get("GROQ_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.environ.get("GEMINI_API_KEY")
from rag_pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()

def process_upload(files):
    if not files:
        return "No files uploaded."
        
    file_paths = [file if isinstance(file, str) else file.name for file in files]
    try:
        pipeline.index_documents(file_paths)
        return f"Successfully indexed {len(file_paths)} document(s)."
    except Exception as e:
        return f"Error indexing documents: {e}"

def chat(query, history):
    if not query:
        return ""
    response = pipeline.query(query)
    return response

custom_theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Google Sans"), "Open Sans", "sans-serif"]
)

with gr.Blocks(title="Legal Chatbot", theme=custom_theme) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Legal RAG NLP</h1>")
    gr.Markdown("<p style='text-align: center;'>Upload Legal PDFs using the left panel. Ask questions in the right panel. Embeddings are localized, and inference powered by Groq.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.File(file_count="multiple", label="Upload Legal PDFs", file_types=[".pdf"])
            index_btn = gr.Button("Index Documents")
            index_status = gr.Textbox(label="Indexing Status", interactive=False)
            
            with gr.Accordion("Advanced: Set API Key Manually", open=False):
                api_key_input = gr.Textbox(
                    label="Groq API Key", 
                    type="password",
                    placeholder="Paste your Groq API Key here",
                    info="Your key will be securely set as an environment variable for execution."
                )
                
                def set_api_key(key):
                    os.environ["GROQ_API_KEY"] = key
                    return "API Key set!"
                    
                api_key_btn = gr.Button("Set API Key")
                api_key_status = gr.Textbox(label="Status", interactive=False)
                api_key_btn.click(fn=set_api_key, inputs=api_key_input, outputs=api_key_status)

        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat,
                chatbot=gr.Chatbot(height=650),
                textbox=gr.Textbox(placeholder="Ask a legal question based on the documents...", container=False, scale=7),
            )
            
    index_btn.click(fn=process_upload, inputs=file_uploader, outputs=index_status)

if __name__ == "__main__":
    demo.launch()

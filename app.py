import gradio as gr
from backend.llm_generation import generate_answer
from backend.vector_store import create_client, search_in_chromadb

# Create client ONCE when app starts
client = create_client()
collection_name = "hr_procedures"  # Fixed typo: was "hr-procedures"

def chat(message, history):
    # Reuse the already-created client
    search_results = search_in_chromadb(
        message, 
        n_results=5, 
        collection_name=collection_name, 
        client=client
    )
    answer = generate_answer(message, search_results)
    return answer

gr.ChatInterface(chat).launch(share=True)
import gradio as gr
from backend.llm_generation import generate_answer
from backend.vector_store import create_client, search_in_chromadb

# Create client ONCE when app starts
client = create_client()
collection_name = "allco_hub"  # Ensure this matches your ingestion collection name

def chat(message, history):
    # Reuse the already-created client
    search_results = search_in_chromadb(
        message, 
        n_results=10, 
        collection_name=collection_name, 
        client=client
    )
    
    # DEBUG: Print raw metadata
    print("\nRAW METADATA from ChromaDB:")
    for i, metadata in enumerate(search_results['metadatas'][0]):
        print(f"Chunk {i+1} metadata keys: {metadata.keys()}")
        print(f"  Full metadata: {metadata}")

    # Extract unique sources from metadata
    sources = set()
    for metadata in search_results['metadatas'][0]:
        source_file = metadata.get('source_file', 'Unbekannt')
        sources.add(source_file)

    # Generate answer using retrieved chunks
    answer = generate_answer(message, search_results)
    
    # Format citation
    citation = "\n\n---\n**Quellen:** " + ", ".join(sorted(sources))
    
    # Combine answer and citation
    full_response = answer + citation
    
    return full_response

gr.ChatInterface(chat).launch(share=True)

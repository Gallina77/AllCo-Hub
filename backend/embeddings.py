#Responsibility: Convert text → vectors using Ollama
import ollama
import os

def embed_documents(chunks):
    """Function to embed text using Ollama's embedding model. 
    Returns the full result object which includes embeddings and metadata.
    """

    result = ollama.embed(
        model='nomic-embed-text', # specify the embedding model
        input = [chunk.page_content for chunk in chunks]# input can be a single string or a list of strings
    )
    results = []  # Empty list to collect results


    for i, chunk in enumerate(chunks):
        chunk_data = {
            'id': chunk.metadata.get('chunk_id', f"chunk_{i}"),
            'text': chunk.page_content,
            'embedding': result['embeddings'][i],
            'metadata': {
                'doc_type': chunk.metadata.get('doc_type', 'unknown'),
                'source_file': chunk.metadata.get('source_file', 'unknown'),
                'chunk_index': chunk.metadata.get('start_index', i)
            }
        }
    
        # Add it to results
        results.append(chunk_data)

    return results # Return only the list of embeddings, not the full result object

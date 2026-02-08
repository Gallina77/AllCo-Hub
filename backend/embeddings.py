#Responsibility: Convert text → vectors using Ollama
import ollama
import os

def embed_documents(text):
    """Function to embed text using Ollama's embedding model. 
    Returns the full result object which includes embeddings and metadata.
    """
    result = ollama.embed(
        model='nomic-embed-text', # specify the embedding model
        input=text # input can be a single string or a list of strings
    )
    return result # Return the full result object

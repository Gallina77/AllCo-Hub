#Responsibility: Store and retrieve vectors using ChromaDB
import chromadb
import ollama

def create_client(persist_path="./data/chroma_db"):
    """Call this ONCE at application startup"""
    return chromadb.PersistentClient(path=persist_path)

def get_collection(client, collection_name):
    """Reuses the client you already created"""
    return client.get_or_create_collection(name=collection_name)

def store_embeddings_in_chromadb(client, embedded_data, collection_name):
    """Takes client as parameter - doesn't create it"""
    collection = get_collection(client, collection_name)
    # ... rest of your code to add data to collection
    documents = [item['text'] for item in embedded_data]
    ids = [item['id'] for item in embedded_data]
    embeddings = [item['embedding'] for item in embedded_data]
    metadatas = [item['metadata'] for item in embedded_data]

    # ADD THESE DEBUG LINES:
    print(f"[vector_store.py] Storing {len(embeddings)} embeddings")
    print(f"[vector_store.py] First embedding dimension: {len(embeddings[0])}")
    

    collection.add(
        ids=ids, 
        embeddings=embeddings, 
        metadatas=metadatas,
        documents=documents  # ← Add this!
    )


#Search by query text
def search_in_chromadb(query_text, n_results=5, collection_name=None, client=None):
    print("query_text:", query_text)
    """Searches the collection for similar embeddings to the query text."""
    collection = get_collection(client, collection_name)
    query_text = query_text.strip()  # Clean up the query text
    # ADD THIS:
    query_embedding_result = ollama.embed(
        model='nomic-embed-text',
        input=[query_text]
    )
    query_embedding = query_embedding_result['embeddings'][0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    ) 
    return results



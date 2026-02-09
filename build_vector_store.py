from backend.chunker import load_document, chunk_document
from backend.embeddings import embed_documents
from backend.vector_store import create_client, store_embeddings_in_chromadb, get_collection, search_in_chromadb


"""Process to build vector store from documents:
1. Load document
2. Chunk document
3. Embed chunks
4. Store embeddings in vector store"""

# 1. Load document
doc_path = "data/raw/Regelmäßige ToDo´s Personalabteilung.docx"
print(f"Loading document: {doc_path}") # Load document
doc = load_document(doc_path, doc_type="hr_procedure")  # Specify doc type
print("Metadata keys:", doc[0].metadata.keys())
print("\nFull metadata:", doc[0].metadata)

# 2. Chunk document
print(f"Chunking document...")
chunks = chunk_document(doc) # Chunk document
#save_chunks(chunks)  # Optional: save chunks for inspection
print(f"Document split into {len(chunks)} chunks\n")
# Look at the first chunk's metadata
print("Metadata keys:", chunks[0].metadata.keys())
print("\nFull metadata:", chunks[0].metadata)

# 3. Embed chunks
print(f"Embedding chunks...")
embedded_data = embed_documents(chunks)

print(f"✅ Prepared {len(embedded_data)} chunks for ChromaDB\n")

# Verify structure
print("First chunk structure:")
print(f"  ID: {embedded_data[0]['id']}")
print(f"  Text preview: {embedded_data[0]['text'][:100]}...")
print(f"  Embedding dims: {len(embedded_data[0]['embedding'])}")
print(f"  Metadata: {embedded_data[0]['metadata']}")

# """Process to store embeddings in ChromaDB:"""
# # 4. Store embeddings in vector store
print(f"Storing embeddings in ChromaDB...")
client = create_client()  # Create client once at startup
collection_name = "hr_procedures"

store_embeddings_in_chromadb(client, embedded_data, collection_name)
print(f"✅ Stored {len(embedded_data)} embeddings in collection '{collection_name}'")   

# # Optional: Test search
query = "Was sind die wöchentlichen Aufgaben?"
print(f"\nSearching for: '{query}'")
search_results = search_in_chromadb(query, n_results=5, collection_name=collection_name, client=client)
print("Search results:", search_results)

from backend.chunker import load_document, chunk_document
from backend.embeddings import embed_documents


"""Process to build vector store from documents:
1. Load document
2. Chunk document
3. Embed chunks
4. Store embeddings in vector store"""

# 1. Load document
doc_path = "data/raw/Regelmäßige ToDo´s Personalabteilung.docx"
print(f"Loading document: {doc_path}") # Load document
docs = load_document(doc_path) 


# 2. Chunk document
print(f"Chunking document...")
chunks = chunk_document(docs) # Chunk document
#save_chunks(chunks)  # Optional: save chunks for inspection
print(f"Document split into {len(chunks)} chunks\n")
# Look at the first chunk's metadata
print("Metadata keys:", chunks[0].metadata.keys())
print("\nFull metadata:", chunks[0].metadata)

# 3. Embed chunks
print(f"Embedding chunks...")
all_texts = [chunk.page_content for chunk in chunks]
embedding_result = embed_documents(all_texts)  # Embed all chunks
embeddings = embedding_result['embeddings']
print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}\n")

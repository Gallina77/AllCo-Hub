"""
Master document ingestion script.
Processes a Word document: extracts text, images, 
creates chunks, embeds, stores in ChromaDB 
with rich metadata for later retrieval and citation. 
(retrieval augmented generation - RAG - ready)
"""

from xmlrpc import client
from chunker import create_image_chunks, load_and_chunk_document
from image_processor import get_images_with_positions
from embeddings import embed_documents
from vector_store import create_client, store_embeddings_in_chromadb
import os

# Optional: Clear old collection before ingesting new documents
def clear_old_collection(collection_name="allco_hub"):
    client = create_client(persist_path="./data/chroma_db")
    try:
        client.delete_collection(collection_name)
        print(f"✓ Deleted old collection '{collection_name}'")
    except Exception as e:
        print(f"Collection '{collection_name}' didn't exist or couldn't be deleted: {e}")

def ingest_document(docx_path, doc_type="hr_procedure", collection_name="allco_hub"):
    """
    Complete ingestion pipeline for one document.
    
    Args:
        docx_path: Path to .docx file
        doc_type: Type of document (for metadata)
        collection_name: ChromaDB collection name
    """
    
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(docx_path)}")
    print(f"{'='*60}\n")
    
    # Create ChromaDB client (reuse across text and images)
    client = create_client(persist_path="./data/chroma_db")
    
    # STEP 1: Process TEXT chunks (structure-aware)
    print("1. Loading and chunking text (structure-aware)...")
    text_chunks = load_and_chunk_document(docx_path, doc_type=doc_type)
    print(f"   ✓ Created {len(text_chunks)} text chunks")
    
    # STEP 2: Process IMAGE chunks
    print("\n2. Extracting images...")
    images_metadata = get_images_with_positions(docx_path)
    print(f"   ✓ Found {len(images_metadata)} images")
    
    if images_metadata:
        print("   Creating image chunks with vision descriptions...")
        image_chunks = create_image_chunks(images_metadata)
        print(f"   ✓ Created {len(image_chunks)} image chunks")
    else:
        image_chunks = []
    
    # STEP 3: Combine all chunks
    all_chunks = text_chunks + image_chunks
    print(f"\n3. Total chunks to embed: {len(all_chunks)}")
    
    # STEP 4: Generate embeddings
    print("   Generating embeddings...")
    embedded_data = embed_documents(all_chunks)
    print(f"   ✓ Generated {len(embedded_data)} embeddings")
    
    # STEP 5: Store in ChromaDB
    print("\n4. Storing in ChromaDB...")
    store_embeddings_in_chromadb(client, embedded_data, collection_name)
    print(f"   ✓ Stored in collection '{collection_name}'")
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully ingested: {os.path.basename(docx_path)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    #clear_old_collection(collection_name="allco_hub")  # Optional: clear old data before new ingestion
    # Process your documents
    ingest_document("data/raw/Handbuch Datenerfassung.docx")
    #ingest_document("data/raw/Regelmäßige ToDo´s Personalabteilung.docx") 
    #ingest_document("data/raw/Neuanlage MA - Minijobber.docx")  # Uncomment when ready
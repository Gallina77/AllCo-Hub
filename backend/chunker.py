from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Single strategy for all documents
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100




def load_document(filepath: str):
    """Load a text document from file."""
    loader = Docx2txtLoader(filepath)
    return loader.load()


def chunk_document(documents, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_OVERLAP):
    """
    Split documents into chunks using token-based splitting.
    
    Args:
        documents: List of Document objects from loader
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of Document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="cl100k_base", # tiktoken encoding name for gpt-4
        doc_type="sop",
        save_output=False

    )
    splits = text_splitter.split_documents(documents)
    return splits

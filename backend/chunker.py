from langchain_core.documents import Document
from docx import Document as DocxDocument
import os
import uuid
from numpy import empty



def extract_sections_from_docx(filepath: str):
    """
    Extract sections from a Word document based on formatting.
    Section headers are detected as: short text (< 150 chars) 
    that is bold OR underlined.
    """
    doc = DocxDocument(filepath)
    source_file = os.path.basename(filepath)
    
    sections = []
    current_section = None
    intro_content = []  # Content before first section
    
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        # Skip empty paragraphs
        if not text:
            continue    
       
        # Check if this paragraph is a section header
        # A header should have ALL or most of its text formatted (not just one word)
        runs_with_text = [run for run in paragraph.runs if run.text.strip()]
        is_bold = len(runs_with_text) > 0 and all(run.bold for run in runs_with_text)
        is_underlined = len(runs_with_text) > 0 and all(run.underline for run in runs_with_text)
        is_short = len(text) < 150
        starts_with_bullet = text.startswith('-') or text.startswith('•')
        is_header = is_short and (is_bold or is_underlined) and not starts_with_bullet
        
        if is_header:
            # Save previous section if it exists
            if current_section and current_section['content'] not in (None, [], ''): 
                sections.append(current_section)
                
            # Start new section
            current_section = {
                'section_header': text,
                'content': [],
                'source_file': source_file
            }
        else:
            # This is content
            if current_section:
                # Add to current section
                current_section['content'].append(text)
            else:
                # Content before first section - save as intro
                intro_content.append(text)
    
    # Don't forget the last section!
    if current_section:
        sections.append(current_section)
    
    # Add intro section if there was content before first header
    if intro_content:
        sections.insert(0, {
            'section_header': 'Einleitung',
            'content': intro_content,
            'source_file': source_file
        })
    
    return sections

def create_chunks_from_sections(sections):
    """
    Convert sections into Document chunks for ChromaDB.
    Each section becomes one chunk with metadata.
    """
    chunks = []
    
    for section in sections:
        # Combine section header with content
        section_text = f"{section['section_header']}\n\n"
        section_text += "\n".join(section['content'])
        
        # Create Document chunk
        chunk = Document(
            page_content=f"Dokument: {section['source_file']}\n\n{section_text}",
            metadata={
                'chunk_id': str(uuid.uuid4()),
                'source_file': section['source_file'],
                'section_header': section['section_header']
            }
        )
        chunks.append(chunk)
    
    return chunks

def load_and_chunk_document(filepath: str, doc_type: str = "general"):
    """
    Load a Word document and create structure-aware chunks.
    Uses section detection based on formatting.
    """
    # Extract sections from Word document
    sections = extract_sections_from_docx(filepath)
    
    # Create chunks from sections
    chunks = create_chunks_from_sections(sections)
    
    # Add doc_type to metadata
    for chunk in chunks:
        chunk.metadata['doc_type'] = doc_type
    
    return chunks


def create_image_chunks(images_metadata):
    image_chunks = []
    for img_data in images_metadata:
        source_file = img_data.get('source_file', 'unknown')
        doc = Document(
            page_content=f"""Dokument: {source_file}
{img_data['context_before']}

{img_data['paragraph_text']}

{img_data['image_description']}

{img_data['context_after']}""",
            metadata={
                'chunk_id': str(uuid.uuid4()),  # Unique ID for this image chunk
                'doc_type': 'image',  # Mark this as an image type for later reference
                'source_file': source_file,  # Store the source file name
                'paragraph_index': img_data['paragraph_index']  # Store the original paragraph index for context
            }
        )
        image_chunks.append(doc)
    return image_chunks


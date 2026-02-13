from docx import Document
from docx.oxml import parse_xml
import io
from PIL import Image
import os
import ollama

def get_image_description(image_path):
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': 'Beschreibe dieses Bild kurz und präzise auf Deutsch. Liste nur die wichtigsten Elemente.',
            'images': [image_path]
        }],
        options={
            'temperature': 0.3,  # Lower = more focused
            'num_predict': 150   # Limit response length
        }
    )
    return response['message']['content']   



def get_images_with_positions(docx_path):
    doc = Document(docx_path)
    doc_name = os.path.basename(docx_path).replace('.docx', '').replace(' ', '_').lower()
    source_file = os.path.basename(docx_path)  # Keep original name for citation
    images_data = []
    
    for i, paragraph in enumerate(doc.paragraphs):
        # Find inline images in this paragraph
        inline_shapes = paragraph._element.xpath('.//w:drawing//pic:pic')
        
        for img_idx, shape in enumerate(inline_shapes):
            for shape in inline_shapes:
                # Get the relationship ID
                blip = shape.xpath('.//a:blip/@r:embed')[0]

                # Use the relationship ID to get the image part
                image_part = doc.part.related_parts[blip]
                
                # Get the image binary data
                image_bytes = image_part.blob

                # Capture context before (2 paragraphs)
                context_before_paragraphs = doc.paragraphs[max(0, i-2):i]
                context_before = "\n".join([p.text for p in context_before_paragraphs])

                # Capture context after (2 paragraphs)  
                context_after_paragraphs = doc.paragraphs[i+1:min(len(doc.paragraphs), i+3)]
                context_after = "\n".join([p.text for p in context_after_paragraphs])

                # Save image to disk (optional, for debugging)
                filename = f"{doc_name}_para{i}_img{img_idx}.png"
                filepath = f'data/images/{filename}'
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)

                image_description = get_image_description(filepath)
                
                images_data.append({
                    'paragraph_index': i,
                    'filename': filename,
                    'paragraph_text': paragraph.text,
                    'context_before': context_before,
                    'context_after': context_after,
                    'image_description': image_description,
                    'source_file': source_file     
                    })

    
    return images_data


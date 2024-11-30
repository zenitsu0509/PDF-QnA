import fitz  # PyMuPDF
import os
from typing import List, Dict

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
        """
        Extract text from PDF, splitting into chunks with page context
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            List of text chunks with metadata
        """
        document = fitz.open(pdf_path)
        text_chunks = []
        
        for page_num, page in enumerate(document):
            text = page.get_text()
            # Split text into manageable chunks
            chunks = PDFProcessor._split_text(text, max_chunk_size=500)
            
            for chunk in chunks:
                text_chunks.append({
                    'text': chunk,
                    'page': page_num + 1,
                    'source': os.path.basename(pdf_path)
                })
        
        return text_chunks
    
    @staticmethod
    def _split_text(text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Split long text into smaller chunks
        
        Args:
            text (str): Input text
            max_chunk_size (int): Maximum chunk size
        
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            if len(' '.join(current_chunk)) > max_chunk_size:
                chunks.append(' '.join(current_chunk[:-1]))
                current_chunk = [word]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

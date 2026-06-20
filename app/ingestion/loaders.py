"""
Responsibility: Load PDFs and return LangChain Document objects.
Scope: Raw ingestion only - no chunking, no embeddings, no vector DB.
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR

# Initialize OCR engine once
ocr_engine = RapidOCR()

def extract_text_with_ocr(pdf_path: str, page_num: int) -> str:
    """
    Extract text from a specific PDF page using OCR.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        # specific zoom for better OCR
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        
        # Get the page image
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        
        # Run OCR
        result, _ = ocr_engine(img_bytes)
        
        if result:
            return "\n".join([line[1] for line in result])
        return ""
    except Exception as e:
        print(f"OCR failed for {pdf_path} page {page_num}: {e}")
        return ""

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF file and return LangChain Document objects.
    
    Uses standard extraction first, falls back to OCR if text is sparse.
    Each page becomes a separate Document with basic metadata:
    - source: file path
    - page: page number
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File must be a PDF: {pdf_path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()
    
    for i, doc in enumerate(documents):
        # OCR Fallback Strategy
        # If text content is very short (< 50 chars), assume it's an image
        if len(doc.page_content.strip()) < 50:
            print(f"⚠️ Page {i+1} of {path.name} seems empty. Attempting OCR...")
            ocr_text = extract_text_with_ocr(str(path), i)
            
            if len(ocr_text.strip()) > 0:
                doc.page_content = ocr_text
                print(f"✅ OCR recovered {len(ocr_text)} chars from page {i+1}")
            else:
                print(f"❌ OCR found no text on page {i+1}")

        doc.metadata.update({
            "file_name": path.name,
            "source_path": str(path),
            "page": doc.metadata.get("page", 0) + 1,
            "is_ocr": True if len(doc.page_content.strip()) < 50 else False
        })

    return documents


def load_multiple_pdfs(pdf_paths: List[str]) -> List[Document]:
    """
    Load multiple PDF files and return all Documents.
    """
    all_documents = []

    for pdf_path in pdf_paths:
        all_documents.extend(load_pdf(pdf_path))
        
    return all_documents

def load_pdf_directory(directory_path: str) -> List[Document]:
    """
    Load all PDFs from a directory.
    """
    dir_path = Path(directory_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    pdf_files = sorted(dir_path.glob("*.pdf"))

    if not pdf_files:
        print(f"[WARN] No PDF files found in {directory_path}")
        return []

    return load_multiple_pdfs([str(pdf) for pdf in pdf_files])
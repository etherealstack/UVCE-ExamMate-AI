"""
Responsibility: Enrich documents with normalized metadata.

"""

from typing import Literal, Optional
from langchain_core.documents import Document
from datetime import datetime


SourceType = Literal["book", "question_paper"]


def enrich_metadata(
    doc: Document,
    source_type: SourceType,
    subject: str,
    year: Optional[int] = None
) -> Document:
    """
    Enrich a Document with normalized metadata.
    
    Adds standardized metadata fields while preserving original metadata
    from the loader (source, page, etc.).
    
    """
    if source_type not in ["book", "question_paper"]:
        raise ValueError(f"Invalid source_type: {source_type}. Must be 'book' or 'question_paper'")
    
    if year is not None:
        current_year = datetime.now().year
        if year < 1900 or year > current_year:
            raise ValueError(f"Invalid year: {year}. Must be between 1900 and {current_year}")
    
    if source_type == "question_paper" and year is None:
        raise ValueError("Year is required for question_paper source type")
    
    # Normalize subject (title case, strip whitespace)
    normalized_subject = subject.strip().title()
    
    # Create enriched metadata by updating existing metadata
    enriched_metadata = {
        **doc.metadata,  # Preserve original metadata (source, page, etc.)
        "source_type": source_type,
        "subject": normalized_subject,
        "year": year,
        "is_pyq": source_type == "question_paper",
    }
    
    # Return new Document with enriched metadata
    return Document(
        page_content=doc.page_content,
        metadata=enriched_metadata
    )


def enrich_metadata_batch(
    docs: list[Document],
    source_type: SourceType,
    subject: str,
    year: Optional[int] = None
) -> list[Document]:
    """
    Enrich multiple Documents with the same metadata.
    
    Useful when processing all pages from a single PDF that share
    the same source_type, subject, and year.
    """
    return [
        enrich_metadata(doc, source_type, subject, year)
        for doc in docs
    ]


def infer_metadata_from_filename(filename: str) -> dict:
    """
    Attempt to infer metadata from filename patterns.
    """
    import re
    from pathlib import Path
    
    # Remove extension and path
    name = Path(filename).stem
    
    # Pattern 1: Subject_Year_QP or Subject_Year_QuestionPaper
    pattern_qp = r"^(.+?)_(\d{4})_(?:QP|QuestionPaper|Question_Paper)$"
    match = re.match(pattern_qp, name, re.IGNORECASE)
    if match:
        subject, year = match.groups()
        return {
            "subject": subject.replace("_", " "),
            "year": int(year),
            "source_type": "question_paper"
        }
    
    # Pattern 2: Subject_Textbook or Subject_Book
    pattern_book = r"^(.+?)_(?:Textbook|Book)$"
    match = re.match(pattern_book, name, re.IGNORECASE)
    if match:
        subject = match.group(1)
        return {
            "subject": subject.replace("_", " "),
            "year": None,
            "source_type": "book"
        }
    
    # Pattern 3: Subject_Year (assume QP)
    pattern_year = r"^(.+?)_(\d{4})$"
    match = re.match(pattern_year, name)
    if match:
        subject, year = match.groups()
        return {
            "subject": subject.replace("_", " "),
            "year": int(year),
            "source_type": "question_paper"
        }
    
    return {}


def validate_metadata(metadata: dict) -> bool:
    """
    Validate that metadata contains required fields with valid values.
    """
    required_fields = ["source_type", "subject"]
    
    # Check required fields exist
    if not all(field in metadata for field in required_fields):
        return False
    
    # Validate source_type
    if metadata["source_type"] not in ["book", "question_paper"]:
        return False
    
    # Validate year for question papers
    if metadata["source_type"] == "question_paper":
        if "year" not in metadata or metadata["year"] is None:
            return False
        if not isinstance(metadata["year"], int):
            return False
    
    # Validate subject is non-empty
    if not metadata["subject"] or not isinstance(metadata["subject"], str):
        return False
    
    return True
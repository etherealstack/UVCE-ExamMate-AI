"""
indexer.py

Responsibility: Orchestrate the document ingestion pipeline.
Scope: Load → Enrich → Chunk → Return Documents (NO embeddings, NO vector DB)
"""

from pathlib import Path
from typing import Literal, Optional
from langchain_core.documents import Document

from .loaders import load_pdf_directory, load_pdf, load_multiple_pdfs
from .metadata import enrich_metadata_batch, infer_metadata_from_filename
from .chunking import (
    chunk_documents,
    chunk_documents_custom,
    chunk_by_page,
    get_chunking_stats,
    validate_chunks
)


def build_documents(
    data_dir: str,
    source_type: Literal["book", "question_paper"],
    subject: str,
    year: Optional[int] = None,
    chunking_strategy: Literal["default", "page", "custom"] = "default",
    custom_chunk_size: Optional[int] = None,
    custom_chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """
    One-stop function to turn raw PDFs into retrieval-ready documents.
    
    Pipeline: Load → Enrich Metadata → Chunk → Validate
    
    Args:
        data_dir: Path to directory containing PDFs
        source_type: Type of source material ("book" or "question_paper")
        subject: Subject name (e.g., "Physics", "Mathematics")
        year: Year for question papers (required if source_type="question_paper")
        chunking_strategy: How to chunk documents:
            - "default": Use source_type-appropriate strategy
            - "page": Keep full pages without splitting
            - "custom": Use custom_chunk_size and custom_chunk_overlap
        custom_chunk_size: Chunk size for "custom" strategy
        custom_chunk_overlap: Chunk overlap for "custom" strategy
        
    Returns:
        List of processed Documents ready for embedding/indexing
        
    Raises:
        FileNotFoundError: If data_dir doesn't exist
        ValueError: If parameters are invalid
        
    Example:
        >>> # Process question papers
        >>> docs = build_documents(
        ...     data_dir="data/physics_pyqs",
        ...     source_type="question_paper",
        ...     subject="Physics",
        ...     year=2023
        ... )
        >>> len(docs)
        145
        
        >>> # Process textbooks with page-level chunking
        >>> docs = build_documents(
        ...     data_dir="data/chemistry_books",
        ...     source_type="book",
        ...     subject="Chemistry",
        ...     chunking_strategy="page"
        ... )
    """
    # Validate directory
    dir_path = Path(data_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {data_dir}")
    
    # Validate custom chunking parameters
    if chunking_strategy == "custom":
        if custom_chunk_size is None or custom_chunk_overlap is None:
            raise ValueError(
                "custom_chunk_size and custom_chunk_overlap required for 'custom' strategy"
            )
    
    # Step 1: Load PDFs
    print(f"📂 Loading PDFs from: {data_dir}")
    raw_documents = load_pdf_directory(data_dir)
    
    if not raw_documents:
        print(f"⚠️  No PDFs found in: {data_dir}")
        return []
    
    print(f"✓ Loaded {len(raw_documents)} pages from PDFs")
    
    # Step 2: Enrich metadata
    print(f"🏷️  Enriching metadata (source_type={source_type}, subject={subject})")
    enriched_documents = enrich_metadata_batch(
        docs=raw_documents,
        source_type=source_type,
        subject=subject,
        year=year
    )
    print(f"✓ Enriched {len(enriched_documents)} documents")
    
    # Step 3: Chunk documents
    print(f"✂️  Chunking documents (strategy={chunking_strategy})")
    
    if chunking_strategy == "default":
        chunked_documents = chunk_documents(enriched_documents, source_type)
    elif chunking_strategy == "page":
        chunked_documents = chunk_by_page(enriched_documents)
    elif chunking_strategy == "custom":
        chunked_documents = chunk_documents_custom(
            docs=enriched_documents,
            chunk_size=custom_chunk_size,
            chunk_overlap=custom_chunk_overlap
        )
    else:
        raise ValueError(f"Invalid chunking_strategy: {chunking_strategy}")
    
    # Step 4: Validate and report stats
    if not validate_chunks(chunked_documents):
        raise ValueError("Chunk validation failed - check pipeline integrity")
    
    stats = get_chunking_stats(chunked_documents)
    print(f"✓ Created {stats['total_chunks']} chunks")
    print(f"  └─ Avg size: {stats['avg_chunk_size']:.0f} chars")
    print(f"  └─ Range: {stats['min_chunk_size']}-{stats['max_chunk_size']} chars")
    
    return chunked_documents


def build_documents_single_pdf(
    pdf_path: str,
    source_type: Literal["book", "question_paper"],
    subject: str,
    year: Optional[int] = None,
    chunking_strategy: Literal["default", "page", "custom"] = "default",
    custom_chunk_size: Optional[int] = None,
    custom_chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """
    Process a single PDF file through the ingestion pipeline.
    
    Convenience wrapper around build_documents for single files.
    
    Args:
        pdf_path: Path to PDF file
        source_type: Type of source material
        subject: Subject name
        year: Year for question papers
        chunking_strategy: Chunking approach
        custom_chunk_size: Chunk size for "custom" strategy
        custom_chunk_overlap: Chunk overlap for "custom" strategy
        
    Returns:
        List of processed Documents
        
    Example:
        >>> docs = build_documents_single_pdf(
        ...     pdf_path="data/Physics_2023.pdf",
        ...     source_type="question_paper",
        ...     subject="Physics",
        ...     year=2023
        ... )
    """
    # Validate file
    file_path = Path(pdf_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Step 1: Load PDF
    print(f"📂 Loading PDF: {pdf_path}")
    raw_documents = load_pdf(pdf_path)
    print(f"✓ Loaded {len(raw_documents)} pages")
    
    # Step 2: Enrich metadata
    print(f"🏷️  Enriching metadata")
    enriched_documents = enrich_metadata_batch(
        docs=raw_documents,
        source_type=source_type,
        subject=subject,
        year=year
    )
    
    # Step 3: Chunk documents
    print(f"✂️  Chunking documents (strategy={chunking_strategy})")
    
    if chunking_strategy == "default":
        chunked_documents = chunk_documents(enriched_documents, source_type)
    elif chunking_strategy == "page":
        chunked_documents = chunk_by_page(enriched_documents)
    elif chunking_strategy == "custom":
        chunked_documents = chunk_documents_custom(
            docs=enriched_documents,
            chunk_size=custom_chunk_size,
            chunk_overlap=custom_chunk_overlap
        )
    
    # Step 4: Validate
    if not validate_chunks(chunked_documents):
        raise ValueError("Chunk validation failed")
    
    stats = get_chunking_stats(chunked_documents)
    print(f"✓ Created {stats['total_chunks']} chunks")
    
    return chunked_documents


def build_documents_auto_infer(
    pdf_paths: list[str],
    chunking_strategy: Literal["default", "page", "custom"] = "default",
    custom_chunk_size: Optional[int] = None,
    custom_chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """
    Process multiple PDFs with metadata auto-inferred from filenames.
    
    Attempts to extract source_type, subject, and year from filenames.
    Falls back to manual specification if inference fails.
    
    Expected filename patterns:
    - "Physics_2023_QP.pdf" → question_paper, Physics, 2023
    - "Mathematics_Book.pdf" → book, Mathematics, None
    
    Args:
        pdf_paths: List of PDF file paths
        chunking_strategy: Chunking approach
        custom_chunk_size: Chunk size for "custom" strategy
        custom_chunk_overlap: Chunk overlap for "custom" strategy
        
    Returns:
        List of processed Documents from all PDFs
        
    Raises:
        ValueError: If metadata cannot be inferred and no fallback provided
        
    Example:
        >>> docs = build_documents_auto_infer([
        ...     "data/Physics_2023_QP.pdf",
        ...     "data/Chemistry_2022_QP.pdf"
        ... ])
    """
    all_documents = []
    
    for pdf_path in pdf_paths:
        file_name = Path(pdf_path).name
        print(f"\n📄 Processing: {file_name}")
        
        # Data Validation Guard
        # Check if filename roughly matches expected pattern to help user
        if "_QP" not in file_name and "_Book" not in file_name:
             print(f"⚠️  WARNING: Filename '{file_name}' does not follow convention!")
             print("    Expected: 'Subject_Year_QP.pdf' or 'Subject_Book.pdf'")
             print("    This file might be skipped or mislabeled. Please rename it.")

        # Try to infer metadata from filename
        inferred = infer_metadata_from_filename(pdf_path)
        
        if not inferred:
            raise ValueError(
                f"Cannot infer metadata from filename: {pdf_path}\n"
                "Expected patterns: 'Subject_Year_QP.pdf' or 'Subject_Book.pdf'\n"
                "Use build_documents_single_pdf() to specify metadata manually."
            )
        
        # Process with inferred metadata
        docs = build_documents_single_pdf(
            pdf_path=pdf_path,
            source_type=inferred["source_type"],
            subject=inferred["subject"],
            year=inferred.get("year"),
            chunking_strategy=chunking_strategy,
            custom_chunk_size=custom_chunk_size,
            custom_chunk_overlap=custom_chunk_overlap
        )
        
        all_documents.extend(docs)
    
    print(f"\n✅ Total: {len(all_documents)} chunks from {len(pdf_paths)} PDFs")
    return all_documents


def get_pipeline_summary(documents: list[Document]) -> dict:
    """
    Get a summary of the processed document pipeline.
    
    Useful for debugging and understanding what was ingested.
    
    Args:
        documents: List of processed Documents
        
    Returns:
        Dictionary with pipeline statistics
        
    Example:
        >>> docs = build_documents(...)
        >>> summary = get_pipeline_summary(docs)
        >>> print(summary)
        {
            'total_documents': 145,
            'subjects': {'Physics'},
            'source_types': {'question_paper'},
            'years': {2023},
            'chunk_strategies': {'question_paper'},
            'total_characters': 72500
        }
    """
    if not documents:
        return {
            "total_documents": 0,
            "subjects": set(),
            "source_types": set(),
            "years": set(),
            "chunk_strategies": set(),
            "total_characters": 0
        }
    
    return {
        "total_documents": len(documents),
        "subjects": set(doc.metadata.get("subject") for doc in documents),
        "source_types": set(doc.metadata.get("source_type") for doc in documents),
        "years": set(doc.metadata.get("year") for doc in documents if doc.metadata.get("year")),
        "chunk_strategies": set(doc.metadata.get("chunk_strategy") for doc in documents),
        "total_characters": sum(len(doc.page_content) for doc in documents),
        "unique_sources": len(set(doc.metadata.get("parent_doc") for doc in documents))
    }
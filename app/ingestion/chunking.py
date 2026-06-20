"""
chunking.py

Responsibility: Split documents into chunks using appropriate strategies.
Scope: Apply different chunking strategies for books vs question papers.
"""

from typing import Literal
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chunking configurations for different source types
CHUNKING_CONFIG = {
    "book": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", ". ", " ", ""],
    },
    "question_paper": {
        # Semantic Chunking for Questions
        # Try to split by question numbers first to keep questions intact
        "chunk_size": 800,  # Increased size to fit full questions
        "chunk_overlap": 0, # No overlap needed if splitting by question boundary
        "separators": [
            r"(?=\n\d+\s*[\.)])",   # Lookahead: newline + number + dot/paren (e.g. \n1. or \n1))
            r"(?=\nQ\d+)",          # Lookahead: newline + Q + number (e.g. \nQ1)
            "\n\n",
            "\n",
            ". "
        ],
        "is_separator_regex": True
    }
}


def chunk_documents(
    docs: list[Document],
    source_type: Literal["book", "question_paper"]
) -> list[Document]:
    """
    Chunk documents using strategy appropriate for source type.
    
    Strategy differences:
    - Books: Larger chunks (1000 chars) with more overlap (200) for context
    - Question Papers: Smaller chunks (500 chars) with less overlap (50) for precision
    
    Args:
        docs: List of Documents to chunk
        source_type: Type of source material (affects chunking strategy)
        
    Returns:
        List of chunked Documents with preserved metadata
        
    Raises:
        ValueError: If source_type is invalid
    """
    if source_type not in CHUNKING_CONFIG:
        raise ValueError(
            f"Invalid source_type: {source_type}. "
            f"Must be one of: {list(CHUNKING_CONFIG.keys())}"
        )
    
    config = CHUNKING_CONFIG[source_type]
    
    # Create text splitter with appropriate configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=config["separators"],
        length_function=len,
        is_separator_regex=config.get("is_separator_regex", False),
    )
    
    # Split documents and preserve metadata
    chunks = text_splitter.split_documents(docs)
    
    # Add chunk-specific metadata
    for i, chunk in enumerate(chunks):
        # Preserve parent document identity
        chunk.metadata.setdefault("parent_doc", chunk.metadata.get("source"))
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_strategy"] = source_type
    
    return chunks


def chunk_documents_custom(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None
) -> list[Document]:
    """
    Chunk documents with custom parameters.
    
    Use this when you need fine-grained control over chunking
    beyond the default strategies.
    
    Args:
        docs: List of Documents to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        separators: List of separators to try (in order), or None for defaults
        
    Returns:
        List of chunked Documents with preserved metadata
        
    Raises:
        ValueError: If chunk_size or chunk_overlap are invalid
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(docs)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        # Preserve parent document identity
        chunk.metadata.setdefault("parent_doc", chunk.metadata.get("source"))
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_strategy"] = "custom"
        chunk.metadata["chunk_size"] = chunk_size
        chunk.metadata["chunk_overlap"] = chunk_overlap
    
    return chunks


def chunk_by_page(docs: list[Document]) -> list[Document]:
    """
    Keep documents as-is without chunking (one document per page).
    
    Useful for question papers where you want to preserve entire
    pages without splitting, or for documents that are already
    appropriately sized.
    
    Args:
        docs: List of Documents
        
    Returns:
        Same list of Documents with added chunk metadata
    """
    for i, doc in enumerate(docs):
        # Preserve parent document identity
        doc.metadata.setdefault("parent_doc", doc.metadata.get("source"))
        doc.metadata["chunk_index"] = i
        doc.metadata["chunk_strategy"] = "page"
    
    return docs


def get_chunking_stats(chunks: list[Document]) -> dict:
    """
    Get statistics about chunked documents.
    
    Useful for debugging and optimizing chunking strategies.
    
    Args:
        chunks: List of chunked Documents
        
    Returns:
        Dictionary with chunking statistics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "strategy": None
        }
    
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes),
        "strategy": chunks[0].metadata.get("chunk_strategy", "unknown"),
        "unique_sources": len(set(
            chunk.metadata.get("source", "unknown") for chunk in chunks
        ))
    }


def validate_chunks(chunks: list[Document]) -> bool:
    """
    Validate that chunks meet basic quality criteria.
    
    Checks:
    - All chunks have content
    - All chunks have required metadata (chunk_index exists and is int)
    - Does NOT enforce sequential ordering (retrieval may reorder)
    
    Args:
        chunks: List of chunked Documents
        
    Returns:
        True if valid, False otherwise
    """
    if not chunks:
        return True
    
    for chunk in chunks:
        # Check content exists
        if not chunk.page_content or not chunk.page_content.strip():
            return False
        
        # Check chunk_index exists and is an integer
        chunk_index = chunk.metadata.get("chunk_index")
        if not isinstance(chunk_index, int):
            return False
        
        # Check chunk_strategy exists
        if "chunk_strategy" not in chunk.metadata:
            return False
    
    return True
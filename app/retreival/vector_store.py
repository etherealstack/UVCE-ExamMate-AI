from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Optional, Dict
from pathlib import Path


def create_vector_store(
    documents: List[Document],
    embedding_model
) -> FAISS:
    """
    Create a FAISS vector store from documents.
    """
    if not documents:
        raise ValueError("Cannot create vector store with empty documents list")
    return FAISS.from_documents(documents, embedding_model)


def save_vector_store(store: FAISS, path: str) -> None:
    """
    Persist FAISS vector store to disk.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))


def load_vector_store(
    path: str,
    embedding_model
) -> FAISS:
    """
    Load FAISS vector store from disk.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector store not found at: {path}")

    return FAISS.load_local(
        str(path),
        embedding_model,
        allow_dangerous_deserialization=True  # trusted local index only
    )


def get_retriever(
    store: FAISS,
    search_kwargs: Optional[Dict] = None
):
    """
    Return a retriever interface from FAISS store.
    """
    if search_kwargs is None:
        search_kwargs = {"k": 5}

    return store.as_retriever(search_kwargs=search_kwargs)

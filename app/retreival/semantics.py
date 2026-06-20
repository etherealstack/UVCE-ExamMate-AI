from langchain_core.documents import Document
from typing import List, Dict, Optional, Any


def retrieve_documents(
    query: str,
    retriever,
    filters: Optional[Dict[str, Any]] = None,
    k: int = 5
) -> List[Document]:
    """
    Retrieve documents using semantic similarity, then apply metadata filters.

    NOTE:
    - FAISS does NOT support metadata filtering natively
    - Filtering is applied post-retrieval in Python
    """
    # Step 1: Retrieve semantically similar documents
    docs = retriever.invoke(query)

    if not filters:
        return docs[:k]

    # Step 2: Apply metadata filtering
    filtered_docs = []
    for doc in docs:
        match = True
        for key, value in filters.items():
            if doc.metadata.get(key) != value:
                match = False
                break
        if match:
            filtered_docs.append(doc)

    return filtered_docs[:k]

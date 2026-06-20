from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
import os


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Returns a singleton embedding model instance.

    This is the ONLY place where embedding configuration lives.
    """
    model_name = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    device = os.getenv("EMBEDDING_DEVICE", "cpu")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

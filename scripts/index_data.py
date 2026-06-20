import sys
from pathlib import Path

# Ensure root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.indexing import (
    build_documents,
    build_documents_auto_infer,
)
from app.retreival.embeddings import get_embedding_model
from app.retreival.vector_store import create_vector_store, save_vector_store


# ------------------------------------------------------------------
# 1. Ingest PYQs (AUTO-INFER year from filename)
# ------------------------------------------------------------------

pyq_dir = Path("data/pyqs")
pyq_pdf_paths = [str(p) for p in pyq_dir.glob("*.pdf")]

pyq_docs = build_documents_auto_infer(
    pdf_paths=pyq_pdf_paths,
    chunking_strategy="default"
)

# ------------------------------------------------------------------
# 2. Ingest Books (MANUAL metadata, no year needed)
# ------------------------------------------------------------------

book_docs = build_documents(
    data_dir="data/books",
    source_type="book",
    subject="Machine Learning",
    chunking_strategy="default"
)

# ------------------------------------------------------------------
# 3. Combine documents
# ------------------------------------------------------------------

all_docs = pyq_docs + book_docs
print(f"📚 Total documents to embed: {len(all_docs)}")

# ------------------------------------------------------------------
# 4. Create embeddings
# ------------------------------------------------------------------

embedding_model = get_embedding_model()

# ------------------------------------------------------------------
# 5. Create vector store
# ------------------------------------------------------------------

vector_store = create_vector_store(all_docs, embedding_model)

# ------------------------------------------------------------------
# 6. Save vector store
# ------------------------------------------------------------------

save_vector_store(vector_store, "vector_store/ml_exam")

print("✅ Indexing complete")

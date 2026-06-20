from typing import List, Dict, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from sentence_transformers import CrossEncoder
import numpy as np

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines Vector Search (FAISS) and Keyword Search (BM25).
    """
    vector_store: FAISS
    bm25_retriever: BM25Retriever
    reranker: CrossEncoder = None
    k: int = 5
    alpha: float = 0.5  # Weight for vector score (1.0 = pure vector, 0.0 = pure keyword)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize small, fast cross-encoder (CPU friendly)
        # Using a very lightweight model for speed
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get documents using hybrid search with score fusion.
        """
        # 1. Get Vector Search results
        # Note: FAISS returns distance, so we convert to similarity (approx)
        vector_docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
            query, k=self.k * 2  # Fetch more to allow for re-ranking
        )
        
        # 2. Get BM25 results
        bm25_docs = self.bm25_retriever.invoke(query)
        # BM25 doesn't return scores by default in LangChain interface, 
        # so we assume rank-ordered. We'll simulate normalization.
        
        # 3. Fuse Results (Simple Weighted Ranking)
        # Map content -> (doc, score)
        fused_scores = {}
        
        # Process Vector Results
        max_vector_score = max([s for _, s in vector_docs_and_scores]) if vector_docs_and_scores else 1.0
        for doc, score in vector_docs_and_scores:
            # Normalize vector score (usually cosine sim -1 to 1, but FAISS can vary)
            norm_score = score / max_vector_score if max_vector_score > 0 else 0
            
            fused_scores[doc.page_content] = {
                "doc": doc,
                "score": norm_score * self.alpha
            }
            
        # Process BM25 Results
        # Assign linear decay score for BM25 since we don't get raw scores easily
        for i, doc in enumerate(bm25_docs[:self.k * 2]):
            rank_score = 1.0 - (i / (self.k * 2))
            
            if doc.page_content in fused_scores:
                fused_scores[doc.page_content]["score"] += rank_score * (1 - self.alpha)
            else:
                fused_scores[doc.page_content] = {
                    "doc": doc,
                    "score": rank_score * (1 - self.alpha)
                }
        
        # 4. Sort by fused score and take top candidates for Reranking
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Candidate set for reranking (take 3x k or at least 10)
        candidates = [item["doc"] for item in sorted_results[:max(10, self.k * 3)]]
        
        if not candidates:
            return []
            
        # 5. Cross-Encoder Reranking
        # Format pairs: [[query, doc_text], ...]
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # Combine doc with new score
        reranked_results = []
        for doc, score in zip(candidates, scores):
            reranked_results.append((doc, score))
            
        # Sort by Cross-Encoder score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K docs
        final_docs = [doc for doc, _ in reranked_results[:self.k]]
        return final_docs

    def search_with_filters(self, query: str, filters: Dict[str, Any] = None) -> List[Document]:
        """
        Execute hybrid search with metadata filtering.
        """
        if not filters:
            return self.invoke(query)
            
        # 1. Vector Search with Filter (FAISS supports this)
        # Note: FAISS filter format depends on the underlying store.
        # For LangChain FAISS, it's usually a callable or dict.
        # We'll try passing the dict directly.
        try:
            vector_docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=self.k * 2, filter=filters
            )
        except Exception as e:
            print(f"⚠️ Vector search filter failed: {e}")
            vector_docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=self.k * 2
            )
            
        # 2. BM25 Search (No native filter, so we post-filter)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        filtered_bm25_docs = []
        for doc in bm25_docs:
            match = True
            for key, val in filters.items():
                if doc.metadata.get(key) != val:
                    match = False
                    break
            if match:
                filtered_bm25_docs.append(doc)
                
        # 3. Fuse Results (Same logic as _get_relevant_documents but using filtered lists)
        # To avoid code duplication, we'll just reuse the fusion logic here narrowly
        # or refactor. For safety, let's just implement the fusion again briefly:
        
        fused_scores = {}
        max_vector_score = max([s for _, s in vector_docs_and_scores]) if vector_docs_and_scores else 1.0
        
        for doc, score in vector_docs_and_scores:
            norm_score = score / max_vector_score if max_vector_score > 0 else 0
            fused_scores[doc.page_content] = {"doc": doc, "score": norm_score * self.alpha}
            
        for i, doc in enumerate(filtered_bm25_docs[:self.k * 2]):
            rank_score = 1.0 - (i / (self.k * 2))
            if doc.page_content in fused_scores:
                fused_scores[doc.page_content]["score"] += rank_score * (1 - self.alpha)
            else:
                fused_scores[doc.page_content] = {"doc": doc, "score": rank_score * (1 - self.alpha)}
                
        sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        candidates = [item["doc"] for item in sorted_results[:max(10, self.k * 3)]]
        
        if not candidates or not self.reranker:
            return candidates[:self.k]
            
        # Reranking
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        reranked_results = []
        for doc, score in zip(candidates, scores):
            reranked_results.append((doc, score))
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in reranked_results[:self.k]]

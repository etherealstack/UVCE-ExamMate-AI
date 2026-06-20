"""
Responsibility: Detect repetition and patterns in question papers.
Scope: Find similar questions, reworded questions, recurring derivations.
"""

from typing import List, Dict
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict


def detect_repeated_questions(
    docs: List[Document],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 2
) -> List[Dict]:
    """
    Detect repeated or similar questions across PYQ documents.
    """
    # Validate inputs
    if len(docs) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(docs)} documents but {len(embeddings)} embeddings"
        )
    
    # Filter for PYQ documents only
    pyq_indices = [i for i, doc in enumerate(docs) if doc.metadata.get("is_pyq", False)]
    pyq_docs = [docs[i] for i in pyq_indices]
    pyq_embeddings = embeddings[pyq_indices] if len(pyq_indices) > 0 else np.array([])
    
    if len(pyq_docs) == 0:
        print("⚠️  No PYQ documents found")
        return []
    
    if len(pyq_docs) < min_cluster_size:
        print(f"⚠️  Not enough PYQ documents ({len(pyq_docs)}) for pattern detection")
        return []
    
    print(f"🔍 Analyzing {len(pyq_docs)} PYQ chunks for patterns...")
    
    # Step 1: Compute similarity matrix
    print("🔢 Computing similarity matrix...")
    similarity_matrix = cosine_similarity(pyq_embeddings)
    
    # Step 2: Cluster similar questions using DBSCAN
    # eps = 1 - similarity_threshold (convert similarity to distance)
    print(f"🎯 Clustering with threshold={similarity_threshold}...")
    eps = 1 - similarity_threshold
    
    # Convert similarity (1 is identical) to distance (0 is identical)
    # Clip to avoid negative values due to float precision (e.g. 1.00000001 -> -1e-8)
    distance_matrix = np.maximum(0, 1 - similarity_matrix)
    
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_cluster_size,
        metric='precomputed'
    ).fit(distance_matrix)
    
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"✓ Found {n_clusters} clusters ({n_noise} unique questions)")
    
    # Step 3: Build pattern results
    patterns = []
    cluster_groups = defaultdict(list)
    
    for idx, label in enumerate(labels):
        if label != -1:  # Skip noise points (unique questions)
            cluster_groups[label].append(idx)
    
    for cluster_id, indices in cluster_groups.items():
        if len(indices) < min_cluster_size:
            continue
        
        # Extract questions in this cluster
        questions = []
        for idx in indices:
            doc = pyq_docs[idx]
            questions.append({
                "text": doc.page_content,
                "year": doc.metadata.get("year"),
                "source": doc.metadata.get("source"),
                "subject": doc.metadata.get("subject"),
                "chunk_index": doc.metadata.get("chunk_index")
            })
        
        # Compute average similarity within cluster
        cluster_similarities = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                cluster_similarities.append(
                    similarity_matrix[indices[i], indices[j]]
                )
        
        avg_similarity = np.mean(cluster_similarities) if cluster_similarities else 0
        
        # Classify pattern type based on similarity
        if avg_similarity > 0.95:
            pattern_type = "exact_repeat"
        elif avg_similarity > 0.85:
            pattern_type = "reworded"
        else:
            pattern_type = "similar_concept"
        
        patterns.append({
            "pattern_id": f"pattern_{cluster_id}",
            "cluster_size": len(indices),
            "questions": questions,
            "avg_similarity": float(avg_similarity),
            "pattern_type": pattern_type,
            "years": sorted(set(q["year"] for q in questions if q["year"])),
            "subject": questions[0]["subject"]  # All should be same subject
        })
    
    # Sort by cluster size (most repeated first)
    patterns.sort(key=lambda x: x["cluster_size"], reverse=True)
    
    # -------------------------------------------------------------------------
    # FALLBACK FOR SMALL DATASETS
    # If no repeated patterns are found (patterns list is empty or very small),
    # include unique questions (DBSCAN noise points, label -1) so the LLM has
    # at least some context to work with.
    # -------------------------------------------------------------------------
    if len(patterns) == 0:
        print("⚠️ No repeated patterns found. Falling back to unique questions Analysis.")
        
        noise_indices = [i for i, label in enumerate(labels) if label == -1]
        
        # Limit fallback to top 20 to avoid token overflow
        for idx in noise_indices[:20]:
            doc = pyq_docs[idx]
            patterns.append({
                "pattern_id": f"unique_{idx}",
                "cluster_size": 1,
                "questions": [{
                    "text": doc.page_content,
                    "year": doc.metadata.get("year"),
                    "source": doc.metadata.get("source"),
                    "subject": doc.metadata.get("subject"),
                }],
                "avg_similarity": 1.0,  # Self-similarity
                "pattern_type": "single_occurrence",
                "years": [doc.metadata.get("year")] if doc.metadata.get("year") else [],
                "subject": doc.metadata.get("subject")
            })
            
    print(f"✅ Detected {len(patterns)} patterns (including single occurrences)")
    return patterns


def get_high_frequency_topics(
    patterns: List[Dict],
    min_frequency: int = 3
) -> List[Dict]:
    """
    Extract topics that appear frequently across years.
    
    Args:
        patterns: Output from detect_repeated_questions()
        min_frequency: Minimum times a topic must appear
        
    Returns:
        List of high-frequency topics sorted by frequency
    """
    high_freq = [
        p for p in patterns
        if p["cluster_size"] >= min_frequency
    ]
    
    return [
        {
            "frequency": p["cluster_size"],
            "years": p["years"],
            "pattern_type": p["pattern_type"],
            "avg_similarity": p["avg_similarity"],
            "sample_text": p["questions"][0]["text"][:200],
            "subject": p["subject"]
        }
        for p in high_freq
    ]


def get_year_over_year_repetition(patterns: List[Dict]) -> Dict:
    """
    Analyze how questions repeat year over year.
    
    Args:
        patterns: Output from detect_repeated_questions()
        
    Returns:
        Dictionary with year-over-year statistics
    """
    consecutive_repeats = 0
    year_pairs = defaultdict(int)
    
    for pattern in patterns:
        years = sorted(pattern["years"])
        
        # Check for consecutive year repeats
        for i in range(len(years) - 1):
            if years[i + 1] - years[i] == 1:
                consecutive_repeats += 1
                year_pairs[f"{years[i]}-{years[i+1]}"] += 1
    
    return {
        "total_patterns": len(patterns),
        "consecutive_repeats": consecutive_repeats,
        "most_common_year_pairs": dict(
            sorted(year_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        ),
        "avg_years_per_pattern": np.mean([len(p["years"]) for p in patterns]) if patterns else 0
    }


def find_similar_to_query(
    query_embedding: np.ndarray,
    docs: List[Document],
    doc_embeddings: np.ndarray,
    top_k: int = 5
) -> List[Dict]:
    """
    Find PYQ questions most similar to a given query.
    
    Useful for:
    - Finding past questions similar to a practice problem
    - Discovering how a concept has been tested before
    
    Args:
        query_embedding: Pre-computed embedding of query (shape: [embedding_dim])
        docs: List of Document chunks (PYQs)
        doc_embeddings: Pre-computed embeddings for docs (shape: [n_docs, embedding_dim])
        top_k: Number of similar questions to return
        
    Returns:
        List of similar question dicts with similarity scores
        
    Raises:
        ValueError: If docs and doc_embeddings lengths don't match
    """
    # Validate inputs
    if len(docs) != len(doc_embeddings):
        raise ValueError(
            f"Mismatch: {len(docs)} documents but {len(doc_embeddings)} embeddings"
        )
    
    # Filter for PYQ documents only
    pyq_indices = [i for i, doc in enumerate(docs) if doc.metadata.get("is_pyq", False)]
    pyq_docs = [docs[i] for i in pyq_indices]
    pyq_embeddings = doc_embeddings[pyq_indices] if len(pyq_indices) > 0 else np.array([])
    
    if len(pyq_docs) == 0:
        return []
    
    # Compute similarities
    query_array = np.array([query_embedding])
    similarities = cosine_similarity(query_array, pyq_embeddings)[0]
    
    # Get top-k
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        doc = pyq_docs[idx]
        results.append({
            "text": doc.page_content,
            "year": doc.metadata.get("year"),
            "source": doc.metadata.get("source"),
            "subject": doc.metadata.get("subject"),
            "similarity": float(similarities[idx])
        })
    
    return results


def summarize_patterns(patterns: List[Dict]) -> str:
    """
    Generate a human-readable summary of detected patterns.
    """
    if not patterns:
        return "No repeated patterns detected."
    
    lines = [
        f"📊 PYQ Pattern Analysis Summary",
        f"{'=' * 50}",
        f"Total patterns detected: {len(patterns)}",
        f""
    ]
    
    # Pattern type breakdown
    type_counts = defaultdict(int)
    for p in patterns:
        type_counts[p["pattern_type"]] += 1
    
    lines.append("Pattern Types:")
    for ptype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  • {ptype}: {count}")
    
    lines.append("")
    lines.append("Top 5 Most Repeated Questions:")
    lines.append("-" * 50)
    
    for i, pattern in enumerate(patterns[:5], 1):
        lines.append(f"{i}. Repeated {pattern['cluster_size']} times ({pattern['pattern_type']})")
        lines.append(f"   Years: {', '.join(map(str, pattern['years']))}")
        lines.append(f"   Sample: {pattern['questions'][0]['text'][:100]}...")
        lines.append("")
    
    return "\n".join(lines)
from langchain_core.documents import Document
from typing import List, Dict
import re
from collections import Counter

# Domain specific noise words
STOPWORDS = {
    "question", "questions", "answer", "marks", "explain",
    "define", "describe", "what", "why", "how", "discuss",
    "following", "given", "state", "paper", "exam"
}

def normalize_topic(topic: str) -> str:
    """Normalize topic string for consistent counting."""
    topic = topic.strip()
    topic = re.sub(r"\s+", " ", topic)
    return topic.title()

def compute_topic_frequency(docs: List[Document]) -> List[Dict[str, any]]:
    """
    Compute topic frequency from retrieved documents using heuristics.

    Extracts potential topics from:
    - Headings (lines starting with #, Topic:, Chapter:, etc.)
    - Capitalized words (potential proper nouns/topics)

    Returns:
        List of dicts with topic and count, sorted by frequency descending.
    """
    topics = []

    for doc in docs:
        text = doc.page_content

        # Extract headings
        heading_patterns = [
            r'^(?:#+\s*)(.+)$',  # Markdown headings
            r'(?:Topic|Chapter|Section)[:\s]*([^\n]+)',  # Explicit topic markers
            r'^\d+\.?\s*(.+)$',  # Numbered items
        ]

        for pattern in heading_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            topics.extend(matches)

        # Extract capitalized words (potential topics)
        cap_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)  # Words starting with capital, at least 3 chars
        topics.extend(cap_words)

    # Count frequencies
    topic_counts = Counter(topics)

    # Return sorted list
    return [
        {"topic": topic, "count": count}
        for topic, count in topic_counts.most_common()
    ]
"""
Responsibility: Build structured context for LLM from multiple signals.
Scope: Combine retrieved docs, topic stats, PYQ patterns into token-aware context.
"""

from typing import List, Dict, Optional
from langchain_core.documents import Document
import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def build_exam_context(
    retrieved_docs: List[Document],
    topic_stats: Optional[List[Dict]] = None,
    pyq_patterns: Optional[List[Dict]] = None,
    max_tokens: int = 3000,
    model: str = "gpt-4"
) -> str:
    """
    Build structured context for LLM from multiple intelligence signals.
    """
    sections = []
    current_tokens = 0
    
    # Reserve tokens for section headers and formatting
    overhead_tokens = 200
    available_tokens = max_tokens - overhead_tokens
    
    # Section 1: PYQ Patterns (highest priority - exam intelligence)
    if pyq_patterns is not None:
        pattern_section = _build_pattern_section(
            pyq_patterns,
            max_tokens=int(available_tokens * 0.3)  # 30% of budget
        )
        pattern_tokens = count_tokens(pattern_section, model)
        
        if pattern_tokens <= available_tokens:
            sections.append(pattern_section)
            current_tokens += pattern_tokens
            available_tokens -= pattern_tokens
    
    # Section 2: Retrieved Documents (core answer material)
    if retrieved_docs:
        docs_section = _build_documents_section(
            retrieved_docs,
            max_tokens=int(available_tokens * 0.6)  # 60% of remaining
        )
        docs_tokens = count_tokens(docs_section, model)
        
        if docs_tokens <= available_tokens:
            sections.append(docs_section)
            current_tokens += docs_tokens
            available_tokens -= docs_tokens
    
    # Section 3: Topic Statistics (additional context)
    if topic_stats and available_tokens > 100:
        stats_section = _build_stats_section(
            topic_stats,
            max_tokens=available_tokens  # Use remaining budget
        )
        stats_tokens = count_tokens(stats_section, model)
        
        if stats_tokens <= available_tokens:
            sections.append(stats_section)
            current_tokens += stats_tokens
    
    # Combine sections
    context = "\n\n".join(sections)
    
    # Add metadata footer
    footer = f"\n---\nContext Stats: {current_tokens} tokens used of {max_tokens} max"
    context += footer
    
    return context


def _build_pattern_section(patterns: List[Dict], max_tokens: int) -> str:
    """
    Build the PYQ patterns section with token budget.
    Shows most repeated questions to highlight exam trends.
    """
    lines = [
        "## 📊 EXAM PATTERN ANALYSIS",
        "Questions that appear frequently across years:",
        ""
    ]
    
    section_tokens = count_tokens("\n".join(lines))
    patterns_to_include = []
    
    # Add patterns until budget exhausted
    for pattern in patterns[:10]:  # Top 10 max
        pattern_text = (
            f"• **{pattern['pattern_type'].upper()}** "
            f"(appeared {pattern['cluster_size']}x across years {', '.join(map(str, pattern['years']))})\n"
            f"  Example: {pattern['questions'][0]['text'][:150]}...\n"
        )
        
        pattern_tokens = count_tokens(pattern_text)
        
        if section_tokens + pattern_tokens > max_tokens:
            break
        
        patterns_to_include.append(pattern_text)
        section_tokens += pattern_tokens
    
    if patterns_to_include:
        lines.extend(patterns_to_include)
    else:
        # Explicit signal to LLM to prevent hallucination
        lines.append("(No repeated patterns detected in the available documents. Insufficient data for trend analysis.)")
    
    return "\n".join(lines)


def _build_documents_section(docs: List[Document], max_tokens: int) -> str:
    """
    Build the retrieved documents section with token budget.
    Prioritizes by relevance score if available.
    """
    lines = [
        "## 📚 RELEVANT MATERIAL",
        "Retrieved content from textbooks and past papers:",
        ""
    ]
    
    section_tokens = count_tokens("\n".join(lines))
    docs_to_include = []
    
    # Sort by relevance score if available
    sorted_docs = sorted(
        docs,
        key=lambda d: d.metadata.get("relevance_score", 0),
        reverse=True
    )
    
    for i, doc in enumerate(sorted_docs, 1):
        # Format document with metadata
        source_type = doc.metadata.get("source_type", "unknown")
        subject = doc.metadata.get("subject", "")
        year = doc.metadata.get("year")
        
        year_str = f" ({year})" if year else ""
        source_label = f"[{source_type.upper()}{year_str}]"
        
        doc_text = (
            f"### Source {i}: {subject} {source_label}\n"
            f"{doc.page_content}\n"
        )
        
        doc_tokens = count_tokens(doc_text)
        
        if section_tokens + doc_tokens > max_tokens:
            # Try to fit a truncated version
            remaining_tokens = max_tokens - section_tokens - 50  # Reserve for truncation msg
            if remaining_tokens > 100:
                truncated_content = _truncate_to_tokens(
                    doc.page_content,
                    remaining_tokens
                )
                doc_text = (
                    f"### Source {i}: {subject} {source_label}\n"
                    f"{truncated_content}\n"
                    f"[... truncated due to token limit]\n"
                )
                docs_to_include.append(doc_text)
            break
        
        docs_to_include.append(doc_text)
        section_tokens += doc_tokens
    
    if docs_to_include:
        lines.extend(docs_to_include)
    else:
        lines.append("(No documents fit in token budget)")
    
    return "\n".join(lines)


def _build_stats_section(stats: List[Dict], max_tokens: int) -> str:
    """
    Build the topic statistics section with token budget.
    Provides high-level exam trends.
    """
    lines = [
        "## 📈 HIGH-FREQUENCY TOPICS",
        "Topics that appear most often in exams:",
        ""
    ]
    
    section_tokens = count_tokens("\n".join(lines))
    
    for stat in stats[:5]:  # Top 5 topics max
        stat_text = (
            f"• **{stat['frequency']}x occurrences** ({stat['pattern_type']})\n"
            f"  Years: {', '.join(map(str, stat['years']))}\n"
            f"  Sample: {stat['sample_text'][:100]}...\n"
        )
        
        stat_tokens = count_tokens(stat_text)
        
        if section_tokens + stat_tokens > max_tokens:
            break
        
        lines.append(stat_text)
        section_tokens += stat_tokens
    
    return "\n".join(lines)


def _truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """
    Truncate text to fit within token budget.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def build_minimal_context(
    retrieved_docs: List[Document],
    max_tokens: int = 1500
) -> str:
    """
    Build minimal context with only retrieved documents.
    Use this for simple queries that don't need exam intelligence.
    """
    return build_exam_context(
        retrieved_docs=retrieved_docs,
        topic_stats=None,
        pyq_patterns=None,
        max_tokens=max_tokens
    )


def build_pattern_focused_context(
    pyq_patterns: List[Dict],
    max_tokens: int = 2000
) -> str:
    """
    Build context focused on exam patterns without retrieved docs.
    Use this for meta-questions like "What topics appear most often?"
    """
    return build_exam_context(
        retrieved_docs=[],
        topic_stats=None,
        pyq_patterns=pyq_patterns,
        max_tokens=max_tokens
    )


def estimate_context_tokens(
    retrieved_docs: List[Document],
    topic_stats: Optional[List[Dict]] = None,
    pyq_patterns: Optional[List[Dict]] = None,
    model: str = "gpt-4"
) -> Dict[str, int]:
    """
    Estimate token usage for each context section without building.
    Useful for deciding whether to include optional sections.
    """
    estimates = {
        "overhead": 200,  # Headers and formatting
        "patterns": 0,
        "documents": 0,
        "stats": 0
    }
    
    if pyq_patterns:
        pattern_section = _build_pattern_section(pyq_patterns, max_tokens=10000)
        estimates["patterns"] = count_tokens(pattern_section, model)
    
    if retrieved_docs:
        docs_section = _build_documents_section(retrieved_docs, max_tokens=10000)
        estimates["documents"] = count_tokens(docs_section, model)
    
    if topic_stats:
        stats_section = _build_stats_section(topic_stats, max_tokens=10000)
        estimates["stats"] = count_tokens(stats_section, model)
    
    estimates["total"] = sum(estimates.values())
    
    return estimates
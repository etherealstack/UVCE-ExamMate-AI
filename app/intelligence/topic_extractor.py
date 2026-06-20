"""
Topic Extractor Module

Responsibility: Extract and map questions to semantic topics.
Uses LLM to identify main topic, sub-topic, and specific concept.
"""

import re
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from collections import defaultdict

from .question_classifier import ClassifiedQuestion


# ============================================================================
# DATA MODELS
# ============================================================================

class TopicInfo(BaseModel):
    """Topic hierarchy for a question."""
    main_topic: str = Field(description="Broad topic area (e.g., 'Supervised Learning')")
    sub_topic: str = Field(description="Specific topic (e.g., 'Decision Trees')")
    specific_concept: Optional[str] = Field(default=None, description="Exact concept (e.g., 'Information Gain')")
    bloom_level: Optional[str] = Field(
        default=None,
        description="Bloom's taxonomy level: remember, understand, apply, analyze, evaluate, create"
    )


class AnalyzedQuestion(BaseModel):
    """Fully analyzed question with classification and topic."""
    question_text: str
    question_type: str
    confidence: float
    
    # Topic information
    main_topic: str
    sub_topic: str
    specific_concept: Optional[str] = None
    bloom_level: Optional[str] = None
    
    # Metadata
    marks: Optional[int] = None
    question_number: Optional[str] = None
    source_year: Optional[int] = None
    source_file: Optional[str] = None
    difficulty_hint: Optional[str] = None
    key_concepts: List[str] = Field(default_factory=list)


class TopicExtractionResult(BaseModel):
    """LLM output for topic extraction."""
    main_topic: str = Field(description="Main topic area")
    sub_topic: str = Field(description="Sub-topic within main area")
    specific_concept: Optional[str] = Field(default=None, description="Specific concept or technique")
    bloom_level: str = Field(default="understand", description="Cognitive level required")


# ============================================================================
# COURSE TOPIC MAPPING (ML-SPECIFIC, EXTENSIBLE)
# ============================================================================

# Default topic hierarchy for Machine Learning
# This will be extended/replaced by extracting from course material
ML_TOPIC_HIERARCHY = {
    "Foundations": [
        "Linear Algebra",
        "Probability & Statistics",
        "Optimization",
        "Gradient Descent",
        "Loss Functions",
    ],
    "Supervised Learning": [
        "Linear Regression",
        "Logistic Regression",
        "Decision Trees",
        "Random Forest",
        "Support Vector Machines",
        "Naive Bayes",
        "K-Nearest Neighbors",
        "Ensemble Methods",
    ],
    "Unsupervised Learning": [
        "K-Means Clustering",
        "Hierarchical Clustering",
        "DBSCAN",
        "PCA",
        "Dimensionality Reduction",
        "Anomaly Detection",
    ],
    "Neural Networks": [
        "Perceptron",
        "Multilayer Perceptron",
        "Backpropagation",
        "Activation Functions",
        "Convolutional Neural Networks",
        "Recurrent Neural Networks",
        "LSTM",
        "Transformers",
    ],
    "Model Evaluation": [
        "Bias-Variance Tradeoff",
        "Cross Validation",
        "Confusion Matrix",
        "ROC Curve",
        "Precision & Recall",
        "Overfitting & Underfitting",
        "Regularization",
    ],
    "Feature Engineering": [
        "Feature Selection",
        "Feature Scaling",
        "One-Hot Encoding",
        "Missing Data Handling",
    ],
    "Advanced Topics": [
        "Reinforcement Learning",
        "Generative Models",
        "GANs",
        "Bayesian Methods",
        "Attention Mechanisms",
    ]
}


def get_all_topics_flat() -> List[str]:
    """Get flat list of all topics."""
    topics = []
    for category, subtopics in ML_TOPIC_HIERARCHY.items():
        topics.append(category)
        topics.extend(subtopics)
    return topics


def find_matching_topic(text: str, topic_list: List[str]) -> Optional[tuple]:
    """
    Find matching topic from list using simple text matching.
    Returns (main_topic, sub_topic) if found.
    """
    text_lower = text.lower()
    
    for main_topic, subtopics in ML_TOPIC_HIERARCHY.items():
        for sub in subtopics:
            if sub.lower() in text_lower:
                return (main_topic, sub)
        if main_topic.lower() in text_lower:
            return (main_topic, subtopics[0] if subtopics else main_topic)
    
    return None


# ============================================================================
# TOPIC EXTRACTION FROM COURSE MATERIAL
# ============================================================================

def extract_topics_from_course_material(
    docs: List[Document],
    model: str = "llama-3.1-8b-instant"
) -> List[str]:
    """
    Extract syllabus topics from course material (books, syllabus PDFs).
    
    Args:
        docs: Book/course documents
        model: LLM model name
        
    Returns:
        List of extracted topic names
    """
    # Filter for non-PYQ docs (books, syllabus)
    book_docs = [d for d in docs if not d.metadata.get("is_pyq", False)]
    
    if not book_docs:
        print("⚠️ No course material found. Using default ML topics.")
        return get_all_topics_flat()
    
    # Combine first few pages (likely TOC/intro)
    combined_text = ""
    for doc in book_docs[:10]:  # First 10 chunks likely contain TOC
        combined_text += doc.page_content + "\n"
        if len(combined_text) > 5000:
            break
    
    # Try to extract topics using LLM
    try:
        topics = _extract_topics_with_llm(combined_text[:5000], model)
        if topics:
            print(f"📚 Extracted {len(topics)} topics from course material")
            return topics
    except Exception as e:
        print(f"⚠️ LLM topic extraction failed: {e}")
    
    return get_all_topics_flat()


def _extract_topics_with_llm(text: str, model: str) -> List[str]:
    """Extract topics from text using LLM."""
    
    TOPIC_EXTRACTION_TEMPLATE = """Extract the main course topics from this text (likely a table of contents or syllabus).

Text:
{text}

Rules:
1. Extract topic NAMES only (not descriptions)
2. Include both chapter names and sub-topics
3. Return as a simple list, one topic per line
4. If this doesn't look like a syllabus/TOC, return "NO_TOPICS"

Topics (one per line):"""

    llm = ChatGroq(model=model, temperature=0)
    prompt = PromptTemplate(
        template=TOPIC_EXTRACTION_TEMPLATE,
        input_variables=["text"]
    )
    
    chain = prompt | llm
    result = chain.invoke({"text": text})
    
    content = result.content.strip()
    
    if "NO_TOPICS" in content:
        return []
    
    # Parse line-by-line
    topics = [
        line.strip().lstrip("- •*0123456789.")
        for line in content.split("\n")
        if line.strip() and len(line.strip()) > 2
    ]
    
    return topics[:50]  # Limit


# ============================================================================
# MAIN TOPIC EXTRACTION
# ============================================================================

def extract_topic(
    question: ClassifiedQuestion,
    course_topics: Optional[List[str]] = None,
    use_llm: bool = True,
    model: str = "llama-3.1-8b-instant"
) -> TopicInfo:
    """
    Extract topic for a single question.
    
    Args:
        question: Classified question
        course_topics: List of valid course topics
        use_llm: Use LLM for extraction
        model: LLM model name
        
    Returns:
        TopicInfo with topic hierarchy
    """
    text = question.question_text
    concepts = question.key_concepts
    
    if course_topics is None:
        course_topics = get_all_topics_flat()
    
    # Try simple matching first using existing concepts
    for concept in concepts:
        match = find_matching_topic(concept, course_topics)
        if match:
            return TopicInfo(
                main_topic=match[0],
                sub_topic=match[1],
                specific_concept=concept,
                bloom_level=_infer_bloom_level(question.question_type)
            )
    
    # Try matching question text directly
    match = find_matching_topic(text, course_topics)
    if match:
        return TopicInfo(
            main_topic=match[0],
            sub_topic=match[1],
            bloom_level=_infer_bloom_level(question.question_type)
        )
    
    # Use LLM for complex cases
    if use_llm:
        try:
            return _extract_topic_with_llm(text, course_topics, model)
        except Exception as e:
            print(f"⚠️ LLM topic extraction failed: {e}")
    
    # Fallback: return generic based on keywords
    return TopicInfo(
        main_topic="General",
        sub_topic="Miscellaneous",
        bloom_level=_infer_bloom_level(question.question_type)
    )


def _infer_bloom_level(question_type: str) -> str:
    """Infer Bloom's taxonomy level from question type."""
    type_to_bloom = {
        "mcq": "remember",
        "short_answer": "remember",
        "theory": "understand",
        "comparison": "analyze",
        "numerical": "apply",
        "derivation": "analyze",
        "algorithm": "apply",
        "code": "create",
        "diagram": "understand",
        "case_study": "evaluate",
    }
    return type_to_bloom.get(question_type, "understand")


def _extract_topic_with_llm(
    text: str, 
    course_topics: List[str], 
    model: str
) -> TopicInfo:
    """Use LLM to extract topic."""
    
    # Create topic list for context
    topic_list = "\n".join(f"- {t}" for t in course_topics[:30])
    
    TOPIC_TEMPLATE = """Given this exam question, identify its topic from the course syllabus.

Question: {question}

Valid Course Topics:
{topics}

{format_instructions}

Return ONLY the JSON."""

    parser = PydanticOutputParser(pydantic_object=TopicExtractionResult)
    
    prompt = PromptTemplate(
        template=TOPIC_TEMPLATE,
        input_variables=["question", "topics"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatGroq(model=model, temperature=0)
    chain = prompt | llm | parser
    
    result = chain.invoke({"question": text[:500], "topics": topic_list})
    
    return TopicInfo(
        main_topic=result.main_topic,
        sub_topic=result.sub_topic,
        specific_concept=result.specific_concept,
        bloom_level=result.bloom_level
    )


def analyze_questions_with_topics(
    questions: List[ClassifiedQuestion],
    course_topics: Optional[List[str]] = None,
    use_llm: bool = False,
    model: str = "llama-3.1-8b-instant"
) -> List[AnalyzedQuestion]:
    """
    Add topic information to all classified questions.
    
    Args:
        questions: List of classified questions
        course_topics: Valid course topics (extracted from syllabus)
        use_llm: Use LLM for topic extraction
        model: LLM model name
        
    Returns:
        List of fully analyzed questions
    """
    print(f"📌 Extracting topics for {len(questions)} questions...")
    
    analyzed = []
    for i, q in enumerate(questions):
        topic = extract_topic(q, course_topics, use_llm=use_llm, model=model)
        
        analyzed.append(AnalyzedQuestion(
            question_text=q.question_text,
            question_type=q.question_type,
            confidence=q.confidence,
            main_topic=topic.main_topic,
            sub_topic=topic.sub_topic,
            specific_concept=topic.specific_concept,
            bloom_level=topic.bloom_level,
            marks=q.marks,
            question_number=q.question_number,
            source_year=q.source_year,
            source_file=q.source_file,
            difficulty_hint=q.difficulty_hint,
            key_concepts=q.key_concepts
        ))
        
        if (i + 1) % 20 == 0:
            print(f"   Analyzed {i + 1}/{len(questions)}...")
    
    print("✅ Topic extraction complete")
    return analyzed


def get_topic_stats(questions: List[AnalyzedQuestion]) -> Dict:
    """Get topic distribution statistics."""
    if not questions:
        return {"total": 0}
    
    main_topic_counts = defaultdict(int)
    sub_topic_counts = defaultdict(int)
    topic_by_year = defaultdict(lambda: defaultdict(int))
    bloom_counts = defaultdict(int)
    
    for q in questions:
        main_topic_counts[q.main_topic] += 1
        sub_topic_counts[q.sub_topic] += 1
        if q.source_year:
            topic_by_year[q.source_year][q.sub_topic] += 1
        if q.bloom_level:
            bloom_counts[q.bloom_level] += 1
    
    return {
        "total": len(questions),
        "main_topics": dict(sorted(main_topic_counts.items(), key=lambda x: x[1], reverse=True)),
        "sub_topics": dict(sorted(sub_topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
        "topic_by_year": {year: dict(topics) for year, topics in topic_by_year.items()},
        "bloom_levels": dict(bloom_counts)
    }

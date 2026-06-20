"""
Question Classifier Module

Responsibility: Classify questions by type (MCQ, numerical, theory, etc.)
Uses rule-based classification first (fast), LLM fallback (accurate).
"""

import re
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .question_extractor import ExtractedQuestion


# ============================================================================
# QUESTION TYPES
# ============================================================================

QuestionType = Literal[
    "mcq",           # Multiple choice question
    "numerical",     # Calculate, find value, solve
    "derivation",    # Derive formula, prove mathematically
    "theory",        # Explain, describe, define
    "comparison",    # Compare A vs B, differentiate
    "diagram",       # Draw, label, illustrate
    "short_answer",  # Brief factual answer
    "algorithm",     # Write algorithm, pseudocode
    "code",          # Write code, program
    "case_study",    # Analyze given scenario
    "unknown"        # Could not classify
]


# Valid question types for normalization
VALID_QUESTION_TYPES = {
    "mcq", "numerical", "derivation", "theory", "comparison",
    "diagram", "short_answer", "algorithm", "code", "case_study", "unknown"
}

# Map common LLM free-text to valid types
TYPE_ALIASES = {
    "multiple choice question": "mcq",
    "multiple choice": "mcq",
    "short answer": "short_answer",
    "case study": "case_study",
    "derive": "derivation",
    "proof": "derivation",
    "explain": "theory",
    "theoretical": "theory",
    "descriptive": "theory",
    "programming": "code",
    "coding": "code",
    "compare": "comparison",
    "draw": "diagram",
    "calculate": "numerical",
    "computation": "numerical",
}


def normalize_question_type(raw_type: Optional[str]) -> Optional[str]:
    """Normalize a raw type string to a valid QuestionType."""
    if raw_type is None:
        return None
    raw = raw_type.strip().lower().replace("-", "_")
    if raw in VALID_QUESTION_TYPES:
        return raw
    if raw in TYPE_ALIASES:
        return TYPE_ALIASES[raw]
    # Fuzzy match: check if any valid type is a substring
    for valid in VALID_QUESTION_TYPES:
        if valid in raw or raw in valid:
            return valid
    return "unknown"


class ClassifiedQuestion(BaseModel):
    """A question with its classification."""
    question_text: str = Field(description="The question text")
    question_type: QuestionType = Field(description="Type of question")
    confidence: float = Field(default=1.0, description="Classification confidence (0-1)")
    secondary_type: Optional[str] = Field(default=None, description="Secondary type if applicable")
    
    # Preserved from extraction
    marks: Optional[int] = None
    question_number: Optional[str] = None
    source_year: Optional[int] = None
    source_file: Optional[str] = None
    
    # Additional analysis
    difficulty_hint: Optional[str] = Field(default=None, description="easy/medium/hard based on marks/keywords")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts identified in question")


class ClassificationResult(BaseModel):
    """LLM classification output."""
    question_type: str = Field(description="Primary type of question")
    secondary_type: Optional[str] = Field(default=None, description="Secondary type if applicable")
    confidence: float = Field(default=0.9, description="Classification confidence")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts in the question")


# ============================================================================
# RULE-BASED CLASSIFICATION
# ============================================================================

# Keywords that strongly indicate question type
TYPE_INDICATORS = {
    "numerical": [
        r"calculate", r"find\s+(?:the\s+)?value", r"compute", r"solve",
        r"determine\s+(?:the\s+)?(?:value|number)", r"evaluate",
        r"how\s+(?:much|many)", r"what\s+is\s+the\s+(?:value|number)",
        r"\d+\s*(?:units?|meters?|kg|seconds?|watts?)",  # Contains numbers with units
    ],
    "derivation": [
        r"derive", r"prove\s+(?:that)?", r"show\s+that", r"establish",
        r"deduce", r"verify\s+(?:that)?", r"demonstrate\s+mathematically",
    ],
    "theory": [
        r"explain", r"describe", r"define", r"what\s+(?:is|are)",
        r"discuss", r"elaborate", r"state\s+(?:and\s+explain)?",
        r"enumerate", r"list\s+(?:the|and\s+explain)?",
    ],
    "comparison": [
        r"compare", r"contrast", r"differentiate", r"difference\s+between",
        r"distinguish", r"similarities?\s+(?:and\s+differences?)?",
        r"(?:advantages?\s+(?:and|or)\s+disadvantages?)",
    ],
    "diagram": [
        r"draw", r"sketch", r"illustrate", r"diagram",
        r"plot\s+(?:the)?", r"label", r"show\s+(?:using\s+)?(?:a\s+)?(?:diagram|figure)",
    ],
    "algorithm": [
        r"algorithm", r"pseudo\s*code", r"step-by-step",
        r"procedure", r"steps\s+(?:to|for|involved)",
    ],
    "code": [
        r"write\s+(?:a\s+)?(?:program|code|function)",
        r"implement", r"code\s+(?:to|for)",
        r"python|java|c\+\+|sql",
    ],
    "mcq": [
        r"\(a\).*\(b\).*\(c\)",  # Multiple options pattern
        r"choose\s+(?:the\s+)?(?:correct|right|best)",
        r"which\s+(?:of\s+the\s+following|one)",
    ],
    "case_study": [
        r"case\s+study", r"scenario", r"given\s+(?:the\s+)?(?:following|below)",
        r"consider\s+(?:the\s+)?(?:following|scenario)",
    ],
    "short_answer": [
        r"what\s+does\s+.+\s+stand\s+for",
        r"name\s+(?:the|any)", r"give\s+(?:an?\s+)?example",
        r"mention", r"write\s+(?:the\s+)?(?:formula|equation)",
    ],
}

# Difficulty indicators based on keywords and marks
DIFFICULTY_PATTERNS = {
    "easy": {
        "keywords": ["define", "list", "name", "state", "what is"],
        "marks_range": (1, 3)
    },
    "medium": {
        "keywords": ["explain", "describe", "compare", "calculate"],
        "marks_range": (4, 7)
    },
    "hard": {
        "keywords": ["derive", "prove", "design", "implement", "analyze"],
        "marks_range": (8, 20)
    }
}


def classify_by_rules(question_text: str) -> tuple[QuestionType, float]:
    """
    Classify question using rule-based pattern matching.
    Returns (type, confidence).
    """
    text_lower = question_text.lower()
    
    type_scores = {}
    
    for qtype, patterns in TYPE_INDICATORS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                score += 1
        if score > 0:
            type_scores[qtype] = score
    
    if not type_scores:
        return "unknown", 0.0
    
    # Get type with highest score
    best_type = max(type_scores, key=type_scores.get)
    max_score = type_scores[best_type]
    
    # Calculate confidence based on score and uniqueness
    total_scores = sum(type_scores.values())
    confidence = max_score / total_scores if total_scores > 0 else 0.0
    
    # Boost confidence if only one type matched
    if len(type_scores) == 1:
        confidence = min(1.0, confidence + 0.3)
    
    return best_type, confidence


def infer_difficulty(question_text: str, marks: Optional[int]) -> str:
    """Infer difficulty from marks and keywords."""
    text_lower = question_text.lower()
    
    # First check marks
    if marks:
        for difficulty, info in DIFFICULTY_PATTERNS.items():
            min_m, max_m = info["marks_range"]
            if min_m <= marks <= max_m:
                return difficulty
        if marks > 10:
            return "hard"
    
    # Fallback to keyword-based
    for difficulty, info in DIFFICULTY_PATTERNS.items():
        for kw in info["keywords"]:
            if kw in text_lower:
                return difficulty
    
    return "medium"  # Default


def extract_key_concepts_simple(question_text: str) -> List[str]:
    """
    Extract key concepts using simple heuristics.
    Looks for capitalized terms, quoted terms, and technical patterns.
    """
    concepts = []
    
    # Find quoted terms
    quoted = re.findall(r'"([^"]+)"', question_text)
    concepts.extend(quoted)
    
    # Find capitalized multi-word terms (e.g., "Support Vector Machine")
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question_text)
    concepts.extend(caps)
    
    # Find acronyms (e.g., SVM, CNN, RNN)
    acronyms = re.findall(r'\b([A-Z]{2,5})\b', question_text)
    concepts.extend(acronyms)
    
    # Find technical terms (common ML/CS terms)
    technical_terms = [
        "neural network", "decision tree", "random forest", "svm",
        "gradient descent", "backpropagation", "regression", "classification",
        "clustering", "dimensionality reduction", "overfitting", "underfitting",
        "cross-validation", "regularization", "hyperparameter",
    ]
    text_lower = question_text.lower()
    for term in technical_terms:
        if term in text_lower:
            concepts.append(term.title())
    
    # Deduplicate while preserving order
    seen = set()
    unique_concepts = []
    for c in concepts:
        c_lower = c.lower()
        if c_lower not in seen and len(c) > 2:
            seen.add(c_lower)
            unique_concepts.append(c)
    
    return unique_concepts[:10]  # Limit to 10


# ============================================================================
# MAIN CLASSIFICATION FUNCTION
# ============================================================================

def classify_question(
    question: ExtractedQuestion,
    use_llm: bool = False,
    model: str = "llama-3.1-8b-instant"
) -> ClassifiedQuestion:
    """
    Classify a single question by type.
    
    Args:
        question: ExtractedQuestion to classify
        use_llm: Use LLM for classification (slower but more accurate)
        model: LLM model name
        
    Returns:
        ClassifiedQuestion with type and analysis
    """
    text = question.question_text
    
    # Rule-based classification (fast)
    rule_type, rule_confidence = classify_by_rules(text)
    
    # If high confidence from rules, use it
    if rule_confidence > 0.7 and not use_llm:
        return ClassifiedQuestion(
            question_text=text,
            question_type=rule_type,
            confidence=rule_confidence,
            marks=question.marks,
            question_number=question.question_number,
            source_year=question.source_year,
            source_file=question.source_file,
            difficulty_hint=infer_difficulty(text, question.marks),
            key_concepts=extract_key_concepts_simple(text)
        )
    
    # LLM classification for low confidence or when requested
    if use_llm or rule_confidence < 0.5:
        try:
            llm_result = _classify_with_llm(text, model)
            # Normalize LLM output to valid types
            primary = normalize_question_type(llm_result.question_type) or "unknown"
            secondary = normalize_question_type(llm_result.secondary_type)
            return ClassifiedQuestion(
                question_text=text,
                question_type=primary,
                confidence=llm_result.confidence,
                secondary_type=secondary,
                marks=question.marks,
                question_number=question.question_number,
                source_year=question.source_year,
                source_file=question.source_file,
                difficulty_hint=infer_difficulty(text, question.marks),
                key_concepts=llm_result.key_concepts or extract_key_concepts_simple(text)
            )
        except Exception as e:
            print(f"⚠️ LLM classification failed: {e}")
    
    # Fallback to rule-based result
    return ClassifiedQuestion(
        question_text=text,
        question_type=rule_type if rule_type != "unknown" else "theory",
        confidence=rule_confidence if rule_confidence > 0 else 0.5,
        marks=question.marks,
        question_number=question.question_number,
        source_year=question.source_year,
        source_file=question.source_file,
        difficulty_hint=infer_difficulty(text, question.marks),
        key_concepts=extract_key_concepts_simple(text)
    )


def _classify_with_llm(text: str, model: str) -> ClassificationResult:
    """Use LLM to classify question."""
    
    CLASSIFICATION_TEMPLATE = """Classify this exam question by type.

Question: {question}

Question Types:
- mcq: Multiple choice question
- numerical: Calculate, solve, find value
- derivation: Derive formula, prove mathematically
- theory: Explain, describe, define concepts
- comparison: Compare A vs B, differentiate
- diagram: Draw, sketch, illustrate
- algorithm: Write algorithm, procedure steps
- code: Write program, implement function
- case_study: Analyze given scenario
- short_answer: Brief factual answer

{format_instructions}

Return ONLY the JSON."""

    parser = PydanticOutputParser(pydantic_object=ClassificationResult)
    
    prompt = PromptTemplate(
        template=CLASSIFICATION_TEMPLATE,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatGroq(model=model, temperature=0)
    chain = prompt | llm | parser
    
    return chain.invoke({"question": text[:1000]})


def classify_questions(
    questions: List[ExtractedQuestion],
    use_llm: bool = False,
    model: str = "llama-3.1-8b-instant"
) -> List[ClassifiedQuestion]:
    """
    Classify multiple questions.
    
    Args:
        questions: List of ExtractedQuestion objects
        use_llm: Use LLM for all classifications
        model: LLM model name
        
    Returns:
        List of ClassifiedQuestion objects
    """
    print(f"🏷️ Classifying {len(questions)} questions...")
    
    classified = []
    for i, q in enumerate(questions):
        classified.append(classify_question(q, use_llm=use_llm, model=model))
        
        if (i + 1) % 20 == 0:
            print(f"   Classified {i + 1}/{len(questions)}...")
    
    print("✅ Classification complete")
    return classified


def get_classification_stats(questions: List[ClassifiedQuestion]) -> dict:
    """Get statistics about classified questions."""
    if not questions:
        return {"total": 0}
    
    type_counts = {}
    difficulty_counts = {}
    
    for q in questions:
        # Type distribution
        type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
        
        # Difficulty distribution
        if q.difficulty_hint:
            difficulty_counts[q.difficulty_hint] = difficulty_counts.get(q.difficulty_hint, 0) + 1
    
    # All unique concepts
    all_concepts = []
    for q in questions:
        all_concepts.extend(q.key_concepts)
    concept_counts = {}
    for c in all_concepts:
        concept_counts[c] = concept_counts.get(c, 0) + 1
    
    top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    return {
        "total": len(questions),
        "type_distribution": type_counts,
        "difficulty_distribution": difficulty_counts,
        "top_concepts": top_concepts,
        "avg_confidence": sum(q.confidence for q in questions) / len(questions)
    }

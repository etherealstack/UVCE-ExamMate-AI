"""
Question Extractor Module

Responsibility: Extract individual questions from PYQ document chunks.
Uses a combination of regex patterns and LLM for robust extraction.
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ============================================================================
# DATA MODELS
# ============================================================================

class ExtractedQuestion(BaseModel):
    """Represents a single extracted question from a PYQ."""
    question_text: str = Field(description="The complete question text")
    marks: Optional[int] = Field(default=None, description="Marks for this question if mentioned")
    question_number: Optional[str] = Field(default=None, description="Question number (e.g., '1', '2a', 'Q3')")
    sub_parts: List[str] = Field(default_factory=list, description="Sub-parts if multi-part question")
    source_year: Optional[int] = Field(default=None, description="Year of the question paper")
    source_file: Optional[str] = Field(default=None, description="Source file name")
    page_number: Optional[int] = Field(default=None, description="Page number in source")


class ExtractionResult(BaseModel):
    """Result of question extraction from a document."""
    questions: List[ExtractedQuestion] = Field(description="List of extracted questions")
    extraction_confidence: float = Field(default=0.0, description="Confidence in extraction quality (0-1)")


# ============================================================================
# REGEX PATTERNS FOR QUESTION DETECTION
# ============================================================================

# Common question number patterns
QUESTION_PATTERNS = [
    # Q1. or Q1) or Q.1 or Q 1
    r"(?:^|\n)\s*Q\.?\s*(\d+)[.\):]?\s*(.+?)(?=(?:\n\s*Q\.?\s*\d+[.\):])|$)",
    # 1. or 1) - numbered questions
    r"(?:^|\n)\s*(\d+)\s*[.\)]\s*(.+?)(?=(?:\n\s*\d+\s*[.\)])|$)",
    # (a), (b), (c) - sub-parts
    r"\(([a-z])\)\s*(.+?)(?=(?:\([a-z]\))|$)",
    # [5 marks] or (5M) patterns
    r"\[(\d+)\s*(?:marks?|M)\]|\((\d+)\s*(?:marks?|M)\)",
]

# Patterns that indicate question boundaries
BOUNDARY_INDICATORS = [
    r"\n\s*Q\.?\s*\d+",       # Q1, Q.1, Q 1
    r"\n\s*\d+\s*[.\)]",      # 1. or 1)
    r"\n\s*(?:OR|PART\s+[AB])", # OR, PART A/B
    r"\[Section",             # [Section A]
]


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_marks(text: str) -> Optional[int]:
    """Extract marks from question text."""
    patterns = [
        r"\[(\d+)\s*(?:marks?|M)\]",
        r"\((\d+)\s*(?:marks?|M)\)",
        r"(\d+)\s*marks?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_question_number(text: str) -> Optional[str]:
    """Extract question number from text."""
    patterns = [
        r"^Q\.?\s*(\d+[a-z]?)",
        r"^(\d+)\s*[.\)]",
        r"^\(([a-z])\)",
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_questions_regex(text: str) -> List[dict]:
    """
    Extract questions using regex patterns.
    Returns raw question data for further processing.
    """
    questions = []
    
    # Split by common question indicators
    # Pattern: Start of line + optional spaces + Q/number pattern
    split_pattern = r"(?=(?:^|\n)\s*(?:Q\.?\s*\d+[.\):]|\d+\s*[.\)]))"
    
    parts = re.split(split_pattern, text, flags=re.MULTILINE)
    
    for part in parts:
        part = part.strip()
        if len(part) < 10:  # Skip very short fragments
            continue
            
        # Check if this looks like a question
        if re.match(r"^(?:Q\.?\s*\d+|\d+\s*[.\)])", part, re.IGNORECASE):
            question_num = extract_question_number(part)
            marks = extract_marks(part)
            
            # Clean the question text (remove marks notation)
            clean_text = re.sub(r"\[?\(?\d+\s*(?:marks?|M)\]?\)?", "", part)
            clean_text = clean_text.strip()
            
            questions.append({
                "text": clean_text,
                "number": question_num,
                "marks": marks
            })
    
    return questions


def extract_questions_from_doc(
    doc: Document,
    use_llm: bool = True,
    model: str = "llama-3.1-8b-instant"
) -> List[ExtractedQuestion]:
    """
    Extract individual questions from a single PYQ document.
    
    Args:
        doc: LangChain Document with PYQ content
        use_llm: Whether to use LLM for extraction (more accurate but slower)
        model: LLM model to use
        
    Returns:
        List of ExtractedQuestion objects
    """
    text = doc.page_content
    metadata = doc.metadata
    
    # Only process PYQ documents
    if not metadata.get("is_pyq", False):
        return []
    
    # First try regex extraction (fast)
    regex_questions = extract_questions_regex(text)
    
    # If regex found questions, use them
    if regex_questions and not use_llm:
        return [
            ExtractedQuestion(
                question_text=q["text"],
                marks=q["marks"],
                question_number=q["number"],
                source_year=metadata.get("year"),
                source_file=metadata.get("file_name"),
                page_number=metadata.get("page")
            )
            for q in regex_questions
        ]
    
    # Use LLM for more accurate extraction
    if use_llm:
        return _extract_with_llm(text, metadata, model)
    
    return []


def _extract_with_llm(
    text: str, 
    metadata: dict, 
    model: str = "llama-3.1-8b-instant"
) -> List[ExtractedQuestion]:
    """
    Use LLM to extract questions from text.
    More accurate but slower than regex.
    """
    
    EXTRACTION_TEMPLATE = """You are an expert at extracting exam questions from question papers.

Given the following text from a question paper, extract ALL individual questions.

Text:
{text}

Rules:
1. Extract EACH question as a separate item
2. If a question has sub-parts (a, b, c), include them in sub_parts
3. Extract marks if mentioned (e.g., [5 marks], (5M))
4. Include the question number if present
5. Keep the FULL question text, don't summarize

{format_instructions}

Return ONLY the JSON, no additional text."""

    try:
        parser = PydanticOutputParser(pydantic_object=ExtractionResult)
        
        prompt = PromptTemplate(
            template=EXTRACTION_TEMPLATE,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        llm = ChatGroq(model=model, temperature=0)
        chain = prompt | llm | parser
        
        result = chain.invoke({"text": text[:3000]})  # Limit text to avoid token overflow
        
        # Enrich with metadata
        for q in result.questions:
            q.source_year = metadata.get("year")
            q.source_file = metadata.get("file_name")
            q.page_number = metadata.get("page")
        
        return result.questions
        
    except Exception as e:
        print(f"⚠️ LLM extraction failed: {e}. Falling back to regex.")
        regex_questions = extract_questions_regex(text)
        return [
            ExtractedQuestion(
                question_text=q["text"],
                marks=q["marks"],
                question_number=q["number"],
                source_year=metadata.get("year"),
                source_file=metadata.get("file_name"),
                page_number=metadata.get("page")
            )
            for q in regex_questions
        ]


def extract_all_questions(
    docs: List[Document],
    use_llm: bool = True,
    model: str = "llama-3.1-8b-instant"
) -> List[ExtractedQuestion]:
    """
    Extract questions from all PYQ documents.
    
    Args:
        docs: List of Documents (filters for is_pyq=True)
        use_llm: Whether to use LLM extraction
        model: LLM model name
        
    Returns:
        List of all extracted questions with metadata
    """
    all_questions = []
    
    # Filter for PYQ docs only
    pyq_docs = [d for d in docs if d.metadata.get("is_pyq", False)]
    
    print(f"📝 Extracting questions from {len(pyq_docs)} PYQ documents...")
    
    for i, doc in enumerate(pyq_docs):
        questions = extract_questions_from_doc(doc, use_llm=use_llm, model=model)
        all_questions.extend(questions)
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(pyq_docs)} documents...")
    
    print(f"✅ Extracted {len(all_questions)} questions total")
    
    # Deduplicate similar questions (within same year)
    unique_questions = _deduplicate_questions(all_questions)
    print(f"📊 After deduplication: {len(unique_questions)} unique questions")
    
    return unique_questions


def _deduplicate_questions(questions: List[ExtractedQuestion]) -> List[ExtractedQuestion]:
    """
    Remove duplicate questions (same text, same year).
    """
    seen = set()
    unique = []
    
    for q in questions:
        # Create a key from normalized text + year
        key = (
            re.sub(r'\s+', ' ', q.question_text.lower().strip())[:100],
            q.source_year
        )
        
        if key not in seen:
            seen.add(key)
            unique.append(q)
    
    return unique


def get_extraction_stats(questions: List[ExtractedQuestion]) -> dict:
    """
    Get statistics about extracted questions.
    """
    if not questions:
        return {"total": 0}
    
    years = [q.source_year for q in questions if q.source_year]
    marks = [q.marks for q in questions if q.marks]
    
    return {
        "total": len(questions),
        "with_marks": len(marks),
        "avg_marks": sum(marks) / len(marks) if marks else 0,
        "years_covered": sorted(set(years)) if years else [],
        "questions_by_year": {
            year: len([q for q in questions if q.source_year == year])
            for year in set(years)
        } if years else {}
    }

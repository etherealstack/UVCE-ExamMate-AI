"""
Answer Key Generator Module

Responsibility: Generate model answers for exam questions using course material.
Creates step-by-step solutions with references to source material.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document

from .mock_exam_generator import MockQuestion, MockExamPaper


# ============================================================================
# DATA MODELS
# ============================================================================

class AnswerStep(BaseModel):
    """A step in the solution."""
    step_number: int = Field(description="Step number")
    content: str = Field(description="Step content/explanation")
    is_key_point: bool = Field(default=False, description="Whether this is a key marking point")


class ModelAnswer(BaseModel):
    """Complete model answer for a question."""
    question_number: int = Field(description="Question number")
    question_text: str = Field(description="The original question")
    answer_type: str = Field(description="Type of answer (definition, explanation, derivation, etc.)")
    brief_answer: str = Field(description="Brief one-line answer (key point)")
    detailed_answer: str = Field(description="Full detailed answer")
    steps: List[AnswerStep] = Field(default_factory=list, description="Step-by-step solution if applicable")
    key_points: List[str] = Field(description="Key points for scoring marks")
    common_mistakes: List[str] = Field(default_factory=list, description="Common mistakes to avoid")
    marks_breakdown: Dict[str, int] = Field(default_factory=dict, description="How marks are distributed")
    references: List[str] = Field(default_factory=list, description="Source references from course material")


class AnswerKey(BaseModel):
    """Complete answer key for an exam."""
    exam_title: str = Field(description="Title of the exam")
    answers: List[ModelAnswer] = Field(description="All model answers")
    total_marks: int = Field(description="Total marks")
    study_tips: List[str] = Field(description="Tips based on the answers")


class LLMAnswerOutput(BaseModel):
    """LLM output for answer generation."""
    brief_answer: str = Field(description="One-line key answer")
    detailed_answer: str = Field(description="Complete detailed answer")
    key_points: List[str] = Field(description="Key scoring points")
    common_mistakes: List[str] = Field(default_factory=list, description="Common mistakes")


# ============================================================================
# ANSWER GENERATION
# ============================================================================

def generate_answer_key(
    exam: MockExamPaper,
    course_docs: Optional[List[Document]] = None,
    model: str = "llama-3.1-8b-instant"
) -> AnswerKey:
    """
    Generate complete answer key for a mock exam.
    
    Args:
        exam: MockExamPaper to generate answers for
        course_docs: Course material documents for context
        model: LLM model name
        
    Returns:
        AnswerKey with all model answers
    """
    print(f"📝 Generating answer key for: {exam.title}")
    
    # Get context from course material
    context = _build_context(course_docs) if course_docs else ""
    
    answers = []
    question_num = 1
    
    for section in exam.sections:
        print(f"   Processing: {section.section_name}")
        
        for question in section.questions:
            answer = generate_model_answer(
                question=question,
                question_number=question_num,
                context=context,
                model=model
            )
            answers.append(answer)
            question_num += 1
    
    # Generate study tips based on answers
    study_tips = _generate_study_tips(answers)
    
    print(f"✅ Generated {len(answers)} model answers")
    
    return AnswerKey(
        exam_title=exam.title,
        answers=answers,
        total_marks=exam.total_marks,
        study_tips=study_tips
    )


def generate_model_answer(
    question: MockQuestion,
    question_number: int,
    context: str = "",
    model: str = "llama-3.1-8b-instant"
) -> ModelAnswer:
    """
    Generate model answer for a single question.
    
    Args:
        question: The question to answer
        question_number: Question number in the exam
        context: Course material context
        model: LLM model name
        
    Returns:
        ModelAnswer with detailed solution
    """
    
    # Determine answer type based on question type
    answer_type = _get_answer_type(question.question_type)
    
    # Generate answer using LLM
    try:
        llm_answer = _generate_with_llm(
            question=question,
            context=context,
            model=model
        )
        
        # Create marks breakdown
        marks_breakdown = _estimate_marks_breakdown(
            question.marks,
            question.question_type,
            llm_answer.key_points
        )
        
        # Generate steps for derivation/numerical
        steps = []
        if question.question_type in ["derivation", "numerical", "algorithm"]:
            steps = _extract_steps(llm_answer.detailed_answer)
        
        return ModelAnswer(
            question_number=question_number,
            question_text=question.question_text,
            answer_type=answer_type,
            brief_answer=llm_answer.brief_answer,
            detailed_answer=llm_answer.detailed_answer,
            steps=steps,
            key_points=llm_answer.key_points,
            common_mistakes=llm_answer.common_mistakes,
            marks_breakdown=marks_breakdown,
            references=[]
        )
        
    except Exception as e:
        print(f"⚠️ LLM answer generation failed: {e}")
        
        # Fallback answer
        return ModelAnswer(
            question_number=question_number,
            question_text=question.question_text,
            answer_type=answer_type,
            brief_answer=f"Answer for: {question.sub_topic}",
            detailed_answer=f"Detailed explanation of {question.sub_topic} concept.",
            key_points=[f"Key point about {question.sub_topic}"],
            common_mistakes=["Not providing specific examples"],
            marks_breakdown={"concept": question.marks},
            references=[]
        )


def _build_context(docs: List[Document]) -> str:
    """Build context from course documents."""
    if not docs:
        return ""
    
    # Filter for non-PYQ docs
    book_docs = [d for d in docs if not d.metadata.get("is_pyq", False)]
    
    context_parts = []
    total_chars = 0
    max_chars = 4000
    
    for doc in book_docs:
        if total_chars >= max_chars:
            break
        snippet = doc.page_content[:600]
        context_parts.append(snippet)
        total_chars += len(snippet)
    
    return "\n---\n".join(context_parts)


def _get_answer_type(question_type: str) -> str:
    """Map question type to answer type."""
    mapping = {
        "theory": "Explanation",
        "derivation": "Step-by-step Derivation",
        "numerical": "Calculation with Steps",
        "comparison": "Comparative Analysis",
        "diagram": "Diagram with Labels",
        "mcq": "Direct Answer with Justification",
        "short_answer": "Brief Definition",
        "algorithm": "Algorithm with Explanation",
        "code": "Code with Comments",
    }
    return mapping.get(question_type, "Detailed Explanation")


def _generate_with_llm(
    question: MockQuestion,
    context: str,
    model: str
) -> LLMAnswerOutput:
    """Use LLM to generate answer."""
    
    ANSWER_TEMPLATE = """You are an expert professor creating model answers for exam questions.

Question: {question}
Topic: {topic}
Marks: {marks}
Question Type: {question_type}

{context_section}

Generate a model answer that would score FULL marks.

Requirements:
1. brief_answer: One clear sentence that captures the key point
2. detailed_answer: Complete answer worth {marks} marks (be proportional)
3. key_points: List the exact points that would earn marks
4. common_mistakes: What students typically get wrong

{format_instructions}

Return ONLY valid JSON."""

    parser = PydanticOutputParser(pydantic_object=LLMAnswerOutput)
    
    context_section = f"\nReference Material:\n{context}" if context else ""
    
    prompt = PromptTemplate(
        template=ANSWER_TEMPLATE,
        input_variables=["question", "topic", "marks", "question_type", "context_section"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatGroq(model=model, temperature=0.2)
    chain = prompt | llm | parser
    
    return chain.invoke({
        "question": question.question_text,
        "topic": question.sub_topic,
        "marks": question.marks,
        "question_type": question.question_type,
        "context_section": context_section
    })


def _estimate_marks_breakdown(
    total_marks: int,
    question_type: str,
    key_points: List[str]
) -> Dict[str, int]:
    """Estimate how marks are distributed."""
    if not key_points:
        return {"answer": total_marks}
    
    # Distribute marks among key points
    marks_per_point = total_marks // len(key_points)
    remainder = total_marks % len(key_points)
    
    breakdown = {}
    for i, point in enumerate(key_points[:5]):  # Max 5 points
        point_marks = marks_per_point + (1 if i < remainder else 0)
        if point_marks > 0:
            # Create short label from point
            label = point[:30].strip().rstrip(",.:")
            breakdown[label] = point_marks
    
    return breakdown


def _extract_steps(detailed_answer: str) -> List[AnswerStep]:
    """Extract steps from a detailed answer."""
    import re
    
    steps = []
    
    # Try to find numbered steps
    step_patterns = [
        r"(?:Step\s*)?(\d+)[.):]\s*(.+?)(?=(?:Step\s*)?\d+[.):]|$)",
        r"(\d+)\.\s*(.+?)(?=\d+\.|$)",
    ]
    
    for pattern in step_patterns:
        matches = re.findall(pattern, detailed_answer, re.DOTALL | re.IGNORECASE)
        if matches and len(matches) >= 2:
            for num, content in matches:
                if content.strip():
                    steps.append(AnswerStep(
                        step_number=int(num),
                        content=content.strip()[:200],
                        is_key_point=True
                    ))
            break
    
    # Fallback: split by sentences
    if not steps:
        sentences = detailed_answer.split(". ")[:5]
        for i, sent in enumerate(sentences, 1):
            if sent.strip():
                steps.append(AnswerStep(
                    step_number=i,
                    content=sent.strip()[:200],
                    is_key_point=i <= 3
                ))
    
    return steps


def _generate_study_tips(answers: List[ModelAnswer]) -> List[str]:
    """Generate study tips based on answer patterns."""
    tips = []
    
    # Analyze common mistakes
    all_mistakes = []
    for a in answers:
        all_mistakes.extend(a.common_mistakes)
    
    if all_mistakes:
        tips.append(f"Common pitfalls to avoid: {all_mistakes[0]}")
    
    # Count answer types
    type_counts = {}
    for a in answers:
        type_counts[a.answer_type] = type_counts.get(a.answer_type, 0) + 1
    
    most_common = max(type_counts.items(), key=lambda x: x[1], default=None)
    if most_common:
        tips.append(f"Focus on {most_common[0]} style questions - they appear frequently.")
    
    tips.append("Practice writing complete answers within time limits.")
    tips.append("Include diagrams/examples wherever applicable for extra clarity.")
    
    return tips


# ============================================================================
# FORMATTING
# ============================================================================

def format_answer_key(answer_key: AnswerKey) -> str:
    """Format answer key as readable text."""
    lines = [
        "=" * 70,
        f"📝 ANSWER KEY: {answer_key.exam_title}",
        "=" * 70,
        "",
    ]
    
    for answer in answer_key.answers:
        lines.append(f"\n{'─' * 60}")
        lines.append(f"Q{answer.question_number}. {answer.question_text[:80]}...")
        lines.append(f"Type: {answer.answer_type}")
        lines.append("")
        
        lines.append("📌 BRIEF ANSWER:")
        lines.append(f"   {answer.brief_answer}")
        lines.append("")
        
        lines.append("📖 DETAILED ANSWER:")
        # Wrap long text
        answer_lines = answer.detailed_answer.split("\n")
        for line in answer_lines[:15]:  # Limit display
            lines.append(f"   {line}")
        lines.append("")
        
        if answer.steps:
            lines.append("📋 STEPS:")
            for step in answer.steps:
                marker = "⭐" if step.is_key_point else "  "
                lines.append(f"   {marker} Step {step.step_number}: {step.content}")
            lines.append("")
        
        lines.append("✅ KEY POINTS (marking criteria):")
        for point in answer.key_points:
            lines.append(f"   • {point}")
        
        if answer.marks_breakdown:
            lines.append("")
            lines.append("📊 MARKS BREAKDOWN:")
            for criterion, marks in answer.marks_breakdown.items():
                lines.append(f"   • {criterion}: {marks} marks")
        
        if answer.common_mistakes:
            lines.append("")
            lines.append("⚠️ COMMON MISTAKES:")
            for mistake in answer.common_mistakes:
                lines.append(f"   ✗ {mistake}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("💡 STUDY TIPS:")
    for tip in answer_key.study_tips:
        lines.append(f"   👉 {tip}")
    
    return "\n".join(lines)


def save_answer_key(answer_key: AnswerKey, filepath: str) -> None:
    """Save answer key to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(format_answer_key(answer_key))
    print(f"💾 Saved answer key to: {filepath}")

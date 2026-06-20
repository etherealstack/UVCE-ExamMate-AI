"""
Mock Exam Generator Module

Responsibility: Generate realistic mock exam papers based on historical patterns.
Creates papers with proper weightage, difficulty distribution, and time allocation.
"""

import random
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from collections import defaultdict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document

from .topic_extractor import AnalyzedQuestion
from .predictor import PatternAnalysis, analyze_patterns


# ============================================================================
# DATA MODELS
# ============================================================================

class MockQuestion(BaseModel):
    """A question in the mock exam."""
    question_number: int = Field(description="Question number in the paper")
    question_text: str = Field(description="The question text")
    topic: str = Field(description="Topic being tested")
    sub_topic: str = Field(description="Specific sub-topic")
    question_type: str = Field(description="Type of question")
    marks: int = Field(description="Marks allocated")
    time_minutes: int = Field(description="Suggested time in minutes")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Difficulty level")
    reference_years: List[int] = Field(default_factory=list, description="Years with similar questions")


class MockExamSection(BaseModel):
    """A section in the mock exam."""
    section_name: str = Field(description="Section name (e.g., 'Part A', 'Section 1')")
    section_type: Literal["compulsory", "choice"] = Field(description="Whether all questions are compulsory")
    questions_to_attempt: int = Field(description="Number of questions to attempt")
    questions: List[MockQuestion] = Field(description="Questions in this section")
    total_marks: int = Field(description="Total marks for this section")


class MockExamPaper(BaseModel):
    """A complete mock exam paper."""
    title: str = Field(description="Exam title")
    subject: str = Field(description="Subject name")
    total_marks: int = Field(description="Total marks")
    duration_minutes: int = Field(description="Exam duration in minutes")
    sections: List[MockExamSection] = Field(description="Exam sections")
    instructions: List[str] = Field(description="General instructions")
    difficulty_distribution: Dict[str, int] = Field(description="Count by difficulty")
    topic_coverage: List[str] = Field(description="Topics covered")


class ExamConfig(BaseModel):
    """Configuration for generating mock exam."""
    total_marks: int = Field(default=100, description="Total marks for the exam")
    duration_minutes: int = Field(default=180, description="Exam duration in minutes")
    num_sections: int = Field(default=2, description="Number of sections")
    easy_percent: float = Field(default=0.2, description="Percentage of easy questions")
    medium_percent: float = Field(default=0.5, description="Percentage of medium questions")
    hard_percent: float = Field(default=0.3, description="Percentage of hard questions")
    include_choice: bool = Field(default=True, description="Include optional questions")
    subject: str = Field(default="Machine Learning", description="Subject name")


# ============================================================================
# EXAM PATTERN TEMPLATES
# ============================================================================

EXAM_TEMPLATES = {
    "standard": {
        "sections": [
            {"name": "Part A - Short Answer", "marks_range": (2, 5), "count": 10, "compulsory": True},
            {"name": "Part B - Long Answer", "marks_range": (8, 15), "count": 5, "compulsory": False, "attempt": 3},
        ]
    },
    "mixed": {
        "sections": [
            {"name": "Section 1 - MCQ", "marks_range": (1, 2), "count": 10, "compulsory": True},
            {"name": "Section 2 - Short Answer", "marks_range": (5, 7), "count": 6, "compulsory": True},
            {"name": "Section 3 - Long Answer", "marks_range": (10, 15), "count": 4, "compulsory": False, "attempt": 2},
        ]
    },
    "comprehensive": {
        "sections": [
            {"name": "Part A - Theory", "marks_range": (5, 10), "count": 5, "compulsory": True},
            {"name": "Part B - Numerical", "marks_range": (10, 15), "count": 4, "compulsory": True},
            {"name": "Part C - Application", "marks_range": (15, 20), "count": 3, "compulsory": False, "attempt": 2},
        ]
    }
}


# ============================================================================
# MOCK EXAM GENERATOR
# ============================================================================

def generate_mock_exam(
    analyzed_questions: List[AnalyzedQuestion],
    config: Optional[ExamConfig] = None,
    template: Literal["standard", "mixed", "comprehensive"] = "standard",
    course_docs: Optional[List[Document]] = None,
    model: str = "llama-3.1-8b-instant"
) -> MockExamPaper:
    """
    Generate a complete mock exam paper.
    
    Args:
        analyzed_questions: Historical questions with topics and types
        config: Exam configuration
        template: Exam structure template to use
        course_docs: Course material for additional context
        model: LLM model for question generation
        
    Returns:
        MockExamPaper with full exam structure
    """
    if config is None:
        config = ExamConfig()
    
    print(f"📝 Generating mock exam: {config.subject}")
    print(f"   Template: {template}, Total Marks: {config.total_marks}")
    
    # Analyze patterns from historical data
    patterns = analyze_patterns(analyzed_questions)
    
    # Get template structure
    exam_template = EXAM_TEMPLATES.get(template, EXAM_TEMPLATES["standard"])
    
    # Generate sections
    sections = []
    remaining_marks = config.total_marks
    
    for section_config in exam_template["sections"]:
        if remaining_marks <= 0:
            break
            
        section = _generate_section(
            section_config=section_config,
            analyzed_questions=analyzed_questions,
            patterns=patterns,
            config=config,
            remaining_marks=remaining_marks,
            model=model
        )
        
        sections.append(section)
        remaining_marks -= section.total_marks
    
    # Calculate distributions
    all_questions = [q for s in sections for q in s.questions]
    difficulty_dist = defaultdict(int)
    topics_covered = set()
    
    for q in all_questions:
        difficulty_dist[q.difficulty] += 1
        topics_covered.add(q.sub_topic)
    
    # Generate instructions
    instructions = _generate_instructions(config, sections)
    
    return MockExamPaper(
        title=f"{config.subject} - Mock Examination",
        subject=config.subject,
        total_marks=sum(s.total_marks for s in sections),
        duration_minutes=config.duration_minutes,
        sections=sections,
        instructions=instructions,
        difficulty_distribution=dict(difficulty_dist),
        topic_coverage=sorted(topics_covered)
    )


def _generate_section(
    section_config: Dict,
    analyzed_questions: List[AnalyzedQuestion],
    patterns: PatternAnalysis,
    config: ExamConfig,
    remaining_marks: int,
    model: str
) -> MockExamSection:
    """Generate a single exam section."""
    
    section_name = section_config["name"]
    marks_range = section_config["marks_range"]
    question_count = section_config["count"]
    is_compulsory = section_config.get("compulsory", True)
    attempt_count = section_config.get("attempt", question_count)
    
    # Determine marks per question
    min_marks, max_marks = marks_range
    
    # Select topics based on frequency
    frequent_topics = list(patterns.topic_frequency.keys())[:15]
    
    questions = []
    current_marks = 0
    question_num = 1
    
    # Group historical questions by topic
    questions_by_topic = defaultdict(list)
    for q in analyzed_questions:
        questions_by_topic[q.sub_topic].append(q)
    
    # Generate questions
    for i in range(question_count):
        if current_marks >= remaining_marks:
            break
            
        # Select topic (weighted by frequency)
        topic = random.choice(frequent_topics) if frequent_topics else "General"
        
        # Get historical questions for this topic
        topic_questions = questions_by_topic.get(topic, [])
        
        if topic_questions:
            # Base on historical question
            source_q = random.choice(topic_questions)
            marks = random.randint(min_marks, min(max_marks, remaining_marks - current_marks))
            
            question = MockQuestion(
                question_number=question_num,
                question_text=_vary_question(source_q.question_text),
                topic=source_q.main_topic,
                sub_topic=source_q.sub_topic,
                question_type=source_q.question_type,
                marks=marks,
                time_minutes=marks * 2,  # 2 minutes per mark
                difficulty=source_q.difficulty_hint or "medium",
                reference_years=[source_q.source_year] if source_q.source_year else []
            )
        else:
            # Generate generic question for topic
            marks = random.randint(min_marks, min(max_marks, remaining_marks - current_marks))
            question = MockQuestion(
                question_number=question_num,
                question_text=f"Explain the concept of {topic} and its applications.",
                topic=topic,
                sub_topic=topic,
                question_type="theory",
                marks=marks,
                time_minutes=marks * 2,
                difficulty="medium",
                reference_years=[]
            )
        
        questions.append(question)
        current_marks += question.marks
        question_num += 1
    
    return MockExamSection(
        section_name=section_name,
        section_type="compulsory" if is_compulsory else "choice",
        questions_to_attempt=attempt_count,
        questions=questions,
        total_marks=sum(q.marks for q in questions)
    )


def _vary_question(original: str) -> str:
    """Slightly vary a question to create a new version."""
    variations = [
        # Keep original
        original,
        # Add context
        f"With a suitable example, {original.lower()}" if not original.lower().startswith("with") else original,
        # Add comparison
        original.replace("Explain", "Explain and discuss") if "Explain" in original else original,
    ]
    return random.choice(variations)


def _generate_instructions(config: ExamConfig, sections: List[MockExamSection]) -> List[str]:
    """Generate exam instructions."""
    instructions = [
        f"Total Marks: {sum(s.total_marks for s in sections)}",
        f"Duration: {config.duration_minutes} minutes",
        "Read all questions carefully before attempting.",
        "Write neatly and clearly.",
    ]
    
    for section in sections:
        if section.section_type == "choice":
            instructions.append(
                f"{section.section_name}: Attempt any {section.questions_to_attempt} "
                f"out of {len(section.questions)} questions."
            )
        else:
            instructions.append(f"{section.section_name}: All {len(section.questions)} questions are compulsory.")
    
    return instructions


# ============================================================================
# FORMATTING
# ============================================================================

def format_mock_exam(paper: MockExamPaper) -> str:
    """Format mock exam as readable text."""
    lines = [
        "=" * 70,
        f"📝 {paper.title}",
        "=" * 70,
        "",
        f"Subject: {paper.subject}",
        f"Total Marks: {paper.total_marks}",
        f"Duration: {paper.duration_minutes} minutes",
        "",
        "📋 INSTRUCTIONS:",
    ]
    
    for inst in paper.instructions:
        lines.append(f"  • {inst}")
    
    lines.append("")
    lines.append("-" * 70)
    
    question_num = 1
    for section in paper.sections:
        lines.append("")
        lines.append(f"📌 {section.section_name} ({section.total_marks} marks)")
        if section.section_type == "choice":
            lines.append(f"   [Attempt any {section.questions_to_attempt} questions]")
        lines.append("-" * 40)
        
        for q in section.questions:
            diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(q.difficulty, "⚪")
            lines.append("")
            lines.append(f"Q{question_num}. [{q.marks} marks] [{q.time_minutes} min] {diff_icon}")
            lines.append(f"    {q.question_text}")
            lines.append(f"    📚 Topic: {q.sub_topic} | Type: {q.question_type}")
            question_num += 1
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    lines.append("📊 PAPER ANALYSIS:")
    lines.append(f"   Difficulty: Easy={paper.difficulty_distribution.get('easy', 0)}, "
                 f"Medium={paper.difficulty_distribution.get('medium', 0)}, "
                 f"Hard={paper.difficulty_distribution.get('hard', 0)}")
    lines.append(f"   Topics Covered: {len(paper.topic_coverage)}")
    
    return "\n".join(lines)


def save_mock_exam(paper: MockExamPaper, filepath: str) -> None:
    """Save mock exam to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(format_mock_exam(paper))
    print(f"💾 Saved mock exam to: {filepath}")

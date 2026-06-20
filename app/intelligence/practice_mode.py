"""
Practice Mode Module

Responsibility: Interactive practice with instant feedback.
Generates practice questions and evaluates answers.
"""

import random
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document

from .topic_extractor import AnalyzedQuestion
from .question_classifier import QuestionType


# ============================================================================
# DATA MODELS
# ============================================================================

class PracticeQuestion(BaseModel):
    """A practice question."""
    id: int = Field(description="Question ID")
    question: str = Field(description="Question text")
    topic: str = Field(description="Topic being tested")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Difficulty")
    marks: int = Field(description="Marks for this question")
    hints: List[str] = Field(default_factory=list, description="Progressive hints")
    model_answer: str = Field(description="Expected answer")
    key_points: List[str] = Field(description="Key points to include")


class AnswerEvaluation(BaseModel):
    """Evaluation of a user's answer."""
    score: float = Field(description="Score out of total marks (0-1)")
    marks_obtained: float = Field(description="Actual marks obtained")
    feedback: str = Field(description="Detailed feedback")
    correct_points: List[str] = Field(description="Points correctly covered")
    missing_points: List[str] = Field(description="Points that were missed")
    suggestions: List[str] = Field(description="Improvement suggestions")
    grade: Literal["excellent", "good", "fair", "needs_improvement"] = Field(description="Grade")


class PracticeSession(BaseModel):
    """A practice session."""
    session_id: str = Field(description="Unique session ID")
    topic: Optional[str] = Field(default=None, description="Topic filter")
    difficulty: Optional[str] = Field(default=None, description="Difficulty filter")
    questions: List[PracticeQuestion] = Field(description="Questions in session")
    current_index: int = Field(default=0, description="Current question index")
    scores: List[float] = Field(default_factory=list, description="Scores per question")
    total_marks: int = Field(description="Total marks in session")
    is_completed: bool = Field(default=False, description="Whether session is complete")


class LLMEvaluationOutput(BaseModel):
    """LLM output for answer evaluation."""
    score_percentage: float = Field(description="Score as percentage 0-100")
    feedback: str = Field(description="Detailed feedback")
    correct_points: List[str] = Field(description="Correct points")
    missing_points: List[str] = Field(description="Missing points")
    suggestions: List[str] = Field(description="Improvement suggestions")


# ============================================================================
# QUESTION GENERATION
# ============================================================================

def generate_practice_questions(
    analyzed_questions: List[AnalyzedQuestion],
    num_questions: int = 5,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    model: str = "llama-3.1-8b-instant"
) -> List[PracticeQuestion]:
    """
    Generate practice questions from historical data.
    
    Args:
        analyzed_questions: Historical questions to base on
        num_questions: Number of questions to generate
        topic: Filter by topic
        difficulty: Filter by difficulty
        model: LLM model for variations
        
    Returns:
        List of practice questions
    """
    # Filter questions
    filtered = analyzed_questions
    
    if topic:
        filtered = [q for q in filtered if topic.lower() in q.sub_topic.lower()]
    
    if difficulty:
        filtered = [q for q in filtered if q.difficulty_hint == difficulty]
    
    if not filtered:
        filtered = analyzed_questions  # Fallback to all
    
    # Select questions
    if len(filtered) < num_questions:
        selected = filtered
    else:
        selected = random.sample(filtered, num_questions)
    
    # Convert to practice questions
    practice_questions = []
    
    for i, q in enumerate(selected, 1):
        # Generate hints
        hints = _generate_hints(q)
        
        # Generate model answer
        model_answer = _generate_model_answer(q, model)
        
        practice_questions.append(PracticeQuestion(
            id=i,
            question=q.question_text,
            topic=q.sub_topic,
            difficulty=q.difficulty_hint or "medium",
            marks=q.marks or 5,
            hints=hints,
            model_answer=model_answer,
            key_points=q.key_concepts if q.key_concepts else [f"Explain {q.sub_topic}"]
        ))
    
    return practice_questions


def _generate_hints(question: AnalyzedQuestion) -> List[str]:
    """Generate progressive hints for a question."""
    hints = []
    
    topic = question.sub_topic
    concepts = question.key_concepts
    
    # Hint 1: Topic area
    hints.append(f"💡 This question is about {topic}.")
    
    # Hint 2: Key concepts
    if concepts:
        hints.append(f"💡 Focus on these concepts: {', '.join(concepts[:2])}")
    
    # Hint 3: Structure
    qtype = question.question_type
    structure_hints = {
        "derivation": "Start from the basic equation and apply the chain rule.",
        "numerical": "Identify the given values, apply the formula, show working.",
        "theory": "Define the concept, explain its importance, give examples.",
        "comparison": "Use a table format to compare key aspects.",
        "algorithm": "List the steps clearly with explanations.",
    }
    if qtype in structure_hints:
        hints.append(f"💡 Structure: {structure_hints[qtype]}")
    
    return hints


def _generate_model_answer(
    question: AnalyzedQuestion,
    model: str
) -> str:
    """Generate a model answer for the question."""
    
    # For now, return a structured placeholder
    # In production, this would use LLM
    qtype = question.question_type
    topic = question.sub_topic
    
    templates = {
        "theory": f"""**{topic}** is a concept that...

Key points:
1. Definition: [Define {topic}]
2. Importance: [Why it matters]
3. Applications: [Where it's used]
4. Example: [Practical example]""",
        
        "numerical": f"""**Solution:**

Given: [List given values]
To Find: [What needs to be calculated]

Step 1: Identify the formula
Step 2: Substitute values
Step 3: Calculate
Step 4: Verify and write final answer""",
        
        "derivation": f"""**Derivation:**

Starting from: [Base equation]

Step 1: [First transformation]
Step 2: [Apply rule/formula]
Step 3: [Simplify]
...
Final Result: [Derived equation]

QED""",
        
        "comparison": f"""**Comparison Table:**

| Aspect | Option A | Option B |
|--------|----------|----------|
| Definition | ... | ... |
| Use Case | ... | ... |
| Advantages | ... | ... |
| Disadvantages | ... | ... |

Conclusion: [When to use which]""",
    }
    
    return templates.get(qtype, f"Answer for {topic}: [Detailed explanation]")


# ============================================================================
# ANSWER EVALUATION
# ============================================================================

def evaluate_answer(
    question: PracticeQuestion,
    user_answer: str,
    model: str = "llama-3.1-8b-instant"
) -> AnswerEvaluation:
    """
    Evaluate a user's answer against the model answer.
    
    Args:
        question: The practice question
        user_answer: User's answer text
        model: LLM model for evaluation
        
    Returns:
        AnswerEvaluation with score and feedback
    """
    
    if not user_answer.strip():
        return AnswerEvaluation(
            score=0.0,
            marks_obtained=0.0,
            feedback="No answer provided. Please attempt the question.",
            correct_points=[],
            missing_points=question.key_points,
            suggestions=["Try to answer even if you're unsure."],
            grade="needs_improvement"
        )
    
    try:
        # Use LLM for evaluation
        evaluation = _evaluate_with_llm(question, user_answer, model)
        
        # Calculate marks
        marks_obtained = (evaluation.score_percentage / 100) * question.marks
        
        # Determine grade
        if evaluation.score_percentage >= 80:
            grade = "excellent"
        elif evaluation.score_percentage >= 60:
            grade = "good"
        elif evaluation.score_percentage >= 40:
            grade = "fair"
        else:
            grade = "needs_improvement"
        
        return AnswerEvaluation(
            score=evaluation.score_percentage / 100,
            marks_obtained=round(marks_obtained, 1),
            feedback=evaluation.feedback,
            correct_points=evaluation.correct_points,
            missing_points=evaluation.missing_points,
            suggestions=evaluation.suggestions,
            grade=grade
        )
        
    except Exception as e:
        print(f"⚠️ LLM evaluation failed: {e}")
        # Fallback: simple keyword matching
        return _simple_evaluate(question, user_answer)


def _evaluate_with_llm(
    question: PracticeQuestion,
    user_answer: str,
    model: str
) -> LLMEvaluationOutput:
    """Use LLM to evaluate the answer."""
    
    EVAL_TEMPLATE = """You are an exam evaluator. Evaluate this student's answer.

Question: {question}
Topic: {topic}
Total Marks: {marks}
Key Points Expected: {key_points}

Model Answer:
{model_answer}

Student's Answer:
{user_answer}

Evaluate based on:
1. Coverage of key points
2. Accuracy of information
3. Clarity of explanation
4. Use of examples (if applicable)

{format_instructions}

Return ONLY valid JSON."""

    parser = PydanticOutputParser(pydantic_object=LLMEvaluationOutput)
    
    prompt = PromptTemplate(
        template=EVAL_TEMPLATE,
        input_variables=["question", "topic", "marks", "key_points", "model_answer", "user_answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatGroq(model=model, temperature=0)
    chain = prompt | llm | parser
    
    return chain.invoke({
        "question": question.question,
        "topic": question.topic,
        "marks": question.marks,
        "key_points": ", ".join(question.key_points),
        "model_answer": question.model_answer[:500],  # Limit for token efficiency
        "user_answer": user_answer[:1000]
    })


def _simple_evaluate(
    question: PracticeQuestion,
    user_answer: str
) -> AnswerEvaluation:
    """Simple keyword-based evaluation as fallback."""
    
    answer_lower = user_answer.lower()
    
    # Check key points coverage
    correct = []
    missing = []
    
    for point in question.key_points:
        point_words = point.lower().split()
        if any(word in answer_lower for word in point_words if len(word) > 3):
            correct.append(point)
        else:
            missing.append(point)
    
    # Calculate score
    if not question.key_points:
        score = 0.5 if len(user_answer) > 50 else 0.2
    else:
        score = len(correct) / len(question.key_points)
    
    marks = score * question.marks
    
    # Grade
    if score >= 0.8:
        grade = "excellent"
    elif score >= 0.6:
        grade = "good"
    elif score >= 0.4:
        grade = "fair"
    else:
        grade = "needs_improvement"
    
    return AnswerEvaluation(
        score=score,
        marks_obtained=round(marks, 1),
        feedback="Basic evaluation based on keyword matching.",
        correct_points=correct,
        missing_points=missing,
        suggestions=["Make sure to cover all key concepts.", "Use specific terminology."],
        grade=grade
    )


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def create_practice_session(
    analyzed_questions: List[AnalyzedQuestion],
    num_questions: int = 5,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None
) -> PracticeSession:
    """Create a new practice session."""
    import uuid
    
    questions = generate_practice_questions(
        analyzed_questions,
        num_questions=num_questions,
        topic=topic,
        difficulty=difficulty
    )
    
    total_marks = sum(q.marks for q in questions)
    
    return PracticeSession(
        session_id=str(uuid.uuid4())[:8],
        topic=topic,
        difficulty=difficulty,
        questions=questions,
        current_index=0,
        scores=[],
        total_marks=total_marks,
        is_completed=False
    )


def get_current_question(session: PracticeSession) -> Optional[PracticeQuestion]:
    """Get the current question in the session."""
    if session.is_completed or session.current_index >= len(session.questions):
        return None
    return session.questions[session.current_index]


def submit_answer(
    session: PracticeSession,
    answer: str,
    model: str = "llama-3.1-8b-instant"
) -> AnswerEvaluation:
    """Submit an answer for the current question."""
    question = get_current_question(session)
    
    if not question:
        return AnswerEvaluation(
            score=0,
            marks_obtained=0,
            feedback="Session completed or no current question.",
            correct_points=[],
            missing_points=[],
            suggestions=[],
            grade="needs_improvement"
        )
    
    # Evaluate
    evaluation = evaluate_answer(question, answer, model)
    
    # Update session
    session.scores.append(evaluation.score)
    session.current_index += 1
    
    if session.current_index >= len(session.questions):
        session.is_completed = True
    
    return evaluation


def get_session_summary(session: PracticeSession) -> Dict:
    """Get summary of a practice session."""
    if not session.scores:
        return {
            "status": "not_started",
            "questions_attempted": 0,
            "total_questions": len(session.questions),
        }
    
    total_marks = sum(
        session.scores[i] * session.questions[i].marks
        for i in range(len(session.scores))
    )
    
    max_marks = sum(session.questions[i].marks for i in range(len(session.scores)))
    percentage = (total_marks / max_marks * 100) if max_marks > 0 else 0
    
    return {
        "status": "completed" if session.is_completed else "in_progress",
        "questions_attempted": len(session.scores),
        "total_questions": len(session.questions),
        "marks_obtained": round(total_marks, 1),
        "max_marks": max_marks,
        "percentage": round(percentage, 1),
        "average_score": round(sum(session.scores) / len(session.scores), 2),
    }


# ============================================================================
# FORMATTING
# ============================================================================

def format_question(question: PracticeQuestion, show_hints: int = 0) -> str:
    """Format a practice question for display."""
    lines = [
        f"{'─' * 60}",
        f"📝 Question {question.id} [{question.marks} marks] [{'🟢🟡🔴'['easy medium hard'.split().index(question.difficulty)]}]",
        f"{'─' * 60}",
        "",
        f"{question.question}",
        "",
        f"📚 Topic: {question.topic}",
    ]
    
    if show_hints > 0:
        lines.append("")
        lines.append("💡 HINTS:")
        for i, hint in enumerate(question.hints[:show_hints], 1):
            lines.append(f"   {hint}")
    
    return "\n".join(lines)


def format_evaluation(evaluation: AnswerEvaluation, question: PracticeQuestion) -> str:
    """Format an evaluation for display."""
    grade_icons = {
        "excellent": "🌟",
        "good": "👍",
        "fair": "📝",
        "needs_improvement": "📚"
    }
    
    lines = [
        "",
        f"{'─' * 60}",
        f"📊 EVALUATION RESULT",
        f"{'─' * 60}",
        "",
        f"{grade_icons.get(evaluation.grade, '📝')} Grade: {evaluation.grade.upper()}",
        f"📈 Score: {evaluation.marks_obtained}/{question.marks} marks ({evaluation.score*100:.0f}%)",
        "",
        f"💬 Feedback:",
        f"   {evaluation.feedback}",
        "",
    ]
    
    if evaluation.correct_points:
        lines.append("✅ Correct Points:")
        for point in evaluation.correct_points:
            lines.append(f"   • {point}")
    
    if evaluation.missing_points:
        lines.append("")
        lines.append("❌ Missing Points:")
        for point in evaluation.missing_points:
            lines.append(f"   • {point}")
    
    if evaluation.suggestions:
        lines.append("")
        lines.append("💡 Suggestions:")
        for sug in evaluation.suggestions:
            lines.append(f"   • {sug}")
    
    return "\n".join(lines)


def format_session_summary(session: PracticeSession) -> str:
    """Format session summary for display."""
    summary = get_session_summary(session)
    
    lines = [
        "",
        "=" * 60,
        "📊 PRACTICE SESSION SUMMARY",
        "=" * 60,
        "",
        f"Status: {summary['status'].upper()}",
        f"Questions Attempted: {summary['questions_attempted']}/{summary['total_questions']}",
    ]
    
    if summary.get("marks_obtained") is not None:
        lines.extend([
            f"Marks: {summary['marks_obtained']}/{summary['max_marks']}",
            f"Percentage: {summary['percentage']}%",
            "",
        ])
        
        # Performance bar
        pct = summary['percentage']
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"[{bar}] {pct}%")
        
        # Grade
        if pct >= 80:
            lines.append("🌟 Excellent performance!")
        elif pct >= 60:
            lines.append("👍 Good job! Keep practicing.")
        elif pct >= 40:
            lines.append("📝 Fair performance. Review weak areas.")
        else:
            lines.append("📚 Needs improvement. Focus on fundamentals.")
    
    return "\n".join(lines)

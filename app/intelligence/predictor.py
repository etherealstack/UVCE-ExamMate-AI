"""
Prediction Engine Module

Responsibility: Generate intelligent question predictions based on patterns.
Analyzes topic trends, question types, and gaps to predict likely exam questions.
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from collections import defaultdict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
import json

from .topic_extractor import AnalyzedQuestion


# ============================================================================
# DATA MODELS
# ============================================================================

class PredictedQuestion(BaseModel):
    """A predicted exam question with reasoning."""
    question: str = Field(description="The predicted exam question")
    topic: str = Field(description="Main topic this question tests")
    sub_topic: str = Field(description="Specific sub-topic")
    question_type: str = Field(description="Type of question")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Difficulty level")
    confidence: Literal["high", "medium", "low"] = Field(description="Prediction confidence")
    reasoning: str = Field(description="Why this question is likely to appear")
    similar_years: List[int] = Field(default_factory=list, description="Years with similar questions")
    study_tip: Optional[str] = Field(default=None, description="How to prepare for this")


class PatternAnalysis(BaseModel):
    """Analysis of question patterns."""
    total_questions: int
    years_covered: List[int]
    topic_frequency: Dict[str, int]
    type_frequency: Dict[str, int]
    yearly_topic_trends: Dict[str, List[int]]  # topic -> list of years it appeared
    gap_topics: List[str]  # Topics not asked recently
    consistent_topics: List[str]  # Topics asked almost every year
    emerging_topics: List[str]  # Topics appearing more in recent years


class PredictionResult(BaseModel):
    """Complete prediction result."""
    predictions: List[PredictedQuestion] = Field(description="Predicted questions")
    high_priority_topics: List[str] = Field(description="Topics to prioritize")
    exam_tips: List[str] = Field(description="Strategic exam tips")
    pattern_summary: str = Field(description="Summary of detected patterns")
    confidence_score: float = Field(description="Overall confidence in predictions (0-1)")


class LLMPredictionOutput(BaseModel):
    """LLM output schema for predictions."""
    predictions: List[PredictedQuestion]
    high_priority_topics: List[str]
    exam_tips: List[str]


# ============================================================================
# PATTERN ANALYSIS
# ============================================================================

def analyze_patterns(questions: List[AnalyzedQuestion]) -> PatternAnalysis:
    """
    Analyze patterns in historical questions.
    
    Args:
        questions: List of analyzed questions with topics
        
    Returns:
        PatternAnalysis with trends and statistics
    """
    if not questions:
        return PatternAnalysis(
            total_questions=0,
            years_covered=[],
            topic_frequency={},
            type_frequency={},
            yearly_topic_trends={},
            gap_topics=[],
            consistent_topics=[],
            emerging_topics=[]
        )
    
    # Gather statistics
    topic_frequency = defaultdict(int)
    type_frequency = defaultdict(int)
    yearly_topic_presence = defaultdict(set)  # topic -> set of years
    topic_by_year = defaultdict(lambda: defaultdict(int))  # year -> topic -> count
    
    years = set()
    
    for q in questions:
        topic_frequency[q.sub_topic] += 1
        type_frequency[q.question_type] += 1
        
        if q.source_year:
            years.add(q.source_year)
            yearly_topic_presence[q.sub_topic].add(q.source_year)
            topic_by_year[q.source_year][q.sub_topic] += 1
    
    sorted_years = sorted(years) if years else []
    
    # Find consistent topics (appear in >70% of years)
    year_count = len(sorted_years)
    consistent_topics = []
    if year_count > 0:
        for topic, topic_years in yearly_topic_presence.items():
            if len(topic_years) / year_count >= 0.7:
                consistent_topics.append(topic)
    
    # Find gap topics (asked before but not in last year/two)
    gap_topics = []
    if len(sorted_years) >= 2:
        recent_years = set(sorted_years[-2:])  # Last 2 years
        older_years = set(sorted_years[:-2]) if len(sorted_years) > 2 else set()
        
        for topic, topic_years in yearly_topic_presence.items():
            if topic_years & older_years and not (topic_years & recent_years):
                gap_topics.append(topic)
    
    # Find emerging topics (appear more in recent years)
    emerging_topics = []
    if len(sorted_years) >= 3:
        mid = len(sorted_years) // 2
        old_years = set(sorted_years[:mid])
        new_years = set(sorted_years[mid:])
        
        for topic, topic_years in yearly_topic_presence.items():
            old_count = len(topic_years & old_years)
            new_count = len(topic_years & new_years)
            if new_count > old_count * 1.5:  # 50% more appearances
                emerging_topics.append(topic)
    
    # Convert yearly presence to list format
    yearly_topic_trends = {
        topic: sorted(years)
        for topic, years in yearly_topic_presence.items()
    }
    
    return PatternAnalysis(
        total_questions=len(questions),
        years_covered=sorted_years,
        topic_frequency=dict(sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)),
        type_frequency=dict(type_frequency),
        yearly_topic_trends=yearly_topic_trends,
        gap_topics=gap_topics[:10],
        consistent_topics=consistent_topics[:10],
        emerging_topics=emerging_topics[:10]
    )


def format_pattern_summary(analysis: PatternAnalysis) -> str:
    """Create human-readable pattern summary."""
    lines = [
        f"📊 Pattern Analysis Summary",
        f"{'=' * 40}",
        f"Total Questions Analyzed: {analysis.total_questions}",
        f"Years Covered: {', '.join(map(str, analysis.years_covered))}",
        "",
        "🔥 Most Frequent Topics:",
    ]
    
    for topic, count in list(analysis.topic_frequency.items())[:10]:
        lines.append(f"  • {topic}: {count} times")
    
    lines.append("")
    lines.append("📈 Consistent Topics (appear every year):")
    for topic in analysis.consistent_topics:
        lines.append(f"  • {topic}")
    
    if analysis.gap_topics:
        lines.append("")
        lines.append("⚠️ Gap Topics (not asked recently):")
        for topic in analysis.gap_topics:
            lines.append(f"  • {topic}")
    
    if analysis.emerging_topics:
        lines.append("")
        lines.append("🆕 Emerging Topics (gaining importance):")
        for topic in analysis.emerging_topics:
            lines.append(f"  • {topic}")
    
    lines.append("")
    lines.append("📝 Question Type Distribution:")
    for qtype, count in analysis.type_frequency.items():
        pct = count / analysis.total_questions * 100 if analysis.total_questions > 0 else 0
        lines.append(f"  • {qtype}: {count} ({pct:.1f}%)")
    
    return "\n".join(lines)


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def predict_questions(
    questions: List[AnalyzedQuestion],
    course_material: Optional[List[Document]] = None,
    num_predictions: int = 10,
    model: str = "llama-3.1-8b-instant",
    prediction_style: Literal["conservative", "exploratory", "both"] = "both"
) -> PredictionResult:
    """
    Generate predicted exam questions based on pattern analysis.
    
    Args:
        questions: Historical analyzed questions
        course_material: Course book documents (for context)
        num_predictions: Number of predictions to generate
        model: LLM model name
        prediction_style:
            - "conservative": High confidence predictions based on clear patterns
            - "exploratory": Include potential surprises and gap topics
            - "both": Mix of both (recommended)
            
    Returns:
        PredictionResult with predictions and analysis
    """
    # Analyze patterns
    print("📊 Analyzing question patterns...")
    analysis = analyze_patterns(questions)
    pattern_summary = format_pattern_summary(analysis)
    
    if analysis.total_questions == 0:
        return PredictionResult(
            predictions=[],
            high_priority_topics=[],
            exam_tips=["No historical data available for prediction."],
            pattern_summary="No questions found in the database.",
            confidence_score=0.0
        )
    
    # Extract course topics from material
    course_topics = _extract_course_context(course_material) if course_material else ""
    
    # Generate predictions using LLM
    print("🔮 Generating predictions...")
    predictions = _generate_predictions_llm(
        analysis=analysis,
        course_context=course_topics,
        num_predictions=num_predictions,
        prediction_style=prediction_style,
        model=model
    )
    
    # Calculate confidence score
    confidence = _calculate_confidence(analysis)
    
    return PredictionResult(
        predictions=predictions.predictions,
        high_priority_topics=predictions.high_priority_topics,
        exam_tips=predictions.exam_tips,
        pattern_summary=pattern_summary,
        confidence_score=confidence
    )


def _extract_course_context(docs: List[Document]) -> str:
    """Extract relevant context from course material."""
    if not docs:
        return ""
    
    # Get non-PYQ docs
    book_docs = [d for d in docs if not d.metadata.get("is_pyq", False)]
    
    if not book_docs:
        return ""
    
    # Combine content (limited)
    context_parts = []
    total_chars = 0
    max_chars = 3000
    
    for doc in book_docs:
        if total_chars >= max_chars:
            break
        snippet = doc.page_content[:500]
        context_parts.append(snippet)
        total_chars += len(snippet)
    
    return "\n---\n".join(context_parts)


def _generate_predictions_llm(
    analysis: PatternAnalysis,
    course_context: str,
    num_predictions: int,
    prediction_style: str,
    model: str
) -> LLMPredictionOutput:
    """Use LLM to generate predictions."""
    
    # Build the prompt based on style
    style_instruction = ""
    if prediction_style == "conservative":
        style_instruction = """
Focus on HIGH CONFIDENCE predictions only:
- Questions from topics that appear EVERY year
- Question types that dominate the exam
- Clear, predictable patterns"""
    elif prediction_style == "exploratory":
        style_instruction = """
Include EXPLORATORY predictions:
- Gap topics that haven't been asked recently (likely to return)
- Emerging topics gaining importance
- Creative variations on common questions"""
    else:  # both
        style_instruction = f"""
Generate a MIX of predictions:
- {num_predictions // 2} HIGH CONFIDENCE predictions (consistent topics, clear patterns)
- {num_predictions - num_predictions // 2} EXPLORATORY predictions (gap topics, emerging trends)"""

    # Format pattern data
    topic_freq_str = "\n".join(f"- {t}: {c} times" for t, c in list(analysis.topic_frequency.items())[:15])
    type_freq_str = "\n".join(f"- {t}: {c}" for t, c in analysis.type_frequency.items())
    
    consistent_str = ", ".join(analysis.consistent_topics) if analysis.consistent_topics else "None identified"
    gap_str = ", ".join(analysis.gap_topics) if analysis.gap_topics else "None"
    emerging_str = ", ".join(analysis.emerging_topics) if analysis.emerging_topics else "None"
    years_str = ", ".join(map(str, analysis.years_covered))

    # Use explicit JSON example instead of PydanticOutputParser format_instructions
    # The small LLM was echoing the $defs schema back instead of generating data
    PREDICTION_TEMPLATE = """You are an expert exam question predictor for Machine Learning exams.

## Historical Pattern Analysis

**Years Analyzed**: {years}
**Total Questions**: {total}

### Topic Frequency:
{topic_freq}

### Question Type Distribution:
{type_freq}

### Consistent Topics (appear almost every year):
{consistent}

### Gap Topics (not asked recently - likely to return):
{gap}

### Emerging Topics (growing in importance):
{emerging}

{course_context}

## Prediction Style
{style}

## Task
Generate exactly {num_predictions} predicted exam questions.

Return a JSON object with this EXACT structure (fill in real data, do NOT repeat the schema):

```json
{{
  "predictions": [
    {{
      "question": "Derive the gradient descent update rule for linear regression with L2 regularization.",
      "topic": "Optimization",
      "sub_topic": "Gradient Descent",
      "question_type": "derivation",
      "difficulty": "hard",
      "confidence": "high",
      "reasoning": "Gradient descent appears in 3 out of 4 years",
      "similar_years": [2022, 2024],
      "study_tip": "Practice the derivation step by step"
    }}
  ],
  "high_priority_topics": ["Gradient Descent", "SVM", "Neural Networks"],
  "exam_tips": ["Focus on derivations - they carry the most marks"]
}}
```

IMPORTANT: Generate {num_predictions} actual predicted questions in the "predictions" array. 
For difficulty use ONLY: "easy", "medium", or "hard".
For confidence use ONLY: "high", "medium", or "low".

Return ONLY the JSON object. No markdown, no explanation, no schema definitions."""

    course_section = f"\n## Course Material Context:\n{course_context}" if course_context else ""
    
    prompt = PromptTemplate(
        template=PREDICTION_TEMPLATE,
        input_variables=[
            "years", "total", "topic_freq", "type_freq",
            "consistent", "gap", "emerging", "course_context",
            "style", "num_predictions"
        ],
    )
    
    llm = ChatGroq(model=model, temperature=0.3)
    chain = prompt | llm
    
    try:
        result = chain.invoke({
            "years": years_str,
            "total": analysis.total_questions,
            "topic_freq": topic_freq_str,
            "type_freq": type_freq_str,
            "consistent": consistent_str,
            "gap": gap_str,
            "emerging": emerging_str,
            "course_context": course_section,
            "style": style_instruction,
            "num_predictions": num_predictions
        })
        
        # Parse the LLM text output manually
        raw_text = result.content if hasattr(result, 'content') else str(result)
        
        # Extract JSON from response (handle markdown code blocks)
        json_text = raw_text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(json_text)
        
        # Build predictions manually with validation
        predictions = []
        for p in parsed.get("predictions", []):
            try:
                predictions.append(PredictedQuestion(
                    question=p.get("question", ""),
                    topic=p.get("topic", ""),
                    sub_topic=p.get("sub_topic", p.get("topic", "")),
                    question_type=p.get("question_type", "theory"),
                    difficulty=p.get("difficulty", "medium") if p.get("difficulty") in ("easy", "medium", "hard") else "medium",
                    confidence=p.get("confidence", "medium") if p.get("confidence") in ("high", "medium", "low") else "medium",
                    reasoning=p.get("reasoning", "Based on historical patterns"),
                    similar_years=p.get("similar_years", []),
                    study_tip=p.get("study_tip")
                ))
            except Exception:
                continue
        
        return LLMPredictionOutput(
            predictions=predictions,
            high_priority_topics=parsed.get("high_priority_topics", []),
            exam_tips=parsed.get("exam_tips", [])
        )
        
    except Exception as e:
        print(f"⚠️ LLM prediction failed: {e}")
        # Return simple fallback
        return LLMPredictionOutput(
            predictions=[
                PredictedQuestion(
                    question=f"Explain {topic} and its applications.",
                    topic=topic.split(" > ")[0] if " > " in topic else topic,
                    sub_topic=topic,
                    question_type="theory",
                    difficulty="medium",
                    confidence="medium",
                    reasoning="High frequency topic in historical data",
                    similar_years=analysis.years_covered[-2:] if analysis.years_covered else []
                )
                for topic in list(analysis.topic_frequency.keys())[:num_predictions]
            ],
            high_priority_topics=list(analysis.topic_frequency.keys())[:5],
            exam_tips=["Focus on high-frequency topics identified in pattern analysis."]
        )


def _calculate_confidence(analysis: PatternAnalysis) -> float:
    """Calculate overall prediction confidence."""
    if analysis.total_questions == 0:
        return 0.0
    
    # Factors that increase confidence:
    # - More questions analyzed
    # - More years covered
    # - Clear patterns (consistent topics)
    
    question_score = min(1.0, analysis.total_questions / 50)  # Max at 50 questions
    year_score = min(1.0, len(analysis.years_covered) / 5)  # Max at 5 years
    pattern_score = min(1.0, len(analysis.consistent_topics) / 5)  # Max at 5 consistent topics
    
    # Weighted average
    confidence = (
        question_score * 0.3 +
        year_score * 0.4 +
        pattern_score * 0.3
    )
    
    return round(confidence, 2)


# ============================================================================
# FULL PIPELINE
# ============================================================================

def run_prediction_pipeline(
    questions: List[AnalyzedQuestion],
    course_docs: Optional[List[Document]] = None,
    num_predictions: int = 10,
    model: str = "llama-3.1-8b-instant"
) -> PredictionResult:
    """
    Run the full prediction pipeline.
    
    This is the main entry point for generating predictions.
    """
    print("\n" + "=" * 60)
    print("🔮 EXAM QUESTION PREDICTION ENGINE")
    print("=" * 60)
    
    result = predict_questions(
        questions=questions,
        course_material=course_docs,
        num_predictions=num_predictions,
        model=model,
        prediction_style="both"
    )
    
    print(f"\n✅ Generated {len(result.predictions)} predictions")
    print(f"📊 Confidence Score: {result.confidence_score * 100:.0f}%")
    
    return result

"""
Marks Predictor Module

Responsibility: Predict marks distribution for upcoming exams.
Analyzes historical marks patterns by topic and question type.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from collections import defaultdict

from .topic_extractor import AnalyzedQuestion
from .predictor import PatternAnalysis, analyze_patterns


# ============================================================================
# DATA MODELS
# ============================================================================

class TopicMarksInfo(BaseModel):
    """Marks information for a topic."""
    topic: str = Field(description="Topic name")
    avg_marks: float = Field(description="Average marks per question")
    total_marks: int = Field(description="Total marks allocated historically")
    question_count: int = Field(description="Number of questions")
    marks_range: str = Field(description="Range of marks (e.g., '5-15')")
    predicted_marks: int = Field(description="Predicted marks in next exam")
    confidence: str = Field(description="Prediction confidence")


class TypeMarksInfo(BaseModel):
    """Marks information for a question type."""
    question_type: str = Field(description="Question type")
    avg_marks: float = Field(description="Average marks")
    typical_count: int = Field(description="Typical count per exam")
    predicted_total: int = Field(description="Predicted total marks")


class MarksPrediction(BaseModel):
    """Complete marks prediction for an exam."""
    total_predicted_marks: int = Field(description="Total predicted marks")
    by_topic: List[TopicMarksInfo] = Field(description="Marks by topic")
    by_type: List[TypeMarksInfo] = Field(description="Marks by question type")
    high_weightage_topics: List[str] = Field(description="Topics with highest marks")
    marks_distribution_summary: Dict[str, int] = Field(description="Category -> marks")
    strategy_tips: List[str] = Field(description="Exam strategy tips")
    confidence_score: float = Field(description="Overall confidence 0-1")


# ============================================================================
# MARKS ANALYSIS
# ============================================================================

def analyze_marks_distribution(
    questions: List[AnalyzedQuestion]
) -> Dict[str, Dict]:
    """
    Analyze historical marks distribution.
    
    Returns statistics for topics and question types.
    """
    topic_marks = defaultdict(list)
    type_marks = defaultdict(list)
    year_topic_marks = defaultdict(lambda: defaultdict(int))
    
    for q in questions:
        marks = q.marks or 5  # Default to 5 if not specified
        topic_marks[q.sub_topic].append(marks)
        type_marks[q.question_type].append(marks)
        
        if q.source_year:
            year_topic_marks[q.source_year][q.sub_topic] += marks
    
    # Calculate statistics
    topic_stats = {}
    for topic, marks_list in topic_marks.items():
        topic_stats[topic] = {
            "count": len(marks_list),
            "total": sum(marks_list),
            "avg": sum(marks_list) / len(marks_list),
            "min": min(marks_list),
            "max": max(marks_list),
        }
    
    type_stats = {}
    for qtype, marks_list in type_marks.items():
        type_stats[qtype] = {
            "count": len(marks_list),
            "total": sum(marks_list),
            "avg": sum(marks_list) / len(marks_list),
        }
    
    return {
        "by_topic": topic_stats,
        "by_type": type_stats,
        "by_year_topic": dict(year_topic_marks),
    }


def predict_marks(
    questions: List[AnalyzedQuestion],
    total_exam_marks: int = 100
) -> MarksPrediction:
    """
    Predict marks distribution for next exam.
    
    Args:
        questions: Historical analyzed questions
        total_exam_marks: Total marks for the exam
        
    Returns:
        MarksPrediction with detailed breakdown
    """
    if not questions:
        return MarksPrediction(
            total_predicted_marks=0,
            by_topic=[],
            by_type=[],
            high_weightage_topics=[],
            marks_distribution_summary={},
            strategy_tips=["No historical data available."],
            confidence_score=0.0
        )
    
    print(f"📊 Predicting marks distribution for {len(questions)} questions...")
    
    # Get marks analysis
    analysis = analyze_marks_distribution(questions)
    patterns = analyze_patterns(questions)
    
    # Get years for analysis
    years = sorted(set(q.source_year for q in questions if q.source_year))
    num_years = len(years)
    
    # === TOPIC MARKS PREDICTION ===
    topic_predictions = []
    total_avg_marks = 0
    
    for topic, stats in sorted(
        analysis["by_topic"].items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )[:15]:
        # Calculate predicted marks based on:
        # 1. Historical average
        # 2. Frequency (consistent topics get more marks)
        # 3. Recent trends
        
        avg_marks_per_exam = stats["total"] / num_years if num_years > 0 else stats["avg"]
        
        # Adjust for consistency
        is_consistent = topic in patterns.consistent_topics
        is_gap = topic in patterns.gap_topics
        
        if is_consistent:
            predicted = int(avg_marks_per_exam * 1.1)  # Slight increase for consistent
            confidence = "high"
        elif is_gap:
            predicted = int(avg_marks_per_exam * 0.8)  # Might appear with lower marks
            confidence = "medium"
        else:
            predicted = int(avg_marks_per_exam)
            confidence = "medium"
        
        # Cap at reasonable portion
        predicted = min(predicted, int(total_exam_marks * 0.25))
        
        topic_predictions.append(TopicMarksInfo(
            topic=topic,
            avg_marks=round(stats["avg"], 1),
            total_marks=stats["total"],
            question_count=stats["count"],
            marks_range=f"{stats['min']}-{stats['max']}",
            predicted_marks=predicted,
            confidence=confidence
        ))
        
        total_avg_marks += predicted
    
    # === TYPE MARKS PREDICTION ===
    type_predictions = []
    
    for qtype, stats in analysis["by_type"].items():
        avg_per_exam = stats["total"] / num_years if num_years > 0 else stats["total"]
        typical_count = stats["count"] // max(num_years, 1)
        
        type_predictions.append(TypeMarksInfo(
            question_type=qtype,
            avg_marks=round(stats["avg"], 1),
            typical_count=max(typical_count, 1),
            predicted_total=int(avg_per_exam)
        ))
    
    # Sort by predicted total
    type_predictions.sort(key=lambda x: x.predicted_total, reverse=True)
    
    # === HIGH WEIGHTAGE TOPICS ===
    high_weightage = [t.topic for t in topic_predictions[:5]]
    
    # === MARKS DISTRIBUTION SUMMARY ===
    distribution = defaultdict(int)
    for tp in topic_predictions:
        if tp.predicted_marks >= 15:
            distribution["High (15+)"] += tp.predicted_marks
        elif tp.predicted_marks >= 8:
            distribution["Medium (8-14)"] += tp.predicted_marks
        else:
            distribution["Low (1-7)"] += tp.predicted_marks
    
    # === STRATEGY TIPS ===
    tips = _generate_marks_strategy_tips(topic_predictions, type_predictions)
    
    # === CONFIDENCE SCORE ===
    confidence = min(num_years / 5, 1.0) * 0.5 + min(len(questions) / 50, 1.0) * 0.5
    
    return MarksPrediction(
        total_predicted_marks=total_exam_marks,
        by_topic=topic_predictions,
        by_type=type_predictions,
        high_weightage_topics=high_weightage,
        marks_distribution_summary=dict(distribution),
        strategy_tips=tips,
        confidence_score=round(confidence, 2)
    )


def _generate_marks_strategy_tips(
    topic_preds: List[TopicMarksInfo],
    type_preds: List[TypeMarksInfo]
) -> List[str]:
    """Generate exam strategy tips based on marks prediction."""
    tips = []
    
    # Top topics tip
    top_topics = [t.topic for t in topic_preds[:3]]
    if top_topics:
        tips.append(f"🎯 Prioritize these high-weightage topics: {', '.join(top_topics)}")
    
    # High marks per question tip
    high_avg = [t for t in topic_preds if t.avg_marks >= 10]
    if high_avg:
        tips.append(f"📝 Topics with 10+ marks/question: {', '.join(t.topic for t in high_avg[:3])}")
    
    # Question type tip
    if type_preds:
        top_type = type_preds[0]
        tips.append(f"📋 Most marks come from '{top_type.question_type}' questions (~{top_type.predicted_total} marks)")
    
    # High confidence topics
    high_conf = [t for t in topic_preds if t.confidence == "high"]
    if high_conf:
        tips.append(f"✅ Almost certain to appear: {', '.join(t.topic for t in high_conf[:3])}")
    
    # General tips
    tips.extend([
        "⏰ Allocate time proportional to marks (1 min per mark is a good rule)",
        "🎯 Attempt all high-weightage questions first",
    ])
    
    return tips


# ============================================================================
# FORMATTING
# ============================================================================

def format_marks_prediction(prediction: MarksPrediction) -> str:
    """Format marks prediction as readable text."""
    lines = [
        "=" * 70,
        "📊 MARKS PREDICTION REPORT",
        "=" * 70,
        "",
        f"Total Exam Marks: {prediction.total_predicted_marks}",
        f"Confidence: {prediction.confidence_score * 100:.0f}%",
        "",
    ]
    
    # High weightage topics
    lines.append("🔥 HIGH WEIGHTAGE TOPICS:")
    for topic in prediction.high_weightage_topics:
        lines.append(f"   • {topic}")
    
    lines.append("")
    
    # Topic-wise breakdown
    lines.append("-" * 50)
    lines.append("📚 TOPIC-WISE MARKS PREDICTION")
    lines.append("-" * 50)
    lines.append(f"{'Topic':<30} {'Predicted':<12} {'Avg':<8} {'Conf':<8}")
    lines.append("-" * 50)
    
    for tp in prediction.by_topic:
        lines.append(
            f"{tp.topic[:30]:<30} {tp.predicted_marks:<12} {tp.avg_marks:<8.1f} {tp.confidence:<8}"
        )
    
    lines.append("")
    
    # Type-wise breakdown
    lines.append("-" * 50)
    lines.append("📝 QUESTION TYPE MARKS PREDICTION")
    lines.append("-" * 50)
    lines.append(f"{'Type':<20} {'Avg Marks':<12} {'Count':<10} {'Total':<10}")
    lines.append("-" * 50)
    
    for tp in prediction.by_type:
        lines.append(
            f"{tp.question_type:<20} {tp.avg_marks:<12.1f} {tp.typical_count:<10} {tp.predicted_total:<10}"
        )
    
    lines.append("")
    
    # Distribution summary
    lines.append("📊 MARKS DISTRIBUTION:")
    for category, marks in prediction.marks_distribution_summary.items():
        bar_len = marks // 5
        bar = "█" * bar_len
        lines.append(f"   {category}: {bar} {marks} marks")
    
    lines.append("")
    
    # Strategy tips
    lines.append("=" * 70)
    lines.append("💡 EXAM STRATEGY TIPS")
    lines.append("=" * 70)
    for tip in prediction.strategy_tips:
        lines.append(f"   {tip}")
    
    return "\n".join(lines)


def save_marks_prediction(prediction: MarksPrediction, filepath: str) -> None:
    """Save marks prediction to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(format_marks_prediction(prediction))
    print(f"💾 Saved marks prediction to: {filepath}")

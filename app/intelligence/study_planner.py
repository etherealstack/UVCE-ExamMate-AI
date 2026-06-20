"""
Smart Study Planner Module

Responsibility: Generate priority-based study schedules.
Allocates time based on topic importance, difficulty, and exam patterns.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from collections import defaultdict
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .topic_extractor import AnalyzedQuestion
from .predictor import PatternAnalysis, analyze_patterns


# ============================================================================
# DATA MODELS
# ============================================================================

class StudyTask(BaseModel):
    """A specific study task."""
    task: str = Field(description="What to study/do")
    topic: str = Field(description="Topic being covered")
    duration_minutes: int = Field(description="Suggested time in minutes")
    priority: str = Field(description="high/medium/low")
    task_type: str = Field(description="read/practice/revise/solve")
    resources: List[str] = Field(default_factory=list, description="Suggested resources")


class StudyDay(BaseModel):
    """A day in the study plan."""
    day_number: int = Field(description="Day number")
    date: Optional[str] = Field(default=None, description="Actual date if provided")
    total_hours: float = Field(description="Total study hours for the day")
    tasks: List[StudyTask] = Field(description="Tasks for the day")
    focus_area: str = Field(description="Main focus for the day")
    notes: str = Field(default="", description="Additional notes")


class StudyPlan(BaseModel):
    """Complete study plan."""
    title: str = Field(description="Plan title")
    exam_date: Optional[str] = Field(default=None, description="Target exam date")
    total_days: int = Field(description="Number of days in the plan")
    hours_per_day: float = Field(description="Average hours per day")
    days: List[StudyDay] = Field(description="Day-by-day plan")
    high_priority_topics: List[str] = Field(description="Topics to prioritize")
    gap_topics: List[str] = Field(description="Topics needing extra attention")
    weekly_goals: List[str] = Field(description="Weekly milestones")
    tips: List[str] = Field(description="Study tips")


class StudyPlanConfig(BaseModel):
    """Configuration for study plan generation."""
    days_until_exam: int = Field(default=30, description="Days until exam")
    hours_per_day: float = Field(default=4, description="Study hours per day")
    include_revision: bool = Field(default=True, description="Include revision days")
    revision_days: int = Field(default=5, description="Days reserved for revision")
    subject: str = Field(default="Machine Learning", description="Subject name")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")


# ============================================================================
# TOPIC PRIORITIZATION
# ============================================================================

def calculate_topic_priority(
    topic: str,
    patterns: PatternAnalysis,
    questions: List[AnalyzedQuestion]
) -> Dict:
    """
    Calculate priority score for a topic.
    
    Factors:
    - Frequency in past exams
    - Recency (recent appearance = lower priority, gap = higher)
    - Difficulty distribution
    - Marks weightage
    """
    
    # Get topic stats
    frequency = patterns.topic_frequency.get(topic, 0)
    total = sum(patterns.topic_frequency.values())
    frequency_score = frequency / total if total > 0 else 0
    
    # Check if gap topic (not asked recently)
    is_gap = topic in patterns.gap_topics
    gap_score = 0.3 if is_gap else 0
    
    # Check if consistent topic
    is_consistent = topic in patterns.consistent_topics
    consistent_score = 0.2 if is_consistent else 0
    
    # Calculate difficulty (harder = more study time needed)
    topic_questions = [q for q in questions if q.sub_topic == topic]
    hard_count = sum(1 for q in topic_questions if q.difficulty_hint == "hard")
    difficulty_score = hard_count / len(topic_questions) if topic_questions else 0
    
    # Calculate marks weightage
    total_marks = sum(q.marks or 5 for q in topic_questions)
    avg_marks = total_marks / len(topic_questions) if topic_questions else 0
    marks_score = min(avg_marks / 10, 1)  # Normalize to 0-1
    
    # Combined priority score
    priority_score = (
        frequency_score * 0.3 +
        gap_score * 0.25 +
        consistent_score * 0.15 +
        difficulty_score * 0.15 +
        marks_score * 0.15
    )
    
    # Determine priority level
    if priority_score > 0.4:
        priority = "high"
    elif priority_score > 0.2:
        priority = "medium"
    else:
        priority = "low"
    
    return {
        "topic": topic,
        "score": priority_score,
        "priority": priority,
        "frequency": frequency,
        "is_gap": is_gap,
        "is_consistent": is_consistent,
        "avg_marks": avg_marks,
        "estimated_hours": max(1, int(priority_score * 8))  # 1-8 hours
    }


# ============================================================================
# STUDY PLAN GENERATION
# ============================================================================

def generate_study_plan(
    questions: List[AnalyzedQuestion],
    config: Optional[StudyPlanConfig] = None,
    model: str = "llama-3.1-8b-instant"
) -> StudyPlan:
    """
    Generate a smart study plan based on exam patterns.
    
    Args:
        questions: Historical analyzed questions
        config: Study plan configuration
        model: LLM model for task generation
        
    Returns:
        StudyPlan with day-by-day schedule
    """
    if config is None:
        config = StudyPlanConfig()
    
    print(f"📅 Generating {config.days_until_exam}-day study plan...")
    
    # Analyze patterns
    patterns = analyze_patterns(questions)
    
    # Get unique topics and prioritize
    topics = list(patterns.topic_frequency.keys())
    topic_priorities = [
        calculate_topic_priority(t, patterns, questions)
        for t in topics
    ]
    
    # Sort by priority score
    topic_priorities.sort(key=lambda x: x["score"], reverse=True)
    
    # Separate high/medium/low priority topics
    high_priority = [t for t in topic_priorities if t["priority"] == "high"]
    medium_priority = [t for t in topic_priorities if t["priority"] == "medium"]
    low_priority = [t for t in topic_priorities if t["priority"] == "low"]
    
    print(f"   High priority topics: {len(high_priority)}")
    print(f"   Medium priority topics: {len(medium_priority)}")
    print(f"   Low priority topics: {len(low_priority)}")
    
    # Generate days
    days = []
    available_days = config.days_until_exam - (config.revision_days if config.include_revision else 0)
    
    # Distribute topics across days
    # High priority: First 40% of days
    # Medium priority: Next 35% of days
    # Low priority: Next 15% of days
    # Revision: Last days
    
    high_days = int(available_days * 0.4)
    medium_days = int(available_days * 0.35)
    low_days = available_days - high_days - medium_days
    
    current_day = 1
    start_date = datetime.strptime(config.start_date, "%Y-%m-%d") if config.start_date else datetime.now()
    
    # High priority days
    for i, priority_info in enumerate(high_priority[:high_days]):
        day = _create_study_day(
            day_number=current_day,
            date=start_date + timedelta(days=current_day - 1),
            topic_info=priority_info,
            hours=config.hours_per_day,
            phase="intensive"
        )
        days.append(day)
        current_day += 1
    
    # Medium priority days
    for i, priority_info in enumerate(medium_priority[:medium_days]):
        day = _create_study_day(
            day_number=current_day,
            date=start_date + timedelta(days=current_day - 1),
            topic_info=priority_info,
            hours=config.hours_per_day,
            phase="reinforcement"
        )
        days.append(day)
        current_day += 1
    
    # Low priority days
    for i, priority_info in enumerate(low_priority[:low_days]):
        day = _create_study_day(
            day_number=current_day,
            date=start_date + timedelta(days=current_day - 1),
            topic_info=priority_info,
            hours=config.hours_per_day * 0.8,  # Slightly less time
            phase="coverage"
        )
        days.append(day)
        current_day += 1
    
    # Revision days
    if config.include_revision:
        for i in range(config.revision_days):
            revision_topics = [t["topic"] for t in topic_priorities[:5]]
            day = StudyDay(
                day_number=current_day,
                date=(start_date + timedelta(days=current_day - 1)).strftime("%Y-%m-%d"),
                total_hours=config.hours_per_day,
                tasks=[
                    StudyTask(
                        task="Revise key formulas and concepts",
                        topic="All Topics",
                        duration_minutes=60,
                        priority="high",
                        task_type="revise",
                        resources=["Notes", "Flashcards"]
                    ),
                    StudyTask(
                        task="Practice previous year questions",
                        topic=", ".join(revision_topics[:3]),
                        duration_minutes=90,
                        priority="high",
                        task_type="practice",
                        resources=["PYQs"]
                    ),
                    StudyTask(
                        task="Mock test",
                        topic="All Topics",
                        duration_minutes=90,
                        priority="high",
                        task_type="solve",
                        resources=["Mock Paper"]
                    ),
                ],
                focus_area="Revision & Practice",
                notes="Focus on weak areas and quick revision"
            )
            days.append(day)
            current_day += 1
    
    # Generate weekly goals
    weekly_goals = _generate_weekly_goals(days)
    
    # Generate tips
    tips = _generate_study_tips(patterns, topic_priorities)
    
    return StudyPlan(
        title=f"{config.subject} - {config.days_until_exam} Day Study Plan",
        exam_date=config.start_date,
        total_days=len(days),
        hours_per_day=config.hours_per_day,
        days=days,
        high_priority_topics=[t["topic"] for t in high_priority[:7]],
        gap_topics=patterns.gap_topics[:5],
        weekly_goals=weekly_goals,
        tips=tips
    )


def _create_study_day(
    day_number: int,
    date: datetime,
    topic_info: Dict,
    hours: float,
    phase: str
) -> StudyDay:
    """Create a study day with tasks."""
    
    topic = topic_info["topic"]
    priority = topic_info["priority"]
    is_gap = topic_info.get("is_gap", False)
    
    tasks = []
    remaining_minutes = int(hours * 60)
    
    # Reading/Understanding phase (30%)
    read_time = int(remaining_minutes * 0.3)
    tasks.append(StudyTask(
        task=f"Read and understand {topic} concepts",
        topic=topic,
        duration_minutes=read_time,
        priority=priority,
        task_type="read",
        resources=["Textbook", "Notes", "Online resources"]
    ))
    remaining_minutes -= read_time
    
    # Practice phase (40%)
    practice_time = int(remaining_minutes * 0.5)
    tasks.append(StudyTask(
        task=f"Solve practice problems on {topic}",
        topic=topic,
        duration_minutes=practice_time,
        priority=priority,
        task_type="practice",
        resources=["Practice problems", "PYQs"]
    ))
    remaining_minutes -= practice_time
    
    # Notes/Summary phase (20%)
    notes_time = int(remaining_minutes * 0.5)
    tasks.append(StudyTask(
        task=f"Create summary notes for {topic}",
        topic=topic,
        duration_minutes=notes_time,
        priority="medium",
        task_type="revise",
        resources=["Notebook", "Flashcards"]
    ))
    remaining_minutes -= notes_time
    
    # Extra focus if gap topic
    if is_gap and remaining_minutes > 15:
        tasks.append(StudyTask(
            task=f"Extra practice (gap topic)",
            topic=topic,
            duration_minutes=remaining_minutes,
            priority="high",
            task_type="practice",
            resources=["Additional problems"]
        ))
    
    notes = ""
    if is_gap:
        notes = "⚠️ Gap topic - give extra attention!"
    elif phase == "intensive":
        notes = "🔥 High-priority day - focus intensely!"
    
    return StudyDay(
        day_number=day_number,
        date=date.strftime("%Y-%m-%d"),
        total_hours=hours,
        tasks=tasks,
        focus_area=topic,
        notes=notes
    )


def _generate_weekly_goals(days: List[StudyDay]) -> List[str]:
    """Generate weekly milestone goals."""
    goals = []
    
    week = 1
    for i in range(0, len(days), 7):
        week_days = days[i:i+7]
        topics_covered = list(set(d.focus_area for d in week_days))[:3]
        goals.append(f"Week {week}: Master {', '.join(topics_covered)}")
        week += 1
    
    return goals


def _generate_study_tips(patterns: PatternAnalysis, priorities: List[Dict]) -> List[str]:
    """Generate study tips based on analysis."""
    tips = [
        "Start each session with a quick review of yesterday's topics.",
        "Use active recall - test yourself frequently.",
        "Take regular breaks (Pomodoro technique: 25 min study, 5 min break).",
    ]
    
    high_priority = [p for p in priorities if p["priority"] == "high"]
    if high_priority:
        tips.append(f"Focus heavily on: {', '.join(p['topic'] for p in high_priority[:3])}")
    
    if patterns.gap_topics:
        tips.append(f"Don't neglect gap topics: {', '.join(patterns.gap_topics[:3])}")
    
    tips.append("Practice with timed mock tests in the final week.")
    
    return tips


# ============================================================================
# FORMATTING
# ============================================================================

def format_study_plan(plan: StudyPlan) -> str:
    """Format study plan as readable text."""
    lines = [
        "=" * 70,
        f"📅 {plan.title}",
        "=" * 70,
        "",
        f"Total Days: {plan.total_days}",
        f"Hours per Day: {plan.hours_per_day}",
        "",
    ]
    
    if plan.exam_date:
        lines.append(f"Target Exam: {plan.exam_date}")
        lines.append("")
    
    # High priority topics
    lines.append("🔥 HIGH PRIORITY TOPICS:")
    for topic in plan.high_priority_topics:
        lines.append(f"   • {topic}")
    
    lines.append("")
    lines.append("⚠️ GAP TOPICS (Need Extra Attention):")
    for topic in plan.gap_topics:
        lines.append(f"   • {topic}")
    
    lines.append("")
    lines.append("🎯 WEEKLY GOALS:")
    for goal in plan.weekly_goals:
        lines.append(f"   {goal}")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("📋 DAY-BY-DAY SCHEDULE")
    lines.append("-" * 70)
    
    for day in plan.days:
        lines.append("")
        lines.append(f"📆 Day {day.day_number}" + (f" ({day.date})" if day.date else ""))
        lines.append(f"   Focus: {day.focus_area} | Total: {day.total_hours}h")
        if day.notes:
            lines.append(f"   📝 {day.notes}")
        
        for task in day.tasks:
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "⚪")
            lines.append(f"   {priority_icon} [{task.duration_minutes}min] {task.task}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("💡 STUDY TIPS:")
    for tip in plan.tips:
        lines.append(f"   👉 {tip}")
    
    return "\n".join(lines)


def save_study_plan(plan: StudyPlan, filepath: str) -> None:
    """Save study plan to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(format_study_plan(plan))
    print(f"💾 Saved study plan to: {filepath}")

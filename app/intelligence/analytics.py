"""
Visual Analytics Module

Responsibility: Generate visual analytics and charts for exam pattern analysis.
Creates topic trends, question type distributions, and difficulty heatmaps.
"""

import os
from typing import List, Dict, Optional
from collections import defaultdict
from pydantic import BaseModel, Field

from .topic_extractor import AnalyzedQuestion
from .predictor import PatternAnalysis, analyze_patterns


# ============================================================================
# DATA MODELS
# ============================================================================

class TopicTrend(BaseModel):
    """Trend data for a topic over years."""
    topic: str
    yearly_counts: Dict[int, int]  # year -> count
    trend: str  # "increasing", "decreasing", "stable", "sporadic"
    total_appearances: int


class TypeDistribution(BaseModel):
    """Distribution of question types."""
    type_name: str
    count: int
    percentage: float


class DifficultyStats(BaseModel):
    """Difficulty statistics by topic."""
    topic: str
    easy_count: int
    medium_count: int
    hard_count: int
    avg_marks: float


class AnalyticsReport(BaseModel):
    """Complete analytics report."""
    total_questions: int
    years_analyzed: List[int]
    topic_trends: List[TopicTrend]
    type_distribution: List[TypeDistribution]
    difficulty_by_topic: List[DifficultyStats]
    year_over_year_growth: Dict[int, int]
    hottest_topics: List[str]
    coldest_topics: List[str]
    recommendations: List[str]


# ============================================================================
# ANALYTICS FUNCTIONS
# ============================================================================

def generate_analytics_report(
    questions: List[AnalyzedQuestion]
) -> AnalyticsReport:
    """
    Generate comprehensive analytics report.
    
    Args:
        questions: List of analyzed questions
        
    Returns:
        AnalyticsReport with all statistics
    """
    if not questions:
        return AnalyticsReport(
            total_questions=0,
            years_analyzed=[],
            topic_trends=[],
            type_distribution=[],
            difficulty_by_topic=[],
            year_over_year_growth={},
            hottest_topics=[],
            coldest_topics=[],
            recommendations=["No data available for analysis."]
        )
    
    print(f"📊 Generating analytics for {len(questions)} questions...")
    
    # Analyze patterns
    patterns = analyze_patterns(questions)
    
    # Calculate topic trends
    topic_trends = _calculate_topic_trends(questions)
    
    # Calculate type distribution
    type_dist = _calculate_type_distribution(questions)
    
    # Calculate difficulty by topic
    difficulty_stats = _calculate_difficulty_stats(questions)
    
    # Year over year counts
    yoy_counts = defaultdict(int)
    for q in questions:
        if q.source_year:
            yoy_counts[q.source_year] += 1
    
    # Find hottest and coldest topics
    sorted_trends = sorted(topic_trends, key=lambda t: t.total_appearances, reverse=True)
    hottest = [t.topic for t in sorted_trends[:5]]
    coldest = [t.topic for t in sorted_trends[-5:] if t.total_appearances < 3]
    
    # Generate recommendations
    recommendations = _generate_recommendations(topic_trends, type_dist, patterns)
    
    return AnalyticsReport(
        total_questions=len(questions),
        years_analyzed=sorted(patterns.years_covered),
        topic_trends=topic_trends,
        type_distribution=type_dist,
        difficulty_by_topic=difficulty_stats,
        year_over_year_growth=dict(yoy_counts),
        hottest_topics=hottest,
        coldest_topics=coldest,
        recommendations=recommendations
    )


def _calculate_topic_trends(questions: List[AnalyzedQuestion]) -> List[TopicTrend]:
    """Calculate trends for each topic over years."""
    
    # Count by topic and year
    topic_year_counts = defaultdict(lambda: defaultdict(int))
    
    for q in questions:
        if q.source_year:
            topic_year_counts[q.sub_topic][q.source_year] += 1
    
    trends = []
    for topic, year_counts in topic_year_counts.items():
        yearly_counts = dict(year_counts)
        total = sum(yearly_counts.values())
        
        # Determine trend direction
        years = sorted(yearly_counts.keys())
        if len(years) >= 2:
            first_half = sum(yearly_counts.get(y, 0) for y in years[:len(years)//2])
            second_half = sum(yearly_counts.get(y, 0) for y in years[len(years)//2:])
            
            if second_half > first_half * 1.5:
                trend = "increasing"
            elif first_half > second_half * 1.5:
                trend = "decreasing"
            elif len(yearly_counts) == len(years):
                trend = "stable"
            else:
                trend = "sporadic"
        else:
            trend = "insufficient_data"
        
        trends.append(TopicTrend(
            topic=topic,
            yearly_counts=yearly_counts,
            trend=trend,
            total_appearances=total
        ))
    
    return sorted(trends, key=lambda t: t.total_appearances, reverse=True)


def _calculate_type_distribution(questions: List[AnalyzedQuestion]) -> List[TypeDistribution]:
    """Calculate distribution of question types."""
    type_counts = defaultdict(int)
    
    for q in questions:
        type_counts[q.question_type] += 1
    
    total = len(questions)
    
    return [
        TypeDistribution(
            type_name=qtype,
            count=count,
            percentage=round(count / total * 100, 1) if total > 0 else 0
        )
        for qtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    ]


def _calculate_difficulty_stats(questions: List[AnalyzedQuestion]) -> List[DifficultyStats]:
    """Calculate difficulty statistics by topic."""
    topic_difficulty = defaultdict(lambda: {"easy": 0, "medium": 0, "hard": 0, "marks": [], "count": 0})
    
    for q in questions:
        diff = q.difficulty_hint or "medium"
        topic_difficulty[q.sub_topic][diff] += 1
        topic_difficulty[q.sub_topic]["count"] += 1
        if q.marks:
            topic_difficulty[q.sub_topic]["marks"].append(q.marks)
    
    stats = []
    for topic, data in topic_difficulty.items():
        avg_marks = sum(data["marks"]) / len(data["marks"]) if data["marks"] else 0
        stats.append(DifficultyStats(
            topic=topic,
            easy_count=data["easy"],
            medium_count=data["medium"],
            hard_count=data["hard"],
            avg_marks=round(avg_marks, 1)
        ))
    
    return sorted(stats, key=lambda s: s.easy_count + s.medium_count + s.hard_count, reverse=True)


def _generate_recommendations(
    trends: List[TopicTrend],
    type_dist: List[TypeDistribution],
    patterns: PatternAnalysis
) -> List[str]:
    """Generate study recommendations based on analytics."""
    recommendations = []
    
    # Topic recommendations
    increasing_topics = [t.topic for t in trends if t.trend == "increasing"][:3]
    if increasing_topics:
        recommendations.append(f"📈 Rising topics to focus on: {', '.join(increasing_topics)}")
    
    stable_topics = [t.topic for t in trends if t.trend == "stable"][:3]
    if stable_topics:
        recommendations.append(f"✅ Consistently important: {', '.join(stable_topics)}")
    
    # Type recommendations
    if type_dist:
        top_type = type_dist[0]
        recommendations.append(f"📝 {top_type.type_name.title()} questions dominate ({top_type.percentage}%). Practice this format.")
    
    # Gap topic recommendations
    if patterns.gap_topics:
        recommendations.append(f"⚠️ Gap topics (likely to return): {', '.join(patterns.gap_topics[:3])}")
    
    # General tips
    recommendations.append("💡 Cover all difficulty levels - exams typically have a mix.")
    
    return recommendations


# ============================================================================
# TEXT-BASED VISUALIZATION
# ============================================================================

def format_analytics_report(report: AnalyticsReport) -> str:
    """Format analytics report as text with ASCII charts."""
    lines = [
        "=" * 70,
        "📊 EXAM ANALYTICS REPORT",
        "=" * 70,
        "",
        f"Total Questions Analyzed: {report.total_questions}",
        f"Years Covered: {', '.join(map(str, report.years_analyzed))}",
        "",
    ]
    
    # Topic Trends Bar Chart
    lines.append("─" * 50)
    lines.append("📈 TOPIC FREQUENCY (Top 15)")
    lines.append("─" * 50)
    
    max_count = max((t.total_appearances for t in report.topic_trends[:15]), default=1)
    for trend in report.topic_trends[:15]:
        bar_len = int(trend.total_appearances / max_count * 30)
        bar = "█" * bar_len
        trend_icon = {"increasing": "📈", "decreasing": "📉", "stable": "➡️", "sporadic": "❓"}.get(trend.trend, "")
        lines.append(f"  {trend.topic[:25]:<25} {bar} {trend.total_appearances} {trend_icon}")
    
    lines.append("")
    
    # Year Over Year
    lines.append("─" * 50)
    lines.append("📅 QUESTIONS PER YEAR")
    lines.append("─" * 50)
    
    if report.year_over_year_growth:
        max_yoy = max(report.year_over_year_growth.values())
        for year, count in sorted(report.year_over_year_growth.items()):
            bar_len = int(count / max_yoy * 40)
            bar = "▓" * bar_len
            lines.append(f"  {year}: {bar} {count}")
    
    lines.append("")
    
    # Question Type Distribution (Pie Chart as text)
    lines.append("─" * 50)
    lines.append("📝 QUESTION TYPE DISTRIBUTION")
    lines.append("─" * 50)
    
    for td in report.type_distribution:
        pct_bar = "●" * int(td.percentage / 5)  # Scale down
        lines.append(f"  {td.type_name:<15} {pct_bar} {td.percentage}%")
    
    lines.append("")
    
    # Difficulty Heatmap
    lines.append("─" * 50)
    lines.append("🎯 DIFFICULTY BY TOPIC (Top 10)")
    lines.append("─" * 50)
    lines.append("  " + "-" * 40)
    lines.append(f"  {'Topic':<20} {'Easy':>6} {'Med':>6} {'Hard':>6}")
    lines.append("  " + "-" * 40)
    
    for ds in report.difficulty_by_topic[:10]:
        easy_bar = "🟢" * min(ds.easy_count, 5)
        med_bar = "🟡" * min(ds.medium_count, 5)
        hard_bar = "🔴" * min(ds.hard_count, 5)
        lines.append(f"  {ds.topic[:20]:<20} {easy_bar:>6} {med_bar:>6} {hard_bar:>6}")
    
    lines.append("")
    
    # Hottest & Coldest Topics
    lines.append("─" * 50)
    lines.append("🔥 HOTTEST TOPICS (Most Frequent)")
    for topic in report.hottest_topics:
        lines.append(f"  🔥 {topic}")
    
    lines.append("")
    lines.append("❄️ COLDEST TOPICS (Least Frequent)")
    for topic in report.coldest_topics:
        lines.append(f"  ❄️ {topic}")
    
    lines.append("")
    
    # Recommendations
    lines.append("=" * 70)
    lines.append("💡 RECOMMENDATIONS")
    lines.append("=" * 70)
    for rec in report.recommendations:
        lines.append(f"  {rec}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def save_analytics_report(report: AnalyticsReport, filepath: str) -> None:
    """Save analytics report to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(format_analytics_report(report))
    print(f"💾 Saved analytics report to: {filepath}")


# ============================================================================
# HTML CHART GENERATION (Optional - for web display)
# ============================================================================

def generate_html_charts(report: AnalyticsReport) -> str:
    """Generate HTML with Chart.js visualizations."""
    
    # Prepare data for charts
    topic_labels = [t.topic for t in report.topic_trends[:10]]
    topic_values = [t.total_appearances for t in report.topic_trends[:10]]
    
    type_labels = [t.type_name for t in report.type_distribution]
    type_values = [t.count for t in report.type_distribution]
    
    year_labels = sorted(report.year_over_year_growth.keys())
    year_values = [report.year_over_year_growth[y] for y in year_labels]
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>UVCEExamMate Ai Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .chart-container {{ background: #16213e; border-radius: 10px; padding: 20px; margin: 20px 0; }}
        h1 {{ color: #e94560; }}
        h2 {{ color: #0f3460; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .recommendation {{ background: #0f3460; padding: 10px; border-radius: 5px; margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 UVCEExamMate Ai Analytics Dashboard</h1>
        <p>Total Questions: {report.total_questions} | Years: {', '.join(map(str, report.years_analyzed))}</p>
        
        <div class="grid">
            <div class="chart-container">
                <canvas id="topicChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="typeChart"></canvas>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="yearChart"></canvas>
        </div>
        
        <h2>💡 Recommendations</h2>
        {"".join(f'<div class="recommendation">{r}</div>' for r in report.recommendations)}
    </div>
    
    <script>
        // Topic Frequency Chart
        new Chart(document.getElementById('topicChart'), {{
            type: 'bar',
            data: {{
                labels: {topic_labels},
                datasets: [{{
                    label: 'Question Count',
                    data: {topic_values},
                    backgroundColor: 'rgba(233, 69, 96, 0.7)',
                    borderColor: 'rgba(233, 69, 96, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Topic Frequency' }} }}
            }}
        }});
        
        // Type Distribution Chart
        new Chart(document.getElementById('typeChart'), {{
            type: 'doughnut',
            data: {{
                labels: {type_labels},
                datasets: [{{
                    data: {type_values},
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)',
                        'rgba(255, 159, 64, 0.7)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Question Types' }} }}
            }}
        }});
        
        // Year Trend Chart
        new Chart(document.getElementById('yearChart'), {{
            type: 'line',
            data: {{
                labels: {[str(y) for y in year_labels]},
                datasets: [{{
                    label: 'Questions per Year',
                    data: {year_values},
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: true,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)'
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Year over Year Trend' }} }}
            }}
        }});
    </script>
</body>
</html>'''
    
    return html


def save_html_charts(report: AnalyticsReport, filepath: str) -> None:
    """Save HTML charts to file."""
    html = generate_html_charts(report)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"💾 Saved HTML charts to: {filepath}")

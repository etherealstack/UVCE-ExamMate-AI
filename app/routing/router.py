def route_query(query: str) -> str:
    """
    Routes the user query to the appropriate handler.

    Returns:
        - "exam_analysis" for exam-oriented queries
        - "concept_qa" for conceptual explanation queries

    NOTE:
    - Defaults to concept_qa if no strong exam intent is detected
    """
    query_lower = query.lower()

    strong_exam_keywords = [
        "exam", "question paper", "previous year", "pyq",
        "marks", "syllabus", "exam pattern", "mock test",
        "predict", "likely", "important", "repeated", "trend", "weightage"
    ]

    weak_exam_keywords = [
        "test", "assessment", "difficulty"
    ]

    # Strong signals → exam analysis
    if any(k in query_lower for k in strong_exam_keywords):
        return "exam_analysis"

    # Weak signals require reinforcement
    if any(k in query_lower for k in weak_exam_keywords):
        if "exam" in query_lower or "paper" in query_lower:
            return "exam_analysis"

    # Default: concept explanation
    return "concept_qa"
"""
Intelligence Module

Provides comprehensive exam analysis, prediction, and study tools.

Features:
- Question Extraction & Classification
- Topic Analysis & Pattern Detection
- Intelligent Question Prediction
- Mock Exam Generation
- Answer Key Generation
- Visual Analytics
- Smart Study Planning
- Concept Graph & Learning Paths
- Multi-Subject Support
- Marks Prediction
- Interactive Practice Mode
"""

# ============================================================================
# CORE: Question Processing
# ============================================================================

from .question_extractor import (
    ExtractedQuestion,
    extract_all_questions,
    extract_questions_from_doc,
    get_extraction_stats,
)

from .question_classifier import (
    ClassifiedQuestion,
    QuestionType,
    classify_question,
    classify_questions,
    get_classification_stats,
)

from .topic_extractor import (
    AnalyzedQuestion,
    TopicInfo,
    analyze_questions_with_topics,
    extract_topic,
    extract_topics_from_course_material,
    get_topic_stats,
)

# ============================================================================
# PREDICTION & ANALYSIS
# ============================================================================

from .predictor import (
    PatternAnalysis,
    PredictedQuestion,
    PredictionResult,
    analyze_patterns,
    predict_questions,
    run_prediction_pipeline,
)

from .pyq_analysis import (
    detect_repeated_questions,
    get_high_frequency_topics,
    summarize_patterns,
)

# ============================================================================
# ENHANCED FEATURES
# ============================================================================

from .mock_exam_generator import (
    MockExamPaper,
    MockQuestion,
    MockExamSection,
    ExamConfig,
    generate_mock_exam,
    format_mock_exam,
    save_mock_exam,
)

from .answer_generator import (
    AnswerKey,
    ModelAnswer,
    generate_answer_key,
    generate_model_answer,
    format_answer_key,
    save_answer_key,
)

from .analytics import (
    AnalyticsReport,
    TopicTrend,
    generate_analytics_report,
    format_analytics_report,
    generate_html_charts,
    save_analytics_report,
    save_html_charts,
)

from .study_planner import (
    StudyPlan,
    StudyDay,
    StudyTask,
    StudyPlanConfig,
    generate_study_plan,
    format_study_plan,
    save_study_plan,
)

from .concept_graph import (
    ConceptGraph,
    ConceptNode,
    ConceptEdge,
    LearningPath,
    build_concept_graph,
    generate_learning_path,
    get_prerequisites,
    find_knowledge_gaps,
    format_concept_graph,
    generate_mermaid_graph,
    save_concept_graph,
)

from .subject_config import (
    SubjectConfig,
    get_subject,
    get_all_subjects,
    register_subject,
    get_subject_topics,
    detect_subject_from_text,
    create_custom_subject,
    list_subjects,
    show_subject_details,
)

from .marks_predictor import (
    MarksPrediction,
    TopicMarksInfo,
    predict_marks,
    format_marks_prediction,
    save_marks_prediction,
)

from .practice_mode import (
    PracticeSession,
    PracticeQuestion,
    AnswerEvaluation,
    create_practice_session,
    get_current_question,
    submit_answer,
    evaluate_answer,
    get_session_summary,
    format_question,
    format_evaluation,
    format_session_summary,
)


# ============================================================================
# ALL EXPORTS
# ============================================================================

__all__ = [
    # Extraction
    "ExtractedQuestion", "extract_all_questions", "extract_questions_from_doc", "get_extraction_stats",
    # Classification
    "ClassifiedQuestion", "QuestionType", "classify_question", "classify_questions", "get_classification_stats",
    # Topics
    "AnalyzedQuestion", "TopicInfo", "analyze_questions_with_topics", "extract_topic", 
    "extract_topics_from_course_material", "get_topic_stats",
    # Prediction
    "PatternAnalysis", "PredictedQuestion", "PredictionResult", "analyze_patterns",
    "predict_questions", "run_prediction_pipeline",
    # Legacy
    "detect_repeated_questions", "get_high_frequency_topics", "summarize_patterns",
    # Mock Exam
    "MockExamPaper", "MockQuestion", "ExamConfig", "generate_mock_exam", 
    "format_mock_exam", "save_mock_exam",
    # Answer Key
    "AnswerKey", "ModelAnswer", "generate_answer_key", "format_answer_key", "save_answer_key",
    # Analytics
    "AnalyticsReport", "generate_analytics_report", "format_analytics_report",
    "generate_html_charts", "save_analytics_report", "save_html_charts",
    # Study Planner
    "StudyPlan", "StudyPlanConfig", "generate_study_plan", "format_study_plan", "save_study_plan",
    # Concept Graph
    "ConceptGraph", "LearningPath", "build_concept_graph", "generate_learning_path",
    "get_prerequisites", "find_knowledge_gaps", "format_concept_graph", "save_concept_graph",
    # Subject Config
    "SubjectConfig", "get_subject", "get_all_subjects", "register_subject",
    "get_subject_topics", "detect_subject_from_text", "list_subjects",
    # Marks Prediction
    "MarksPrediction", "predict_marks", "format_marks_prediction", "save_marks_prediction",
    # Practice Mode
    "PracticeSession", "PracticeQuestion", "AnswerEvaluation", "create_practice_session",
    "get_current_question", "submit_answer", "evaluate_answer", "format_question",
    "format_evaluation", "format_session_summary",
]

"""
UVCEExamMate Ai Enhanced CLI

Unified interface for all exam intelligence features.
"""

import sys
import json
from app.routing.router import route_query
from app.retreival.embeddings import get_embedding_model
from app.retreival.vector_store import load_vector_store
from app.retreival.semantics import retrieve_documents
from app.context.builder import build_exam_context
from app.llm.chains import auto_route_query
from dotenv import load_dotenv

# Import all intelligence features
from app.intelligence import (
    # Core pipeline
    extract_all_questions,
    classify_questions,
    analyze_questions_with_topics,
    run_prediction_pipeline,
    # Enhanced features
    generate_mock_exam,
    format_mock_exam,
    save_mock_exam,
    ExamConfig,
    generate_answer_key,
    format_answer_key,
    save_answer_key,
    generate_analytics_report,
    format_analytics_report,
    save_html_charts,
    generate_study_plan,
    format_study_plan,
    save_study_plan,
    StudyPlanConfig,
    build_concept_graph,
    format_concept_graph,
    generate_learning_path,
    predict_marks,
    format_marks_prediction,
    create_practice_session,
    format_question,
    format_evaluation,
    format_session_summary,
    submit_answer,
    get_current_question,
    # Utils
    list_subjects,
    get_subject,
)

load_dotenv()

# ============================================================================
# MENU
# ============================================================================

MENU = """
╔══════════════════════════════════════════════════════════════════╗
║                      UVCEExamMate Ai                             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   1. Predict Exam Questions                                      ║
║   2. Generate Mock Exam                                          ║
║   3. Generate Answer Key                                         ║
║   4. View Analytics Dashboard                                    ║
║   5. Create Study Plan                                           ║
║   6. Explore Concept Graph                                       ║
║   7. Predict Marks Distribution                                  ║
║   8. Practice Mode                                               ║
║   9. Ask a Question (RAG)                                        ║
║   0. Exit                                                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# GLOBAL STATE
# ============================================================================

vector_store = None
all_docs = None
analyzed_questions = None
hybrid_retriever = None

def initialize():
    """Initialize the system."""
    global vector_store, all_docs, analyzed_questions, hybrid_retriever
    
    print("\n🔄 Initializing UVCEExamMate Ai...")
    
    # Load vector store
    embedding_model = get_embedding_model()
    vector_store = load_vector_store("vector_store/ml_exam", embedding_model)
    
    # Get all documents
    all_docs = list(vector_store.docstore._dict.values())
    pyq_docs = [d for d in all_docs if d.metadata.get("is_pyq", False)]
    
    print(f"   📚 Loaded {len(all_docs)} documents ({len(pyq_docs)} PYQs)")
    
    # Initialize Hybrid Search for RAG queries
    try:
        from langchain_community.retrievers import BM25Retriever
        from app.retrieval.hybrid import HybridRetriever
        
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            k=15,
            alpha=0.6
        )
        print("   🔍 Hybrid search initialized")
    except Exception as e:
        print(f"   ⚠️ Hybrid search init failed: {e}")
        hybrid_retriever = None
    
    # Process questions
    if pyq_docs:
        print("   📝 Extracting questions...")
        extracted = extract_all_questions(pyq_docs, use_llm=False)
        
        print("   🏷️  Classifying questions...")
        classified = classify_questions(extracted, use_llm=False)
        
        print("   📌 Analyzing topics...")
        analyzed_questions = analyze_questions_with_topics(classified, use_llm=False)
        
        print(f"   ✅ Processed {len(analyzed_questions)} questions")
    else:
        print("   ⚠️ No PYQ documents found")
        analyzed_questions = []
    
    print("✅ Initialization complete!\n")


# ============================================================================
# FEATURE HANDLERS
# ============================================================================

def handle_prediction():
    """Generate question predictions."""
    global analyzed_questions, all_docs
    
    print("\n🔮 QUESTION PREDICTION")
    print("=" * 60)
    
    if not analyzed_questions:
        print("⚠️ No analyzed questions. Please ensure PYQs are indexed.")
        return
    
    num = input("Number of predictions (default 10): ").strip() or "10"
    book_docs = [d for d in all_docs if not d.metadata.get("is_pyq", False)]
    
    result = run_prediction_pipeline(
        questions=analyzed_questions,
        course_docs=book_docs,
        num_predictions=int(num)
    )
    
    print(f"\n🎓 PREDICTION REPORT (Confidence: {result.confidence_score*100:.0f}%)")
    print("=" * 60)
    
    print("\n🔥 HIGH PRIORITY TOPICS:")
    for topic in result.high_priority_topics[:7]:
        print(f"  • {topic}")
    
    print("\n🔮 PREDICTED QUESTIONS:")
    for i, q in enumerate(result.predictions, 1):
        conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(q.confidence, "⚪")
        print(f"\n{i}. {q.question}")
        print(f"   {conf_icon} {q.confidence.upper()} | 📚 {q.topic} > {q.sub_topic}")
        print(f"   💡 {q.reasoning}")


def handle_mock_exam():
    """Generate a mock exam."""
    global analyzed_questions
    
    print("\n📝 MOCK EXAM GENERATOR")
    print("=" * 60)
    
    if not analyzed_questions:
        print("⚠️ No analyzed questions available.")
        return
    
    template = input("Template (standard/mixed/comprehensive) [standard]: ").strip() or "standard"
    
    config = ExamConfig(
        subject="Machine Learning",
        total_marks=100,
        duration_minutes=180
    )
    
    paper = generate_mock_exam(
        analyzed_questions=analyzed_questions,
        config=config,
        template=template
    )
    
    print(format_mock_exam(paper))
    
    save = input("\nSave to file? (y/n): ").strip().lower()
    if save == 'y':
        save_mock_exam(paper, "mock_exam.txt")


def handle_answer_key():
    """Generate answer key for mock exam."""
    global analyzed_questions, all_docs
    
    print("\n✏️ ANSWER KEY GENERATOR")
    print("=" * 60)
    
    # First generate a mock exam
    config = ExamConfig(subject="Machine Learning")
    paper = generate_mock_exam(analyzed_questions, config=config)
    
    book_docs = [d for d in all_docs if not d.metadata.get("is_pyq", False)]
    
    print("Generating answers (this may take a moment)...")
    answer_key = generate_answer_key(paper, course_docs=book_docs)
    
    print(format_answer_key(answer_key))
    
    save = input("\nSave to file? (y/n): ").strip().lower()
    if save == 'y':
        save_answer_key(answer_key, "answer_key.txt")


def handle_analytics():
    """Show analytics dashboard."""
    global analyzed_questions
    
    print("\n📊 ANALYTICS DASHBOARD")
    print("=" * 60)
    
    if not analyzed_questions:
        print("⚠️ No data available for analytics.")
        return
    
    report = generate_analytics_report(analyzed_questions)
    print(format_analytics_report(report))
    
    save = input("\nSave HTML charts? (y/n): ").strip().lower()
    if save == 'y':
        save_html_charts(report, "analytics.html")
        print("📊 Open analytics.html in your browser!")


def handle_study_plan():
    """Generate study plan."""
    global analyzed_questions
    
    print("\n📅 SMART STUDY PLANNER")
    print("=" * 60)
    
    if not analyzed_questions:
        print("⚠️ No data available.")
        return
    
    days = input("Days until exam (default 30): ").strip() or "30"
    hours = input("Study hours per day (default 4): ").strip() or "4"
    
    config = StudyPlanConfig(
        days_until_exam=int(days),
        hours_per_day=float(hours),
        subject="Machine Learning"
    )
    
    plan = generate_study_plan(analyzed_questions, config=config)
    print(format_study_plan(plan))
    
    save = input("\nSave to file? (y/n): ").strip().lower()
    if save == 'y':
        save_study_plan(plan, "study_plan.txt")


def handle_concept_graph():
    """Explore concept graph."""
    global analyzed_questions
    
    print("\n🔗 CONCEPT GRAPH")
    print("=" * 60)
    
    graph = build_concept_graph(analyzed_questions)
    print(format_concept_graph(graph))
    
    target = input("\nGenerate learning path for concept (or press Enter to skip): ").strip()
    if target:
        path = generate_learning_path(graph, target)
        print(f"\n📚 Learning Path: {path.title}")
        print(f"   Estimated Hours: {path.estimated_hours}")
        print(f"   Concepts ({path.total_concepts}):")
        for i, c in enumerate(path.concepts, 1):
            print(f"      {i}. {c}")


def handle_marks_prediction():
    """Predict marks distribution."""
    global analyzed_questions
    
    print("\n💰 MARKS PREDICTOR")
    print("=" * 60)
    
    if not analyzed_questions:
        print("⚠️ No data available.")
        return
    
    prediction = predict_marks(analyzed_questions)
    print(format_marks_prediction(prediction))


def handle_practice_mode():
    """Interactive practice mode."""
    global analyzed_questions
    
    print("\n🎯 PRACTICE MODE")
    print("=" * 60)
    
    if not analyzed_questions:
        print("⚠️ No questions available.")
        return
    
    num = input("Number of questions (default 5): ").strip() or "5"
    topic = input("Topic filter (or press Enter for all): ").strip() or None
    
    session = create_practice_session(
        analyzed_questions,
        num_questions=int(num),
        topic=topic
    )
    
    print(f"\n📝 Starting practice session: {session.session_id}")
    print(f"   Questions: {len(session.questions)} | Total Marks: {session.total_marks}")
    
    while not session.is_completed:
        question = get_current_question(session)
        if not question:
            break
        
        print(format_question(question, show_hints=0))
        print("\n" + "-" * 40)
        print("(Type 'hint' for hints, 'skip' to skip, 'quit' to end)")
        
        hints_shown = 0
        while True:
            answer = input("\n📝 Your Answer:\n").strip()
            
            if answer.lower() == 'hint':
                hints_shown += 1
                print(format_question(question, show_hints=hints_shown))
            elif answer.lower() == 'skip':
                session.scores.append(0)
                session.current_index += 1
                break
            elif answer.lower() == 'quit':
                session.is_completed = True
                break
            else:
                evaluation = submit_answer(session, answer)
                print(format_evaluation(evaluation, question))
                input("\nPress Enter for next question...")
                break
    
    print(format_session_summary(session))

def handle_rag_query(query: str):
    """Handle a RAG query using the full retrieval pipeline."""
    global vector_store, hybrid_retriever
    
    print(f"\n🔍 Searching for: {query}")
    print("=" * 60)
    
    if not hybrid_retriever:
        print("⚠️ Hybrid search not available. Cannot answer queries.")
        return
    
    try:
        # 1. Route the query
        route = route_query(query)
        print(f"🧭 Routed to: {route}")
        
        # 2. Retrieve relevant documents
        retrieved_docs = retrieve_documents(
            query=query,
            retriever=hybrid_retriever,
            filters=None
        )
        print(f"📚 Retrieved {len(retrieved_docs)} relevant documents")
        
        # 3. Build context
        context = build_exam_context(
            retrieved_docs=retrieved_docs,
            topic_stats=None,
            pyq_patterns=None
        )
        
        # 4. Generate response via LLM
        print("🧠 Generating answer...\n")
        response = auto_route_query(context, query)
        
        # 5. Display response
        response_type = response.get("type", "general")
        result = response.get("result", {})
        
        print(f"📋 Response Type: {response_type.upper()}")
        print("-" * 60)
        
        if isinstance(result, dict):
            # Handle structured responses
            if "answer" in result:
                print(f"\n{result['answer']}")
            elif "explanation" in result:
                print(f"\n{result['explanation']}")
            else:
                for key, value in result.items():
                    if isinstance(value, list):
                        print(f"\n{key.replace('_', ' ').title()}:")
                        for item in value:
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    print(f"  • {k}: {v}")
                            else:
                                print(f"  • {item}")
                    elif isinstance(value, str) and len(value) > 10:
                        print(f"\n{key.replace('_', ' ').title()}:")
                        print(f"  {value}")
                    else:
                        print(f"  {key}: {value}")
        elif isinstance(result, str):
            print(f"\n{result}")
        else:
            print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"\n⚠️ Error answering query: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("\n" + "=" * 66)
    print("   🎓 Welcome to UVCEExamMate Ai - Intelligent Exam Preparation System")
    print("=" * 66)
    
    try:
        initialize()
    except Exception as e:
        print(f"⚠️ Initialization error: {e}")
        print("Make sure you've run 'python scripts/index_data.py' first.")
        return
    
    handlers = {
        '1': handle_prediction,
        '2': handle_mock_exam,
        '3': handle_answer_key,
        '4': handle_analytics,
        '5': handle_study_plan,
        '6': handle_concept_graph,
        '7': handle_marks_prediction,
        '8': handle_practice_mode,
    }
    
    while True:
        print(MENU)
        choice = input("Select option: ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye! Good luck with your exams!\n")
            break
        elif choice == '9':
            query = input("\n❓ Enter your question: ").strip()
            if not query:
                continue
            handle_rag_query(query)
        elif choice in handlers:
            try:
                handlers[choice]()
            except Exception as e:
                print(f"\n⚠️ Error: {e}")
        else:
            print("Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

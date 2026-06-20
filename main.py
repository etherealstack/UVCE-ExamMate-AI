"""
UVCEExamMate Ai Main Entry Point

Intelligent exam preparation system with pattern analysis and prediction.
"""

from app.routing.router import route_query
from app.retreival.embeddings import get_embedding_model
from app.retreival.vector_store import load_vector_store, get_retriever
from app.retreival.semantics import retrieve_documents

from app.context.builder import build_exam_context
from app.llm.chains import auto_route_query
from dotenv import load_dotenv

# NEW: Enhanced Intelligence Pipeline
from app.intelligence import (
    extract_all_questions,
    classify_questions,
    analyze_questions_with_topics,
    run_prediction_pipeline,
    get_extraction_stats,
    get_classification_stats,
    get_topic_stats,
)

# Legacy Pattern Detection (fallback)
from app.intelligence.pyq_analysis import (
    detect_repeated_questions,
    get_high_frequency_topics
)
import numpy as np

load_dotenv()

# --------------------------------------------------
# Load vector store & Hybrid Search
# --------------------------------------------------
embedding_model = get_embedding_model()
vector_store = load_vector_store("vector_store/ml_exam", embedding_model)

# Initialize Hybrid Search
from langchain_community.retrievers import BM25Retriever
from app.retrieval.hybrid import HybridRetriever

# We need raw documents for BM25
all_docs = list(vector_store.docstore._dict.values())
bm25_retriever = BM25Retriever.from_documents(all_docs)

# Create Hybrid Retriever
retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_retriever=bm25_retriever,
    k=15,
    alpha=0.6  # Slightly favor semantic search
)

# --------------------------------------------------
# USER QUERY
# --------------------------------------------------
query = input("❓ Ask your exam question: ")

# --------------------------------------------------
# ROUTING
# --------------------------------------------------
route = route_query(query)
print(f"\n🧭 Routed to: {route}")

# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------
retrieved_docs = retrieve_documents(
    query=query,
    retriever=retriever,
    filters=None
)

# --------------------------------------------------
# INTELLIGENCE (EXAM ANALYSIS WITH NEW PIPELINE)
# --------------------------------------------------
topic_stats = None
pyq_patterns = None
prediction_result = None

if route == "exam_analysis":
    print("\n" + "=" * 60)
    print("🎓 EXAM INTELLIGENCE ENGINE")
    print("=" * 60)
    
    # Load ALL documents from vector store
    print("\n📚 Loading all documents...")
    all_docs = list(vector_store.docstore._dict.values())
    
    # Separate PYQs and Books
    pyq_docs = [d for d in all_docs if d.metadata.get("is_pyq", False)]
    book_docs = [d for d in all_docs if not d.metadata.get("is_pyq", False)]
    
    print(f"   Found {len(pyq_docs)} PYQ chunks, {len(book_docs)} book chunks")
    
    if len(pyq_docs) > 0:
        # ========== NEW INTELLIGENT PIPELINE ==========
        try:
            # 1. Extract Questions from PYQs
            print("\n📝 Step 1: Extracting questions...")
            extracted_questions = extract_all_questions(pyq_docs, use_llm=False)
            
            if extracted_questions:
                print(f"   Stats: {get_extraction_stats(extracted_questions)}")
                
                # 2. Classify Questions
                print("\n🏷️  Step 2: Classifying questions...")
                classified_questions = classify_questions(extracted_questions, use_llm=False)
                print(f"   Type distribution: {get_classification_stats(classified_questions)['type_distribution']}")
                
                # 3. Extract Topics
                print("\n📌 Step 3: Analyzing topics...")
                analyzed_questions = analyze_questions_with_topics(classified_questions, use_llm=False)
                print(f"   Topic coverage: {len(get_topic_stats(analyzed_questions)['sub_topics'])} unique topics")
                
                # 4. Run Prediction Pipeline
                print("\n🔮 Step 4: Generating predictions...")
                prediction_result = run_prediction_pipeline(
                    questions=analyzed_questions,
                    course_docs=book_docs,
                    num_predictions=10
                )
        
        except Exception as e:
            print(f"\n⚠️ New pipeline failed: {e}")
            print("📊 Falling back to legacy pattern detection...")
            
            # FALLBACK: Legacy Pattern Detection
            doc_embeddings = embedding_model.embed_documents(
                [doc.page_content for doc in pyq_docs]
            )
            pyq_patterns = detect_repeated_questions(
                docs=pyq_docs,
                embeddings=np.array(doc_embeddings)
            )
            topic_stats = get_high_frequency_topics(pyq_patterns)
    else:
        print("\n⚠️ No PYQ documents found in the vector store.")

# --------------------------------------------------
# CONTEXT BUILDING
# --------------------------------------------------
context = build_exam_context(
    retrieved_docs=retrieved_docs,
    topic_stats=topic_stats,
    pyq_patterns=pyq_patterns
)

# --------------------------------------------------
# RESPONSE FORMATTING
# --------------------------------------------------
def format_prediction_response(result):
    """Format the prediction result for display."""
    print(f"\n🎓 EXAM PREDICTION REPORT (Confidence: {result.confidence_score*100:.0f}%)")
    print("=" * 60)
    
    print("\n🔥 HIGH PRIORITY TOPICS:")
    for topic in result.high_priority_topics[:7]:
        print(f"  • {topic}")
    
    print("\n🔮 PREDICTED QUESTIONS:")
    for i, q in enumerate(result.predictions, 1):
        conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(q.confidence, "⚪")
        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(q.difficulty, "⚪")
        
        print(f"\n{i}. {q.question}")
        print(f"   {diff_icon} {q.difficulty.upper()} | {conf_icon} {q.confidence.upper()} CONFIDENCE")
        print(f"   📚 Topic: {q.topic} > {q.sub_topic}")
        print(f"   💡 Reasoning: {q.reasoning}")
        if q.similar_years:
            print(f"   📅 Similar to: {', '.join(map(str, q.similar_years))}")
        if q.study_tip:
            print(f"   📖 Tip: {q.study_tip}")
    
    print("\n✨ EXAM TIPS:")
    for tip in result.exam_tips[:5]:
        print(f"  👉 {tip}")
    
    print("\n" + "=" * 60)
    print("📊 PATTERN ANALYSIS:")
    print(result.pattern_summary)


def format_response(response_type: str, result: dict):
    """Format the output for human readability."""
    if response_type == "exam_analysis":
        print(f"\n🎓 EXAM ANALYSIS REPORT (Confidence: {result.get('confidence_score', 0)*100:.0f}%)")
        print("=" * 60)
        
        print("\n🔥 HIGH PRIORITY TOPICS:")
        for topic in result.get("high_priority_topics", []):
            print(f"  • {topic}")
            
        print("\n🔮 PREDICTED QUESTIONS:")
        for i, q in enumerate(result.get("predicted_questions", []), 1):
            difficulty_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(q['difficulty'], "⚪")
            print(f"\n{i}. {q['question']}")
            print(f"   {difficulty_icon} {q['difficulty'].upper()} | 📚 {q['topic']}")
            print(f"   💡 Reasoning: {q['reasoning']}")
            if q.get('similar_years'):
                print(f"   📅 Similar to: {', '.join(map(str, q['similar_years']))}")

        print("\n✨ EXAM TIPS:")
        for tip in result.get("exam_tips", []):
            print(f"  👉 {tip}")
            
    else:
        # Fallback for other types
        import json
        print(json.dumps(result, indent=2))

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
if prediction_result:
    # Use new prediction result
    format_prediction_response(prediction_result)
else:
    # Use LLM for other query types
    response = auto_route_query(context, query)
    
    print("\n🧠 RESPONSE TYPE:", response["type"])
    print("\n📘 RESULT:\n")
    
    format_response(response["type"], response["result"])

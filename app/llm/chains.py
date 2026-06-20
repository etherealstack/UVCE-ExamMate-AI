"""
Responsibility: LLM reasoning on prepared context.
Scope: Apply prompt templates, return structured outputs.
Note: Pure reasoning - no retrieval, no indexing, no intelligence logic.
"""

from typing import Dict, List, Literal, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json


# ============================================================================
# OUTPUT SCHEMAS (Structured Responses)
# ============================================================================

class PredictedQuestion(BaseModel):
    """Schema for a predicted exam question."""
    question: str = Field(description="The predicted exam question")
    topic: str = Field(description="Main topic/concept being tested")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Difficulty level")
    reasoning: str = Field(description="Why this question is likely to appear")
    similar_years: List[int] = Field(description="Years when similar questions appeared")


class ExamAnalysisOutput(BaseModel):
    """Schema for exam analysis output."""
    predicted_questions: List[PredictedQuestion] = Field(
        description="List of predicted exam questions"
    )
    high_priority_topics: List[str] = Field(
        description="Topics to prioritize for exam prep"
    )
    exam_tips: List[str] = Field(
        description="Strategic tips for the exam"
    )
    confidence_score: float = Field(
        description="Confidence in predictions (0-1)",
        ge=0,
        le=1
    )


class StudyPlanItem(BaseModel):
    """Schema for a study plan item."""
    day: int = Field(description="Day number in the plan")
    topic: str = Field(description="Topic to study")
    tasks: List[str] = Field(description="Specific tasks for the day")
    estimated_hours: float = Field(description="Estimated study hours")
    resources: List[str] = Field(description="Recommended resources")


class StudyPlanOutput(BaseModel):
    """Schema for study plan output."""
    total_days: int = Field(description="Total days in the plan")
    plan: List[StudyPlanItem] = Field(description="Day-by-day study plan")
    key_topics: List[str] = Field(description="Key topics covered")
    exam_date_assumptions: str = Field(description="Assumptions about exam timing")


class ConceptExplanation(BaseModel):
    """Schema for concept explanation."""
    concept: str = Field(description="Name of the concept")
    simple_explanation: str = Field(description="Simple explanation in plain language")
    key_points: List[str] = Field(description="Key points to remember")
    common_mistakes: List[str] = Field(description="Common mistakes to avoid")
    exam_relevance: str = Field(description="How this appears in exams")
    practice_tip: str = Field(description="How to practice this concept")


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

EXAM_ANALYSIS_TEMPLATE = """You are an expert exam analyst with deep knowledge of exam patterns and question trends.

Context (from textbooks and past papers):
{context}

Student's Query: {query}

CRITICAL RULES:
1. **NO HALLUCINATION**: Only use questions and years EXPLICITLY mentioned in the Context.
2. **STRICT YEAR CHECK**: If the Context only mentions 2022, 2023, 2024, do NOT invent 2018, 2019, 2020.
3. **INSUFFICIENT DATA**: If the Context does not contain pattern data or specific questions, state "Insufficient data to predict" instead of guessing.
4. **EVIDENCE BASED**: Every predicted question must interpret a specific part of the Context.

Based on the context above, analyze the exam patterns and provide:
1. Predicted questions most likely to appear (based on frequency and patterns)
2. High-priority topics to focus on
3. Strategic exam tips
4. Your confidence level in these predictions (if patterns are missing, confidence should be low)

{format_instructions}

Provide ONLY the JSON output, no additional text."""


STUDY_PLAN_TEMPLATE = """You are an expert study planner who creates personalized, effective study schedules.

Context (from textbooks and past papers):
{context}

Student's Request: {query}

Create a day-by-day study plan that:
- Prioritizes high-frequency exam topics
- Builds from fundamentals to advanced concepts
- Includes specific, actionable tasks
- Estimates realistic study hours
- Recommends specific resources from the context

{format_instructions}

Provide ONLY the JSON output, no additional text."""


CONCEPT_EXPLANATION_TEMPLATE = """You are an expert tutor who explains complex concepts clearly and relates them to exams.

Context (from textbooks and past papers):
{context}

Student's Question: {query}

Explain the concept by:
1. Providing a simple, clear explanation
2. Highlighting key points to remember
3. Warning about common mistakes
4. Explaining how this appears in exams (based on context)
5. Giving practical tips for mastering it

{format_instructions}

Provide ONLY the JSON output, no additional text."""


GENERAL_ANSWER_TEMPLATE = """You are a knowledgeable exam preparation tutor.

Context (from textbooks and past papers):
{context}

Student's Question: {query}

Provide a clear, accurate answer based ONLY on the context provided. If the context doesn't contain enough information, say so clearly.

Structure your answer to:
- Be concise and exam-focused
- Highlight key points
- Reference specific examples from past papers if relevant
- Suggest what to focus on for exam preparation

Answer:"""


HYDE_TEMPLATE = """You are an AI assistant helping a student find relevant exam questions.
Please generate 3 hypothetical exam questions that are semantically similar to the user's query but phrased differently.
These questions will be used to expanded the search query to find more relevant documents.

User Query: {query}

Provide ONLY the 3 questions separated by newlines. No numbering or extra text."""


# ============================================================================
# CHAIN FUNCTIONS
# ============================================================================

def run_exam_analysis(
    context: str,
    query: str,
    model: str = "llama3-8b-8192",
    temperature: float = 0.2
) -> Dict:
    """
    Analyze exam patterns and predict likely questions.
    """
    # Setup parser
    parser = PydanticOutputParser(pydantic_object=ExamAnalysisOutput)
    
    # Create prompt
    prompt = PromptTemplate(
        template=EXAM_ANALYSIS_TEMPLATE,
        input_variables=["context", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Create chain
    llm = ChatGroq(model=model, temperature=temperature)
    chain = prompt | llm | parser
    
    # Run
    result = chain.invoke({"context": context, "query": query})
    
    return result.dict()


def run_study_planner(
    context: str,
    query: str,
    model: str = "gpt-4o",
    temperature: float = 0.5
) -> Dict:
    """
    Generate a personalized study plan.
    """
    parser = PydanticOutputParser(pydantic_object=StudyPlanOutput)
    
    prompt = PromptTemplate(
        template=STUDY_PLAN_TEMPLATE,
        input_variables=["context", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatGroq(model=model, temperature=temperature)
    chain = prompt | llm | parser
    
    result = chain.invoke({"context": context, "query": query})
    
    return result.dict()


def run_concept_explainer(
    context: str,
    query: str,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.4
) -> Dict:
    """
    Explain a concept clearly with exam focus.
    """
    parser = PydanticOutputParser(pydantic_object=ConceptExplanation)
    
    prompt = PromptTemplate(
        template=CONCEPT_EXPLANATION_TEMPLATE,
        input_variables=["context", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatGroq(model=model, temperature=temperature)
    chain = prompt | llm | parser
    
    result = chain.invoke({"context": context, "query": query})
    
    return result.dict()


def run_general_answer(
    context: str,
    query: str,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.3
) -> str:
    """
    Get a general answer to any question (fallback for unstructured queries).
    """
    prompt = PromptTemplate(
        template=GENERAL_ANSWER_TEMPLATE,
        input_variables=["context", "query"]
    )
    
    llm = ChatGroq(model=model, temperature=temperature)
    chain = prompt | llm
    
    result = chain.invoke({"context": context, "query": query})
    
    return result.content


def run_query_expansion(
    query: str,
    model: str = "llama-3.1-8b-instant"
) -> str:
    """
    Expand user query into multiple variations (HyDE).
    Returns the original query concatenated with generated variations.
    """
    prompt = PromptTemplate(
        template=HYDE_TEMPLATE,
        input_variables=["query"]
    )
    
    llm = ChatGroq(model=model, temperature=0.7) # Higher temp for creativity
    chain = prompt | llm
    
    try:
        response = chain.invoke({"query": query})
        variations = response.content.replace("\n", " ")
        print(f"🧠 HyDE Expanded: {variations}")
        # Combine original query (weighted more) with variations
        return f"{query} {query} {variations}"
    except Exception as e:
        print(f"⚠️ HyDE failed, using original query: {e}")
        return query


def auto_route_query(
    context: str,
    query: str,
    model: str = "llama-3.1-8b-instant"
) -> Dict:
    """
    Automatically route query to the appropriate chain based on intent.
    
    Intent detection:
    - "predict", "likely", "will appear" → exam_analysis
    - "plan", "schedule", "prepare" → study_planner  
    - "explain", "what is", "how does" → concept_explainer
    - Otherwise → general_answer
    """
    query_lower = query.lower()
    
    # Intent detection (simple keyword matching)
    if any(word in query_lower for word in ["predict", "likely", "will appear", "questions"]):
        return {
            "type": "exam_analysis",
            "result": run_exam_analysis(context, query, model)
        }
    
    elif any(word in query_lower for word in ["plan", "schedule", "prepare", "study"]):
        return {
            "type": "study_plan",
            "result": run_study_planner(context, query, model)
        }
    
    elif any(word in query_lower for word in ["explain", "what is", "how does", "why"]):
        return {
            "type": "concept_explanation",
            "result": run_concept_explainer(context, query, model)
        }
    
    else:
        return {
            "type": "general_answer",
            "result": run_general_answer(context, query, model)
        }


def get_available_chains() -> List[str]:
    """
    List all available reasoning chains.
    
    Returns:
        List of chain names
    """
    return [
        "exam_analysis",
        "study_planner",
        "concept_explainer",
        "general_answer"
    ]


def validate_chain_output(output: Dict, chain_type: str) -> bool:
    """
    Validate that chain output matches expected schema.
    
    Args:
        output: Chain output dictionary
        chain_type: Type of chain used
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if chain_type == "exam_analysis":
            ExamAnalysisOutput(**output)
        elif chain_type == "study_plan":
            StudyPlanOutput(**output)
        elif chain_type == "concept_explanation":
            ConceptExplanation(**output)
        else:
            return True  # general_answer is just string
        
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False
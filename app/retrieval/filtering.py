
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class SearchFilters(BaseModel):
    year: Optional[int] = Field(description="Year if specified (e.g. 2023), else null")
    subject: Optional[str] = Field(description="Subject name if specified, else null")
    source_type: Optional[str] = Field(description="'question_paper' or 'book' if clear intent, else null")

FILTER_EXTRACTION_TEMPLATE = """You are a smart search query parser.
Extract search filters from the user's query.

User Query: {query}

Rules:
1. Extract 'year' if a specific year is mentioned (e.g., "in 2023").
2. Extract 'subject' if mentioned (e.g., "Machine Learning").
3. Set 'source_type' to 'question_paper' if the user asks for "questions", "papers", "PYQs", "predict".
4. Set 'source_type' to 'book' if the user asks for "book", "textbook", "concepts".
5. Return JSON only. keys: year, subject, source_type. Use null if not found.

{format_instructions}
"""

def extract_filters_from_query(
    query: str,
    model: str = "llama-3.1-8b-instant"
) -> Dict[str, Any]:
    """
    Extract metadata filters from natural language query.
    """
    parser = JsonOutputParser(pydantic_object=SearchFilters)
    
    prompt = PromptTemplate(
        template=FILTER_EXTRACTION_TEMPLATE,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatGroq(model=model, temperature=0) # Deterministic
    chain = prompt | llm | parser
    
    try:
        filters = chain.invoke({"query": query})
        # Remove nulls
        clean_filters = {k: v for k, v in filters.items() if v is not None}
        if clean_filters:
            print(f"🔒 Active Filters: {clean_filters}")
        return clean_filters
    except Exception as e:
        print(f"⚠️ Filter extraction failed: {e}")
        return {}

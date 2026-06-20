"""
Concept Graph Module

Responsibility: Build and analyze knowledge graphs for concept dependencies.
Maps prerequisite relationships between topics and concepts.
"""

from typing import List, Dict, Optional, Set, Tuple
from pydantic import BaseModel, Field
from collections import defaultdict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .topic_extractor import AnalyzedQuestion


# ============================================================================
# DATA MODELS
# ============================================================================

class ConceptNode(BaseModel):
    """A concept in the knowledge graph."""
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Concept name")
    category: str = Field(description="Category/main topic")
    difficulty: str = Field(default="medium", description="Difficulty level")
    description: Optional[str] = Field(default=None, description="Brief description")
    exam_frequency: int = Field(default=0, description="Times appeared in exams")


class ConceptEdge(BaseModel):
    """A relationship between concepts."""
    source: str = Field(description="Source concept ID")
    target: str = Field(description="Target concept ID")
    relationship: str = Field(description="Type of relationship")
    strength: float = Field(default=1.0, description="Relationship strength 0-1")


class ConceptGraph(BaseModel):
    """Knowledge graph of concepts."""
    nodes: List[ConceptNode] = Field(description="All concept nodes")
    edges: List[ConceptEdge] = Field(description="All relationships")
    categories: List[str] = Field(description="All categories")


class LearningPath(BaseModel):
    """A recommended learning path."""
    title: str = Field(description="Path title")
    concepts: List[str] = Field(description="Ordered list of concepts to learn")
    total_concepts: int = Field(description="Number of concepts")
    estimated_hours: float = Field(description="Estimated learning time")
    reason: str = Field(description="Why this path is recommended")


class PrerequisiteOutput(BaseModel):
    """LLM output for prerequisite detection."""
    prerequisites: List[str] = Field(description="List of prerequisite concepts")
    related_concepts: List[str] = Field(description="List of related concepts")


# ============================================================================
# PREDEFINED CONCEPT RELATIONSHIPS (ML Domain)
# ============================================================================

# Core prerequisite relationships for Machine Learning
ML_PREREQUISITES = {
    # Foundations
    "Linear Algebra": [],
    "Probability": ["Linear Algebra"],
    "Statistics": ["Probability"],
    "Optimization": ["Linear Algebra", "Calculus"],
    "Calculus": [],
    
    # Core ML
    "Gradient Descent": ["Calculus", "Optimization"],
    "Loss Functions": ["Calculus", "Statistics"],
    "Regularization": ["Loss Functions", "Optimization"],
    "Bias-Variance Tradeoff": ["Statistics", "Loss Functions"],
    
    # Supervised Learning
    "Linear Regression": ["Linear Algebra", "Gradient Descent", "Loss Functions"],
    "Logistic Regression": ["Linear Regression", "Probability"],
    "Decision Trees": ["Statistics", "Information Theory"],
    "Random Forest": ["Decision Trees", "Ensemble Methods"],
    "Ensemble Methods": ["Decision Trees", "Bias-Variance Tradeoff"],
    "SVM": ["Optimization", "Linear Algebra", "Kernel Methods"],
    "Kernel Methods": ["Linear Algebra"],
    "Naive Bayes": ["Probability", "Bayes Theorem"],
    "Bayes Theorem": ["Probability"],
    "KNN": ["Distance Metrics", "Statistics"],
    "Distance Metrics": ["Linear Algebra"],
    
    # Unsupervised Learning
    "K-Means": ["Distance Metrics", "Optimization"],
    "Hierarchical Clustering": ["Distance Metrics"],
    "DBSCAN": ["Distance Metrics", "Density Estimation"],
    "Density Estimation": ["Statistics"],
    "PCA": ["Linear Algebra", "Eigenvalues"],
    "Eigenvalues": ["Linear Algebra"],
    "Dimensionality Reduction": ["Linear Algebra", "PCA"],
    
    # Neural Networks
    "Perceptron": ["Linear Algebra", "Gradient Descent"],
    "Multilayer Perceptron": ["Perceptron", "Activation Functions"],
    "Activation Functions": ["Calculus"],
    "Backpropagation": ["Multilayer Perceptron", "Chain Rule", "Gradient Descent"],
    "Chain Rule": ["Calculus"],
    "CNN": ["Backpropagation", "Convolution"],
    "Convolution": ["Linear Algebra"],
    "RNN": ["Backpropagation", "Sequence Modeling"],
    "Sequence Modeling": ["Time Series"],
    "LSTM": ["RNN", "Gradient Descent"],
    "Transformers": ["Attention Mechanism", "Backpropagation"],
    "Attention Mechanism": ["RNN", "Softmax"],
    "Softmax": ["Probability"],
    
    # Evaluation
    "Cross Validation": ["Bias-Variance Tradeoff"],
    "Confusion Matrix": ["Classification Basics"],
    "Classification Basics": ["Probability"],
    "ROC Curve": ["Confusion Matrix", "Threshold Selection"],
    "Threshold Selection": ["Probability"],
    "Precision Recall": ["Confusion Matrix"],
    
    # Advanced
    "Reinforcement Learning": ["Probability", "Optimization", "MDP"],
    "MDP": ["Probability"],
    "GANs": ["Backpropagation", "Generative Models"],
    "Generative Models": ["Probability", "Neural Networks"],
    "Bayesian Methods": ["Probability", "Bayes Theorem"],
}


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_concept_graph(
    questions: List[AnalyzedQuestion],
    subject: str = "Machine Learning",
    use_llm: bool = False,
    model: str = "llama-3.1-8b-instant"
) -> ConceptGraph:
    """
    Build a knowledge graph from analyzed questions and predefined relationships.
    
    Args:
        questions: Historical analyzed questions
        subject: Subject name for predefined relationships
        use_llm: Use LLM to discover relationships
        model: LLM model name
        
    Returns:
        ConceptGraph with nodes and edges
    """
    print(f"🔗 Building concept graph for {subject}...")
    
    # Collect concepts from questions
    concepts_from_questions = set()
    concept_frequency = defaultdict(int)
    concept_category = {}
    
    for q in questions:
        concepts_from_questions.add(q.sub_topic)
        concept_frequency[q.sub_topic] += 1
        concept_category[q.sub_topic] = q.main_topic
        
        for concept in q.key_concepts:
            concepts_from_questions.add(concept)
            concept_frequency[concept] += 1
    
    # Create nodes
    nodes = []
    all_concepts = set(ML_PREREQUISITES.keys()) | concepts_from_questions
    
    for concept in all_concepts:
        node = ConceptNode(
            id=concept.lower().replace(" ", "_"),
            name=concept,
            category=concept_category.get(concept, _infer_category(concept)),
            difficulty=_infer_difficulty(concept),
            exam_frequency=concept_frequency.get(concept, 0)
        )
        nodes.append(node)
    
    # Create edges from predefined relationships
    edges = []
    for concept, prereqs in ML_PREREQUISITES.items():
        for prereq in prereqs:
            if concept in all_concepts and prereq in all_concepts:
                edges.append(ConceptEdge(
                    source=prereq.lower().replace(" ", "_"),
                    target=concept.lower().replace(" ", "_"),
                    relationship="prerequisite",
                    strength=1.0
                ))
    
    # Discover additional relationships from co-occurrence
    cooccurrence_edges = _find_cooccurrence_relationships(questions)
    edges.extend(cooccurrence_edges)
    
    # Use LLM for additional relationships if enabled
    if use_llm:
        llm_edges = _discover_relationships_llm(
            list(concepts_from_questions - set(ML_PREREQUISITES.keys()))[:10],
            model
        )
        edges.extend(llm_edges)
    
    # Get unique categories
    categories = list(set(n.category for n in nodes))
    
    print(f"   Created {len(nodes)} nodes, {len(edges)} edges")
    
    return ConceptGraph(
        nodes=nodes,
        edges=edges,
        categories=categories
    )


def _infer_category(concept: str) -> str:
    """Infer category from concept name."""
    concept_lower = concept.lower()
    
    categories = {
        "Foundations": ["algebra", "calculus", "probability", "statistics", "optimization"],
        "Supervised Learning": ["regression", "classification", "svm", "tree", "forest", "naive"],
        "Unsupervised Learning": ["clustering", "k-means", "pca", "dimensionality"],
        "Neural Networks": ["perceptron", "neural", "cnn", "rnn", "lstm", "transformer", "attention"],
        "Evaluation": ["validation", "cross", "roc", "precision", "recall", "confusion"],
    }
    
    for category, keywords in categories.items():
        if any(kw in concept_lower for kw in keywords):
            return category
    
    return "General"


def _infer_difficulty(concept: str) -> str:
    """Infer difficulty from concept name."""
    advanced_keywords = ["transformer", "gan", "lstm", "reinforcement", "bayesian", "attention"]
    intermediate_keywords = ["cnn", "rnn", "svm", "random forest", "pca", "backpropagation"]
    
    concept_lower = concept.lower()
    
    if any(kw in concept_lower for kw in advanced_keywords):
        return "hard"
    elif any(kw in concept_lower for kw in intermediate_keywords):
        return "medium"
    else:
        return "easy"


def _find_cooccurrence_relationships(questions: List[AnalyzedQuestion]) -> List[ConceptEdge]:
    """Find relationships based on co-occurrence in questions."""
    edges = []
    concept_pairs = defaultdict(int)
    
    for q in questions:
        concepts = [q.sub_topic] + q.key_concepts
        # Count pairs that appear together
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                if c1 != c2:
                    pair = tuple(sorted([c1, c2]))
                    concept_pairs[pair] += 1
    
    # Create edges for frequently co-occurring concepts
    for (c1, c2), count in concept_pairs.items():
        if count >= 2:  # At least 2 co-occurrences
            edges.append(ConceptEdge(
                source=c1.lower().replace(" ", "_"),
                target=c2.lower().replace(" ", "_"),
                relationship="related",
                strength=min(count / 5, 1.0)
            ))
    
    return edges


def _discover_relationships_llm(
    concepts: List[str],
    model: str
) -> List[ConceptEdge]:
    """Use LLM to discover prerequisite relationships."""
    if not concepts:
        return []
    
    edges = []
    
    for concept in concepts[:5]:  # Limit to 5 to avoid too many API calls
        try:
            prereqs = _get_prerequisites_llm(concept, model)
            for prereq in prereqs.prerequisites:
                edges.append(ConceptEdge(
                    source=prereq.lower().replace(" ", "_"),
                    target=concept.lower().replace(" ", "_"),
                    relationship="prerequisite",
                    strength=0.8
                ))
            for related in prereqs.related_concepts:
                edges.append(ConceptEdge(
                    source=concept.lower().replace(" ", "_"),
                    target=related.lower().replace(" ", "_"),
                    relationship="related",
                    strength=0.6
                ))
        except Exception as e:
            print(f"⚠️ Failed to get prerequisites for {concept}: {e}")
    
    return edges


def _get_prerequisites_llm(concept: str, model: str) -> PrerequisiteOutput:
    """Get prerequisites for a concept using LLM."""
    
    PREREQ_TEMPLATE = """What are the prerequisite concepts needed to understand "{concept}" in Machine Learning?

{format_instructions}

Return ONLY valid JSON with:
- prerequisites: concepts you MUST understand first
- related_concepts: concepts that are similar or complementary"""

    parser = PydanticOutputParser(pydantic_object=PrerequisiteOutput)
    
    prompt = PromptTemplate(
        template=PREREQ_TEMPLATE,
        input_variables=["concept"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatGroq(model=model, temperature=0)
    chain = prompt | llm | parser
    
    return chain.invoke({"concept": concept})


# ============================================================================
# GRAPH ANALYSIS
# ============================================================================

def get_prerequisites(graph: ConceptGraph, concept_id: str) -> List[str]:
    """Get all prerequisites for a concept (BFS)."""
    prerequisites = set()
    to_visit = [concept_id]
    visited = set()
    
    while to_visit:
        current = to_visit.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        for edge in graph.edges:
            if edge.target == current and edge.relationship == "prerequisite":
                prerequisites.add(edge.source)
                to_visit.append(edge.source)
    
    return list(prerequisites)


def get_dependent_concepts(graph: ConceptGraph, concept_id: str) -> List[str]:
    """Get all concepts that depend on this concept."""
    dependents = set()
    to_visit = [concept_id]
    visited = set()
    
    while to_visit:
        current = to_visit.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        for edge in graph.edges:
            if edge.source == current and edge.relationship == "prerequisite":
                dependents.add(edge.target)
                to_visit.append(edge.target)
    
    return list(dependents)


def generate_learning_path(
    graph: ConceptGraph,
    target_concept: str,
    max_concepts: int = 15
) -> LearningPath:
    """
    Generate optimal learning path to master a target concept.
    Uses topological sort considering prerequisites.
    """
    target_id = target_concept.lower().replace(" ", "_")
    
    # Get all prerequisites
    prereqs = get_prerequisites(graph, target_id)
    
    # Add the target itself
    all_concepts = prereqs + [target_id]
    
    # Topological sort
    ordered = _topological_sort(graph, all_concepts)
    
    # Add the target at the end if not already included
    if target_id not in ordered:
        ordered.append(target_id)
    
    # Map back to names
    id_to_name = {n.id: n.name for n in graph.nodes}
    ordered_names = [id_to_name.get(cid, cid) for cid in ordered[:max_concepts]]
    
    # Estimate time (2 hours per concept average)
    estimated_hours = len(ordered_names) * 2
    
    return LearningPath(
        title=f"Path to {target_concept}",
        concepts=ordered_names,
        total_concepts=len(ordered_names),
        estimated_hours=estimated_hours,
        reason=f"Covers all {len(prereqs)} prerequisites in optimal order"
    )


def _topological_sort(graph: ConceptGraph, concepts: List[str]) -> List[str]:
    """Topological sort of concepts based on prerequisites."""
    # Build adjacency list
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    
    concept_set = set(concepts)
    
    for edge in graph.edges:
        if edge.relationship == "prerequisite":
            if edge.source in concept_set and edge.target in concept_set:
                adj[edge.source].append(edge.target)
                in_degree[edge.target] += 1
    
    # Initialize queue with nodes having no prerequisites
    queue = [c for c in concepts if in_degree[c] == 0]
    result = []
    
    while queue:
        current = queue.pop(0)
        result.append(current)
        
        for neighbor in adj[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Add any remaining nodes (cycles or disconnected)
    for c in concepts:
        if c not in result:
            result.append(c)
    
    return result


def find_knowledge_gaps(
    graph: ConceptGraph,
    questions: List[AnalyzedQuestion]
) -> List[str]:
    """Find concepts that are prerequisites but rarely tested directly."""
    
    # Get all tested concepts
    tested = set()
    for q in questions:
        tested.add(q.sub_topic.lower().replace(" ", "_"))
    
    # Find prerequisites that aren't directly tested
    prereqs_not_tested = set()
    for edge in graph.edges:
        if edge.relationship == "prerequisite":
            if edge.source not in tested and edge.target in tested:
                prereqs_not_tested.add(edge.source)
    
    # Map back to names
    id_to_name = {n.id: n.name for n in graph.nodes}
    
    return [id_to_name.get(pid, pid) for pid in prereqs_not_tested]


# ============================================================================
# VISUALIZATION
# ============================================================================

def format_concept_graph(graph: ConceptGraph) -> str:
    """Format concept graph as text."""
    lines = [
        "=" * 70,
        "🔗 CONCEPT KNOWLEDGE GRAPH",
        "=" * 70,
        "",
        f"Total Concepts: {len(graph.nodes)}",
        f"Total Relationships: {len(graph.edges)}",
        f"Categories: {', '.join(graph.categories)}",
        "",
    ]
    
    # Group by category
    by_category = defaultdict(list)
    for node in graph.nodes:
        by_category[node.category].append(node)
    
    for category, nodes in by_category.items():
        lines.append(f"📚 {category}")
        lines.append("-" * 40)
        for node in sorted(nodes, key=lambda n: n.exam_frequency, reverse=True)[:10]:
            freq = f"[{node.exam_frequency}x]" if node.exam_frequency > 0 else ""
            diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(node.difficulty, "")
            lines.append(f"  {diff_icon} {node.name} {freq}")
        lines.append("")
    
    # Show key relationships
    lines.append("🔗 KEY PREREQUISITE CHAINS")
    lines.append("-" * 40)
    
    # Find concepts with most dependencies
    prereq_count = defaultdict(int)
    for edge in graph.edges:
        if edge.relationship == "prerequisite":
            prereq_count[edge.target] += 1
    
    top_complex = sorted(prereq_count.items(), key=lambda x: x[1], reverse=True)[:5]
    id_to_name = {n.id: n.name for n in graph.nodes}
    
    for concept_id, count in top_complex:
        prereqs = get_prerequisites(graph, concept_id)[:3]
        prereq_names = [id_to_name.get(p, p) for p in prereqs]
        concept_name = id_to_name.get(concept_id, concept_id)
        if prereq_names:
            lines.append(f"  {' → '.join(prereq_names)} → {concept_name}")
    
    return "\n".join(lines)


def generate_mermaid_graph(graph: ConceptGraph) -> str:
    """Generate Mermaid diagram code for the graph."""
    lines = ["```mermaid", "graph LR"]
    
    # Add nodes by category with styling
    categories_added = set()
    for node in graph.nodes[:30]:  # Limit nodes for readability
        style = ""
        if node.difficulty == "hard":
            style = ":::hard"
        elif node.difficulty == "easy":
            style = ":::easy"
        lines.append(f"    {node.id}[{node.name}]{style}")
    
    # Add edges
    for edge in graph.edges[:40]:  # Limit edges
        if edge.relationship == "prerequisite":
            lines.append(f"    {edge.source} --> {edge.target}")
        else:
            lines.append(f"    {edge.source} -.- {edge.target}")
    
    # Add styling
    lines.extend([
        "",
        "    classDef hard fill:#ff6b6b,stroke:#333,stroke-width:2px",
        "    classDef easy fill:#69db7c,stroke:#333,stroke-width:2px",
    ])
    
    lines.append("```")
    
    return "\n".join(lines)


def save_concept_graph(graph: ConceptGraph, filepath: str) -> None:
    """Save concept graph to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(format_concept_graph(graph))
        f.write("\n\n")
        f.write("MERMAID DIAGRAM:\n")
        f.write(generate_mermaid_graph(graph))
    print(f"💾 Saved concept graph to: {filepath}")

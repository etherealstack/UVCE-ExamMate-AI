"""
Multi-Subject Configuration Module

Responsibility: Support multiple subjects with dynamic topic hierarchies.
Allows adding new subjects with their own topic structures and patterns.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# DATA MODELS
# ============================================================================

class SubjectConfig(BaseModel):
    """Configuration for a subject."""
    name: str = Field(description="Subject name")
    code: str = Field(description="Subject code (e.g., ML, DBMS, CN)")
    topic_hierarchy: Dict[str, List[str]] = Field(description="Main topics -> sub-topics")
    question_patterns: Dict[str, str] = Field(
        default_factory=dict,
        description="Pattern name -> regex pattern"
    )
    typical_marks_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Question type -> typical marks"
    )
    exam_duration_minutes: int = Field(default=180, description="Standard exam duration")
    total_marks: int = Field(default=100, description="Total exam marks")
    keywords: List[str] = Field(default_factory=list, description="Subject-specific keywords")


class SubjectRegistry(BaseModel):
    """Registry of all subjects."""
    subjects: Dict[str, SubjectConfig] = Field(default_factory=dict)
    default_subject: str = Field(default="ML", description="Default subject code")


# ============================================================================
# PREDEFINED SUBJECT CONFIGURATIONS
# ============================================================================

ML_CONFIG = SubjectConfig(
    name="Machine Learning",
    code="ML",
    topic_hierarchy={
        "Foundations": [
            "Linear Algebra",
            "Probability & Statistics",
            "Optimization",
            "Gradient Descent",
            "Loss Functions",
        ],
        "Supervised Learning": [
            "Linear Regression",
            "Logistic Regression",
            "Decision Trees",
            "Random Forest",
            "Support Vector Machines",
            "Naive Bayes",
            "K-Nearest Neighbors",
            "Ensemble Methods",
        ],
        "Unsupervised Learning": [
            "K-Means Clustering",
            "Hierarchical Clustering",
            "DBSCAN",
            "PCA",
            "Dimensionality Reduction",
            "Anomaly Detection",
        ],
        "Neural Networks": [
            "Perceptron",
            "Multilayer Perceptron",
            "Backpropagation",
            "Activation Functions",
            "CNN",
            "RNN",
            "LSTM",
            "Transformers",
        ],
        "Model Evaluation": [
            "Bias-Variance Tradeoff",
            "Cross Validation",
            "Confusion Matrix",
            "ROC Curve",
            "Precision & Recall",
            "Regularization",
        ],
    },
    question_patterns={
        "derivation": r"derive|proof|show that|prove",
        "numerical": r"calculate|find|compute|solve",
        "theory": r"explain|describe|what is|define",
        "comparison": r"compare|differentiate|distinguish",
        "algorithm": r"algorithm|steps|procedure",
    },
    typical_marks_distribution={
        "mcq": 1,
        "short_answer": 5,
        "theory": 10,
        "numerical": 10,
        "derivation": 15,
        "algorithm": 12,
    },
    exam_duration_minutes=180,
    total_marks=100,
    keywords=[
        "regression", "classification", "clustering", "neural", "gradient",
        "training", "validation", "overfitting", "bias", "variance",
        "backpropagation", "activation", "loss function", "optimizer"
    ]
)

DBMS_CONFIG = SubjectConfig(
    name="Database Management Systems",
    code="DBMS",
    topic_hierarchy={
        "Fundamentals": [
            "Database Concepts",
            "Data Models",
            "ER Model",
            "Relational Model",
            "Database Architecture",
        ],
        "SQL": [
            "DDL",
            "DML",
            "Joins",
            "Subqueries",
            "Views",
            "Stored Procedures",
            "Triggers",
        ],
        "Normalization": [
            "1NF",
            "2NF",
            "3NF",
            "BCNF",
            "4NF",
            "5NF",
            "Functional Dependencies",
        ],
        "Transaction Management": [
            "ACID Properties",
            "Concurrency Control",
            "Locking",
            "Deadlock",
            "Recovery",
            "Serializability",
        ],
        "Indexing & Hashing": [
            "B-Trees",
            "B+ Trees",
            "Hashing Techniques",
            "Index Selection",
        ],
        "Query Processing": [
            "Query Optimization",
            "Cost Estimation",
            "Relational Algebra",
            "Query Execution Plans",
        ],
    },
    question_patterns={
        "sql": r"write.*query|sql|select|insert|update|delete",
        "er_diagram": r"er diagram|entity|relationship|draw",
        "normalization": r"normalize|normal form|nf|decompose",
        "theory": r"explain|describe|what is",
        "numerical": r"calculate|find|compute",
    },
    typical_marks_distribution={
        "mcq": 1,
        "sql": 8,
        "er_diagram": 10,
        "normalization": 12,
        "theory": 8,
        "numerical": 10,
    },
    exam_duration_minutes=180,
    total_marks=100,
    keywords=[
        "table", "query", "join", "primary key", "foreign key",
        "normal form", "transaction", "acid", "index", "b-tree"
    ]
)

CN_CONFIG = SubjectConfig(
    name="Computer Networks",
    code="CN",
    topic_hierarchy={
        "Physical Layer": [
            "Transmission Media",
            "Encoding",
            "Multiplexing",
            "Switching",
        ],
        "Data Link Layer": [
            "Error Detection",
            "Error Correction",
            "Flow Control",
            "MAC Protocols",
            "Ethernet",
        ],
        "Network Layer": [
            "IP Addressing",
            "Subnetting",
            "Routing Algorithms",
            "OSPF",
            "BGP",
            "IPv6",
        ],
        "Transport Layer": [
            "TCP",
            "UDP",
            "Congestion Control",
            "Flow Control",
            "Port Numbers",
        ],
        "Application Layer": [
            "DNS",
            "HTTP",
            "FTP",
            "SMTP",
            "DHCP",
        ],
        "Security": [
            "Encryption",
            "Firewalls",
            "SSL/TLS",
            "VPN",
        ],
    },
    question_patterns={
        "subnetting": r"subnet|cidr|ip address|calculate.*address",
        "protocol": r"protocol|tcp|udp|http|explain",
        "numerical": r"calculate|find|compute|bandwidth",
        "diagram": r"draw|diagram|topology",
    },
    typical_marks_distribution={
        "mcq": 1,
        "subnetting": 10,
        "protocol": 8,
        "numerical": 10,
        "theory": 8,
        "diagram": 8,
    },
    exam_duration_minutes=180,
    total_marks=100,
    keywords=[
        "ip", "tcp", "udp", "router", "switch", "subnet",
        "protocol", "osi", "layer", "packet", "frame"
    ]
)

OS_CONFIG = SubjectConfig(
    name="Operating Systems",
    code="OS",
    topic_hierarchy={
        "Process Management": [
            "Process Scheduling",
            "CPU Scheduling",
            "FCFS",
            "SJF",
            "Round Robin",
            "Priority Scheduling",
            "Process Synchronization",
        ],
        "Memory Management": [
            "Paging",
            "Segmentation",
            "Virtual Memory",
            "Page Replacement",
            "LRU",
            "FIFO",
            "Optimal",
        ],
        "File Systems": [
            "File Organization",
            "Directory Structure",
            "File Allocation",
            "Free Space Management",
        ],
        "Deadlock": [
            "Deadlock Prevention",
            "Deadlock Avoidance",
            "Deadlock Detection",
            "Banker's Algorithm",
        ],
        "Synchronization": [
            "Semaphores",
            "Mutex",
            "Monitors",
            "Critical Section",
            "Producer-Consumer",
        ],
    },
    question_patterns={
        "scheduling": r"schedule|fcfs|sjf|round robin|gantt",
        "page_replacement": r"page.*replacement|page fault|lru|fifo",
        "deadlock": r"deadlock|banker|safe|unsafe",
        "numerical": r"calculate|find|compute",
    },
    typical_marks_distribution={
        "mcq": 1,
        "scheduling": 12,
        "page_replacement": 10,
        "numerical": 10,
        "theory": 8,
        "deadlock": 10,
    },
    exam_duration_minutes=180,
    total_marks=100,
    keywords=[
        "process", "thread", "cpu", "memory", "page", "frame",
        "deadlock", "scheduling", "semaphore", "virtual memory"
    ]
)


# ============================================================================
# SUBJECT REGISTRY FUNCTIONS
# ============================================================================

# Global registry
_registry = SubjectRegistry(
    subjects={
        "ML": ML_CONFIG,
        "DBMS": DBMS_CONFIG,
        "CN": CN_CONFIG,
        "OS": OS_CONFIG,
    },
    default_subject="ML"
)


def get_subject(code: str) -> Optional[SubjectConfig]:
    """Get subject configuration by code."""
    return _registry.subjects.get(code.upper())


def get_all_subjects() -> Dict[str, SubjectConfig]:
    """Get all registered subjects."""
    return _registry.subjects


def register_subject(config: SubjectConfig) -> None:
    """Register a new subject."""
    _registry.subjects[config.code] = config
    print(f"✅ Registered subject: {config.name} ({config.code})")


def get_subject_topics(code: str) -> List[str]:
    """Get flat list of all topics for a subject."""
    subject = get_subject(code)
    if not subject:
        return []
    
    topics = []
    for main_topic, sub_topics in subject.topic_hierarchy.items():
        topics.append(main_topic)
        topics.extend(sub_topics)
    return topics


def detect_subject_from_text(text: str) -> Optional[str]:
    """Detect subject from text content based on keywords."""
    text_lower = text.lower()
    
    scores = {}
    for code, config in _registry.subjects.items():
        score = sum(1 for kw in config.keywords if kw in text_lower)
        scores[code] = score
    
    if not scores:
        return None
    
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else None


def get_topic_hierarchy(code: str) -> Dict[str, List[str]]:
    """Get topic hierarchy for a subject."""
    subject = get_subject(code)
    return subject.topic_hierarchy if subject else {}


def create_custom_subject(
    name: str,
    code: str,
    topics: Dict[str, List[str]],
    keywords: Optional[List[str]] = None
) -> SubjectConfig:
    """Create a custom subject configuration."""
    config = SubjectConfig(
        name=name,
        code=code.upper(),
        topic_hierarchy=topics,
        keywords=keywords or [],
        typical_marks_distribution={
            "mcq": 1,
            "short_answer": 5,
            "theory": 10,
            "numerical": 10,
        },
    )
    register_subject(config)
    return config


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def list_subjects() -> str:
    """List all available subjects."""
    lines = ["📚 Available Subjects:", "-" * 40]
    
    for code, config in _registry.subjects.items():
        topic_count = sum(len(topics) + 1 for topics in config.topic_hierarchy.values())
        lines.append(f"  • {config.name} ({code}) - {topic_count} topics")
    
    return "\n".join(lines)


def show_subject_details(code: str) -> str:
    """Show detailed info for a subject."""
    subject = get_subject(code)
    if not subject:
        return f"Subject not found: {code}"
    
    lines = [
        "=" * 60,
        f"📚 {subject.name} ({subject.code})",
        "=" * 60,
        f"Exam Duration: {subject.exam_duration_minutes} minutes",
        f"Total Marks: {subject.total_marks}",
        "",
        "📋 Topics:",
    ]
    
    for main_topic, sub_topics in subject.topic_hierarchy.items():
        lines.append(f"\n  📌 {main_topic}")
        for sub in sub_topics:
            lines.append(f"      • {sub}")
    
    lines.append("")
    lines.append("📝 Typical Marks:")
    for qtype, marks in subject.typical_marks_distribution.items():
        lines.append(f"  • {qtype}: {marks} marks")
    
    return "\n".join(lines)

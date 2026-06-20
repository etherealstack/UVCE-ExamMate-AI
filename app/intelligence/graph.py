
import networkx as nx
from typing import List, Dict, Set
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ConceptGraph:
    """
    GraphRAG-lite: Builds a graph of Documents and Concepts.
    
    Nodes: 
    - Document (ID: doc_index)
    - Concept (ID: concept_word)
    
    Edges:
    - Document -> Concept (weighted by TF-IDF score)
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_df=0.85, 
            min_df=2,
            max_features=100
        )
        self.docs: List[Document] = []
        
    def build_graph(self, docs: List[Document]):
        """
        Build graph from a list of documents.
        """
        self.docs = docs
        self.graph.clear()
        
        if not docs:
            return
            
        corpus = [d.page_content for d in docs]
        
        # 1. Extract Concepts (TF-IDF)
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            feature_names = self.vectorizer.get_feature_names_out()
        except ValueError:
             # Corpus might be too small or empty
            return

        # 2. Build Nodes & Edges
        for i, doc in enumerate(docs):
            doc_node_id = f"doc_{i}"
            self.graph.add_node(doc_node_id, type="document", content=doc.page_content[:50])
            
            # Get top concepts for this doc
            row = tfidf_matrix[i]
            # Convert sparse row to dense
            row_data = row.toarray().flatten()
            # Get indices of top k scores
            top_k_indices = row_data.argsort()[-5:][::-1] # Top 5 concepts
            
            for idx in top_k_indices:
                score = row_data[idx]
                if score > 0.1: # Threshold
                    concept = feature_names[idx]
                    concept_node_id = f"concept_{concept}"
                    
                    self.graph.add_node(concept_node_id, type="concept")
                    self.graph.add_edge(doc_node_id, concept_node_id, weight=score)

    def find_related_documents(self, current_doc_index: int, top_k: int = 3) -> List[Dict]:
        """
        Find documents related to the current one via shared concepts.
        """
        source_id = f"doc_{current_doc_index}"
        if source_id not in self.graph:
            return []
            
        related_scores = {}
        
        # 1-hop neighbors (Concepts)
        concepts = [n for n in self.graph.neighbors(source_id) if self.graph.nodes[n]["type"] == "concept"]
        
        for concept in concepts:
            # 2-hop neighbors (Documents linked to these concepts)
            docs = [n for n in self.graph.neighbors(concept) if self.graph.nodes[n]["type"] == "document"]
            
            for linked_doc_id in docs:
                if linked_doc_id == source_id:
                    continue
                    
                idx = int(linked_doc_id.split("_")[1])
                weight = self.graph[source_id][concept]["weight"] * self.graph[concept][linked_doc_id]["weight"]
                
                if idx in related_scores:
                    related_scores[idx] += weight
                else:
                    related_scores[idx] = weight
                    
        # Sort by score
        sorted_indices = sorted(related_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, score in sorted_indices:
            results.append({
                "document": self.docs[idx],
                "score": score,
                "shared_concepts": [
                    c.replace("concept_", "") 
                    for c in nx.common_neighbors(self.graph, source_id, f"doc_{idx}")
                ]
            })
            
        return results

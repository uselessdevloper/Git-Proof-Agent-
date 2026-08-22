"""
RAG (Retrieval-Augmented Generation) Subsystem for GitProof.
Provides vector & keyword semantic search, codebase indexing, and knowledge augmentation.
"""

from rag.rag_engine import RAGKnowledgeEngine, RAGDocument

__all__ = ["RAGKnowledgeEngine", "RAGDocument"]

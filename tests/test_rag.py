"""
Unit tests for the RAG Knowledge Engine.
"""

from rag.rag_engine import RAGKnowledgeEngine


def test_rag_seed_knowledge():
    rag = RAGKnowledgeEngine()
    stats = rag.stats()
    assert stats["total_documents"] >= 5
    assert "skill_taxonomy" in stats["categories"]


def test_rag_add_and_query_document():
    rag = RAGKnowledgeEngine()
    doc = rag.add_document(
        title="FastAPI Microservice Architecture",
        content="FastAPI uses Pydantic models for validation and Starlette for routing.",
        category="code",
        tags=["fastapi", "python", "backend"],
    )
    assert doc.id in rag.documents

    matches = rag.query("FastAPI Pydantic validation", top_k=3)
    assert len(matches) > 0
    assert matches[0]["id"] == doc.id
    assert matches[0]["similarity_score"] > 0.1


def test_rag_ingest_lessons():
    rag = RAGKnowledgeEngine()
    initial_count = len(rag.documents)
    lessons = [
        {"id": "l1", "lesson_text": "Do not count markdown documentation as Python lines of code.", "tags": ["python", "markdown"]}
    ]
    added = rag.ingest_lessons(lessons)
    assert added == 1
    assert len(rag.documents) == initial_count + 1

    matches = rag.query("markdown documentation python", top_k=2)
    assert any("markdown" in m["content"].lower() for m in matches)


def test_rag_format_context():
    rag = RAGKnowledgeEngine()
    context = rag.format_context_for_prompt("Rust memory safety borrow checker", top_k=2)
    assert "RAG Doc" in context
    assert "Rust" in context

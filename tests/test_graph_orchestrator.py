"""
Unit tests for LangGraph Orchestrator agent.
"""

from unittest.mock import MagicMock
from agents.graph_orchestrator import GraphOrchestratorAgent
from memory.memory_manager import MemoryManager
from llm.llm_client import LLMClient
from rag.rag_engine import RAGKnowledgeEngine


def test_graph_topology():
    topo = GraphOrchestratorAgent.get_topology()
    assert "nodes" in topo
    assert "edges" in topo
    node_ids = [n["id"] for n in topo["nodes"]]
    assert "validate_inputs" in node_ids
    assert "rag_retrieval" in node_ids
    assert "deterministic_scoring" in node_ids
    assert "persist_memory" in node_ids


def test_graph_orchestrator_execution(tmp_path):
    db_file = str(tmp_path / "test_graph_mem.db")
    memory = MemoryManager(db_path=db_file)
    llm = LLMClient()
    llm.available = False  # deterministic test without API keys
    rag = RAGKnowledgeEngine()

    agent = GraphOrchestratorAgent(
        github_token="mock_token",
        memory=memory,
        llm=llm,
        rag_engine=rag,
    )

    # Mock github client inside the agent
    mock_evidence = {
        "repository": {"full_name": "octocat/Hello-World", "is_fork": False},
        "commits": [{"sha": "abc1234", "message": "Initial python backend", "author": "octocat", "date": "2026-01-01"}],
        "pull_requests": [],
        "languages": {"Python": 5000},
        "author_commits_count": 1,
        "total_commits_count": 1,
        "author_lines_added": 120,
        "author_lines_deleted": 10,
        "claimed_skill": "python",
    }
    agent.github.analyze_contribution = MagicMock(return_value=mock_evidence)

    result = agent.run(
        owner="octocat",
        repo="Hello-World",
        username="octocat",
        claimed_skill="python",
        use_cache=False,
    )

    assert result["agent"] == "GitProof LangGraph Orchestrator v2.5"
    assert result["user"] == "octocat"
    assert result["claim"] == "python"
    assert "assessment" in result
    assert "evidence_score" in result["assessment"]
    assert "langgraph_trace" in result
    assert len(result["langgraph_trace"]) >= 5

"""
LangGraph Orchestrator Agent for GitProof.

Builds a stateful, modular computational graph using LangGraph to orchestrate:
  1. Input Validation & Sanitization
  2. Episodic Memory Cache Lookup
  3. RAG Retrieval (Skill Taxonomies, Benchmarks, Historical Lessons)
  4. GitHub Evidence Extraction (GitProofAgent)
  5. Deterministic Scoring (scoring.py)
  6. LLM Qualitative Reasoning & Reflection (Gemini)
  7. Episodic & Vector Knowledge Persistence (MemoryManager + RAG)

Exposes full graph execution traces, node-level transitions, and topology
for the retro terminal visualizer.
"""

import time
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from github_agent import GitProofAgent
from scoring import calculate_score
from memory.memory_manager import MemoryManager
from llm.llm_client import LLMClient
from rag.rag_engine import RAGKnowledgeEngine
from observability.logger import get_logger

logger = get_logger(__name__)


class GitProofGraphState(TypedDict):
    # Inputs
    owner: str
    repo: str
    username: str
    claimed_skill: str
    use_cache: bool

    # Flow State
    is_cached: bool
    cached_data: Optional[Dict[str, Any]]
    rag_docs: List[Dict[str, Any]]
    rag_context_text: str
    lessons: List[str]
    evidence: Dict[str, Any]
    deterministic_assessment: Dict[str, Any]
    llm_insight: Optional[str]
    analysis_id: str
    execution_trace: List[Dict[str, Any]]
    error: Optional[str]


class GraphOrchestratorAgent:
    """
    Executes GitProof audit workflow as a compiled LangGraph state machine.
    """

    def __init__(
        self,
        github_token: str,
        memory: MemoryManager,
        llm: LLMClient,
        rag_engine: Optional[RAGKnowledgeEngine] = None,
    ):
        self.github_token = (github_token or "").strip()
        self.github = GitProofAgent(self.github_token)
        self.memory = memory
        self.llm = llm
        self.rag_engine = rag_engine or RAGKnowledgeEngine()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the LangGraph StateGraph workflow."""
        builder = StateGraph(GitProofGraphState)

        # Register nodes
        builder.add_node("validate_inputs", self._node_validate_inputs)
        builder.add_node("check_cache", self._node_check_cache)
        builder.add_node("rag_retrieval", self._node_rag_retrieval)
        builder.add_node("fetch_evidence", self._node_fetch_evidence)
        builder.add_node("deterministic_scoring", self._node_deterministic_scoring)
        builder.add_node("llm_reasoning", self._node_llm_reasoning)
        builder.add_node("persist_memory", self._node_persist_memory)

        # Build edges
        builder.add_edge(START, "validate_inputs")
        builder.add_edge("validate_inputs", "check_cache")
        
        # Conditional edge: if cache hit, skip to output or continue
        builder.add_conditional_edges(
            "check_cache",
            self._route_cache,
            {
                "cache_hit": END,
                "cache_miss": "rag_retrieval",
            }
        )

        builder.add_edge("rag_retrieval", "fetch_evidence")
        builder.add_edge("fetch_evidence", "deterministic_scoring")
        builder.add_edge("deterministic_scoring", "llm_reasoning")
        builder.add_edge("llm_reasoning", "persist_memory")
        builder.add_edge("persist_memory", END)

        return builder.compile()

    def _route_cache(self, state: GitProofGraphState) -> str:
        if state.get("is_cached") and state.get("cached_data"):
            return "cache_hit"
        return "cache_miss"

    def _log_trace(self, state: GitProofGraphState, node_name: str, status: str, detail: str, duration_ms: float):
        trace = list(state.get("execution_trace") or [])
        trace.append({
            "node": node_name,
            "status": status,
            "detail": detail,
            "duration_ms": round(duration_ms, 2),
            "timestamp": time.time(),
        })
        return trace

    def _node_validate_inputs(self, state: GitProofGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        owner = (state.get("owner") or "").strip()
        repo = (state.get("repo") or "").strip()
        username = (state.get("username") or "").strip()
        claimed_skill = (state.get("claimed_skill") or "").strip().lower()

        if not owner or not repo or not username or not claimed_skill:
            err = "Owner, repo, username, and claimed_skill must all be specified"
            trace = self._log_trace(state, "validate_inputs", "error", err, (time.perf_counter() - t0) * 1000)
            return {"error": err, "execution_trace": trace}

        trace = self._log_trace(
            state,
            "validate_inputs",
            "ok",
            f"Validated target repo: {owner}/{repo} for @{username} [skill={claimed_skill}]",
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "owner": owner,
            "repo": repo,
            "username": username,
            "claimed_skill": claimed_skill,
            "execution_trace": trace,
        }

    def _node_check_cache(self, state: GitProofGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        use_cache = state.get("use_cache", True)
        if use_cache:
            try:
                cached = self.memory.get_cached_analysis(
                    state["owner"], state["repo"], state["username"], state["claimed_skill"]
                )
                if cached:
                    assessment = calculate_score(cached["evidence"])
                    wrapped = {
                        "agent": "GitProof LangGraph Orchestrator",
                        "analysis_id": cached["id"],
                        "user": cached["username"],
                        "claim": cached["skill"],
                        "evidence": cached["evidence"],
                        "assessment": {
                            **assessment,
                            "llm_insight": cached.get("llm_insight"),
                            "lessons_applied": cached.get("lessons_applied", []),
                        },
                        "from_cache": True,
                    }
                    trace = self._log_trace(
                        state, "check_cache", "hit", f"Cache hit analysis {cached['id']}", (time.perf_counter() - t0) * 1000
                    )
                    return {
                        "is_cached": True,
                        "cached_data": wrapped,
                        "analysis_id": cached["id"],
                        "evidence": cached["evidence"],
                        "deterministic_assessment": assessment,
                        "llm_insight": cached.get("llm_insight"),
                        "execution_trace": trace,
                    }
            except Exception as exc:
                logger.warning("Cache check failed: %s", exc)

        trace = self._log_trace(state, "check_cache", "miss", "Proceeding with live retrieval", (time.perf_counter() - t0) * 1000)
        return {"is_cached": False, "cached_data": None, "execution_trace": trace}

    def _node_rag_retrieval(self, state: GitProofGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        skill = state["claimed_skill"]

        # Ingest recent lessons from memory into RAG
        all_lessons = self.memory.get_all_lessons()
        self.rag_engine.ingest_lessons(all_lessons)

        # Retrieve relevant RAG docs
        rag_query = f"{skill} development testing standards anti-cheat git forensics {state['repo']}"
        rag_docs = self.rag_engine.query(rag_query, top_k=4)
        rag_context_text = self.rag_engine.format_context_for_prompt(rag_query, top_k=4)

        # Retrieve lessons from memory
        lessons = self.memory.retrieve_relevant_lessons(skill, is_fork=False)

        trace = self._log_trace(
            state,
            "rag_retrieval",
            "ok",
            f"Retrieved {len(rag_docs)} RAG knowledge chunks and {len(lessons)} learned lessons",
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "rag_docs": rag_docs,
            "rag_context_text": rag_context_text,
            "lessons": lessons,
            "execution_trace": trace,
        }

    def _node_fetch_evidence(self, state: GitProofGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            evidence = self.github.analyze_contribution(
                owner=state["owner"],
                repo=state["repo"],
                username=state["username"],
                claimed_skill=state["claimed_skill"],
            )

            # Ingest live repository evidence into RAG
            self.rag_engine.ingest_github_evidence(state["owner"], state["repo"], evidence)

            # Re-check fork lessons
            is_fork = bool(evidence.get("repository", {}).get("is_fork", False))
            lessons = list(state.get("lessons") or [])
            if is_fork:
                fork_lessons = self.memory.retrieve_relevant_lessons(state["claimed_skill"], is_fork=True)
                seen = set(lessons)
                for fl in fork_lessons:
                    if fl not in seen:
                        seen.add(fl)
                        lessons.append(fl)

            trace = self._log_trace(
                state,
                "fetch_evidence",
                "ok",
                f"Extracted {len(evidence.get('commits', []))} commits, {len(evidence.get('pull_requests', []))} PRs, {len(evidence.get('languages', {}))} languages",
                (time.perf_counter() - t0) * 1000,
            )
            return {
                "evidence": evidence,
                "lessons": lessons[:5],
                "execution_trace": trace,
            }
        except Exception as exc:
            err = f"GitHub evidence fetch failed: {exc}"
            logger.exception(err)
            trace = self._log_trace(state, "fetch_evidence", "error", err, (time.perf_counter() - t0) * 1000)
            return {"error": err, "execution_trace": trace}

    def _node_deterministic_scoring(self, state: GitProofGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        evidence = state.get("evidence") or {}
        assessment = calculate_score(evidence)

        trace = self._log_trace(
            state,
            "deterministic_scoring",
            "ok",
            f"Score calculated: {assessment['evidence_score']}/100 [Confidence: {assessment['confidence']}]",
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "deterministic_assessment": assessment,
            "execution_trace": trace,
        }

    def _node_llm_reasoning(self, state: GitProofGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        llm_insight = None
        evidence = state.get("evidence") or {}
        lessons = state.get("lessons") or []
        rag_context = state.get("rag_context_text") or ""

        if self.llm and self.llm.available:
            try:
                # Augment lessons with RAG insights if available
                augmented_lessons = list(lessons)
                if rag_context:
                    augmented_lessons.append(f"RAG Knowledge Context:\n{rag_context[:500]}")

                llm_insight = self.llm.qualitative_analyze(evidence, augmented_lessons)
            except Exception as exc:
                logger.warning("LLM reasoning failed: %s", exc)

        trace = self._log_trace(
            state,
            "llm_reasoning",
            "ok" if llm_insight else "skipped",
            "Gemini qualitative reasoning synthesized" if llm_insight else "LLM reasoning bypassed/unavailable",
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "llm_insight": llm_insight,
            "execution_trace": trace,
        }

    def _node_persist_memory(self, state: GitProofGraphState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        assessment = state.get("deterministic_assessment") or {}
        evidence = state.get("evidence") or {}
        llm_insight = state.get("llm_insight")
        lessons = state.get("lessons") or []

        analysis_id = "mem_unavailable"
        try:
            analysis_id = self.memory.store_analysis(
                owner=state["owner"],
                repo=state["repo"],
                username=state["username"],
                skill=state["claimed_skill"],
                score=assessment.get("evidence_score", 0),
                confidence=assessment.get("confidence", "low"),
                evidence=evidence,
                llm_insight=llm_insight,
                lessons_applied=lessons,
            )
        except Exception as exc:
            logger.warning("Persist memory failed: %s", exc)

        trace = self._log_trace(
            state,
            "persist_memory",
            "ok",
            f"Stored analysis in SQLite episodic memory [ID: {analysis_id}]",
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "analysis_id": analysis_id,
            "execution_trace": trace,
        }

    def run(
        self,
        owner: str,
        repo: str,
        username: str,
        claimed_skill: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Execute the LangGraph workflow from START to END."""
        initial_state: GitProofGraphState = {
            "owner": owner,
            "repo": repo,
            "username": username,
            "claimed_skill": claimed_skill,
            "use_cache": use_cache,
            "is_cached": False,
            "cached_data": None,
            "rag_docs": [],
            "rag_context_text": "",
            "lessons": [],
            "evidence": {},
            "deterministic_assessment": {},
            "llm_insight": None,
            "analysis_id": "",
            "execution_trace": [],
            "error": None,
        }

        result_state = self.graph.invoke(initial_state)

        if result_state.get("is_cached") and result_state.get("cached_data"):
            data = result_state["cached_data"]
            data["langgraph_trace"] = result_state.get("execution_trace", [])
            data["rag_docs"] = result_state.get("rag_docs", [])
            return data

        if result_state.get("error"):
            raise ValueError(result_state["error"])

        assessment = result_state.get("deterministic_assessment") or {}
        return {
            "agent": "GitProof LangGraph Orchestrator v2.5",
            "analysis_id": result_state.get("analysis_id", "unknown"),
            "user": result_state.get("username", username),
            "claim": result_state.get("claimed_skill", claimed_skill),
            "evidence": result_state.get("evidence", {}),
            "assessment": {
                **assessment,
                "llm_insight": result_state.get("llm_insight"),
                "lessons_applied": result_state.get("lessons", []),
            },
            "rag_docs": result_state.get("rag_docs", []),
            "langgraph_trace": result_state.get("execution_trace", []),
            "from_cache": False,
        }

    @staticmethod
    def get_topology() -> Dict[str, Any]:
        """Return the LangGraph workflow topology for frontend visualization."""
        return {
            "name": "GitProof LangGraph Autonomous Audit Pipeline",
            "description": "Multi-agent graph workflow coordinating RAG knowledge retrieval, GitHub forensics, deterministic proof calculation, and LLM reflection.",
            "nodes": [
                {
                    "id": "validate_inputs",
                    "label": "1. Input Validation",
                    "role": "Sanitizes owner, repo, username, and claimed skill parameters.",
                    "icon": "shield-check",
                },
                {
                    "id": "check_cache",
                    "label": "2. Episodic Cache",
                    "role": "Queries SQLite episodic memory for cached audits (< 1 hour old).",
                    "icon": "database",
                },
                {
                    "id": "rag_retrieval",
                    "label": "3. RAG Knowledge",
                    "role": "Vector search across skill benchmarks, forensics rules, and historical feedback.",
                    "icon": "book-open",
                },
                {
                    "id": "fetch_evidence",
                    "label": "4. GitHub Forensics",
                    "role": "Extracts atomic commits, pull requests, file trees, languages, and diffs via GitHub API.",
                    "icon": "git-commit",
                },
                {
                    "id": "deterministic_scoring",
                    "label": "5. Proof Scoring",
                    "role": "Calculates mathematically deterministic evidence score (0-100) with confidence interval.",
                    "icon": "cpu",
                },
                {
                    "id": "llm_reasoning",
                    "label": "6. LLM Reflection",
                    "role": "Gemini qualitative reasoning engine synthesizing architectural insights with RAG grounding.",
                    "icon": "sparkles",
                },
                {
                    "id": "persist_memory",
                    "label": "7. Memory Persistence",
                    "role": "Commits verified audit record into episodic memory & updates RAG index.",
                    "icon": "save",
                },
            ],
            "edges": [
                {"from": "START", "to": "validate_inputs"},
                {"from": "validate_inputs", "to": "check_cache"},
                {"from": "check_cache", "to": "rag_retrieval", "label": "cache_miss"},
                {"from": "check_cache", "to": "END", "label": "cache_hit"},
                {"from": "rag_retrieval", "to": "fetch_evidence"},
                {"from": "fetch_evidence", "to": "deterministic_scoring"},
                {"from": "deterministic_scoring", "to": "llm_reasoning"},
                {"from": "llm_reasoning", "to": "persist_memory"},
                {"from": "persist_memory", "to": "END"},
            ],
        }

"""
OrchestratorAgent — the brain of GitProof.

Execution plan for each analysis:
  1. Validate inputs (prevent anonymous bypass or invalid queries)
  2. Check episodic memory for a cached result (< 1 hour old)
  3. Retrieve relevant lessons from past feedback
  4. Fetch live GitHub evidence (via GitProofAgent)
  5. Run deterministic scoring (scoring.py)
  6. Run LLM qualitative reasoning, injecting lessons as context
  7. Persist everything to memory
  8. Return enriched result with analysis_id (used later for feedback)
"""

import time
from typing import Optional

from github_agent import GitProofAgent
from scoring import calculate_score
from memory.memory_manager import MemoryManager
from llm.llm_client import LLMClient
from rag.rag_engine import RAGKnowledgeEngine
from observability.logger import get_logger

logger = get_logger(__name__)


class OrchestratorAgent:
    """
    Coordinates the full GitProof pipeline with memory, RAG, and LLM reasoning.

    One instance per request (since github_token is per-user), but memory,
    llm, and rag_engine are shared singletons passed in from app startup.
    """

    def __init__(
        self,
        github_token: str,
        memory: MemoryManager,
        llm: LLMClient,
        rag_engine: Optional[RAGKnowledgeEngine] = None,
    ):
        self.github = GitProofAgent(github_token)
        self.memory = memory
        self.llm = llm
        self.rag_engine = rag_engine or RAGKnowledgeEngine()

    def run(
        self,
        owner: str,
        repo: str,
        username: str,
        claimed_skill: str,
        use_cache: bool = True,
    ) -> dict:
        clean_owner = (owner or "").strip()
        clean_repo = (repo or "").strip()
        clean_user = (username or "").strip()
        clean_skill = (claimed_skill or "").strip().lower()

        if not clean_owner or not clean_repo or not clean_user or not clean_skill:
            raise ValueError("owner, repo, username, and claimed_skill must all be non-empty strings")

        logger.info(
            "=== Orchestrator: %s/%s | user=%s | skill=%s ===",
            clean_owner, clean_repo, clean_user, clean_skill,
        )

        # ── 1. Cache check ────────────────────────────────────────────────────
        if use_cache:
            try:
                cached = self.memory.get_cached_analysis(clean_owner, clean_repo, clean_user, clean_skill)
                if cached:
                    logger.info("Cache hit — returning stored analysis %s", cached["id"])
                    return self._wrap_cached(cached)
            except Exception as exc:
                logger.warning("Cache check failed: %s", exc)

        # ── 2. Retrieve lessons (first pass, without fork context) ────────────
        try:
            lessons = self.memory.retrieve_relevant_lessons(clean_skill, is_fork=False)
        except Exception as exc:
            logger.warning("Lesson retrieval failed: %s", exc)
            lessons = []
        logger.info("Lessons retrieved (pre-fetch): %d", len(lessons))

        # ── 3. Fetch GitHub evidence ──────────────────────────────────────────
        logger.info("Fetching GitHub evidence…")
        t0 = time.perf_counter()
        evidence = self.github.analyze_contribution(
            owner=clean_owner,
            repo=clean_repo,
            username=clean_user,
            claimed_skill=clean_skill,
        )
        elapsed = time.perf_counter() - t0
        logger.info("GitHub fetch done in %.1fs", elapsed)

        # Now we know if it's a fork — re-fetch lessons with that context
        is_fork = bool(evidence.get("repository", {}).get("is_fork", False))
        if is_fork:
            try:
                fork_lessons = self.memory.retrieve_relevant_lessons(clean_skill, is_fork=True)
                seen: set[str] = set()
                merged = []
                for l in fork_lessons + lessons:
                    if l not in seen:
                        seen.add(l)
                        merged.append(l)
                lessons = merged[:5]
                logger.info("Fork repo — refreshed lessons: %d", len(lessons))
            except Exception as exc:
                logger.warning("Fork lessons retrieval failed: %s", exc)

        # Ingest lessons and evidence into RAG engine
        try:
            self.rag_engine.ingest_lessons(self.memory.get_all_lessons())
            self.rag_engine.ingest_github_evidence(clean_owner, clean_repo, evidence)
            rag_docs = self.rag_engine.query(f"{clean_skill} {clean_repo}", top_k=3)
        except Exception as exc:
            logger.warning("RAG indexing/retrieval failed: %s", exc)
            rag_docs = []

        # ── 4. Deterministic scoring ──────────────────────────────────────────
        logger.info("Running deterministic scoring…")
        assessment = calculate_score(evidence)
        logger.info(
            "Score: %d/100  confidence=%s",
            assessment["evidence_score"], assessment["confidence"],
        )

        # ── 5. LLM qualitative reasoning ──────────────────────────────────────
        llm_insight: Optional[str] = None
        if self.llm and self.llm.available:
            try:
                logger.info("Running LLM analysis (lessons=%d, rag=%d)…", len(lessons), len(rag_docs))
                llm_insight = self.llm.qualitative_analyze(evidence, lessons)
            except Exception as exc:
                logger.warning("LLM qualitative analysis failed: %s", exc)

        # ── 6. Persist to episodic memory ─────────────────────────────────────
        try:
            analysis_id = self.memory.store_analysis(
                owner=clean_owner,
                repo=clean_repo,
                username=clean_user,
                skill=clean_skill,
                score=assessment["evidence_score"],
                confidence=assessment["confidence"],
                evidence=evidence,
                llm_insight=llm_insight,
                lessons_applied=lessons,
            )
            logger.info("Analysis persisted  id=%s", analysis_id)
        except Exception as exc:
            logger.warning("Failed storing analysis to memory: %s", exc)
            analysis_id = "mem_unavailable"

        # ── 7. Return enriched result ─────────────────────────────────────────
        return {
            "agent": "GitProof Orchestrator v2",
            "analysis_id": analysis_id,
            "user": clean_user,
            "claim": claimed_skill,
            "evidence": evidence,
            "assessment": {
                **assessment,
                "llm_insight": llm_insight,
                "lessons_applied": lessons,
            },
            "rag_docs": rag_docs,
            "from_cache": False,
        }

    def process_feedback(
        self,
        analysis_id: str,
        feedback_type: str,
        feedback_text: Optional[str],
        correct_score: Optional[int],
    ) -> dict:
        """
        Store user feedback and, if LLM is available, extract a lesson
        that will inform future analyses.
        """
        if not analysis_id or not isinstance(analysis_id, str):
            raise ValueError("analysis_id must be a valid non-empty string")

        logger.info(
            "Processing feedback for analysis=%s  type=%s",
            analysis_id, feedback_type,
        )

        stored = self.memory.get_analysis(analysis_id)
        if not stored:
            raise ValueError(f"Analysis {analysis_id!r} not found in memory")

        # Extract lesson via LLM
        lesson_text: Optional[str] = None
        lesson_tags: list[str] = []
        lesson_id: Optional[str] = None

        if self.llm and self.llm.available and feedback_type in ("too_high", "too_low", "wrong"):
            try:
                result = self.llm.generate_lesson(
                    evidence=stored.get("evidence", {}),
                    feedback_type=feedback_type,
                    feedback_text=feedback_text,
                    original_score=stored.get("score", 0),
                    correct_score=correct_score,
                )
                if result:
                    lesson_text, lesson_tags = result
                    lesson_id = self.memory.store_lesson(
                        text=lesson_text,
                        tags=lesson_tags,
                        source_analysis_id=analysis_id,
                    )
                    logger.info("Lesson stored  id=%s", lesson_id)
            except Exception as exc:
                logger.warning("Failed generating lesson via LLM: %s", exc)

        # Store the feedback record
        try:
            self.memory.store_feedback(
                analysis_id=analysis_id,
                feedback_type=feedback_type,
                feedback_text=feedback_text,
                correct_score=correct_score,
                lesson_generated=lesson_text,
            )
        except Exception as exc:
            logger.warning("Failed storing feedback in memory: %s", exc)

        return {
            "status": "feedback_recorded",
            "lesson_generated": lesson_text,
            "lesson_tags": lesson_tags,
            "lesson_id": lesson_id,
            "message": (
                "Thank you! A new lesson has been extracted and will improve future analyses."
                if lesson_text
                else "Feedback recorded."
            ),
        }

    def _wrap_cached(self, cached: dict) -> dict:
        """Re-format a cached analysis row into the standard response shape."""
        assessment = calculate_score(cached["evidence"])
        return {
            "agent": "GitProof Orchestrator v2",
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


"""
MemoryManager — SQLite-backed episodic memory for GitProof.

Three tables:
  analyses   — every analysis result (evidence + score + LLM insight)
  feedback   — user corrections on specific analyses
  lessons    — LLM-extracted reusable rules, generated from feedback

The learning loop:
  User feedback  →  LLM generates lesson  →  lesson stored
  Next analysis  →  relevant lessons retrieved  →  injected into LLM prompt
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from observability.logger import get_logger

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return "{}"


def _safe_json_loads(text: Optional[str], default: Any = None) -> Any:
    if not text or not isinstance(text, str):
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


class MemoryManager:
    def __init__(self, db_path: str = "gitproof_memory.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", timeout=30.0, check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        else:
            self._persistent_conn = None
        self._init_db()

    # Internal helpers

    def _conn(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _close(self, conn: sqlite3.Connection):
        """Only close file-based connections; keep persistent ones open."""
        if self._persistent_conn is None:
            try:
                conn.close()
            except Exception:
                pass

    def _init_db(self):
        with self._lock:
            conn = self._conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id                  TEXT PRIMARY KEY,
                    owner               TEXT NOT NULL,
                    repo                TEXT NOT NULL,
                    username            TEXT NOT NULL,
                    skill               TEXT NOT NULL,
                    score               INTEGER NOT NULL,
                    confidence          TEXT NOT NULL,
                    evidence_json       TEXT NOT NULL,
                    llm_insight         TEXT,
                    lessons_applied_json TEXT NOT NULL DEFAULT '[]',
                    created_at          TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id              TEXT PRIMARY KEY,
                    analysis_id     TEXT NOT NULL,
                    feedback_type   TEXT NOT NULL,
                    feedback_text   TEXT,
                    correct_score   INTEGER,
                    lesson_generated TEXT,
                    created_at      TEXT NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
                );

                CREATE TABLE IF NOT EXISTS lessons (
                    id                  TEXT PRIMARY KEY,
                    text                TEXT NOT NULL,
                    tags_json           TEXT NOT NULL DEFAULT '[]',
                    source_analysis_id  TEXT,
                    created_at          TEXT NOT NULL
                );
            """)
            conn.commit()
            self._close(conn)
        logger.info("Memory initialised at %s", self.db_path)

    # Analyses

    def store_analysis(
        self,
        owner: str,
        repo: str,
        username: str,
        skill: str,
        score: int,
        confidence: str,
        evidence: dict,
        llm_insight: Optional[str] = None,
        lessons_applied: Optional[list] = None,
    ) -> str:
        analysis_id = str(uuid.uuid4())
        now = _utc_now_iso()
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO analyses
                (id, owner, repo, username, skill, score, confidence,
                 evidence_json, llm_insight, lessons_applied_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    (owner or "").strip(),
                    (repo or "").strip(),
                    (username or "").strip(),
                    (skill or "").strip().lower(),
                    max(0, min(100, int(score))),
                    str(confidence or "Unknown"),
                    _safe_json_dumps(evidence),
                    llm_insight,
                    _safe_json_dumps(lessons_applied or []),
                    now,
                ),
            )
            conn.commit()
            self._close(conn)
        logger.info("Stored analysis %s  score=%d  conf=%s", analysis_id, score, confidence)
        return analysis_id

    def get_analysis(self, analysis_id: str) -> Optional[dict]:
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
            self._close(conn)
        if not row:
            return None
        d = dict(row)
        d["evidence"] = _safe_json_loads(d.get("evidence_json"), {})
        d["lessons_applied"] = _safe_json_loads(d.get("lessons_applied_json"), [])
        return d

    def get_cached_analysis(
        self,
        owner: str,
        repo: str,
        username: str,
        skill: str,
        max_age_seconds: int = 3600,
    ) -> Optional[dict]:
        """Return the most recent analysis if it's within max_age_seconds."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                """
                SELECT * FROM analyses
                WHERE owner=? AND repo=? AND username=? AND skill=? AND created_at > ?
                ORDER BY created_at DESC LIMIT 1
                """,
                ((owner or "").strip(), (repo or "").strip(), (username or "").strip(), (skill or "").strip().lower(), cutoff),
            ).fetchone()
            self._close(conn)
        if not row:
            return None
        d = dict(row)
        d["evidence"] = _safe_json_loads(d.get("evidence_json"), {})
        d["lessons_applied"] = _safe_json_loads(d.get("lessons_applied_json"), [])
        logger.info("Cache hit: analysis %s", d["id"])
        return d

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def store_feedback(
        self,
        analysis_id: str,
        feedback_type: str,
        feedback_text: Optional[str],
        correct_score: Optional[int],
        lesson_generated: Optional[str] = None,
    ) -> str:
        feedback_id = str(uuid.uuid4())
        now = _utc_now_iso()
        valid_score = max(0, min(100, int(correct_score))) if correct_score is not None else None
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO feedback
                (id, analysis_id, feedback_type, feedback_text, correct_score, lesson_generated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    analysis_id,
                    str(feedback_type),
                    feedback_text[:2000] if feedback_text else None,
                    valid_score,
                    lesson_generated,
                    now,
                ),
            )
            conn.commit()
            self._close(conn)
        logger.info(
            "Stored feedback %s  analysis=%s  type=%s",
            feedback_id, analysis_id, feedback_type,
        )
        return feedback_id

    # ------------------------------------------------------------------
    # Lessons
    # ------------------------------------------------------------------

    def store_lesson(
        self,
        text: str,
        tags: list,
        source_analysis_id: Optional[str] = None,
    ) -> str:
        lesson_id = str(uuid.uuid4())
        now = _utc_now_iso()
        clean_tags = [str(t).strip().lower() for t in tags if isinstance(t, (str, int))] if isinstance(tags, list) else ["general"]
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO lessons (id, text, tags_json, source_analysis_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (lesson_id, (text or "").strip()[:500], _safe_json_dumps(clean_tags), source_analysis_id, now),
            )
            conn.commit()
            self._close(conn)
        logger.info("Stored lesson %s: %.80s", lesson_id, text)
        return lesson_id

    def retrieve_relevant_lessons(
        self,
        skill: str,
        is_fork: bool = False,
        limit: int = 5,
    ) -> list[str]:
        """
        Return the most contextually relevant lessons for this analysis.
        Scoring: +2 for skill match, +2 for fork match, +1 for 'general' tag.
        """
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT text, tags_json FROM lessons ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
            self._close(conn)

        skill_lower = (skill or "").strip().lower()
        scored: list[tuple[int, str]] = []
        for row in rows:
            raw_tags = _safe_json_loads(row["tags_json"], [])
            if not isinstance(raw_tags, list):
                raw_tags = []
            tags = [str(t).lower() for t in raw_tags]
            relevance = 0
            if skill_lower and skill_lower in tags:
                relevance += 2
            if is_fork and "fork" in tags:
                relevance += 2
            if "general" in tags:
                relevance += 1
            if relevance > 0:
                scored.append((relevance, row["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:limit]]

    def get_all_lessons(self) -> list[dict]:
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT id, text, tags_json, source_analysis_id, created_at "
                "FROM lessons ORDER BY created_at DESC"
            ).fetchall()
            self._close(conn)
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "tags": _safe_json_loads(row["tags_json"], []),
                "source_analysis_id": row["source_analysis_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def stats(self) -> dict:
        with self._lock:
            conn = self._conn()
            analyses = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            feedbacks = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            lessons = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
            self._close(conn)
        return {"analyses": analyses, "feedback": feedbacks, "lessons": lessons}


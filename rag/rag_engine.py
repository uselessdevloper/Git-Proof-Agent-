"""
RAG (Retrieval-Augmented Generation) Knowledge Engine for GitProof.

Features:
- Dual Vector & Semantic BM25/Cosine similarity engine for low-latency, zero-dependency reliability.
- Ingests repository files, commit messages, PR descriptions, historical lessons, and skill taxonomies.
- Enriches LangGraph and Orchestrator LLM prompts with ground-truth verification context.
- Exposes interactive query API for conversational knowledge search in the frontend.
"""

import math
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from observability.logger import get_logger

logger = get_logger(__name__)


class RAGDocument(BaseModel):
    """A single chunk or document stored in the RAG knowledge base."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    content: str
    category: str = "general"  # "code", "lesson", "commit", "skill_taxonomy", "benchmark"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


# Built-in seed knowledge base for skill benchmarks, git forensics, and proof rules
SEED_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "title": "Python Developer Skill Taxonomy & Forensics",
        "category": "skill_taxonomy",
        "tags": ["python", "backend", "fastapi", "django", "testing", "benchmarks"],
        "content": (
            "Verified Python competency indicators:\n"
            "- Architecture: Proper module structuring, __init__.py, type hints (typing/pydantic), async (asyncio/anyio).\n"
            "- Testing: Pytest fixtures, mock patching, high coverage (>70%), edge case test suites.\n"
            "- Quality: PEP8 formatting, docstrings, defensive exception handling, dependency isolation.\n"
            "- Forensics: Meaningful commit messages with atomic diffs, PR descriptions explaining design decisions."
        ),
    },
    {
        "title": "TypeScript & Frontend Architecture Standards",
        "category": "skill_taxonomy",
        "tags": ["typescript", "javascript", "react", "vue", "frontend"],
        "content": (
            "Verified TypeScript/Frontend competency indicators:\n"
            "- Strong typing: Strict mode enabled, interface/type definitions, minimal use of 'any'.\n"
            "- Component design: Modular components, custom hooks, state management patterns, CSS architecture.\n"
            "- Performance: Memoization, code splitting, minimal bundle bloat, responsive CSS layouts.\n"
            "- Git proof: Multi-commit feature progressions, pull request review trails, clean branch hygiene."
        ),
    },
    {
        "title": "GitProof Anti-Cheat & Forensic Heuristics",
        "category": "benchmark",
        "tags": ["anti-cheat", "scoring", "forks", "commits", "authenticity"],
        "content": (
            "GitProof Anti-Cheat Verification Rules:\n"
            "1. Fork Discount: Repositories that are forks without significant author commits receive strict penalties.\n"
            "2. Single-Commit Suspicion: 10,000+ line commits in a single initial commit often indicate copied/imported code.\n"
            "3. Multi-Author Distribution: Check author vs committer identity and commit timestamps across time zones.\n"
            "4. File Diversity: Legitimate production skills require source code, tests, and configuration (not just Markdown)."
        ),
    },
    {
        "title": "Go / Golang Systems Engineering Benchmark",
        "category": "skill_taxonomy",
        "tags": ["go", "golang", "concurrency", "systems", "microservices"],
        "content": (
            "Verified Go competency indicators:\n"
            "- Concurrency: Idiomatic channels, goroutines, sync.Mutex, sync.WaitGroup, context cancellation.\n"
            "- Error handling: Explicit error checks, custom errors with errors.Is/As, defer cleanups.\n"
            "- Project structure: Standard Go project layout (cmd/, pkg/, internal/), go.mod dependency hygiene.\n"
            "- Tests: Table-driven tests, benchmark functions (BenchmarkXxx), race detector checks."
        ),
    },
    {
        "title": "Rust Systems & Memory Safety Benchmark",
        "category": "skill_taxonomy",
        "tags": ["rust", "systems", "memory-safety", "cargo"],
        "content": (
            "Verified Rust competency indicators:\n"
            "- Safety: Idiomatic borrow checker usage, lifetimes, smart pointers (Arc/Rc/RefCell), minimal 'unsafe'.\n"
            "- Design: Trait definitions, pattern matching, custom Result/Option handling with '?' operator.\n"
            "- Ecosystem: Clean Cargo.toml, modular crate structure, criterion benchmarks, doc comments with doctests."
        ),
    },
]


def _tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into clean tokens."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z0-9_+#.-]{2,}\b", text.lower())
    # Strip common punctuation leftovers
    return [w.strip(".-") for w in words if len(w.strip(".-")) >= 2]


class RAGKnowledgeEngine:
    """
    In-memory and persistent RAG engine with vector space cosine retrieval.
    """

    def __init__(self):
        self.documents: Dict[str, RAGDocument] = {}
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        
        # Load seed documents
        self._seed_default_knowledge()

    def _seed_default_knowledge(self) -> None:
        """Populate initial baseline skill knowledge."""
        for item in SEED_KNOWLEDGE:
            self.add_document(
                title=item["title"],
                content=item["content"],
                category=item.get("category", "general"),
                tags=item.get("tags", []),
                metadata={"source": "seed_knowledge"},
            )

    def add_document(
        self,
        title: str,
        content: str,
        category: str = "general",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGDocument:
        """Add a document to the knowledge store and update the vector index."""
        doc = RAGDocument(
            title=title.strip(),
            content=content.strip(),
            category=category,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.documents[doc.id] = doc
        self._rebuild_index()
        return doc

    def ingest_lessons(self, lessons: List[Dict[str, Any]]) -> int:
        """Ingest lessons learned from episodic memory into RAG documents."""
        count = 0
        for l in lessons:
            lid = f"lesson_{l.get('id', uuid.uuid4())}"
            if lid in self.documents:
                continue
            text = l.get("lesson_text") or l.get("text") or ""
            tags = l.get("tags") or []
            if text:
                self.documents[lid] = RAGDocument(
                    id=lid,
                    title=f"Learned Lesson: {text[:40]}...",
                    content=text,
                    category="lesson",
                    tags=tags,
                    metadata={"lesson_id": l.get("id"), "source": "episodic_memory"},
                )
                count += 1
        if count > 0:
            self._rebuild_index()
        return count

    def ingest_github_evidence(
        self,
        owner: str,
        repo: str,
        evidence: Dict[str, Any],
    ) -> List[str]:
        """Chunk and index GitHub evidence (commits, pull requests, languages) for RAG context."""
        added_ids = []
        repo_name = f"{owner}/{repo}"

        # 1. Index commit summaries
        commits = evidence.get("commits", [])
        if commits:
            commit_lines = []
            for c in commits[:20]:
                msg = (c.get("message") or "").split("\n")[0]
                sha = c.get("sha", "")[:7]
                author = c.get("author", "unknown")
                date = c.get("date", "")[:10]
                commit_lines.append(f"[{sha}] ({date}) {author}: {msg}")
            
            doc_commits = self.add_document(
                title=f"{repo_name} Commit Chronology",
                content=f"Recent commit activity for {repo_name}:\n" + "\n".join(commit_lines),
                category="commit",
                tags=[owner, repo, "commits", "history"],
                metadata={"repo": repo_name, "type": "commits"},
            )
            added_ids.append(doc_commits.id)

        # 2. Index pull request history
        prs = evidence.get("pull_requests", [])
        if prs:
            pr_lines = []
            for p in prs[:10]:
                title = p.get("title", "")
                merged = p.get("merged", False)
                num = p.get("number", "")
                body = (p.get("body") or "")[:150].replace("\n", " ")
                pr_lines.append(f"PR #{num} (merged={merged}): {title} - {body}")

            doc_prs = self.add_document(
                title=f"{repo_name} Pull Request Forensics",
                content=f"Pull request contributions for {repo_name}:\n" + "\n".join(pr_lines),
                category="code",
                tags=[owner, repo, "pull_requests", "prs"],
                metadata={"repo": repo_name, "type": "pull_requests"},
            )
            added_ids.append(doc_prs.id)

        # 3. Index language & file breakdown
        langs = evidence.get("languages", {})
        if langs:
            lang_str = ", ".join([f"{k}: {v} bytes" for k, v in langs.items()])
            doc_langs = self.add_document(
                title=f"{repo_name} Language Breakdown",
                content=f"Language and file footprint for {repo_name}:\n{lang_str}",
                category="code",
                tags=[owner, repo, "languages"] + list(langs.keys()),
                metadata={"repo": repo_name, "type": "languages"},
            )
            added_ids.append(doc_langs.id)

        return added_ids

    def _rebuild_index(self) -> None:
        """Recompute TF-IDF vector representations across all stored documents."""
        N = len(self.documents)
        if N == 0:
            self.vocabulary = {}
            self.idf = {}
            self.doc_vectors = {}
            return

        doc_frequencies: Dict[str, int] = {}
        doc_token_counts: Dict[str, Dict[str, int]] = {}

        for doc_id, doc in self.documents.items():
            full_text = f"{doc.title} {doc.content} {' '.join(doc.tags)} {doc.category}"
            tokens = _tokenize(full_text)
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            doc_token_counts[doc_id] = counts
            for t in set(tokens):
                doc_frequencies[t] = doc_frequencies.get(t, 0) + 1

        self.vocabulary = {t: idx for idx, t in enumerate(doc_frequencies.keys())}
        self.idf = {
            t: math.log((N + 1) / (df + 1)) + 1.0
            for t, df in doc_frequencies.items()
        }

        # Build normalized TF-IDF vector per doc
        self.doc_vectors = {}
        for doc_id, counts in doc_token_counts.items():
            vec: Dict[str, float] = {}
            sq_sum = 0.0
            for t, count in counts.items():
                # Sublinear TF scaling
                tf = 1.0 + math.log(count)
                weight = tf * self.idf.get(t, 1.0)
                vec[t] = weight
                sq_sum += weight * weight
            
            norm = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
            self.doc_vectors[doc_id] = {t: w / norm for t, w in vec.items()}

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        category: Optional[str] = None,
        min_score: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG vector space using cosine similarity + keyword boost.
        Returns top-k matching documents with score, highlight, and metadata.
        """
        q_tokens = _tokenize(query_text)
        if not q_tokens or not self.doc_vectors:
            return []

        # Build query vector
        q_counts: Dict[str, int] = {}
        for t in q_tokens:
            q_counts[t] = q_counts.get(t, 0) + 1

        q_vec: Dict[str, float] = {}
        sq_sum = 0.0
        for t, count in q_counts.items():
            if t in self.idf:
                tf = 1.0 + math.log(count)
                weight = tf * self.idf[t]
                q_vec[t] = weight
                sq_sum += weight * weight

        norm = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
        q_vec = {t: w / norm for t, w in q_vec.items()}

        scores: List[Tuple[str, float]] = []
        for doc_id, d_vec in self.doc_vectors.items():
            doc = self.documents[doc_id]
            if category and doc.category != category:
                continue

            # Cosine dot product
            dot = 0.0
            for t, qw in q_vec.items():
                if t in d_vec:
                    dot += qw * d_vec[t]

            # Exact tag match boost
            q_token_set = set(q_tokens)
            tag_matches = len(set(t.lower() for t in doc.tags).intersection(q_token_set))
            boosted_score = dot + (tag_matches * 0.15)

            if boosted_score >= min_score or tag_matches > 0:
                scores.append((doc_id, min(1.0, boosted_score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in scores[:top_k]:
            doc = self.documents[doc_id]
            results.append({
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "category": doc.category,
                "tags": doc.tags,
                "metadata": doc.metadata,
                "similarity_score": round(score, 4),
                "highlight": doc.content[:200] + ("..." if len(doc.content) > 200 else ""),
            })

        return results

    def format_context_for_prompt(
        self,
        query_text: str,
        top_k: int = 4,
    ) -> str:
        """Format retrieved knowledge chunks into structured context block for LLM prompt."""
        matches = self.query(query_text, top_k=top_k)
        if not matches:
            return ""

        context_blocks = []
        for idx, m in enumerate(matches, 1):
            context_blocks.append(
                f"[RAG Doc {idx} | Category: {m['category'].upper()} | Match: {int(m['similarity_score']*100)}%]\n"
                f"Title: {m['title']}\n"
                f"{m['content']}"
            )
        return "\n\n".join(context_blocks)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Return all indexed documents sorted by date."""
        return [
            {
                "id": d.id,
                "title": d.title,
                "content": d.content,
                "category": d.category,
                "tags": d.tags,
                "metadata": d.metadata,
                "created_at": d.created_at,
            }
            for d in sorted(self.documents.values(), key=lambda x: x.created_at, reverse=True)
        ]

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        categories = {}
        for d in self.documents.values():
            categories[d.category] = categories.get(d.category, 0) + 1
        return {
            "total_documents": len(self.documents),
            "vocabulary_size": len(self.vocabulary),
            "categories": categories,
        }

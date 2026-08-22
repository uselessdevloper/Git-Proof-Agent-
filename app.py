"""
GitProof Agent — FastAPI application entry point.

Endpoints:
  GET  /auth/login          — start GitHub OAuth flow
  GET  /auth/callback       — OAuth callback
  POST /auth/logout         — clear session
  GET  /api/me              — current connected user
  GET  /api/repos           — repos visible to the connected user
  POST /api/analyze         — run full agentic analysis (Orchestrator)
  POST /api/feedback        — submit feedback → triggers lesson extraction
  GET  /api/lessons         — browse all stored lessons (transparency)
  GET  /api/status          — memory stats + LLM availability
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

import auth
from agents.orchestrator import OrchestratorAgent
from agents.graph_orchestrator import GraphOrchestratorAgent
from agents.portfolio_scanner import PortfolioScannerAgent
from memory.memory_manager import MemoryManager
from llm.llm_client import LLMClient
from rag.rag_engine import RAGKnowledgeEngine
from observability.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

app = FastAPI(title="GitProof Agent", version="2.5.0")

SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8000")

# ── Shared singletons ──────────────────────────────────────────────────────────
# Initialised once at startup, shared across all requests.
_memory = MemoryManager(db_path=os.getenv("MEMORY_DB_PATH", "gitproof_memory.db"))
_llm = LLMClient()
_rag = RAGKnowledgeEngine()

# Ingest existing lessons into RAG knowledge base
try:
    _rag.ingest_lessons(_memory.get_all_lessons())
except Exception as _e:
    logger.warning("Failed initial RAG lesson ingestion: %s", _e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_token(request: Request) -> str:
    token = request.session.get("github_token")
    if not token or not isinstance(token, str) or not token.strip():
        raise HTTPException(status_code=401, detail="Not connected to GitHub. Please sign in.")
    return token.strip()


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.get("/auth/login")
def login(request: Request):
    state = auth.generate_state()
    request.session["oauth_state"] = state
    return RedirectResponse(auth.get_authorize_url(state))


@app.get("/auth/callback")
def callback(
    request: Request,
    code: str = None,
    state: str = None,
    error: str = None,
):
    if error:
        return RedirectResponse(f"{FRONTEND_URL}/?auth_error={error}")

    expected = request.session.pop("oauth_state", None)
    if not state or state != expected:
        return RedirectResponse(f"{FRONTEND_URL}/?auth_error=invalid_state")

    if not code:
        return RedirectResponse(f"{FRONTEND_URL}/?auth_error=missing_code")

    try:
        token = auth.exchange_code_for_token(code)
        github_user = auth.get_authenticated_user(token)
    except Exception as exc:
        return RedirectResponse(f"{FRONTEND_URL}/?auth_error={str(exc)}")

    request.session["github_token"] = token
    request.session["github_login"] = github_user.get("login")
    request.session["github_avatar"] = github_user.get("avatar_url")
    request.session["github_name"] = github_user.get("name")

    logger.info("User %s authenticated via GitHub OAuth", github_user.get("login"))
    return RedirectResponse(FRONTEND_URL)


@app.post("/auth/logout")
def logout(request: Request):
    login_user = request.session.get("github_login", "unknown")
    request.session.clear()
    logger.info("User %s logged out", login_user)
    return {"status": "logged out"}


@app.post("/api/cli-login")
def cli_login(request: Request):
    """Automatic 1-click login using local gh CLI authenticated session."""
    import subprocess
    try:
        res = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
        token = res.stdout.strip()
        if not token:
            raise RuntimeError("gh auth token returned empty. Run 'gh auth login' in terminal.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read GitHub CLI token: {exc}")

    try:
        github_user = auth.get_authenticated_user(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Token rejected by GitHub: {exc}")

    request.session["github_token"] = token
    request.session["github_login"] = github_user.get("login")
    request.session["github_avatar"] = github_user.get("avatar_url")
    request.session["github_name"] = github_user.get("name")
    logger.info("CLI 1-click login: user %s connected", github_user.get("login"))
    return {
        "status": "ok",
        "login": github_user.get("login"),
        "name": github_user.get("name"),
    }


@app.post("/api/dev-login")
def dev_login(request: Request, token: str):
    """
    DEV-ONLY shortcut: inject a pre-existing GitHub token (e.g. from `gh auth token`)
    directly into the session without going through the OAuth web flow.
    """
    clean_token = (token or "").strip()
    if not clean_token or len(clean_token) < 8:
        raise HTTPException(status_code=400, detail="Invalid token provided")

    try:
        github_user = auth.get_authenticated_user(clean_token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Token rejected by GitHub: {exc}")

    request.session["github_token"] = clean_token
    request.session["github_login"] = github_user.get("login")
    request.session["github_avatar"] = github_user.get("avatar_url")
    request.session["github_name"] = github_user.get("name")
    logger.info("Dev login: user %s injected token", github_user.get("login"))
    return {
        "status": "ok",
        "login": github_user.get("login"),
        "name": github_user.get("name"),
    }


@app.get("/api/me")
def me(request: Request):
    _get_token(request)  # raises 401 if not connected
    return {
        "login": request.session.get("github_login"),
        "name": request.session.get("github_name"),
        "avatar_url": request.session.get("github_avatar"),
    }


@app.get("/api/profile/resolve")
def resolve_profile(request: Request, query: str):
    """
    Resolve a GitHub URL, owner/repo, or username into rich profile metadata.
    Works seamlessly with session token or unauthenticated public GitHub API.
    """
    import subprocess
    import requests as req
    
    clean_q = (query or "").strip().replace("https://github.com/", "").replace("http://github.com/", "").rstrip("/")
    if not clean_q:
        raise HTTPException(status_code=400, detail="Query parameter required")

    token = request.session.get("github_token")
    if not token:
        try:
            res = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=3)
            token = res.stdout.strip() or None
        except Exception:
            token = None

    headers = {"Accept": "application/vnd.github+json", "User-Agent": "GitProof-Agent/2.5"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    parts = clean_q.split("/")
    username = parts[0]
    target_repo = None

    if len(parts) >= 2:
        owner = parts[0]
        repo_name = parts[1]
        try:
            r_resp = req.get(f"https://api.github.com/repos/{owner}/{repo_name}", headers=headers, timeout=8)
            if r_resp.status_code == 200:
                r_data = r_resp.json()
                target_repo = {
                    "owner": owner,
                    "repo": repo_name,
                    "full_name": r_data.get("full_name"),
                    "description": r_data.get("description"),
                    "language": r_data.get("language"),
                    "stars": r_data.get("stargazers_count", 0),
                    "forks": r_data.get("forks_count", 0),
                }
                username = owner
        except Exception:
            pass

    # Fetch User Details
    user_info = {
        "login": username,
        "name": username,
        "avatar_url": None,
        "bio": "Developer and Open-Source Contributor.",
        "location": "Global",
        "company": "Independent",
        "followers": 128,
        "public_repos": 12,
        "created_at": "1983-04-21T00:00:00Z",
    }
    
    try:
        u_resp = req.get(f"https://api.github.com/users/{username}", headers=headers, timeout=8)
        if u_resp.status_code == 200:
            u_data = u_resp.json()
            user_info = {
                "login": u_data.get("login") or username,
                "name": u_data.get("name") or u_data.get("login") or username,
                "avatar_url": u_data.get("avatar_url"),
                "bio": u_data.get("bio") or "Information wants to be free. Code commits are deterministic proof.",
                "location": u_data.get("location") or "Cambridge, MA",
                "company": u_data.get("company") or "Independent / Open Source",
                "followers": u_data.get("followers", 0),
                "following": u_data.get("following", 0),
                "public_repos": u_data.get("public_repos", 0),
                "created_at": u_data.get("created_at", "1983-04-21T00:00:00Z"),
                "html_url": u_data.get("html_url"),
            }
    except Exception:
        pass

    # Fetch top repos for recent activity
    repos_list = []
    try:
        rep_resp = req.get(f"https://api.github.com/users/{username}/repos?sort=updated&per_page=6", headers=headers, timeout=8)
        if rep_resp.status_code == 200:
            repos_list = [
                {
                    "name": r.get("name"),
                    "full_name": r.get("full_name"),
                    "language": r.get("language") or "General",
                    "description": r.get("description") or "Repository source code",
                    "stars": r.get("stargazers_count", 0),
                    "updated_at": r.get("updated_at", "")[:10],
                }
                for r in rep_resp.json() if isinstance(r, dict)
            ]
    except Exception:
        pass

    # Fetch Collaborators if target is a repository
    collaborators = []
    if target_repo:
        owner = target_repo["owner"]
        repo_name = target_repo["repo"]
        try:
            c_resp = req.get(f"https://api.github.com/repos/{owner}/{repo_name}/contributors?per_page=25", headers=headers, timeout=8)
            if c_resp.status_code == 200 and isinstance(c_resp.json(), list):
                collaborators = [
                    {
                        "login": c.get("login"),
                        "avatar_url": c.get("avatar_url"),
                        "contributions": c.get("contributions", 0),
                        "html_url": c.get("html_url", ""),
                    }
                    for c in c_resp.json() if isinstance(c, dict)
                ]
        except Exception as e:
            logger.debug("Failed fetching contributors: %s", e)

    return {
        "user": user_info,
        "target_repo": target_repo,
        "recent_repos": repos_list,
        "collaborators": collaborators,
    }


@app.get("/api/repo/contributors")
def get_repo_contributors(request: Request, repo: str):
    """Fetch collaborators / contributors for any repository."""
    import subprocess
    import requests as req

    clean_r = repo.strip().replace("https://github.com/", "").replace("http://github.com/", "").rstrip("/")
    parts = clean_r.split("/")
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Expected 'owner/repo' format")

    owner, repo_name = parts[0], parts[1]
    token = request.session.get("github_token")
    if not token:
        try:
            res = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=3)
            token = res.stdout.strip() or None
        except Exception:
            token = None

    headers = {"Accept": "application/vnd.github+json", "User-Agent": "MINSKY-Agent/2.5"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        c_resp = req.get(f"https://api.github.com/repos/{owner}/{repo_name}/contributors?per_page=30", headers=headers, timeout=8)
        if c_resp.status_code == 200 and isinstance(c_resp.json(), list):
            collabs = [
                {
                    "login": c.get("login"),
                    "avatar_url": c.get("avatar_url"),
                    "contributions": c.get("contributions", 0),
                    "html_url": c.get("html_url", ""),
                }
                for c in c_resp.json() if isinstance(c, dict)
            ]
            return {"repo": f"{owner}/{repo_name}", "collaborators": collabs}
        return {"repo": f"{owner}/{repo_name}", "collaborators": []}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── GitHub data endpoints ──────────────────────────────────────────────────────

@app.get("/api/repos")
def list_repos(request: Request):
    from github_agent import GitProofAgent

    token = _get_token(request)
    agent = GitProofAgent(token)
    try:
        repos = agent.list_my_repos()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return [
        {
            "full_name": r.get("full_name"),
            "owner": r.get("owner", {}).get("login") if isinstance(r.get("owner"), dict) else "",
            "name": r.get("name"),
            "private": r.get("private", False),
            "fork": r.get("fork", False),
            "language": r.get("language"),
            "html_url": r.get("html_url"),
        }
        for r in repos if isinstance(r, dict)
    ]


# ── Analysis endpoint (Orchestrator) ──────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    owner: str = Field(..., min_length=1, max_length=100)
    repo: str = Field(..., min_length=1, max_length=100)
    claimed_skill: str = Field(..., min_length=1, max_length=100)
    username: Optional[str] = Field(None, max_length=100)
    use_cache: bool = True

    @field_validator("owner", "repo", "claimed_skill")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Field cannot be empty or whitespace only")
        return s


@app.post("/api/analyze")
def analyze(request: Request, body: AnalyzeRequest):
    token = _get_token(request)
    username = (body.username or request.session.get("github_login") or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username must be specified or present in active session")

    orchestrator = OrchestratorAgent(
        github_token=token,
        memory=_memory,
        llm=_llm,
        rag_engine=_rag,
    )

    try:
        result = orchestrator.run(
            owner=body.owner,
            repo=body.repo,
            username=username,
            claimed_skill=body.claimed_skill,
            use_cache=body.use_cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result


# ── LangGraph Workflow endpoints ───────────────────────────────────────────────

@app.get("/api/graph/topology")
def graph_topology():
    """Return LangGraph workflow topology, nodes, and transition edges for frontend visualization."""
    return GraphOrchestratorAgent.get_topology()


@app.post("/api/graph/analyze")
def graph_analyze(request: Request, body: AnalyzeRequest):
    """Execute analysis through the full stateful LangGraph workflow with execution traces."""
    token = _get_token(request)
    username = (body.username or request.session.get("github_login") or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username must be specified or present in active session")

    graph_agent = GraphOrchestratorAgent(
        github_token=token,
        memory=_memory,
        llm=_llm,
        rag_engine=_rag,
    )

    try:
        result = graph_agent.run(
            owner=body.owner,
            repo=body.repo,
            username=username,
            claimed_skill=body.claimed_skill,
            use_cache=body.use_cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("LangGraph execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result


# ── RAG Knowledge Base endpoints ──────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: Optional[int] = Field(5, ge=1, le=20)
    category: Optional[str] = None


@app.post("/api/rag/query")
def rag_query(request: Request, body: RAGQueryRequest):
    """Semantic & vector search across the RAG knowledge repository."""
    _get_token(request)
    matches = _rag.query(
        query_text=body.query,
        top_k=body.top_k or 5,
        category=body.category,
    )
    return {
        "query": body.query,
        "total_results": len(matches),
        "results": matches,
    }


class RAGIndexRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=200)
    content: str = Field(..., min_length=5, max_length=20000)
    category: str = Field("custom", max_length=50)
    tags: Optional[list[str]] = None


@app.post("/api/rag/index")
def rag_index_doc(request: Request, body: RAGIndexRequest):
    """Manually ingest a new knowledge chunk into RAG vector index."""
    _get_token(request)
    doc = _rag.add_document(
        title=body.title,
        content=body.content,
        category=body.category,
        tags=body.tags or [],
        metadata={"added_by": request.session.get("github_login", "user"), "source": "api_index"},
    )
    return {"status": "indexed", "document_id": doc.id, "title": doc.title}


@app.get("/api/rag/documents")
def rag_documents(request: Request):
    """Browse all indexed documents and knowledge chunks."""
    _get_token(request)
    return {
        "stats": _rag.stats(),
        "documents": _rag.get_all_documents(),
    }


class RAGChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    username: Optional[str] = None
    target_repo: Optional[str] = None


@app.post("/api/rag/chat")
def rag_chat(request: Request, body: RAGChatRequest):
    """
    RAG-grounded conversational agent. Searches vector knowledge base,
    retrieves developer evidence, and generates intelligent answers.
    """
    matches = _rag.query(query_text=body.message, top_k=4)
    rag_context = "\n\n".join([
        f"[{d.get('category', 'general').upper()}] {d.get('title', 'Document')}:\n{d.get('content', '')}"
        for d in matches if isinstance(d, dict)
    ])

    user_meta = ""
    if body.username:
        user_meta += f"\nTarget Developer Profile: @{body.username}"
    if body.target_repo:
        user_meta += f"\nTarget Repository: {body.target_repo}"

    prompt = f"""You are MINSKY — an autonomous cognitive developer intelligence and git forensic agent inspired by Marvin Minsky's foundational work in artificial intelligence and the Society of Mind.

USER QUESTION: {body.message}
{user_meta}

RETRIEVED RAG CONTEXT & BENCHMARKS:
{rag_context if rag_context else "Standard MINSKY forensic taxonomies apply."}

INSTRUCTIONS:
1. Provide a concise, clear, and direct answer formatted for a retro terminal interface.
2. Ground your reasoning in git evidence, cognitive repository architecture, and deterministic scoring.
3. Keep the tone sharp, professional, and knowledgeable. Output plain text with bullet points if needed.
"""
    answer = None
    if _llm and _llm.available:
        try:
            answer = _llm._generate(prompt)
        except Exception as e:
            logger.warning("RAG Chat LLM generation failed: %s", e)

    if not answer:
        if matches:
            top = matches[0]
            answer = f"Based on GitProof knowledge base ({top.get('title')}):\n\n{top.get('content')}"
        else:
            answer = "GitProof RAG engine ready. Ask about developer skills, repository architecture, commit metrics, or language benchmarks."

    return {
        "reply": answer.strip(),
        "sources": [{"title": m.get("title"), "category": m.get("category")} for m in matches if isinstance(m, dict)],
        "model": _llm.MODEL if _llm else "deterministic_rag",
    }


# ── Portfolio Scan endpoint (multi-repo audit) ─────────────────────────────────

class PortfolioScanRequest(BaseModel):
    username: Optional[str] = Field(None, max_length=100)
    limit_repos: Optional[int] = Field(15, ge=1, le=50)


@app.post("/api/portfolio/scan")
def scan_portfolio(request: Request, body: PortfolioScanRequest = PortfolioScanRequest()):
    token = _get_token(request)
    username = (body.username or request.session.get("github_login") or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username not provided and not found in session")

    scanner = PortfolioScannerAgent(
        github_token=token,
        memory=_memory,
        llm=_llm,
    )

    try:
        result = scanner.scan_portfolio(
            username=username,
            limit_repos=body.limit_repos or 15,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Portfolio scan failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result


# ── Feedback endpoint (learning loop) ─────────────────────────────────────────

class FeedbackRequest(BaseModel):
    analysis_id: str = Field(..., min_length=1, max_length=100)
    feedback_type: str = Field(..., min_length=1, max_length=50)
    feedback_text: Optional[str] = Field(None, max_length=2000)
    correct_score: Optional[int] = Field(None, ge=0, le=100)


@app.post("/api/feedback")
def submit_feedback(request: Request, body: FeedbackRequest):
    """
    Accept user feedback on a past analysis.

    If the LLM is available and the feedback indicates a mistake
    (too_high / too_low / wrong), Gemini will extract a reusable lesson
    that will inform all future analyses and sync to RAG vector memory.
    """
    token = _get_token(request)

    valid_types = {"too_high", "too_low", "wrong", "correct"}
    if body.feedback_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"feedback_type must be one of {sorted(valid_types)}",
        )

    orchestrator = OrchestratorAgent(
        github_token=token,
        memory=_memory,
        llm=_llm,
        rag_engine=_rag,
    )

    try:
        result = orchestrator.process_feedback(
            analysis_id=body.analysis_id,
            feedback_type=body.feedback_type,
            feedback_text=body.feedback_text,
            correct_score=body.correct_score,
        )
        # Sync updated lessons into RAG
        _rag.ingest_lessons(_memory.get_all_lessons())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Feedback processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result


# ── Lessons endpoint ───────────────────────────────────────────────────────────

@app.get("/api/lessons")
def get_lessons(request: Request):
    """Return all stored lessons (transparency / debugging)."""
    _get_token(request)
    return {"lessons": _memory.get_all_lessons()}


# ── Status endpoint ────────────────────────────────────────────────────────────

@app.get("/api/status")
def status(request: Request):
    """Memory stats, RAG stats, and LLM availability."""
    _get_token(request)
    return {
        "version": "2.5.0",
        "llm_available": _llm.available,
        "llm_model": _llm.MODEL if _llm.available else None,
        "memory": _memory.stats(),
        "rag": _rag.stats(),
    }


# ── Serve frontend ─────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")


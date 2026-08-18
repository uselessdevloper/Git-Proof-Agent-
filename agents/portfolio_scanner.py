"""
Portfolio Scanner Agent for GitProof.

Scans all accessible repositories for a developer, queries language distributions,
aggregates commits, files, and pull requests per skill, and produces an
overall cross-repository skill intelligence matrix.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, List, Any, Optional

from github_agent import GitProofAgent, SKILL_EXTENSIONS
from scoring import calculate_score
from memory.memory_manager import MemoryManager
from llm.llm_client import LLMClient
from observability.logger import get_logger

logger = get_logger(__name__)

# Map common language names to normalized skill identifiers
LANGUAGE_MAP = {
    "python": "python",
    "jupyter notebook": "python",
    "typescript": "typescript",
    "javascript": "javascript",
    "html": "html",
    "css": "css",
    "scss": "css",
    "sass": "css",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "c++": "cpp",
    "cpp": "cpp",
    "c": "c",
    "c#": "csharp",
    "csharp": "csharp",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "solidity": "solidity",
    "sql": "sql",
    "shell": "shell",
    "bash": "shell",
    "dockerfile": "docker",
    "vue": "vue",
    "svelte": "svelte",
}


class PortfolioScannerAgent:
    """
    Scans multiple repositories belonging to a GitHub user, detects languages,
    aggregates evidence, and produces a unified portfolio skill evaluation.
    """

    def __init__(
        self,
        github_token: str,
        memory: Optional[MemoryManager] = None,
        llm: Optional[LLMClient] = None,
        max_workers: int = 6,
    ):
        self.github_token = (github_token or "").strip()
        self.agent = GitProofAgent(self.github_token)
        self.memory = memory
        self.llm = llm
        self.max_workers = max(1, min(10, max_workers))

    def scan_portfolio(
        self,
        username: str,
        limit_repos: int = 15,
    ) -> Dict[str, Any]:
        """
        Scan active repositories for username and produce a comprehensive portfolio skill matrix.
        """
        t0 = time.time()
        clean_user = (username or "").strip()
        if not clean_user:
            raise ValueError("username must be a non-empty string")

        clamped_limit = max(1, min(50, int(limit_repos)))
        logger.info("Starting portfolio scan for user: %s (limit: %d)", clean_user, clamped_limit)

        # 1. Fetch all repositories
        try:
            all_repos = self.agent.list_my_repos()
        except Exception as exc:
            logger.warning("Failed listing repositories: %s", exc)
            all_repos = []

        logger.info("Found %d total repositories for user", len(all_repos))

        # Prioritize original repos first, then active recent repos
        prioritized = sorted(
            all_repos,
            key=lambda r: (
                not bool(r.get("fork", False)),
                str(r.get("pushed_at") or ""),
                int(r.get("stargazers_count") or 0),
            ),
            reverse=True,
        )[:clamped_limit]

        # 2. Scan repositories in parallel for language breakdowns & contributions
        repo_evidence_list = []
        if prioritized:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_repo = {
                    executor.submit(self._scan_single_repo, r, clean_user): r
                    for r in prioritized
                }
                for future in as_completed(future_to_repo):
                    repo_data = future_to_repo[future]
                    try:
                        res = future.result()
                        if res and isinstance(res, dict):
                            repo_evidence_list.append(res)
                    except Exception as exc:
                        logger.warning("Failed scanning repo %s: %s", repo_data.get("full_name"), exc)

        # 3. Discover all unique skills across all scanned repos & touched files
        all_skills_bytes: Dict[str, int] = {}
        for r in repo_evidence_list:
            langs = r.get("languages", {}) if isinstance(r.get("languages"), dict) else {}
            for lang_name, byte_count in langs.items():
                norm = str(lang_name).lower()
                skill_key = LANGUAGE_MAP.get(norm, norm)
                all_skills_bytes[skill_key] = all_skills_bytes.get(skill_key, 0) + int(byte_count or 0)

            # Also ensure skills from explicitly modified user files are represented
            user_files = r.get("user_files", [])
            for f in user_files:
                f_lower = f.lower()
                for skill_name, exts in SKILL_EXTENSIONS.items():
                    if any(f_lower.endswith(e) for e in exts):
                        all_skills_bytes[skill_name] = all_skills_bytes.get(skill_name, 0) + 1000

        # 4. Aggregate metrics per skill with strict attribution
        skill_matrix = {}
        flagship_projects = []

        # Sort skills by total code volume
        sorted_skills = sorted(all_skills_bytes.items(), key=lambda x: x[1], reverse=True)

        for skill, total_bytes in sorted_skills:
            if total_bytes < 300:
                continue

            exts = set(SKILL_EXTENSIONS.get(skill.lower(), []))
            total_commits = 0
            total_skill_files = 0
            total_additions = 0
            total_deletions = 0
            total_verified = 0
            contributing_repos = []
            max_days = 0

            for r in repo_evidence_list:
                user_commits = int(r.get("user_commits_count") or 0)
                if user_commits == 0:
                    continue

                repo_langs = {str(k).lower(): v for k, v in (r.get("languages") or {}).items()}
                skill_bytes_in_repo = 0
                for lang_k, v in repo_langs.items():
                    if LANGUAGE_MAP.get(lang_k, lang_k) == skill:
                        skill_bytes_in_repo += int(v or 0)

                # Check if user touched files for this skill
                user_files = r.get("user_files", []) or []
                matched_user_files = [
                    f for f in user_files
                    if any(f.lower().endswith(e) for e in exts)
                ]

                # Determine if user contributed to this skill in this repo
                has_user_skill_files = len(matched_user_files) > 0
                has_repo_language = skill_bytes_in_repo > 0

                # Accurate attribution logic:
                # If we have explicit user files for this repo, use exact matched files.
                # If user_files is empty (e.g. shallow scan), only attribute if repo language matches.
                if has_user_skill_files or (not user_files and has_repo_language):
                    skill_files_count = len(matched_user_files) if has_user_skill_files else max(1, skill_bytes_in_repo // 2000)
                    total_commits += user_commits
                    total_skill_files += skill_files_count
                    total_additions += int(r.get("user_additions") or 0)
                    total_deletions += int(r.get("user_deletions") or 0)
                    total_verified += int(r.get("user_verified_commits") or 0)
                    max_days = max(max_days, int(r.get("user_days_active") or 0))

                    contributing_repos.append({
                        "name": r.get("name") or "repo",
                        "full_name": r.get("full_name") or "repo",
                        "url": r.get("html_url") or "",
                        "is_fork": bool(r.get("is_fork", False)),
                        "commits": user_commits,
                        "skill_bytes": skill_bytes_in_repo,
                        "skill_files_count": skill_files_count,
                        "additions": int(r.get("user_additions") or 0),
                        "stars": int(r.get("stars") or 0),
                    })

            if contributing_repos:
                contributing_repos.sort(key=lambda x: (x["commits"], x["skill_bytes"]), reverse=True)

                composite_evidence = {
                    "skill": skill,
                    "repository": {
                        "name": f"{clean_user}'s GitHub Portfolio",
                        "owner": clean_user,
                        "url": f"https://github.com/{clean_user}",
                        "is_fork": False,
                        "parent": None,
                    },
                    "contribution": {
                        "commits": total_commits,
                        "files_changed": total_skill_files,
                        "skill_files": total_skill_files,
                        "additions": total_additions,
                        "deletions": total_deletions,
                        "verified_commits": total_verified,
                        "contribution_days": max_days,
                        "pull_requests": 0,
                        "merged_pull_requests": 0,
                    },
                    "commits": [],
                    "pull_request_details": [],
                    "skill_files": [],
                }

                score_res = calculate_score(composite_evidence)

                skill_matrix[skill] = {
                    "skill": skill,
                    "score": score_res["evidence_score"],
                    "confidence": score_res["confidence"],
                    "total_bytes": total_bytes,
                    "reasons": score_res["reasons"],
                    "warnings": score_res["warnings"],
                    "stats": {
                        "total_commits": total_commits,
                        "skill_files": total_skill_files,
                        "total_additions": total_additions,
                        "repos_count": len(contributing_repos),
                        "verified_commits": total_verified,
                    },
                    "top_repositories": contributing_repos[:3],
                }

        # 5. Build Flagship Projects List
        for r in repo_evidence_list:
            if (int(r.get("user_commits_count") or 0) > 0) or (int(r.get("stars") or 0) > 0) or r.get("languages"):
                primary_lang = r.get("language")
                langs = r.get("languages") or {}
                if not primary_lang and langs:
                    primary_lang = max(langs.items(), key=lambda x: x[1])[0]

                flagship_projects.append({
                    "name": r.get("name") or "repo",
                    "full_name": r.get("full_name") or "repo",
                    "url": r.get("html_url") or "",
                    "description": r.get("description") or "No description provided",
                    "language": primary_lang or "Mixed",
                    "languages_breakdown": langs,
                    "is_fork": bool(r.get("is_fork", False)),
                    "user_commits": int(r.get("user_commits_count") or 0),
                    "additions": int(r.get("user_additions") or 0),
                    "stars": int(r.get("stars") or 0),
                })

        flagship_projects.sort(
            key=lambda x: (not x["is_fork"], x["user_commits"], x["stars"]),
            reverse=True,
        )

        # 6. Retrieve relevant lessons from episodic memory
        past_lessons = []
        if self.memory:
            try:
                for s in list(skill_matrix.keys())[:5]:
                    lessons = self.memory.retrieve_relevant_lessons(s, is_fork=False)
                    for l in lessons:
                        if l not in past_lessons:
                            past_lessons.append(l)
            except Exception as exc:
                logger.warning("Failed retrieving lessons for portfolio: %s", exc)

        # 7. Gemini Developer Portfolio Intelligence Synthesis
        portfolio_summary = None
        if self.llm and self.llm.available and skill_matrix:
            try:
                portfolio_summary = self.llm.synthesize_portfolio(
                    username=clean_user,
                    total_repos=len(all_repos),
                    scanned_repos_count=len(repo_evidence_list),
                    skill_matrix=skill_matrix,
                    flagship_projects=flagship_projects[:6],
                    lessons=past_lessons[:5],
                )
            except Exception as exc:
                logger.warning("Failed synthesizing portfolio LLM: %s", exc)

        duration = round(time.time() - t0, 2)
        logger.info("Portfolio scan completed in %ss. Evaluated %d skills.", duration, len(skill_matrix))

        return {
            "username": clean_user,
            "total_repos_discovered": len(all_repos),
            "repos_scanned": len(repo_evidence_list),
            "scan_duration_seconds": duration,
            "skills": skill_matrix,
            "flagship_projects": flagship_projects[:6],
            "ai_synthesis": portfolio_summary,
            "lessons_applied": past_lessons,
        }

    def _scan_single_repo(self, repo: dict, username: str) -> Optional[dict]:
        """Scan a single repository to fetch languages, user commits, changed files, and metadata."""
        if not isinstance(repo, dict):
            return None

        owner_obj = repo.get("owner")
        owner = owner_obj.get("login") if isinstance(owner_obj, dict) else username
        repo_name = repo.get("name")
        if not owner or not repo_name:
            return None

        try:
            # 1. Fetch exact language breakdown
            languages = {}
            try:
                raw_langs = self.agent.get(f"/repos/{owner}/{repo_name}/languages")
                if isinstance(raw_langs, dict):
                    languages = raw_langs
            except Exception:
                languages = {}

            # 2. Fetch commits
            commits = []
            try:
                commits = self.agent.get_commits(owner, repo_name, author=username)
            except Exception:
                commits = []

            # If author filter returned nothing and user is owner of the repo, get recent commits
            if not commits and owner.lower() == username.lower():
                try:
                    all_c = self.agent.get(f"/repos/{owner}/{repo_name}/commits", params={"per_page": 20})
                    if isinstance(all_c, list):
                        commits = all_c
                except Exception:
                    commits = []

            # 3. Extract stats and touched files
            additions = 0
            deletions = 0
            verified = 0
            dates = []
            user_files = set()

            for c in commits[:8]:
                if not isinstance(c, dict):
                    continue
                commit_info = c.get("commit") or {}
                author_obj = commit_info.get("author") or {}
                author_date = author_obj.get("date")
                if author_date:
                    dates.append(author_date)
                verification = commit_info.get("verification") or {}
                if verification.get("verified"):
                    verified += 1

            # Fetch detailed stats & files on up to 3 commits
            for c in commits[:3]:
                if not isinstance(c, dict):
                    continue
                sha = c.get("sha")
                if sha:
                    try:
                        detail = self.agent.get(f"/repos/{owner}/{repo_name}/commits/{sha}")
                        if isinstance(detail, dict):
                            stats = detail.get("stats") or {}
                            additions += int(stats.get("additions") or 0)
                            deletions += int(stats.get("deletions") or 0)
                            for f in (detail.get("files") or []):
                                if isinstance(f, dict) and f.get("filename"):
                                    user_files.add(f.get("filename"))
                    except Exception:
                        pass

            days_active = 0
            if len(dates) >= 2:
                try:
                    from datetime import datetime
                    d_parsed = [datetime.fromisoformat(d.replace("Z", "+00:00")) for d in dates]
                    days_active = max(0, (max(d_parsed) - min(d_parsed)).days)
                except Exception:
                    days_active = 0

            return {
                "name": repo_name,
                "full_name": repo.get("full_name") or f"{owner}/{repo_name}",
                "html_url": repo.get("html_url") or f"https://github.com/{owner}/{repo_name}",
                "is_fork": bool(repo.get("fork", False)),
                "language": repo.get("language"),
                "languages": languages,
                "description": repo.get("description"),
                "stars": int(repo.get("stargazers_count") or 0),
                "user_commits_count": len(commits),
                "user_files": sorted(list(user_files)),
                "user_additions": additions,
                "user_deletions": deletions,
                "user_verified_commits": verified,
                "user_days_active": days_active,
            }
        except Exception as exc:
            logger.debug("Error scanning repo %s: %s", repo.get("full_name"), exc)
            return None


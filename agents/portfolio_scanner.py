"""
Portfolio Scanner Agent for GitProof.

Scans all accessible repositories for a developer, queries language distributions,
aggregates commits, files, and pull requests per skill, and produces an
overall cross-repository skill intelligence matrix.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
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

        # 1. Fetch all repositories for target user
        try:
            all_repos = self.agent.list_user_repos(clean_user)
        except Exception as exc:
            logger.warning("Failed listing repositories for %s: %s", clean_user, exc)
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

        # 4. Aggregate metrics per skill with strict attribution & deduplication
        skill_matrix = {}
        flagship_projects = []

        # Sort canonical skills by total code volume
        sorted_skills = sorted(all_skills_bytes.items(), key=lambda x: x[1], reverse=True)
        seen_canonical = set()

        for skill, total_bytes in sorted_skills:
            # Map to canonical deduplicated name
            canon_skill = LANGUAGE_MAP.get(skill.lower(), skill.lower())
            if canon_skill in seen_canonical or total_bytes < 300:
                continue
            seen_canonical.add(canon_skill)

            exts = set(SKILL_EXTENSIONS.get(canon_skill, []))
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
                    if LANGUAGE_MAP.get(lang_k, lang_k) == canon_skill:
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

                # Calibrated Forensic Evidence Scoring Model
                # Anchored strictly on verified commit history, volume, and multi-repo consistency
                # Low commit counts (< 15) cannot receive high scores (e.g. 22 commits -> ~5.4/10)
                if total_commits <= 0:
                    raw_scaled_score = 0
                    score_10 = 0.0
                    conf = "Low"
                else:
                    # 1. Commit volume base anchor (piecewise calibrated curve)
                    if total_commits <= 3:
                        c_base = 1.0 + (total_commits * 0.5)            # 1.5 - 2.5
                    elif total_commits <= 10:
                        c_base = 2.5 + ((total_commits - 3) * 0.25)     # 2.75 - 4.25
                    elif total_commits <= 30:
                        c_base = 4.25 + ((total_commits - 10) * 0.075)  # 4.32 - 5.75 (e.g. 22 commits -> 5.15)
                    elif total_commits <= 80:
                        c_base = 5.75 + ((total_commits - 30) * 0.035)  # 5.78 - 7.50 (e.g. 45 commits -> 6.27)
                    elif total_commits <= 150:
                        c_base = 7.50 + ((total_commits - 80) * 0.018)  # 7.52 - 8.76 (e.g. 100 commits -> 7.86)
                    else:
                        c_base = 8.76 + min(0.95, (total_commits - 150) * 0.009) # 8.77 - 9.71 (e.g. 230 commits -> 9.48)

                    # 2. Code Volume Modifier (capped per commit to guard against bundled libraries/node_modules)
                    effective_loc_per_commit = min(total_additions / max(1, total_commits), 500.0)
                    loc_mod = (math.log10(max(10.0, effective_loc_per_commit)) - 1.0) * 0.25 # -0.25 to +0.42

                    # 3. Multi-Repo Breadth Bonus
                    repo_bonus = min(0.5, max(0.0, (len(contributing_repos) - 1) * 0.15))

                    raw_score_10 = c_base + loc_mod + repo_bonus

                    # Strict Forensic Ceiling Guards based on verified commit count
                    if total_commits < 5:
                        raw_score_10 = min(raw_score_10, 3.5)
                    elif total_commits < 15:
                        raw_score_10 = min(raw_score_10, 4.8)
                    elif total_commits < 30:
                        raw_score_10 = min(raw_score_10, 5.9) # 22 commits strictly capped <= 5.9
                    elif total_commits < 60:
                        raw_score_10 = min(raw_score_10, 7.2)
                    elif total_commits < 120:
                        raw_score_10 = min(raw_score_10, 8.5)

                    score_10 = round(max(0.5, min(9.8, raw_score_10)), 1)
                    raw_scaled_score = int(round(score_10 * 10.0))

                    if total_commits >= 50 and len(contributing_repos) >= 2:
                        conf = "High"
                    elif total_commits >= 15:
                        conf = "Medium"
                    else:
                        conf = "Low"

                skill_matrix[canon_skill] = {
                    "skill": canon_skill,
                    "score": raw_scaled_score,
                    "score_out_of_10": score_10,
                    "confidence": conf,
                    "total_bytes": total_bytes,
                    "reasons": [
                        f"Commit Depth: {total_commits} verified commits across {len(contributing_repos)} repo(s)",
                        f"Code Velocity: {total_additions:,} LOC added in {total_skill_files} skill file(s)",
                    ],
                    "warnings": [],
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

        # Compute aggregate metrics
        total_commits = sum(s.get("stats", {}).get("total_commits", 0) for s in skill_matrix.values())
        total_files = sum(s.get("stats", {}).get("skill_files", 0) for s in skill_matrix.values())

        if not skill_matrix or total_commits == 0:
            composite_score = 0
            overall_confidence = "low"
            archetype = "No Public Commit Activity"
        else:
            # Weighted average based on commit count and score
            total_weight = sum(max(1, s.get("stats", {}).get("total_commits", 1)) for s in skill_matrix.values())
            weighted_sum = sum(s.get("score", 0) * max(1, s.get("stats", {}).get("total_commits", 1)) for s in skill_matrix.values())
            composite_score = int(round(weighted_sum / total_weight)) if total_weight > 0 else 0
            
            # Confidence
            if total_commits >= 20 and len(skill_matrix) >= 2:
                overall_confidence = "high"
            elif total_commits >= 5:
                overall_confidence = "medium"
            else:
                overall_confidence = "low"

            # Derive Archetype
            top_skills = sorted(skill_matrix.keys(), key=lambda k: skill_matrix[k].get("score", 0), reverse=True)
            if len(top_skills) >= 3:
                archetype = f"Polyglot ({'/'.join(top_skills[:2]).upper()})"
            elif len(top_skills) == 1:
                archetype = f"{top_skills[0].capitalize()} Specialist"
            elif len(top_skills) == 2:
                archetype = f"{top_skills[0].capitalize()} & {top_skills[1].capitalize()} Engineer"
            else:
                archetype = "General Developer"

        if portfolio_summary and isinstance(portfolio_summary, dict) and portfolio_summary.get("archetype"):
            archetype = portfolio_summary["archetype"]

        duration = round(time.time() - t0, 2)
        logger.info("Portfolio scan completed in %ss. Score: %d (%s)", duration, composite_score, archetype)

        return {
            "username": clean_user,
            "total_repos_discovered": len(all_repos),
            "total_repos_scanned": len(repo_evidence_list),
            "repos_scanned": len(repo_evidence_list),
            "total_commits_indexed": total_commits,
            "portfolio_composite_score": composite_score,
            "confidence": overall_confidence,
            "developer_archetype": archetype,
            "scan_duration_seconds": duration,
            "skills": skill_matrix,
            "skills_detected": skill_matrix,
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


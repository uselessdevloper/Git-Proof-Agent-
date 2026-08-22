import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import requests

GITHUB_API = "https://api.github.com"


SKILL_EXTENSIONS: Dict[str, List[str]] = {
    "python": [".py", ".ipynb", ".pyi"],
    "jupyter": [".ipynb"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx", ".mts", ".cts"],
    "html": [".html", ".htm"],
    "css": [".css", ".scss", ".sass", ".less"],
    "java": [".java"],
    "cpp": [".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"],
    "c++": [".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"],
    "c": [".c", ".h"],
    "csharp": [".cs"],
    "c#": [".cs"],
    "go": [".go"],
    "rust": [".rs"],
    "php": [".php"],
    "ruby": [".rb", ".erb"],
    "swift": [".swift"],
    "kotlin": [".kt", ".kts"],
    "solidity": [".sol"],
    "sql": [".sql"],
    "shell": [".sh", ".bash", ".zsh"],
    "bash": [".sh", ".bash", ".zsh"],
    "docker": ["dockerfile", ".dockerfile", "docker-compose.yml", "docker-compose.yaml"],
    "dockerfile": ["dockerfile", ".dockerfile"],
    "vue": [".vue"],
    "svelte": [".svelte"],
}


class GitProofAgent:
    """
    Wraps the GitHub REST API with defensive error handling, timeout guarantees,
    null-safety, and multi-extension awareness.
    """

    def __init__(self, token: str, timeout: int = 15):
        if not token or not isinstance(token, str) or not token.strip():
            raise ValueError("No valid GitHub access token provided (user is not connected)")

        self.token = token.strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "GitProof-Agent/2.5",
        })

    def get(self, endpoint: str, params: Optional[dict] = None) -> Any:
        url = f"{GITHUB_API}{endpoint}" if not endpoint.startswith("http") else endpoint
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise TimeoutError(f"GitHub API timed out requesting {endpoint}")
        except requests.exceptions.RequestException as exc:
            raise ConnectionError(f"GitHub API connection error on {endpoint}: {exc}")

        if response.status_code == 404:
            raise FileNotFoundError(f"GitHub resource not found: {endpoint}")

        if response.status_code == 401:
            raise PermissionError("GitHub token is invalid or expired — please reconnect your account")

        if response.status_code == 403:
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining == "0":
                raise PermissionError("GitHub API rate limit exceeded — try again shortly")
            raise PermissionError("GitHub API permission denied (missing scope for this repo?)")

        if response.status_code == 409:
            # Git Repository is empty
            return []

        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return response.text

    def get_authenticated_user(self) -> dict:
        data = self.get("/user")
        return data if isinstance(data, dict) else {}

    def list_my_repos(self, per_page: int = 100) -> List[dict]:
        repos = []
        page = 1
        while True:
            try:
                data = self.get("/user/repos", params={"per_page": min(100, max(1, per_page)), "page": page, "sort": "updated"})
                if not data or not isinstance(data, list):
                    break
                repos.extend(data)
                if len(data) < per_page or page >= 5:  # safety limit: up to 500 repos
                    break
                page += 1
            except Exception:
                break
        return repos

    def list_user_repos(self, username: Optional[str] = None, per_page: int = 100) -> List[dict]:
        if not username:
            return self.list_my_repos(per_page=per_page)
        repos = []
        page = 1
        while True:
            try:
                data = self.get(f"/users/{username}/repos", params={"per_page": min(100, max(1, per_page)), "page": page, "sort": "updated"})
                if not data or not isinstance(data, list):
                    break
                repos.extend(data)
                if len(data) < per_page or page >= 5:
                    break
                page += 1
            except Exception:
                break
        if not repos:
            try:
                return self.list_my_repos(per_page=per_page)
            except Exception:
                return []
        return repos

    def get_repository(self, owner: str, repo: str) -> dict:
        data = self.get(f"/repos/{owner}/{repo}")
        return data if isinstance(data, dict) else {}

    def get_contributors(self, owner: str, repo: str, per_page: int = 30) -> List[dict]:
        """Fetch list of collaborators / contributors for a repository."""
        try:
            data = self.get(f"/repos/{owner}/{repo}/contributors", params={"per_page": min(100, max(1, per_page))})
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def get_commits(self, owner: str, repo: str, author: Optional[str] = None, per_page: int = 100) -> List[dict]:
        commits = []
        page = 1
        params = {"per_page": min(100, max(1, per_page)), "page": page}
        if author:
            params["author"] = author

        while True:
            try:
                data = self.get(f"/repos/{owner}/{repo}/commits", params=params)
                if not data or not isinstance(data, list):
                    break
                commits.extend(data)
                if len(data) < per_page or page >= 10:  # safety limit
                    break
                page += 1
                params["page"] = page
            except Exception:
                break
        return commits

    def get_commit(self, owner: str, repo: str, sha: str) -> dict:
        try:
            data = self.get(f"/repos/{owner}/{repo}/commits/{sha}")
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def get_pull_requests(self, owner: str, repo: str, username: str) -> Tuple[List[dict], int]:
        if not username or not username.strip():
            return [], 0

        clean_user = username.strip()
        try:
            data = self.get(
                "/search/issues",
                params={
                    "q": f"repo:{owner}/{repo} type:pr author:{clean_user}",
                    "per_page": 100,
                },
            )
        except Exception:
            return [], 0

        pr_details = []
        merged_count = 0

        items = data.get("items", []) if isinstance(data, dict) else []
        for item in items[:25]:  # limit PR detail lookups to avoid rate limiting
            pr_number = item.get("number")
            if not pr_number:
                continue

            try:
                pr = self.get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
                if not isinstance(pr, dict):
                    continue

                merged = bool(pr.get("merged")) or pr.get("merged_at") is not None
                if merged:
                    merged_count += 1

                pr_details.append({
                    "number": pr_number,
                    "title": pr.get("title") or "Untitled PR",
                    "state": pr.get("state"),
                    "merged": merged,
                    "additions": pr.get("additions") or 0,
                    "deletions": pr.get("deletions") or 0,
                    "changed_files": pr.get("changed_files") or 0,
                    "review_comments": pr.get("review_comments") or 0,
                    "url": pr.get("html_url") or "",
                })
            except Exception:
                continue

        return pr_details, merged_count

    def analyze_contribution(self, owner: str, repo: str, username: str, claimed_skill: str) -> dict:
        if not owner or not repo or not username:
            raise ValueError("owner, repo, and username must be non-empty strings")

        clean_skill = (claimed_skill or "").strip().lower()
        target_extensions = SKILL_EXTENSIONS.get(clean_skill, [])

        repository = self.get_repository(owner, repo)
        commits = self.get_commits(owner, repo, author=username)

        total_additions = 0
        total_deletions = 0
        total_files = set()
        skill_files = set()
        verified_commits = 0
        dates = []
        detailed_commits = []

        for commit_summary in commits:
            if not isinstance(commit_summary, dict):
                continue
            sha = commit_summary.get("sha")
            if not sha:
                continue

            commit = self.get_commit(owner, repo, sha)
            if not isinstance(commit, dict):
                continue

            stats = commit.get("stats") or {}
            additions = stats.get("additions") or 0
            deletions = stats.get("deletions") or 0
            total_additions += additions
            total_deletions += deletions

            c_info = commit.get("commit") or {}
            c_author = c_info.get("author") or {}
            commit_date = c_author.get("date")
            if commit_date:
                dates.append(commit_date)

            verification = c_info.get("verification") or {}
            is_verified = bool(verification.get("verified"))
            if is_verified:
                verified_commits += 1

            files_list = commit.get("files") or []
            if isinstance(files_list, list):
                for file_item in files_list:
                    if isinstance(file_item, dict):
                        filename = file_item.get("filename") or ""
                        if filename:
                            total_files.add(filename)
                            lower_name = filename.lower()
                            for extension in target_extensions:
                                if lower_name.endswith(extension) or (extension.startswith("dockerfile") and "dockerfile" in lower_name):
                                    skill_files.add(filename)

            detailed_commits.append({
                "sha": sha,
                "message": c_info.get("message") or "",
                "date": commit_date,
                "verified": is_verified,
                "additions": additions,
                "deletions": deletions,
            })

        # Contribution duration calculation
        contribution_days = 0
        if len(dates) >= 2:
            parsed_dates = []
            for d in dates:
                try:
                    cleaned_date = d.replace("Z", "+00:00")
                    parsed_dates.append(datetime.fromisoformat(cleaned_date))
                except Exception:
                    pass
            if len(parsed_dates) >= 2:
                contribution_days = max(0, (max(parsed_dates) - min(parsed_dates)).days)

        # Fork analysis
        is_fork = bool(repository.get("fork", False))
        parent_repo = None
        parent_obj = repository.get("parent")
        if is_fork and isinstance(parent_obj, dict):
            parent_repo = {
                "full_name": parent_obj.get("full_name", ""),
                "html_url": parent_obj.get("html_url", ""),
            }

        # Pull request evidence
        try:
            pr_details, merged_prs = self.get_pull_requests(owner, repo, username)
        except Exception:
            pr_details, merged_prs = [], 0

        repo_owner = repository.get("owner")
        owner_login = repo_owner.get("login", owner) if isinstance(repo_owner, dict) else owner

        return {
            "skill": claimed_skill,
            "repository": {
                "name": repository.get("full_name") or f"{owner}/{repo}",
                "owner": owner_login,
                "url": repository.get("html_url") or f"https://github.com/{owner}/{repo}",
                "is_fork": is_fork,
                "parent": parent_repo,
            },
            "contribution": {
                "commits": len(commits),
                "files_changed": len(total_files),
                "skill_files": len(skill_files),
                "additions": total_additions,
                "deletions": total_deletions,
                "verified_commits": verified_commits,
                "contribution_days": contribution_days,
                "pull_requests": len(pr_details),
                "merged_pull_requests": merged_prs,
            },
            "skill_files": sorted(list(skill_files)),
            "commits": detailed_commits,
            "pull_request_details": pr_details,
        }
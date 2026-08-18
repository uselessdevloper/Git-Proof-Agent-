import os
import json
import pytest
from unittest.mock import patch, MagicMock
from cli import main


class TestCLI:
    """Automated tests for GitProof CLI commands."""

    def test_cli_status(self, capsys):
        with patch("sys.argv", ["cli.py", "status"]):
            main()
        captured = capsys.readouterr()
        assert "GITPROOF SYSTEM & MEMORY STATUS" in captured.out
        assert "SQLite Database Path" in captured.out

    def test_cli_lessons(self, capsys):
        with patch("sys.argv", ["cli.py", "lessons"]):
            main()
        captured = capsys.readouterr()
        assert "EPISODIC LESSONS IN MEMORY" in captured.out

    def test_cli_lessons_json(self, capsys):
        with patch("sys.argv", ["cli.py", "lessons", "--json"]):
            main()
        captured = capsys.readouterr()
        # Find JSON array in stdout
        start = captured.out.find("[")
        assert start != -1
        parsed = json.loads(captured.out[start:])
        assert isinstance(parsed, list)

    def test_cli_analyze_command(self, capsys):
        mock_result = {
            "analysis_id": "test-uuid-123",
            "from_cache": False,
            "evidence": {
                "skill": "python",
                "repository": {
                    "name": "octocat/hello-world",
                    "owner": "octocat",
                    "url": "https://github.com/octocat/hello-world",
                    "is_fork": False,
                },
                "contribution": {
                    "commits": 15,
                    "files_changed": 10,
                    "skill_files": 8,
                    "additions": 1200,
                    "deletions": 100,
                    "verified_commits": 12,
                    "contribution_days": 30,
                    "pull_requests": 2,
                    "merged_pull_requests": 2,
                },
                "skill_files": ["app.py", "utils.py"],
            },
            "assessment": {
                "evidence_score": 85,
                "confidence": "High",
                "score_breakdown": {
                    "commits_score": 25,
                    "volume_score": 25,
                    "pr_score": 15,
                    "duration_score": 10,
                    "verification_bonus": 10,
                },
                "penalties": [],
                "llm_insight": "Consistent, high-impact contributions in Python.",
            },
            "applied_lessons": [],
        }

        with patch("agents.orchestrator.OrchestratorAgent.run", return_value=mock_result):
            with patch("sys.argv", [
                "cli.py", "analyze", "octocat/hello-world", "octocat", "python", "--token", "fake-token"
            ]):
                main()

        captured = capsys.readouterr()
        assert "VERIFICATION RESULTS FOR PYTHON" in captured.out
        assert "85.0/100" in captured.out
        assert "High" in captured.out
        assert "Consistent, high-impact contributions" in captured.out

    def test_cli_scan_command(self, capsys):
        mock_report = {
            "username": "octocat",
            "total_repos_scanned": 5,
            "total_commits_found": 120,
            "total_prs_found": 8,
            "scan_duration_seconds": 1.5,
            "skill_matrix": {
                "python": {
                    "evidence_score": 88.0,
                    "tier": "Senior / Lead Contributor",
                    "total_commits": 100,
                    "total_skill_files": 35,
                    "repos": ["repo-a", "repo-b"],
                }
            },
            "highlight_projects": [
                {
                    "name": "octocat/repo-a",
                    "stars": 42,
                    "forks": 10,
                    "language": "Python",
                    "description": "Awesome python project",
                }
            ],
            "portfolio_summary": "Seasoned Python developer with solid contribution velocity.",
        }

        with patch("agents.portfolio_scanner.PortfolioScannerAgent.scan_portfolio", return_value=mock_report):
            with patch("sys.argv", [
                "cli.py", "scan", "octocat", "--limit", "5", "--token", "fake-token"
            ]):
                main()

        captured = capsys.readouterr()
        assert "PORTFOLIO SKILL INTELLIGENCE MATRIX: octocat" in captured.out
        assert "Senior / Lead Contributor" in captured.out
        assert "Awesome python project" in captured.out

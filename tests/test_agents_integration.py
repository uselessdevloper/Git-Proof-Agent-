import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app, _memory, _llm
from memory.memory_manager import MemoryManager
from llm.llm_client import LLMClient
from agents.orchestrator import OrchestratorAgent
from agents.portfolio_scanner import PortfolioScannerAgent


@pytest.fixture
def client():
    return TestClient(app)


# 1. ORCHESTRATOR & LEARNING LOOP E2E TESTS
class TestOrchestratorIntegration:
    """Tests the full orchestrator pipeline and feedback learning loop end-to-end."""

    def test_full_pipeline_with_feedback_learning_loop(self, tmp_path):
        db_path = str(tmp_path / "e2e_memory.db")
        memory = MemoryManager(db_path=db_path)
        llm = LLMClient()
        llm.available = True

        agent = OrchestratorAgent(
            github_token="fake-oauth-token",
            memory=memory,
            llm=llm,
        )

        mock_evidence = {
            "skill": "python",
            "repository": {
                "name": "octocat/hello-world",
                "owner": "octocat",
                "url": "https://github.com/octocat/hello-world",
                "is_fork": True,
                "parent": {"full_name": "upstream/hello-world", "html_url": "https://github.com/upstream/hello-world"},
            },
            "contribution": {
                "commits": 20,
                "files_changed": 15,
                "skill_files": 12,
                "additions": 3000,
                "deletions": 200,
                "verified_commits": 10,
                "contribution_days": 45,
                "pull_requests": 2,
                "merged_pull_requests": 1,
            },
            "skill_files": ["app.py", "main.py"],
            "commits": [{"sha": "123", "message": "Initial commit", "verified": True}],
            "pull_request_details": [{"number": 1, "title": "Feature PR", "merged": True}],
        }

        # Mock GitHub fetch and LLM reasoning
        with patch.object(agent.github, "analyze_contribution", return_value=mock_evidence):
            with patch.object(llm, "_generate", return_value="High velocity contributor on fork with verified commits."):
                # 1. First run (cache miss)
                res1 = agent.run(
                    owner="octocat",
                    repo="hello-world",
                    username="octocat",
                    claimed_skill="python",
                    use_cache=True,
                )

                assert res1["from_cache"] is False
                assert res1["analysis_id"] is not None
                assert res1["assessment"]["evidence_score"] > 0
                assert res1["assessment"]["confidence"] in ["Medium-High", "High"]
                assert res1["assessment"]["llm_insight"] is not None
                analysis_id = res1["analysis_id"]

                # 2. Second run (cache hit)
                res2 = agent.run(
                    owner="octocat",
                    repo="hello-world",
                    username="octocat",
                    claimed_skill="python",
                    use_cache=True,
                )
                assert res2["from_cache"] is True
                assert res2["analysis_id"] == analysis_id

                # 3. Submit feedback to trigger lesson extraction
                lesson_json_response = (
                    '{"lesson": "When repository is a fork, verify PRs are merged upstream.", "tags": ["fork", "python", "prs"]}'
                )
                with patch.object(llm, "_generate", return_value=lesson_json_response):
                    feedback_res = agent.process_feedback(
                        analysis_id=analysis_id,
                        feedback_type="too_high",
                        feedback_text="Most commits were inherited before the fork was created.",
                        correct_score=45,
                    )
                    assert feedback_res["status"] == "feedback_recorded"
                    assert feedback_res["lesson_generated"] is not None
                    assert feedback_res["lesson_id"] is not None

                # 4. Third run with cache disabled - should retrieve and apply the new lesson!
                res3 = agent.run(
                    owner="octocat",
                    repo="hello-world",
                    username="octocat",
                    claimed_skill="python",
                    use_cache=False,
                )
                assert res3["from_cache"] is False
                lessons_applied = res3["assessment"]["lessons_applied"]
                assert len(lessons_applied) > 0
                assert "fork" in lessons_applied[0].lower() or "upstream" in lessons_applied[0].lower()


# ==============================================================================
# 2. FASTAPI ENDPOINT INTEGRATION & SECURITY TESTS
# ==============================================================================

class TestFastAPIEndpoints:
    """Tests all API endpoints for authentication guards and input sanitization."""

    def test_unauthenticated_requests_fail(self, client):
        # Must return 401 when no session exists
        assert client.get("/api/me").status_code == 401
        assert client.get("/api/repos").status_code == 401
        assert client.post("/api/analyze", json={"owner": "a", "repo": "b", "claimed_skill": "python"}).status_code == 401
        assert client.post("/api/portfolio/scan", json={}).status_code == 401
        assert client.post("/api/feedback", json={"analysis_id": "1", "feedback_type": "correct"}).status_code == 401
        assert client.get("/api/lessons").status_code == 401
        assert client.get("/api/status").status_code == 401

    def test_dev_login_and_authenticated_status(self, client):
        with patch("auth.get_authenticated_user", return_value={"login": "testuser", "name": "Test User", "avatar_url": "https://avatar.url"}):
            login_resp = client.post("/api/dev-login?token=gho_1234567890abcdef")
            assert login_resp.status_code == 200
            assert login_resp.json()["login"] == "testuser"

            # Now session is authenticated
            me_resp = client.get("/api/me")
            assert me_resp.status_code == 200
            assert me_resp.json()["login"] == "testuser"

            status_resp = client.get("/api/status")
            assert status_resp.status_code == 200
            assert "memory" in status_resp.json()

            # Logout
            logout_resp = client.post("/auth/logout")
            assert logout_resp.status_code == 200
            assert client.get("/api/me").status_code == 401

    def test_analyze_validation_errors(self, client):
        with patch("auth.get_authenticated_user", return_value={"login": "testuser", "name": "Test User", "avatar_url": ""}):
            client.post("/api/dev-login?token=gho_1234567890abcdef")

            # Missing or empty fields
            resp = client.post("/api/analyze", json={"owner": "", "repo": "r", "claimed_skill": "python"})
            assert resp.status_code == 422 or resp.status_code == 400

            resp2 = client.post("/api/analyze", json={"owner": "o", "repo": "", "claimed_skill": "python"})
            assert resp2.status_code == 422 or resp2.status_code == 400

            resp3 = client.post("/api/analyze", json={"owner": "o", "repo": "r", "claimed_skill": ""})
            assert resp3.status_code == 422 or resp3.status_code == 400

    def test_feedback_validation_errors(self, client):
        with patch("auth.get_authenticated_user", return_value={"login": "testuser", "name": "Test User", "avatar_url": ""}):
            client.post("/api/dev-login?token=gho_1234567890abcdef")

            # Invalid feedback type
            resp = client.post("/api/feedback", json={
                "analysis_id": "test-id",
                "feedback_type": "invalid_type",
            })
            assert resp.status_code == 400

            # Out of bounds score
            resp2 = client.post("/api/feedback", json={
                "analysis_id": "test-id",
                "feedback_type": "too_high",
                "correct_score": 150,
            })
            assert resp2.status_code == 422 or resp2.status_code == 400

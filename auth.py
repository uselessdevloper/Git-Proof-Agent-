"""
GitHub OAuth helper functions.

This implements the standard "Authorization Code" OAuth flow:

1. We redirect the user to GitHub's authorize URL.
2. GitHub redirects back to our /auth/callback with a `code`.
3. We exchange that `code` (+ client secret) for an access_token.
4. We use that access_token to call the GitHub API *as the user*.

You need a GitHub OAuth App to get a client id/secret:
https://github.com/settings/developers -> "New OAuth App"

  Homepage URL:               http://localhost:8000
  Authorization callback URL: http://localhost:8000/auth/callback

IMPORTANT: Env vars are read lazily (inside each function) so that
load_dotenv() in app.py runs before any value is consumed.
"""

import os
import secrets
import requests

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API = "https://api.github.com"

# Scopes needed: read profile + access repos for commit/PR data.
SCOPES = "read:user repo"


def _creds():
    """Read OAuth creds lazily — load_dotenv() must run in app.py first."""
    client_id = os.getenv("GITHUB_CLIENT_ID")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    redirect_uri = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/callback")
    return client_id, client_secret, redirect_uri


def generate_state() -> str:
    """Random per-login value to protect against CSRF on the OAuth callback."""
    return secrets.token_urlsafe(24)


def get_authorize_url(state: str) -> str:
    client_id, _, redirect_uri = _creds()
    if not client_id:
        raise RuntimeError(
            "GITHUB_CLIENT_ID is not set. Create a GitHub OAuth App and "
            "put its client id in your .env file."
        )

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "state": state,
        "allow_signup": "true",
    }
    query = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    return f"{GITHUB_AUTHORIZE_URL}?{query}"


def exchange_code_for_token(code: str) -> str:
    client_id, client_secret, redirect_uri = _creds()
    if not client_id or not client_secret:
        raise RuntimeError(
            "GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET are not set in .env"
        )

    response = requests.post(
        GITHUB_TOKEN_URL,
        headers={"Accept": "application/json"},
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()

    if "error" in payload:
        raise Exception(payload.get("error_description", payload["error"]))

    if "access_token" not in payload:
        raise Exception("GitHub did not return an access_token")

    return payload["access_token"]


def get_authenticated_user(token: str) -> dict:
    r = requests.get(
        f"{GITHUB_API}/user",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()

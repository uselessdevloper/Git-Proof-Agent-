# GitProof Agent (OAuth version)

Lets a user click **Connect GitHub**, authorizes via real GitHub OAuth (no
manual personal access token needed), then runs a deterministic evidence
scan against a repo they picked, for a claimed skill.

## 1. Create a GitHub OAuth App

Go to https://github.com/settings/developers → **New OAuth App**, and set:

| Field | Value |
|---|---|
| Homepage URL | `http://localhost:8000` |
| Authorization callback URL | `http://localhost:8000/auth/callback` |

Save it, then copy the **Client ID** and generate a **Client Secret**.

## 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
SESSION_SECRET=<random string>
```

Generate a random session secret with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## 3. Install and run

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open **http://localhost:8000** — click **Connect GitHub**, authorize the
app, pick a repo you have access to, enter a claimed skill (e.g. `python`),
and click **Run GitProof Analysis**.

## What actually happens

1. `/auth/login` redirects the browser to GitHub's OAuth consent screen.
2. GitHub redirects back to `/auth/callback` with a one-time `code`.
3. The backend exchanges that code (+ client secret) for an **access
   token**, scoped to `read:user repo` — this token belongs to the
   logged-in user, not a static token in `.env`.
4. The token is stored in a signed session cookie (see the note in
   `app.py` about swapping this for server-side session storage —
   Redis/DB — before shipping this beyond a demo).
5. `/api/repos` lists repos the token can see (owned, collaborator, or org).
6. `/api/analyze` spins up a `GitProofAgent` using *that user's token* and
   walks: repo metadata → commits by that user → per-commit stats/files/
   verification → pull requests authored by that user (via the search API)
   → a deterministic score (`scoring.py`).

## Scopes / permissions

- `read:user` — basic profile (name, avatar, login).
- `repo` — required to read commits/PRs on **private** repos the user can
  access. If you only ever want public repos, change `SCOPES` in `auth.py`
  to `"read:user public_repo"` — a narrower grant.

## Rate limits

Every commit is fetched individually (`GET /repos/{owner}/{repo}/commits/{sha}`)
to get per-file stats, so large repos burn through GitHub's API rate limit
(5,000 req/hr for OAuth tokens) quickly. The `get_commits` safety limit caps
this at 10 pages (up to 1,000 commits) — tune `page > 10` in
`github_agent.py` if needed.

## Known limitations (same ones called out in the original MVP)

- GitHub commit **author** attribution isn't cryptographic proof — it's
  metadata GitHub associates from the commit's author email. Commit
  **signature verification** (GPG/SSH signed commits) is stronger evidence
  and is tracked separately (`verified_commits`).
- This produces an **evidence score**, not a skill certification. Pair it
  with academic evidence and a practical assessment, as noted in the
  original design doc's architecture diagram.

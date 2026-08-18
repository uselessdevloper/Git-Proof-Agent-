#!/usr/bin/env python3
"""
GitProof CLI — Command-line interface for GitHub skill verification and portfolio scanning.

Connects to GitHub directly and prints rich analysis results, deterministic scores,
anti-gaming checks, and LLM reasoning directly in the terminal without needing the web frontend.

Usage:
  python cli.py analyze <owner/repo> <username> <skill> [options]
  python cli.py scan <username> [options]
  python cli.py status
  python cli.py lessons [--skill <skill>]
  python cli.py feedback --id <analysis_id> --type <type> --notes <notes>
  python cli.py interactive
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

from github_agent import GitProofAgent
from agents.orchestrator import OrchestratorAgent
from agents.portfolio_scanner import PortfolioScannerAgent
from memory.memory_manager import MemoryManager
from llm.llm_client import LLMClient
from observability.logger import get_logger

logger = get_logger("gitproof_cli")

# ANSI Terminal Colors
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def get_token(explicit_token: Optional[str] = None) -> str:
    """
    Resolve GitHub token in order of priority:
    1. Explicit CLI argument (--token / -t)
    2. GITHUB_TOKEN environment variable
    3. GITHUB_PAT environment variable
    4. gh CLI (`gh auth token`)
    5. Prompt the user interactively
    """
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()

    env_token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
    if env_token and env_token.strip():
        return env_token.strip()

    # Try gh auth token if gh CLI is installed
    try:
        gh_token = subprocess.check_output(
            ["gh", "auth", "token"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        if gh_token:
            return gh_token
    except Exception:
        pass

    # If stdin is interactive, prompt
    if sys.stdin.isatty():
        print(f"{Colors.YELLOW}No GitHub token found in env (GITHUB_TOKEN).{Colors.RESET}")
        try:
            token = input(f"{Colors.BOLD}Enter your GitHub Personal Access Token (PAT): {Colors.RESET}").strip()
            if token:
                return token
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            sys.exit(1)

    print(
        f"{Colors.RED}Error: GitHub token is required.{Colors.RESET}\n"
        "Provide one via:\n"
        "  - CLI option: --token <your_token>\n"
        "  - Environment variable: export GITHUB_TOKEN=<your_token>\n"
        "  - Or login with GitHub CLI: gh auth login",
        file=sys.stderr,
    )
    sys.exit(1)


def format_score_badge(score: float) -> str:
    if score >= 80:
        return f"{Colors.GREEN}{Colors.BOLD}{score:.1f}/100 [HIGH PROOF]{Colors.RESET}"
    elif score >= 50:
        return f"{Colors.YELLOW}{Colors.BOLD}{score:.1f}/100 [MODERATE]{Colors.RESET}"
    else:
        return f"{Colors.RED}{Colors.BOLD}{score:.1f}/100 [LOW/INSUFFICIENT]{Colors.RESET}"


def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════════════════╗
║                       GITPROOF AGENT — CLI ENGINE                        ║
║            Autonomous Evidence-Based Developer Skill Verification        ║
╚══════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def cmd_analyze(args):
    """Run full single-repo agentic analysis."""
    token = get_token(args.token)
    owner = args.owner
    repo = args.repo
    username = args.username
    skill = args.skill.lower()

    if "/" in owner and not repo:
        parts = owner.split("/", 1)
        owner, repo = parts[0], parts[1]

    if not owner or not repo or not username or not skill:
        print(f"{Colors.RED}Error: owner, repo, username, and skill are required.{Colors.RESET}")
        sys.exit(1)

    memory_path = args.memory_db or os.getenv("MEMORY_DB_PATH", "gitproof_memory.db")
    memory = MemoryManager(db_path=memory_path)
    llm = LLMClient()

    agent = OrchestratorAgent(github_token=token, memory=memory, llm=llm)

    if not args.json:
        print(f"\n{Colors.BLUE}🔍 Analyzing repository contribution...{Colors.RESET}")
        print(f"   Target: {Colors.BOLD}{owner}/{repo}{Colors.RESET}")
        print(f"   User:   {Colors.BOLD}{username}{Colors.RESET}")
        print(f"   Skill:  {Colors.BOLD}{skill}{Colors.RESET}")
        print(f"   LLM:    {'🟢 Connected' if llm.available else '⚪ Disabled / No API Key'}\n")

    try:
        result = agent.run(
            owner=owner,
            repo=repo,
            username=username,
            claimed_skill=skill,
            use_cache=not args.no_cache,
        )
    except Exception as exc:
        print(f"{Colors.RED}Analysis Failed: {exc}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    assessment = result.get("assessment", {})
    evidence = result.get("evidence", {})
    contrib = evidence.get("contribution", {})
    score = assessment.get("evidence_score", 0)
    confidence = assessment.get("confidence", "Unknown")
    breakdown = assessment.get("score_breakdown", {})
    penalties = assessment.get("penalties", [])

    print(f"{Colors.BOLD}══════════════════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f" 📊 {Colors.BOLD}VERIFICATION RESULTS FOR {skill.upper()}{Colors.RESET}")
    print(f"{Colors.BOLD}══════════════════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"  • Overall Evidence Score : {format_score_badge(score)}")
    print(f"  • Confidence Level       : {Colors.CYAN}{confidence}{Colors.RESET}")
    print(f"  • From Episodic Cache    : {'Yes (Cached Result)' if result.get('from_cache') else 'No (Live Fetched)'}")
    print(f"  • Analysis ID            : {Colors.DIM}{result.get('analysis_id')}{Colors.RESET}")

    print(f"\n{Colors.BOLD}📈 Score Breakdown:{Colors.RESET}")
    for k, v in breakdown.items():
        name = k.replace("_", " ").title()
        print(f"    - {name:<26}: {v}")

    if penalties:
        print(f"\n{Colors.YELLOW}⚠️  Anti-Gaming Penalties Applied:{Colors.RESET}")
        for p in penalties:
            print(f"    - {p.get('reason')} (Factor: {p.get('factor')})")

    print(f"\n{Colors.BOLD}📦 Live GitHub Contribution Evidence:{Colors.RESET}")
    print(f"    - Commits Found         : {contrib.get('commits', 0)} ({contrib.get('verified_commits', 0)} GPG-verified)")
    print(f"    - Additions / Deletions : +{contrib.get('additions', 0):,} / -{contrib.get('deletions', 0):,}")
    print(f"    - Total Files Touched   : {contrib.get('files_changed', 0)}")
    print(f"    - Target Skill Files    : {contrib.get('skill_files', 0)}")
    print(f"    - Pull Requests         : {contrib.get('pull_requests', 0)} ({contrib.get('merged_pull_requests', 0)} merged)")
    print(f"    - Contribution Span     : {contrib.get('contribution_days', 0)} days")
    print(f"    - Is Fork Repository    : {'Yes' if evidence.get('repository', {}).get('is_fork') else 'No'}")

    skill_files = evidence.get("skill_files", [])
    if skill_files:
        print(f"\n{Colors.BOLD}📁 Verified Skill Files ({len(skill_files)}):{Colors.RESET}")
        for f in skill_files[:10]:
            print(f"    • {f}")
        if len(skill_files) > 10:
            print(f"    • ... and {len(skill_files) - 10} more files")

    applied_lessons = result.get("applied_lessons", [])
    if applied_lessons:
        print(f"\n{Colors.CYAN}🧠 Episodic Memory Lessons Applied ({len(applied_lessons)}):{Colors.RESET}")
        for lesson in applied_lessons:
            print(f"    • [{lesson.get('skill', 'general')}] {lesson.get('lesson_text')}")

    llm_insight = assessment.get("llm_insight")
    if llm_insight:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🤖 AI Agent Qualitative Assessment:{Colors.RESET}")
        print(f"  {llm_insight}\n")

    print(f"{Colors.DIM}To submit feedback on this evaluation and refine agent memory:{Colors.RESET}")
    print(f"{Colors.DIM}  python cli.py feedback --id {result.get('analysis_id')} --type accurate --notes \"verified manually\"{Colors.RESET}\n")


def cmd_scan(args):
    """Run full multi-repository portfolio scan."""
    token = get_token(args.token)
    username = args.username
    limit = args.limit or 15

    if not username:
        print(f"{Colors.RED}Error: username is required.{Colors.RESET}")
        sys.exit(1)

    memory_path = args.memory_db or os.getenv("MEMORY_DB_PATH", "gitproof_memory.db")
    memory = MemoryManager(db_path=memory_path)
    llm = LLMClient()

    scanner = PortfolioScannerAgent(github_token=token, memory=memory, llm=llm)

    if not args.json:
        print(f"\n{Colors.BLUE}🔍 Scanning GitHub portfolio for {Colors.BOLD}{username}{Colors.RESET}{Colors.BLUE}...{Colors.RESET}")
        print(f"   Max Repos: {limit} | LLM: {'🟢 Connected' if llm.available else '⚪ Disabled'}\n")

    try:
        report = scanner.scan_portfolio(username=username, limit_repos=limit)
    except Exception as exc:
        print(f"{Colors.RED}Portfolio Scan Failed: {exc}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"{Colors.BOLD}══════════════════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f" 🌐 {Colors.BOLD}PORTFOLIO SKILL INTELLIGENCE MATRIX: {username}{Colors.RESET}")
    print(f"{Colors.BOLD}══════════════════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"  • Total Repositories Evaluated : {report.get('total_repos_scanned', 0)}")
    print(f"  • Total Commits Discovered     : {report.get('total_commits_found', 0):,}")
    print(f"  • Total Pull Requests          : {report.get('total_prs_found', 0)}")
    print(f"  • Scan Duration                : {report.get('scan_duration_seconds', 0):.1f}s")

    matrix = report.get("skill_matrix", {})
    if matrix:
        print(f"\n{Colors.BOLD}{'SKILL / LANGUAGE':<20} {'SCORE':<16} {'COMMITS':<10} {'FILES':<8} {'REPOS':<8} {'TIER'}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 70}{Colors.RESET}")
        for skill_name, data in matrix.items():
            sc = data.get("evidence_score", 0)
            score_str = f"{sc:.1f}/100"
            commits = data.get("total_commits", 0)
            files = data.get("total_skill_files", 0)
            repos_count = len(data.get("repos", []))
            tier = data.get("tier", "Proficient")

            tier_color = Colors.GREEN if "Senior" in tier or "Lead" in tier or "High" in tier else Colors.YELLOW
            print(f" {skill_name.title():<19} {score_str:<15} {commits:<10} {files:<8} {repos_count:<8} {tier_color}{tier}{Colors.RESET}")

    highlights = report.get("highlight_projects", [])
    if highlights:
        print(f"\n{Colors.BOLD}🌟 Top Highlight Projects:{Colors.RESET}")
        for proj in highlights[:5]:
            p_name = proj.get("name", "Unknown")
            p_stars = proj.get("stars", 0)
            p_forks = proj.get("forks", 0)
            p_lang = proj.get("language", "Code")
            p_desc = proj.get("description") or "No description provided."
            print(f"    • {Colors.CYAN}{Colors.BOLD}{p_name}{Colors.RESET} ({p_lang}) — ⭐ {p_stars} | 🍴 {p_forks}")
            print(f"      {Colors.DIM}{p_desc}{Colors.RESET}")

    profile_summary = report.get("portfolio_summary")
    if profile_summary:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🤖 Comprehensive AI Developer Summary:{Colors.RESET}")
        print(f"  {profile_summary}\n")


def cmd_status(args):
    """Display system and memory statistics."""
    memory_path = args.memory_db or os.getenv("MEMORY_DB_PATH", "gitproof_memory.db")
    memory = MemoryManager(db_path=memory_path)
    llm = LLMClient()

    stats = memory.stats()
    token_present = bool(os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT"))

    print(f"\n{Colors.BOLD}🛠️  GITPROOF SYSTEM & MEMORY STATUS{Colors.RESET}")
    print(f"{Colors.BOLD}──────────────────────────────────────────{Colors.RESET}")
    print(f"  • SQLite Database Path   : {memory.db_path}")
    print(f"  • Episodic Analyses Stored: {stats.get('analyses', 0)}")
    print(f"  • Feedbacks Received     : {stats.get('feedback', 0)}")
    print(f"  • Memory Lessons Learned : {stats.get('lessons', 0)}")
    print(f"  • LLM Engine (Gemini)    : {'🟢 Connected & Available' if llm.available else '⚪ Unavailable (No GEMINI_API_KEY)'}")
    print(f"  • Default GITHUB_TOKEN   : {'🟢 Configured in env' if token_present else '⚪ Not in env (pass via CLI or prompt)'}\n")


def cmd_lessons(args):
    """List lessons stored in memory."""
    memory_path = args.memory_db or os.getenv("MEMORY_DB_PATH", "gitproof_memory.db")
    memory = MemoryManager(db_path=memory_path)

    lessons = memory.get_all_lessons()
    if args.limit:
        lessons = lessons[:args.limit]

    if args.json:
        print(json.dumps(lessons, indent=2))
        return

    print(f"\n{Colors.BOLD}🧠 EPISODIC LESSONS IN MEMORY ({len(lessons)}){Colors.RESET}")
    print(f"{Colors.BOLD}──────────────────────────────────────────────────────────────────────────{Colors.RESET}")
    if not lessons:
        print(f"  {Colors.DIM}No lessons recorded yet. Submit feedback on an analysis to create lessons.{Colors.RESET}\n")
        return

    for idx, item in enumerate(lessons, 1):
        text = item.get("text", "")
        tags = item.get("tags") or []
        tags_str = f" [{', '.join(tags)}]" if tags else ""
        print(f"  {idx}. {text}{Colors.CYAN}{tags_str}{Colors.RESET}")
    print()


def cmd_feedback(args):
    """Submit human feedback on an analysis to trigger learning loop."""
    token = get_token(args.token)
    memory_path = args.memory_db or os.getenv("MEMORY_DB_PATH", "gitproof_memory.db")
    memory = MemoryManager(db_path=memory_path)
    llm = LLMClient()

    agent = OrchestratorAgent(github_token=token, memory=memory, llm=llm)

    try:
        res = agent.process_feedback(
            analysis_id=args.id,
            feedback_type=args.type,
            feedback_text=args.notes or "",
        )
        print(f"\n{Colors.GREEN}✅ Feedback saved successfully for analysis {args.id}!{Colors.RESET}")
        if res.get("lesson_extracted"):
            lesson = res.get("lesson", {})
            print(f"{Colors.BOLD}🧠 New Lesson Extracted:{Colors.RESET} {lesson.get('lesson_text')}")
            print(f"   Tags: {lesson.get('tags')}\n")
        else:
            print(f"{Colors.DIM}No new lesson extracted (LLM disabled or not triggered).{Colors.RESET}\n")
    except Exception as exc:
        print(f"{Colors.RED}Feedback submission failed: {exc}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)


def cmd_interactive(args):
    """Interactive mode in the terminal."""
    print_banner()
    while True:
        print(f"\n{Colors.BOLD}GitProof Interactive Menu:{Colors.RESET}")
        print("  1. Analyze single repository contribution")
        print("  2. Scan full developer portfolio")
        print("  3. Check system & memory status")
        print("  4. View learned lessons")
        print("  5. Exit")

        try:
            choice = input(f"\n{Colors.BOLD}Select an option [1-5]: {Colors.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == "1":
            target = input("Enter repo (owner/repo): ").strip()
            user = input("Enter username: ").strip()
            skill = input("Enter claimed skill (e.g. python, typescript, rust): ").strip()
            if "/" in target:
                owner, repo = target.split("/", 1)
            else:
                owner = target
                repo = input("Enter repo name: ").strip()

            args.owner = owner
            args.repo = repo
            args.username = user
            args.skill = skill
            args.no_cache = False
            args.json = False
            cmd_analyze(args)

        elif choice == "2":
            user = input("Enter username to scan: ").strip()
            limit_str = input("Max repos to scan [default 15]: ").strip()
            args.username = user
            args.limit = int(limit_str) if limit_str.isdigit() else 15
            args.json = False
            cmd_scan(args)

        elif choice == "3":
            cmd_status(args)

        elif choice == "4":
            args.json = False
            args.limit = 50
            cmd_lessons(args)

        elif choice == "5" or choice.lower() in ["exit", "q", "quit"]:
            print("Goodbye!")
            break
        else:
            print(f"{Colors.YELLOW}Invalid choice.{Colors.RESET}")


def main():
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--token", "-t", help="GitHub Personal Access Token (or set GITHUB_TOKEN env var)")
    common_parser.add_argument("--memory-db", help="Path to SQLite memory DB (defaults to gitproof_memory.db)")

    parser = argparse.ArgumentParser(
        description="GitProof CLI — Developer skill verification and portfolio scanning without a frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common_parser],
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    p_analyze = subparsers.add_parser("analyze", parents=[common_parser], help="Analyze repository contribution for a specific skill")
    p_analyze.add_argument("owner_or_target", nargs="?", help="Repository in 'owner/repo' or 'owner' format")
    p_analyze.add_argument("username_arg", nargs="?", help="GitHub username")
    p_analyze.add_argument("skill_arg", nargs="?", help="Skill to verify (e.g. python, typescript)")
    p_analyze.add_argument("--owner", "-o", help="Repository owner")
    p_analyze.add_argument("--repo", "-r", help="Repository name")
    p_analyze.add_argument("--user", "-u", dest="username", help="GitHub username")
    p_analyze.add_argument("--skill", "-s", help="Skill to verify")
    p_analyze.add_argument("--no-cache", action="store_true", help="Bypass cached analyses")
    p_analyze.add_argument("--json", action="store_true", help="Output raw JSON")

    # Scan command
    p_scan = subparsers.add_parser("scan", parents=[common_parser], help="Scan developer portfolio across all repositories")
    p_scan.add_argument("username_arg", nargs="?", help="GitHub username")
    p_scan.add_argument("--user", "-u", dest="username", help="GitHub username")
    p_scan.add_argument("--limit", "-l", type=int, default=15, help="Max repositories to scan")
    p_scan.add_argument("--json", action="store_true", help="Output raw JSON")

    # Status command
    p_status = subparsers.add_parser("status", parents=[common_parser], help="Show system status, LLM availability, and memory stats")

    # Lessons command
    p_lessons = subparsers.add_parser("lessons", parents=[common_parser], help="View lessons learned in episodic memory")
    p_lessons.add_argument("--skill", help="Filter lessons by skill")
    p_lessons.add_argument("--limit", type=int, default=50, help="Max lessons to return")
    p_lessons.add_argument("--json", action="store_true", help="Output raw JSON")

    # Feedback command
    p_feedback = subparsers.add_parser("feedback", parents=[common_parser], help="Submit human feedback on an analysis")
    p_feedback.add_argument("--id", required=True, help="Analysis ID")
    p_feedback.add_argument("--type", required=True, choices=["too_high", "too_low", "accurate", "wrong_language"], help="Feedback type")
    p_feedback.add_argument("--notes", help="Detailed notes or explanation")

    # Interactive command
    subparsers.add_parser("interactive", parents=[common_parser], help="Start interactive terminal mode")

    parsed = parser.parse_args()

    # Normalize positional arguments if passed
    if parsed.command == "analyze":
        if parsed.owner_or_target:
            if "/" in parsed.owner_or_target:
                parsed.owner, parsed.repo = parsed.owner_or_target.split("/", 1)
            else:
                parsed.owner = parsed.owner_or_target
        if parsed.username_arg:
            parsed.username = parsed.username_arg
        if parsed.skill_arg:
            parsed.skill = parsed.skill_arg
        cmd_analyze(parsed)

    elif parsed.command == "scan":
        if parsed.username_arg:
            parsed.username = parsed.username_arg
        cmd_scan(parsed)

    elif parsed.command == "status":
        cmd_status(parsed)

    elif parsed.command == "lessons":
        cmd_lessons(parsed)

    elif parsed.command == "feedback":
        cmd_feedback(parsed)

    elif parsed.command == "interactive":
        cmd_interactive(parsed)

    else:
        # Default to interactive or help
        if len(sys.argv) == 1:
            cmd_interactive(parsed)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()

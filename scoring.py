"""
GitProof Physics Dynamics Scoring Engine  (v2).

Each observable maps to a physically motivated equation so the score
remains deterministic, interpretable, and hard to game.

──────────────────────────────────────────────────────────────────────
PHYSICS MODEL SUMMARY
──────────────────────────────────────────────────────────────────────

1. INERTIAL MASS  M  (max 30 pts)
   Skill-weighted body of code.  Real repositories follow Zipf / power-law
   size distributions, so raw counts are sublinearly compressed:

       M_files  = A · skill_files ^ α       α=0.60 (Pareto exponent)
       M_volume = B · log1p(additions / Ω)  Ω=400 LOC (half-saturation)
       M        = clamp(M_files + M_volume, 0, 30)

2. RELATIVISTIC MOMENTUM  p  (max 25 pts)
   Sustained commit cadence ≙ particle velocity.
   At very high velocity (burst commits) a Lorentz gamma factor damps
   the score — mirroring special relativity where "faster" costs more
   energy but yields diminishing momentum gains:

       v  = commits / max(days, 1)             (commits per day)
       c  = 2.0                                (speed-of-light analogue)
       β  = v / c
       1/γ= sqrt(1 - β²)                       Lorentz correction in (0,1]
       p  = 25 · tanh(commits/τ) · (1/γ) · tanh(days/τ_T)

3. BOLTZMANN ENTROPY  S  (max 15 pts)
   Model commit arrival as Poisson(λ=commits/days).
   High entropy ≙ spread-out iterative development.
   Single-session dumps produce low λ→∞ and collapse entropy:

       H  = -Σ P(k) log P(k)  over truncated Poisson PMF
       S_norm = H / log(commits + 2)
       S  = 15 · S_norm

4. CARNOT EFFICIENCY  η  (max 20 pts)
   Pull request pipeline efficiency as a thermodynamic heat engine.
   η = 1 − T_cold/T_hot  ↔  η_PR = merged_prs / max(prs, 1).
   A perfect review pipeline (all PRs merged) has η → 1 (Carnot limit):

       W  = W_max · η_PR · tanh(merged_prs / τ_PR)

5. YUKAWA INTEGRITY FIELD  Φ  (max 10 pts)
   Cryptographic commit verification ≙ short-range Yukawa potential:

       Φ(n) = g · (1 − e^(−n/λ))     g=10, λ=4

   Saturates quickly (even 1 verified commit gives a real signal).

6. SKILL CONCENTRATION BONUS  (max 5 pts)
   Concentration ratio  ρ = skill_files / files_changed
   Rewards specialists; penalises broad-brush repos where the claimed
   skill is incidental:

       bonus = 5 · ρ²   (quadratic — steeper specialist reward)

──────────────────────────────────────────────────────────────────────
DAMPING / DRAG TERMS
──────────────────────────────────────────────────────────────────────
A. STOKES VISCOUS FORK DRAG    — score capped at 70 for forks
B. LORENTZ BURST DAMPING       — embedded in relativistic p term
──────────────────────────────────────────────────────────────────────

TOTAL HAMILTONIAN  E = M + p + S + W + Φ + bonus → clamp to [0, 100]
Theoretical max: 30+25+15+20+10+5 = 105 → clamped to 100.
"""

import math

# ─── Physical constants (tuning knobs) ────────────────────────────────────────
_ALPHA         = 0.60    # Pareto / Zipf exponent for file-count mass
_A_FILES       = 12.0    # scale for file mass component
_B_VOLUME      = 10.0    # scale for volume mass component
_OMEGA_LOC     = 400.0   # LOC half-saturation constant (Michaelis-Menten)
_C_LIGHT       = 2.0     # "speed of light" analogue — commits/day
_TAU_COMMIT    = 20.0    # commit count half-saturation (tanh scale)
_TAU_DAYS      = 60.0    # day-duration half-saturation (~2 months)
_YUKAWA_G      = 10.0    # Yukawa coupling constant
_YUKAWA_LAMBDA = 4.0     # Yukawa decay length (# verified commits)
_CARNOT_W_MAX  = 20.0    # Maximum Carnot work output (PR ceiling)
_CARNOT_TAU    = 3.0     # merged-PR tanh half-saturation
_ENTROPY_W     = 15.0    # Boltzmann entropy max contribution
_FORK_CAP      = 70      # Stokes drag score cap for fork repos
_BONUS_MAX     = 5.0     # Concentration bonus ceiling


def _poisson_entropy(lam: float, k_max: int) -> float:
    """
    Shannon entropy H of a truncated Poisson(λ) PMF.

        H = -Σ_{k=0}^{k_max} P(k) * log P(k)
        log P(k) = -λ + k*log(λ) - log(k!)   (log-space recurrence)
    """
    if lam <= 0.0:
        return 0.0
    log_lam = math.log(lam)
    H = 0.0
    log_pk = -lam           # log P(0) = -λ
    for k in range(k_max + 1):
        p = math.exp(log_pk)
        if p > 1e-15:
            H -= p * log_pk     # contribution −P(k)·log P(k)
        if k < k_max:
            log_pk += log_lam - math.log(k + 1)
    return max(0.0, H)


def calculate_score(data: dict) -> dict:
    """
    Deterministic 0-100 evidence score from GitHub contribution data.

    Returns dict:
      evidence_score  – int [0, 100]
      confidence      – str label
      reasons         – list[str] human-readable factor breakdown
      warnings        – list[str] red-flag annotations
      physics         – dict of raw physics quantities for UI / debugging
    """
    if not isinstance(data, dict):
        data = {}

    raw_contribution = data.get("contribution")
    contribution = raw_contribution if isinstance(raw_contribution, dict) else {}

    def _safe_float(val, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            return max(0.0, float(val))
        except (ValueError, TypeError):
            return default

    # ── Raw observables ────────────────────────────────────────────────────────
    commits       = _safe_float(contribution.get("commits"))
    skill_files   = _safe_float(contribution.get("skill_files"))
    files_changed = _safe_float(contribution.get("files_changed")) or max(skill_files, 1.0)
    additions     = _safe_float(contribution.get("additions"))
    days          = _safe_float(contribution.get("contribution_days"))
    verified      = _safe_float(contribution.get("verified_commits"))
    prs           = _safe_float(contribution.get("pull_requests"))
    merged_prs    = _safe_float(contribution.get("merged_pull_requests"))

    reasons: list = []
    warnings: list = []

    # ══════════════════════════════════════════════════════════════════════════
    # 1.  INERTIAL MASS  M  — max 30 pts
    #     M_files  = A · n^α            (Pareto power-law)
    #     M_volume = B · log1p(v / Ω)   (Michaelis-Menten saturation)
    # ══════════════════════════════════════════════════════════════════════════
    M_files  = _A_FILES * (skill_files ** _ALPHA) if skill_files > 0 else 0.0
    M_volume = _B_VOLUME * math.log1p(additions / _OMEGA_LOC) if additions > 0 else 0.0
    M = min(M_files + M_volume, 30.0)

    if skill_files > 0:
        reasons.append(
            f"Inertial Mass M={M:.1f} "
            f"[{int(skill_files)} skill files · {int(additions):,} LOC added]"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 2.  RELATIVISTIC MOMENTUM  p  — max 25 pts
    #     β  = v / c = (commits/days) / c
    #     1/γ= sqrt(1 − β²)                    Lorentz correction ∈ (0, 1]
    #     p  = 25 · tanh(N/τ) · (1/γ) · tanh(T/τ_T)
    # ══════════════════════════════════════════════════════════════════════════
    v          = commits / max(days, 1.0)
    beta       = min(v / _C_LIGHT, 0.995)    # β = v/c
    inv_gamma  = math.sqrt(1.0 - beta ** 2)  # 1/γ; burst → near 0
    p = (25.0
         * math.tanh(commits / _TAU_COMMIT)
         * inv_gamma
         * math.tanh(days / _TAU_DAYS))

    burst_detected = bool(commits > 5 and days < 3)
    if burst_detected:
        warnings.append(
            f"High-velocity burst: {int(commits)} commits in {int(days)} day(s) "
            f"(β={beta:.3f} → Lorentz 1/γ={inv_gamma:.3f} applied)."
        )
    if commits > 0:
        reasons.append(
            f"Relativistic Momentum p={p:.1f} "
            f"[N={int(commits)}, T={int(days)} d, β={beta:.3f}, 1/γ={inv_gamma:.3f}]"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 3.  BOLTZMANN ENTROPY  S  — max 15 pts
    #     λ  = commits / max(days, 1)   (Poisson event rate)
    #     H  = Shannon entropy of truncated Poisson(λ) PMF
    #     S  = 15 · H / log(commits + 2)
    # ══════════════════════════════════════════════════════════════════════════
    lam   = v
    k_max = max(int(commits * 2), 5)
    H_raw = _poisson_entropy(lam, k_max) if lam > 0 else 0.0
    H_norm = min(H_raw / math.log(commits + 2), 1.0) if commits > 0 else 0.0
    S = _ENTROPY_W * H_norm

    if commits > 0:
        reasons.append(
            f"Boltzmann Entropy S={S:.1f} "
            f"[H_norm={H_norm:.3f}, λ={lam:.2f} commits/day]"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 4.  CARNOT EFFICIENCY  η  — max 20 pts
    #     η_PR = merged_prs / max(prs, 1)      (heat-engine efficiency)
    #     W    = W_max · η_PR · tanh(N_merged/τ)
    # ══════════════════════════════════════════════════════════════════════════
    eta_carnot = merged_prs / max(prs, 1.0) if prs > 0 else 0.0
    W_pr = _CARNOT_W_MAX * eta_carnot * math.tanh(merged_prs / _CARNOT_TAU)

    if prs > 0:
        reasons.append(
            f"Carnot Efficiency η={eta_carnot:.2f} → PR Work W={W_pr:.1f} "
            f"[{int(merged_prs)}/{int(prs)} PRs merged]"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 5.  YUKAWA INTEGRITY FIELD  Φ  — max 10 pts
    #     Φ(n) = g · (1 − e^(−n/λ))
    # ══════════════════════════════════════════════════════════════════════════
    Phi = _YUKAWA_G * (1.0 - math.exp(-verified / _YUKAWA_LAMBDA)) if verified > 0 else 0.0
    if verified > 0:
        reasons.append(f"Yukawa Field Φ={Phi:.1f} [{int(verified)} signed commits]")

    # ══════════════════════════════════════════════════════════════════════════
    # 6.  SKILL CONCENTRATION BONUS  — max 5 pts
    #     ρ     = skill_files / files_changed   (skill density ratio)
    #     bonus = B_max · ρ²                    (quadratic specialist reward)
    # ══════════════════════════════════════════════════════════════════════════
    rho   = min(skill_files / max(files_changed, 1.0), 1.0)
    bonus = _BONUS_MAX * (rho ** 2)
    if rho > 0:
        reasons.append(f"Concentration Bonus={bonus:.1f} [ρ={rho:.2f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # TOTAL HAMILTONIAN  E = M + p + S + W_pr + Φ + bonus
    # ══════════════════════════════════════════════════════════════════════════
    E = M + p + S + W_pr + Phi + bonus

    # ── STOKES VISCOUS FORK DRAG ───────────────────────────────────────────────
    repo_meta = data.get("repository")
    is_fork = bool(repo_meta.get("is_fork")) if isinstance(repo_meta, dict) else False
    if is_fork:
        warnings.append(f"Stokes drag applied (fork): score capped at {_FORK_CAP}.")

    score = int(round(E))
    score = max(0, min(100, score))
    if is_fork:
        score = min(score, _FORK_CAP)

    # ── Confidence buckets ─────────────────────────────────────────────────────
    if score >= 80:
        confidence = "High"
    elif score >= 60:
        confidence = "Medium-High"
    elif score >= 40:
        confidence = "Medium"
    elif score >= 20:
        confidence = "Low"
    else:
        confidence = "Insufficient"

    # ── Physics telemetry (for UI / debugging) ─────────────────────────────────
    physics = {
        "inertial_mass_M":          round(M, 3),
        "relativistic_momentum_p":  round(p, 3),
        "lorentz_beta":             round(beta, 4),
        "lorentz_inv_gamma":        round(inv_gamma, 4),
        "boltzmann_entropy_S":      round(S, 3),
        "entropy_H_norm":           round(H_norm, 4),
        "carnot_efficiency_eta":    round(eta_carnot, 4),
        "carnot_work_W":            round(W_pr, 3),
        "yukawa_field_Phi":         round(Phi, 3),
        "skill_concentration_rho":  round(rho, 4),
        "concentration_bonus":      round(bonus, 3),
        "total_energy_E":           round(E, 3),
        "stokes_fork_drag":         is_fork,
        "lorentz_burst_detected":   burst_detected,
    }

    return {
        "evidence_score": score,
        "confidence":     confidence,
        "reasons":        reasons,
        "warnings":       warnings,
        "physics":        physics,
    }
"""Final composite score combining evidence, JD fit, behavior, and trap signals.

Scoring formula::

    within_score = evidence_grade * (1 + ALPHA * jd_fit_score) * behavior_multiplier * trap_penalty
    composite    = evidence_tier + squash(within_score / WITHIN_SCORE_DIVISOR)

The **evidence tier** is added as a whole number (0-5), so it dominates the
ordering: no amount of location/notice/behavior bonus can push a Tier-3
candidate above a Tier-4 one. The continuous ``within_score`` only refines
ordering within the same tier.

Design rationale: an early version used a flat weighted sum, and it let a
hyperactive Tier-2 candidate outrank a quiet Tier-4 — which is obviously
wrong. The tier-lexicographic approach fixed this.

Honeypots are **excluded before ranking**, not penalized within it. Sorting
is deterministic: composite descending, candidate_id ascending on ties.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import configuration as config
from . import data_access
from .behavior_analyzer import BehavioralScore, compute_behavioral_multiplier
from .evidence_scorer import AggregatedCandidateEvidence, aggregate_candidate_evidence
from .job_fit_scorer import JobFitScore, compute_jd_fit_score
from .template_audit import grade_to_tier
from .time_anchor import TimeAnchor
from .trap_detector import TrapDetectionResult, detect_trap_signals


@dataclass
class ScoredCandidateResult:
    """Complete scoring result for a single candidate."""

    candidate_id: str
    candidate_data: dict
    evidence: AggregatedCandidateEvidence
    job_fit: JobFitScore
    behavior: BehavioralScore
    trap: TrapDetectionResult
    evidence_tier: int
    within_tier_score: float
    composite_score: float
    confidence: float


def _compute_confidence(
    evidence: AggregatedCandidateEvidence,
    job_fit: JobFitScore,
    behavior: BehavioralScore,
    trap: TrapDetectionResult,
) -> float:
    """Estimate confidence [0, 1] that this ranking is well-supported.

    High confidence when strong, corroborated evidence aligns with decent
    JD-fit and engagement. Low when signals disagree (keyword stuffer, or
    a strong profile that is stale/unreachable). Used for reasoning tone,
    not ordering.
    """
    confidence = 0.35 + 0.5 * evidence.grade

    if evidence.strong_role_count >= 2:
        confidence += 0.08

    if job_fit.score >= 0.0:
        confidence += 0.05
    else:
        confidence -= 0.05

    if behavior.multiplier >= 1.0:
        confidence += 0.05
    elif behavior.multiplier < 0.8:
        confidence -= 0.10

    if trap.is_keyword_stuffer:
        confidence -= 0.25

    return round(max(0.0, min(1.0, confidence)), 4)


def compute_candidate_score(candidate: dict, anchor: TimeAnchor) -> ScoredCandidateResult:
    """Compute the full composite score for a single candidate.

    Runs all four scoring subsystems (evidence, JD-fit, behavior, traps)
    and combines them into a tier-lexicographic composite.

    Args:
        candidate: A raw candidate dictionary.
        anchor: The data-derived time anchor.

    Returns:
        A ``ScoredCandidateResult`` containing all intermediate scores and
        the final composite.
    """
    evidence = aggregate_candidate_evidence(candidate, anchor)
    job_fit = compute_jd_fit_score(candidate)
    behavior = compute_behavioral_multiplier(candidate, anchor)
    trap = detect_trap_signals(candidate, anchor, evidence.grade)

    trap_penalty = config.STUFFER_PENALTY_MULTIPLIER if trap.is_keyword_stuffer else 1.0
    within = (
        evidence.grade
        * (1.0 + config.JD_FIT_ALPHA * job_fit.score)
        * behavior.multiplier
        * trap_penalty
    )
    within = max(0.0, within)

    evidence_tier = grade_to_tier(evidence.grade)

    within_normalized = min(0.999999, within / config.WITHIN_SCORE_DIVISOR)
    composite = evidence_tier + within_normalized

    return ScoredCandidateResult(
        candidate_id=data_access.extract_candidate_id(candidate),
        candidate_data=candidate,
        evidence=evidence,
        job_fit=job_fit,
        behavior=behavior,
        trap=trap,
        evidence_tier=evidence_tier,
        within_tier_score=round(within, 6),
        composite_score=round(composite, 6),
        confidence=_compute_confidence(evidence, job_fit, behavior, trap),
    )


def rank_candidates(
    candidates: list[dict],
    anchor: TimeAnchor,
    top_n: int = 100,
    exclude_honeypots: bool = True,
) -> list[ScoredCandidateResult]:
    """Score all candidates and return the deterministic top-N shortlist.

    Honeypot candidates are dropped before ranking (not penalized within).
    Ordering is deterministic: composite score descending, candidate_id
    ascending on ties — matching the validator's tie-break rule.

    Args:
        candidates: Full list of candidate dictionaries.
        anchor: The data-derived time anchor.
        top_n: Number of candidates to return in the shortlist (default 100).
        exclude_honeypots: If True, drop honeypot candidates entirely.

    Returns:
        A list of ``ScoredCandidateResult`` for the top-N candidates, sorted
        by composite score descending (with deterministic tie-breaking).
    """
    scored: list[ScoredCandidateResult] = []
    for candidate in candidates:
        result = compute_candidate_score(candidate, anchor)
        if exclude_honeypots and result.trap.is_honeypot:
            continue
        scored.append(result)

    scored.sort(key=lambda s: (-s.composite_score, s.candidate_id))
    return scored[:top_n]

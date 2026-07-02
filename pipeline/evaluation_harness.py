"""Local proxy evaluation harness as a stand-in for the hidden leaderboard.

With no public leaderboard or feedback during the competition, we build our
own "ground truth" from the template audit (best audited tier across a
candidate's roles, forced to 0 for honeypots) and compute the same metrics
the specification says the judges will use:

    composite = 0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10

**Limitation**: this proxy ground truth comes from the same audit that drives
the evidence scorer, so a composite of 1.0 against it mostly proves internal
consistency of the tiering logic. It says nothing about whether the ordering
*within* strong candidates matches real judges. Use this for sanity checks
and weight sweeps, not as proof of a good leaderboard score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import data_access
from .template_audit import match_description_to_audit
from .time_anchor import TimeAnchor
from .trap_detector import detect_trap_signals


RELEVANT_TIER_THRESHOLD: int = 3


def compute_proxy_tier(candidate: dict, anchor: TimeAnchor) -> int:
    """Determine the best audited tier across the candidate's roles.

    Returns the maximum tier from the template audit across all career
    descriptions. Honeypots are forced to tier 0. Unknown descriptions
    (not matching any audit entry) contribute 0.

    Args:
        candidate: A raw candidate dictionary.
        anchor: The data-derived time anchor.

    Returns:
        An integer tier in [0, 5].
    """
    best_tier = 0
    for role in data_access.get_career_history(candidate):
        audit_entry = match_description_to_audit(role.get("description", ""))
        if audit_entry is not None:
            best_tier = max(best_tier, audit_entry.tier)

    trap_result = detect_trap_signals(
        candidate,
        anchor,
        evidence_grade=1.0 if best_tier >= 4 else 0.0,
    )
    if trap_result.is_honeypot:
        return 0
    return best_tier


def build_proxy_ground_truth(
    candidates: list[dict], anchor: TimeAnchor
) -> dict[str, int]:
    """Build a proxy ground-truth mapping candidate_id -> tier.

    Iterates all candidates and assigns the best audited tier (or 0 for
    honeypots/unknown profiles).

    Args:
        candidates: Full list of candidate dictionaries.
        anchor: The data-derived time anchor.

    Returns:
        A dict mapping ``candidate_id`` -> ``tier`` (int 0-5).
    """
    return {
        data_access.extract_candidate_id(candidate): compute_proxy_tier(candidate, anchor)
        for candidate in candidates
    }


def _compute_dcg(gains: list[float]) -> float:
    """Compute Discounted Cumulative Gain from a list of relevance gains."""
    return sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))


def compute_ndcg_at_k(
    ranked_candidate_ids: list[str], ground_truth: dict[str, int], k: int
) -> float:
    """Compute Normalized Discounted Cumulative Gain at cutoff k."""
    gains = [float(ground_truth.get(candidate_id, 0)) for candidate_id in ranked_candidate_ids[:k]]
    dcg = _compute_dcg(gains)
    ideal_gains = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = _compute_dcg([float(gain) for gain in ideal_gains])
    return dcg / idcg if idcg > 0 else 0.0


def compute_precision_at_k(
    ranked_candidate_ids: list[str], ground_truth: dict[str, int], k: int
) -> float:
    """Compute precision at cutoff k (fraction of tier >= RELEVANT_TIER)."""
    if k <= 0:
        return 0.0
    hits = sum(
        1
        for candidate_id in ranked_candidate_ids[:k]
        if ground_truth.get(candidate_id, 0) >= RELEVANT_TIER_THRESHOLD
    )
    return hits / k


def compute_mean_average_precision(
    ranked_candidate_ids: list[str], ground_truth: dict[str, int]
) -> float:
    """Compute Mean Average Precision across all relevant candidates."""
    total_relevant = sum(1 for tier in ground_truth.values() if tier >= RELEVANT_TIER_THRESHOLD)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, candidate_id in enumerate(ranked_candidate_ids, start=1):
        if ground_truth.get(candidate_id, 0) >= RELEVANT_TIER_THRESHOLD:
            hits += 1
            precision_sum += hits / i
    return precision_sum / min(total_relevant, len(ranked_candidate_ids))


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""

    ndcg_at_10: float
    ndcg_at_50: float
    mean_average_precision: float
    precision_at_10: float
    precision_at_5: float
    composite_score: float


def evaluate_ranking(
    ranked_candidate_ids: list[str], ground_truth: dict[str, int]
) -> EvaluationMetrics:
    """Compute all evaluation metrics for a ranked list against the proxy ground truth.

    The composite formula matches the competition spec::

        composite = 0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10

    Args:
        ranked_candidate_ids: Ordered list of candidate IDs (best first).
        ground_truth: Dict mapping candidate_id -> tier (from
            ``build_proxy_ground_truth``).

    Returns:
        An ``EvaluationMetrics`` with all computed metrics.
    """
    ndcg_10 = compute_ndcg_at_k(ranked_candidate_ids, ground_truth, 10)
    ndcg_50 = compute_ndcg_at_k(ranked_candidate_ids, ground_truth, 50)
    map_value = compute_mean_average_precision(ranked_candidate_ids, ground_truth)
    precision_10 = compute_precision_at_k(ranked_candidate_ids, ground_truth, 10)
    precision_5 = compute_precision_at_k(ranked_candidate_ids, ground_truth, 5)

    composite = 0.50 * ndcg_10 + 0.30 * ndcg_50 + 0.15 * map_value + 0.05 * precision_10

    return EvaluationMetrics(
        ndcg_at_10=round(ndcg_10, 4),
        ndcg_at_50=round(ndcg_50, 4),
        mean_average_precision=round(map_value, 4),
        precision_at_10=round(precision_10, 4),
        precision_at_5=round(precision_5, 4),
        composite_score=round(composite, 4),
    )

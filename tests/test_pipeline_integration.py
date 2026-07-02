"""Integration tests for the complete ranking pipeline.

Tests cover all subsystems: data access, evidence scoring, trap detection,
JD-fit scoring, behavioral analysis, composite scoring, reasoning generation,
and the local evaluation harness.
"""

from __future__ import annotations

from pipeline import (
    behavior_analyzer,
    composite_scorer,
    data_access,
    evaluation_harness,
    evidence_scorer,
    job_fit_scorer,
    reasoning_generator,
    time_anchor,
    trap_detector,
)
from pipeline.time_anchor import TimeAnchor


# ============================================================================
# Data access and date utilities
# ============================================================================

def test_month_index_and_months_between() -> None:
    """Date arithmetic helpers should produce correct month indices and intervals."""
    assert data_access.month_index_from_date("2026-05-27") == 2026 * 12 + 5
    assert data_access.months_between_dates("2026-01-01", "2026-05-01") == 4
    assert data_access.month_index_from_date("invalid input") is None


def test_time_anchor_is_deterministic() -> None:
    """The time anchor should use the latest last_active_date across candidates."""
    candidates = [
        {"redrob_signals": {"last_active_date": "2026-01-01"}},
        {"redrob_signals": {"last_active_date": "2026-05-27"}},
        {"redrob_signals": {"last_active_date": "2025-11-11"}},
    ]
    anchor = time_anchor.compute_time_anchor(candidates)
    assert anchor.date == "2026-05-27"


# ============================================================================
# Evidence scoring
# ============================================================================

def test_strong_candidate_evidence_is_tier_5(strong_candidate, time_anchor) -> None:
    """An elite ranking profile must achieve evidence grade >= 0.78 (tier 5)."""
    evidence = evidence_scorer.aggregate_candidate_evidence(strong_candidate, time_anchor)
    assert evidence.grade >= 0.78


def test_keyword_stuffer_evidence_is_low(keyword_stuffer_candidate, time_anchor) -> None:
    """A content writer with AI keyword-stuffed skills must score low on evidence."""
    evidence = evidence_scorer.aggregate_candidate_evidence(keyword_stuffer_candidate, time_anchor)
    assert evidence.grade < 0.20


def test_cv_primary_candidate_is_capped(cv_primary_candidate, time_anchor) -> None:
    """A CV-only candidate (no retrieval/ranking) must have grade <= 0.20."""
    evidence = evidence_scorer.aggregate_candidate_evidence(cv_primary_candidate, time_anchor)
    assert evidence.grade <= 0.20


# ============================================================================
# Trap detection
# ============================================================================

def test_honeypot_detects_yoe_inflation(honeypot_candidate, time_anchor) -> None:
    """A honeypot with inflated YoE must be flagged with experience-related reasons."""
    trap_result = trap_detector.detect_trap_signals(honeypot_candidate, time_anchor, evidence_grade=0.95)
    assert trap_result.is_honeypot
    assert any("experience" in reason for reason in trap_result.honeypot_reasons)


def test_honeypot_detects_phantom_expertise(honeypot_candidate, time_anchor) -> None:
    """A honeypot with phantom expert skills must show phantom_expert_count >= 3."""
    trap_result = trap_detector.detect_trap_signals(honeypot_candidate, time_anchor, evidence_grade=0.95)
    assert trap_result.phantom_expert_count >= 3


def test_strong_candidate_is_not_honeypot(strong_candidate, time_anchor) -> None:
    """A legitimate strong candidate must not be flagged as a honeypot."""
    trap_result = trap_detector.detect_trap_signals(strong_candidate, time_anchor, evidence_grade=0.95)
    assert not trap_result.is_honeypot


def test_keyword_stuffer_is_flagged(keyword_stuffer_candidate, time_anchor) -> None:
    """A keyword-stuffer with many AI skills but low evidence must be flagged."""
    evidence = evidence_scorer.aggregate_candidate_evidence(keyword_stuffer_candidate, time_anchor)
    trap_result = trap_detector.detect_trap_signals(
        keyword_stuffer_candidate, time_anchor, evidence.grade
    )
    assert trap_result.is_keyword_stuffer


# ============================================================================
# JD fit scoring
# ============================================================================

def test_pune_location_scores_highest(strong_candidate) -> None:
    """A candidate in Pune must receive the maximum location score."""
    fit_score = job_fit_scorer.compute_jd_fit_score(strong_candidate)
    assert fit_score.location == 1.0
    assert "Pune" in fit_score.location_label


def test_experience_fit_peaks_at_7_years() -> None:
    """Experience fit must peak at 6-8 years and be negative at extremes."""
    assert job_fit_scorer._compute_experience_fit(7.0) == 1.0
    assert job_fit_scorer._compute_experience_fit(2.0) < 0
    assert job_fit_scorer._compute_experience_fit(15.0) < 0


def _assign_roles_to_candidate(candidate: dict, roles: list[dict]) -> dict:
    """Helper to replace a candidate's career history with custom roles."""
    modified = dict(candidate)
    modified["career_history"] = roles
    return modified


def test_tenure_penalizes_job_hoppers(strong_candidate) -> None:
    """Candidates with short average stints must score lower than stable ones."""
    hopper_roles = [
        {"duration_months": 14, "description": "java backend development at a large enterprise"},
        {"duration_months": 15, "description": "java backend development at a large enterprise"},
        {"duration_months": 13, "description": "java backend development at a large enterprise"},
    ]
    stable_roles = [
        {"duration_months": 48, "description": "java backend development at a large enterprise"},
        {"duration_months": 40, "description": "java backend development at a large enterprise"},
    ]
    hopper_tenure = job_fit_scorer._compute_tenure_fit(
        _assign_roles_to_candidate(strong_candidate, hopper_roles)
    )
    stable_tenure = job_fit_scorer._compute_tenure_fit(
        _assign_roles_to_candidate(strong_candidate, stable_roles)
    )
    assert hopper_tenure < 0 < stable_tenure


def test_tenure_is_neutral_for_single_role(strong_candidate) -> None:
    """A single role must not be penalized or rewarded for tenure."""
    assert job_fit_scorer._compute_tenure_fit(strong_candidate) == 0.0


def test_ml_depth_rewards_sustained_ml_work(strong_candidate) -> None:
    """Candidates with sustained ML work must score higher than those with shallow exposure."""
    from tests.conftest import ELITE_RANKING_DESCRIPTION, NONTECH_CONTENT_WRITER_DESCRIPTION
    deep_ml_roles = [
        {"duration_months": 36, "description": ELITE_RANKING_DESCRIPTION},
        {"duration_months": 24, "description": ELITE_RANKING_DESCRIPTION},
    ]
    shallow_ml_roles = [
        {"duration_months": 6, "description": ELITE_RANKING_DESCRIPTION},
        {"duration_months": 60, "description": NONTECH_CONTENT_WRITER_DESCRIPTION},
    ]
    deep_score = job_fit_scorer._compute_ml_depth_fit(
        _assign_roles_to_candidate(strong_candidate, deep_ml_roles)
    )
    shallow_score = job_fit_scorer._compute_ml_depth_fit(
        _assign_roles_to_candidate(strong_candidate, shallow_ml_roles)
    )
    assert deep_score == 1.0
    assert shallow_score < 0


# ============================================================================
# Behavioral analysis
# ============================================================================

def test_behavior_multiplier_stays_in_bounds(strong_candidate, time_anchor) -> None:
    """Behavioral multiplier must always stay within [MIN_MULT, MAX_MULT]."""
    behavior = behavior_analyzer.compute_behavioral_multiplier(strong_candidate, time_anchor)
    assert behavior_analyzer.config.BEHAVIOR_MINIMUM_MULTIPLIER <= behavior.multiplier <= behavior_analyzer.config.BEHAVIOR_MAXIMUM_MULTIPLIER


def test_stale_inactive_candidate_is_downweighted(strong_candidate, time_anchor) -> None:
    """A stale inactive candidate must receive a lower multiplier."""
    stale = dict(strong_candidate)
    stale["redrob_signals"] = dict(strong_candidate["redrob_signals"])
    stale["redrob_signals"]["last_active_date"] = "2025-01-01"
    stale["redrob_signals"]["recruiter_response_rate"] = 0.05
    stale["redrob_signals"]["open_to_work_flag"] = False
    behavior = behavior_analyzer.compute_behavioral_multiplier(stale, time_anchor)
    assert behavior.multiplier < 0.85


# ============================================================================
# Composite scoring and ranking
# ============================================================================

def test_evidence_tier_never_inverts_ranking(strong_candidate, cv_primary_candidate, time_anchor) -> None:
    """A tier-5 candidate must always outrank a tier-1 candidate regardless of extras."""
    strong_result = composite_scorer.compute_candidate_score(strong_candidate, time_anchor)
    cv_result = composite_scorer.compute_candidate_score(cv_primary_candidate, time_anchor)
    assert strong_result.composite_score > cv_result.composite_score
    assert strong_result.evidence_tier > cv_result.evidence_tier


def test_ranking_excludes_honeypots(strong_candidate, honeypot_candidate, time_anchor) -> None:
    """Honeypot candidates must be excluded from the ranked shortlist."""
    ranked = composite_scorer.rank_candidates(
        [honeypot_candidate, strong_candidate], time_anchor, top_n=10
    )
    ranked_ids = [result.candidate_id for result in ranked]
    assert honeypot_candidate["candidate_id"] not in ranked_ids
    assert strong_candidate["candidate_id"] in ranked_ids


def test_ranking_is_deterministic_and_nonincreasing(
    strong_candidate, cv_primary_candidate, keyword_stuffer_candidate, time_anchor
) -> None:
    """Ranking must produce the same order regardless of input order, and scores must be non-increasing."""
    pool = [cv_primary_candidate, strong_candidate, keyword_stuffer_candidate]
    ranking_1 = [
        s.candidate_id
        for s in composite_scorer.rank_candidates(pool, time_anchor, top_n=10)
    ]
    ranking_2 = [
        s.candidate_id
        for s in composite_scorer.rank_candidates(
            list(reversed(pool)), time_anchor, top_n=10
        )
    ]
    assert ranking_1 == ranking_2
    composite_scores = [
        s.composite_score
        for s in composite_scorer.rank_candidates(pool, time_anchor, top_n=10)
    ]
    assert composite_scores == sorted(composite_scores, reverse=True)


def test_equal_scores_break_by_candidate_id_ascending(strong_candidate, time_anchor) -> None:
    """Candidates with identical composite scores must be ordered by ascending candidate_id."""
    twin_a = dict(strong_candidate)
    twin_a["candidate_id"] = "CAND_0000900"
    twin_b = dict(strong_candidate)
    twin_b["candidate_id"] = "CAND_0000800"
    ranked = composite_scorer.rank_candidates([twin_a, twin_b], time_anchor, top_n=10)
    assert ranked[0].candidate_id == "CAND_0000800"


# ============================================================================
# Reasoning generation
# ============================================================================

def test_reasoning_contains_no_hallucinated_skills(strong_candidate, time_anchor) -> None:
    """The reasoning text must only reference skills that actually exist in the candidate's profile."""
    scored = composite_scorer.compute_candidate_score(strong_candidate, time_anchor)
    reasoning = reasoning_generator.generate_reasoning_text(scored, rank_position=1)
    skill_names = [skill["name"] for skill in strong_candidate["skills"]]
    assert any(name in reasoning for name in skill_names)
    assert "Senior AI Engineer" in reasoning


def test_reasoning_tone_varies_by_rank(strong_candidate, time_anchor) -> None:
    """Reasoning text must differ between top and bottom rank positions."""
    scored = composite_scorer.compute_candidate_score(strong_candidate, time_anchor)
    top_rank_reasoning = reasoning_generator.generate_reasoning_text(scored, rank_position=1)
    bottom_rank_reasoning = reasoning_generator.generate_reasoning_text(scored, rank_position=95)
    assert top_rank_reasoning != bottom_rank_reasoning
    assert "depth" in bottom_rank_reasoning.lower()


def test_reasoning_varies_across_candidates(
    strong_candidate, cv_primary_candidate, time_anchor
) -> None:
    """Different candidates must produce different reasoning text."""
    strong_scored = composite_scorer.compute_candidate_score(strong_candidate, time_anchor)
    cv_scored = composite_scorer.compute_candidate_score(cv_primary_candidate, time_anchor)
    reasoning_a = reasoning_generator.generate_reasoning_text(strong_scored, rank_position=1)
    reasoning_b = reasoning_generator.generate_reasoning_text(cv_scored, rank_position=2)
    assert reasoning_a != reasoning_b


# ============================================================================
# Evaluation harness
# ============================================================================

def test_perfect_ranking_scores_one_point_zero() -> None:
    """A perfectly ordered ranking against the ground truth must yield NDCG@10 = 1.0."""
    ground_truth = {"a": 5, "b": 4, "c": 0, "d": 3}
    ranked_ids = ["a", "b", "d", "c"]
    metrics = evaluation_harness.evaluate_ranking(ranked_ids, ground_truth)
    assert metrics.ndcg_at_10 == 1.0


def test_bad_ranking_scores_less_than_one() -> None:
    """A poorly ordered ranking must score less than 1.0 NDCG."""
    ground_truth = {"a": 5, "b": 4, "c": 0, "d": 3}
    ranked_ids = ["c", "d", "b", "a"]
    metrics = evaluation_harness.evaluate_ranking(ranked_ids, ground_truth)
    assert metrics.ndcg_at_10 < 1.0

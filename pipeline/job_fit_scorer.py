"""Job Description logistics fit scoring.

While ``evidence_scorer.py`` answers "did this candidate actually do the kind
of work the JD describes?", this module answers the more mundane question of
whether the rest of the JD requirements line up: years of experience, location,
notice period, product vs. consulting background, assessment score trust,
tenure stability, and depth of ML experience.

This is deliberately a **small correction** on top of evidence (weighted by
``JD_FIT_ALPHA = 0.30``), not a second vote of equal weight. A candidate with
amazing evidence but a slightly long notice period should still clearly outrank
a mediocre one sitting in Noida.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import configuration as config
from . import data_access
from .evidence_scorer import grade_text_block


@dataclass
class JobFitScore:
    """Result of JD logistics fit scoring for a candidate."""

    score: float
    experience: float
    location: float
    notice: float
    product: float
    skill_trust: float
    tenure: float = 0.0
    ml_depth: float = 0.0
    location_label: str = ""
    breakdown: dict = field(default_factory=dict)


def _compute_experience_fit(years: float) -> float:
    """Score how well the candidate's years of experience match the JD's 6-8y band.

    Returns a float in [-1, 1]:
        - Peak (1.0) at 6-8 years
        - Gentle shoulders (0.7) at 5 and 9 years
        - Negative for extreme under/over-seniority
    """
    if years <= 0:
        return -1.0
    if config.EXPERIENCE_PEAK_LOW <= years <= config.EXPERIENCE_PEAK_HIGH:
        return 1.0
    if 5.0 <= years < 6.0 or 8.0 < years <= 9.0:
        return 0.7
    if 4.0 <= years < 5.0 or 9.0 < years <= 11.0:
        return 0.3
    if 3.0 <= years < 4.0 or 11.0 < years <= 13.0:
        return -0.1
    if years < 3.0:
        return -0.6
    return -0.4


def _compute_location_fit(candidate: dict) -> tuple[float, str]:
    """Score location fit against Pune/Noida office preference.

    Location tiers:
        1.0 — Pune or Noida (office city)
        0.6 — India Tier-1 city
        0.3 — Elsewhere in India
       -0.1 — Abroad but willing to relocate
       -0.6 — Abroad and not willing to relocate
    """
    candidate_profile = data_access.get_profile(candidate)
    location = str(candidate_profile.get("location", "")).lower()
    country = str(candidate_profile.get("country", "")).lower()
    redrob_signals = data_access.get_redrob_signals(candidate)
    willing_to_relocate = bool(redrob_signals.get("willing_to_relocate"))

    if any(city in location for city in config.PUNE_NOIDA_CITIES):
        return 1.0, "Pune/Noida (office location)"
    if any(city in location for city in config.INDIA_TIER_ONE_CITIES):
        return 0.6, "India Tier-1 city"
    if country == "india":
        return 0.3, "elsewhere in India"
    if willing_to_relocate:
        return -0.1, "outside India, willing to relocate"
    return -0.6, "outside India, not open to relocation"


def _compute_notice_fit(candidate: dict) -> float:
    """Score notice period fit: shorter is better for the JD's sub-30d preference."""
    notice = data_access.get_redrob_signals(candidate).get("notice_period_days")
    if not isinstance(notice, (int, float)):
        return 0.0
    if notice <= 30:
        return 0.6
    if notice <= 60:
        return 0.2
    if notice <= 90:
        return 0.0
    if notice <= 150:
        return -0.3
    return -0.5


def _compute_product_fit(candidate: dict) -> float:
    """Score product vs. consulting background fit from evidence text.

    The JD explicitly says "not pure services" — product-company experience
    is preferred. Consulting backgrounds are penalized.
    """
    combined_text = " ".join(data_access.get_evidence_text_blocks(candidate)).lower()
    has_product_background = any(marker in combined_text for marker in config.PRODUCT_COMPANY_MARKERS)
    has_consulting_background = config.CONSULTING_COMPANY_MARKER in combined_text
    if has_product_background and not has_consulting_background:
        return 0.4
    if has_consulting_background and not has_product_background:
        return -0.5
    return 0.0


def _compute_role_duration_months(role: dict) -> float:
    """Compute the duration of a single role in months.

    Uses the explicit ``duration_months`` field when available, falling
    back to date arithmetic on ``start_date`` and ``end_date``.
    """
    explicit_duration = role.get("duration_months")
    if isinstance(explicit_duration, (int, float)) and explicit_duration > 0:
        return float(explicit_duration)
    months = data_access.months_between_dates(
        role.get("start_date"), role.get("end_date")
    )
    return float(months) if months and months > 0 else 0.0


def _compute_tenure_fit(candidate: dict) -> float:
    """Score tenure stability based on average role duration.

    The JD warns against "switching companies every 1.5 years" and wants
    someone who "plans to be here for 3+ years". A single role is treated
    as neutral (no hopping pattern to observe).
    """
    durations = [
        _compute_role_duration_months(role)
        for role in data_access.get_career_history(candidate)
    ]
    valid_durations = [d for d in durations if d > 0]

    if len(valid_durations) < 2:
        return 0.0

    average_duration = sum(valid_durations) / len(valid_durations)
    if average_duration >= config.TENURE_EXCELLENT_MONTHS:
        return 1.0
    if average_duration >= config.TENURE_GOOD_MONTHS:
        return 0.5
    if average_duration >= config.TENURE_NEUTRAL_MONTHS:
        return 0.0
    if average_duration >= config.TENURE_MINIMUM_MONTHS:
        return -0.5
    return -0.8


def _compute_ml_depth_fit(candidate: dict) -> float:
    """Score how many years the candidate spent in ML roles.

    The JD specifies "4-5 years in applied ML/AI roles". A role counts as
    ML time if its career description grades at tier 3+ (evidence grade >=
    0.36) in the evidence scorer. The scorer is memoized, so re-scoring
    descriptions here adds negligible cost.
    """
    ml_duration_months = 0.0
    for role in data_access.get_career_history(candidate):
        description = role.get("description")
        if isinstance(description, str):
            text_grade = grade_text_block(description)
            if text_grade.grade >= config.ML_ROLE_GRADE_THRESHOLD:
                ml_duration_months += _compute_role_duration_months(role)

    ml_years = ml_duration_months / 12.0
    if ml_years >= config.ML_DEPTH_EXCELLENT_YEARS:
        return 1.0
    if ml_years >= config.ML_DEPTH_GOOD_YEARS:
        return 0.6
    if ml_years >= config.ML_DEPTH_FAIR_YEARS:
        return 0.2
    if ml_years >= config.ML_DEPTH_MINIMUM_YEARS:
        return -0.1
    return -0.4


def _compute_skill_trust(candidate: dict) -> float:
    """Score trust in claimed skill proficiency via assessment scores.

    Never penalizes missing assessments (many genuine candidates have none).
    Only rewards or lightly penalizes demonstrated, verified skill depth.
    Maps average assessment score linearly: 50 -> 0, 80 -> ~0.6, 30 -> ~-0.4.
    """
    scores = data_access.get_redrob_signals(candidate).get("skill_assessment_scores") or {}
    if not isinstance(scores, dict) or not scores:
        return 0.0
    valid_scores = [v for v in scores.values() if isinstance(v, (int, float))]
    if not valid_scores:
        return 0.0
    average = sum(valid_scores) / len(valid_scores)
    return max(-0.5, min(1.0, (average - 50.0) / 50.0))


def compute_jd_fit_score(candidate: dict) -> JobFitScore:
    """Compute the full JD logistics fit score for a candidate.

    Blends seven sub-scores (experience, location, notice, product,
    skill_trust, tenure, ml_depth) using the weights defined in
    ``configuration.JD_FIT_SUBSCORE_WEIGHTS``. The result is clamped to
    [-1, 1] and represents a small correction on top of the evidence grade.

    Args:
        candidate: A raw candidate dictionary.

    Returns:
        A ``JobFitScore`` with the blended score and all sub-scores.
    """
    candidate_profile = data_access.get_profile(candidate)
    years = candidate_profile.get("years_of_experience")
    years = float(years) if isinstance(years, (int, float)) else 0.0

    location_score, location_label = _compute_location_fit(candidate)

    sub_scores = {
        "experience": _compute_experience_fit(years),
        "location": location_score,
        "notice": _compute_notice_fit(candidate),
        "product": _compute_product_fit(candidate),
        "skill_trust": _compute_skill_trust(candidate),
        "tenure": _compute_tenure_fit(candidate),
        "ml_depth": _compute_ml_depth_fit(candidate),
    }

    total_weight = sum(config.JD_FIT_SUBSCORE_WEIGHTS.values())
    blended = sum(
        config.JD_FIT_SUBSCORE_WEIGHTS[k] * v
        for k, v in sub_scores.items()
    ) / total_weight
    blended = max(-1.0, min(1.0, blended))

    return JobFitScore(
        score=round(blended, 6),
        experience=sub_scores["experience"],
        location=location_score,
        notice=sub_scores["notice"],
        product=sub_scores["product"],
        skill_trust=sub_scores["skill_trust"],
        tenure=sub_scores["tenure"],
        ml_depth=sub_scores["ml_depth"],
        location_label=location_label,
        breakdown=sub_scores,
    )

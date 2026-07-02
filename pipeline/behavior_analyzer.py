"""Behavioral availability multiplier.

The JD makes clear that a perfect-on-paper candidate who hasn't logged in for
6 months and ignores 95% of recruiter messages isn't actually available. This
module produces a multiplier (roughly in [0.6, 1.1]) combining activity
recency, recruiter response rate, open-to-work status, and interview completion
rate.

The multiplier is kept deliberately narrow — it should reorder candidates
within a tier, not let a hyperactive Tier-2 candidate leapfrog a quiet Tier-5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import configuration as config
from . import data_access
from .time_anchor import TimeAnchor


@dataclass
class BehavioralScore:
    """Result of behavioral availability analysis for a candidate."""

    multiplier: float
    months_since_last_active: float
    component_scores: dict = field(default_factory=dict)


def _compute_recency_component(
    candidate: dict, anchor: TimeAnchor
) -> tuple[float, float]:
    """Compute recency component and months since last active.

    Returns ``(component_value, months_since_active)`` where:
        - component is in [-1, 1], with <=1 month = +1, ~6 months = 0, >=12 months = -1
        - months_since_active is a float or -1 if unknown
    """
    last_active = data_access.get_redrob_signals(candidate).get("last_active_date")
    months = data_access.months_between_dates(last_active, anchor.date)
    if months is None:
        return 0.0, -1.0
    months = max(0.0, float(months))
    component = max(-1.0, min(1.0, 1.0 - months / 6.0))
    return component, months


def _compute_response_component(candidate: dict) -> float:
    """Score recruiter response rate: 50% is neutral, 90% is +0.8, 10% is -0.8."""
    rate = data_access.get_redrob_signals(candidate).get("recruiter_response_rate")
    if not isinstance(rate, (int, float)):
        return 0.0
    return max(-1.0, min(1.0, (rate - 0.5) / 0.5))


def _compute_open_to_work_component(candidate: dict) -> float:
    """Score open-to-work flag: +1 if open, -0.4 if not."""
    return 1.0 if data_access.get_redrob_signals(candidate).get("open_to_work_flag") else -0.4


def _compute_interview_component(candidate: dict) -> float:
    """Score interview completion rate: 50% is neutral."""
    rate = data_access.get_redrob_signals(candidate).get("interview_completion_rate")
    if not isinstance(rate, (int, float)):
        return 0.0
    return max(-1.0, min(1.0, (rate - 0.5) / 0.5))


def compute_behavioral_multiplier(candidate: dict, anchor: TimeAnchor) -> BehavioralScore:
    """Compute a behavioral availability multiplier for the candidate.

    Blends four components (recency, response rate, open-to-work, interview
    rate) into a single multiplier in [MIN_MULT, MAX_MULT]. A perfectly
    neutral profile maps to ~1.0 (no adjustment).

    Args:
        candidate: A raw candidate dictionary.
        anchor: The data-derived time anchor.

    Returns:
        A ``BehavioralScore`` with the final multiplier and component details.
    """
    recency, months = _compute_recency_component(candidate, anchor)
    response = _compute_response_component(candidate)
    open_to_work = _compute_open_to_work_component(candidate)
    interview = _compute_interview_component(candidate)

    component_scores = {
        "recency": recency,
        "response": response,
        "open_to_work": open_to_work,
        "interview": interview,
    }

    total_weight = sum(config.BEHAVIOR_COMPONENT_WEIGHTS.values())
    blended = sum(
        config.BEHAVIOR_COMPONENT_WEIGHTS[k] * v
        for k, v in component_scores.items()
    ) / total_weight
    blended = max(-1.0, min(1.0, blended))

    if blended >= 0:
        multiplier = 1.0 + blended * (config.BEHAVIOR_MAXIMUM_MULTIPLIER - 1.0)
    else:
        multiplier = 1.0 + blended * (1.0 - config.BEHAVIOR_MINIMUM_MULTIPLIER)

    multiplier = round(
        max(config.BEHAVIOR_MINIMUM_MULTIPLIER,
            min(config.BEHAVIOR_MAXIMUM_MULTIPLIER, multiplier)),
        6,
    )

    return BehavioralScore(
        multiplier=multiplier,
        months_since_last_active=round(months, 1),
        component_scores=component_scores,
    )

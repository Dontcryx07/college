"""Honeypot and keyword-stuffer trap detection.

The competition specification warns of ~80 honeypot candidates with subtly
impossible profiles, and states that a >10% honeypot rate in the top 100
results in disqualification. This module detects two reliable internal
inconsistencies:

  1. **Experience inflation**: The stated ``years_of_experience`` exceeds the
     actual career-history span by more than the configured threshold.
  2. **Phantom expertise**: Multiple skills claimed at "expert" proficiency
     with zero months of logged usage.

Additionally, **keyword stuffing** (many AI/ML skill names listed but no
evidence of that work in career descriptions) is flagged as a soft signal
used for reasoning and within-tier penalties, not hard exclusion.

Early experimentation showed that company-name-based checks (founding dates,
product/consulting identity) are unreliable because company assignment in the
synthetic dataset is completely independent of role content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import configuration as config
from . import data_access
from .time_anchor import TimeAnchor


# Pattern to detect AI/ML skill names in the skills list. Used to identify
# keyword-stuffers who list dense AI skills but whose described work shows
# no corresponding evidence.
_AI_SKILL_NAME_PATTERN: re.Pattern = re.compile(
    r"(?<![a-z0-9])(?:"
    r"nlp|llm|rag|ml|machine learning|deep learning|pytorch|tensorflow|"
    r"fine-?tuning|fine-?tuning llms|langchain|pinecone|faiss|milvus|weaviate|"
    r"qdrant|vector|embedding|embeddings|transformer|transformers|hugging ?face|"
    r"bert|gpt|lora|semantic search|recommendation|ranking|learning to rank|"
    r"information retrieval|mlops|feature engineering|gans|speech recognition|"
    r"image classification|object detection|tts|statistical modeling"
    r")(?![a-z0-9])",
    re.IGNORECASE,
)


@dataclass
class TrapDetectionResult:
    """Result of honeypot and keyword-stuffer detection for a candidate."""

    is_honeypot: bool
    honeypot_reasons: list[str] = field(default_factory=list)
    is_keyword_stuffer: bool = False
    ai_skill_count: int = 0
    years_of_experience_gap: float = 0.0
    phantom_expert_count: int = 0


def _compute_career_span_years(candidate: dict, anchor: TimeAnchor) -> float:
    """Compute the number of years from earliest role start to the anchor date.

    Returns 0.0 when no role has a parseable start date.
    """
    start_months = [
        data_access.month_index_from_date(role.get("start_date"))
        for role in data_access.get_career_history(candidate)
    ]
    valid_starts = [month for month in start_months if month is not None]
    if not valid_starts:
        return 0.0
    return max(0.0, (anchor.month_index - min(valid_starts)) / 12.0)


def _count_phantom_expert_skills(candidate: dict) -> int:
    """Count skills claimed 'expert' with exactly 0 months of use."""
    count = 0
    for skill in data_access.get_skills(candidate):
        proficiency = str(skill.get("proficiency", "")).lower()
        duration = skill.get("duration_months")
        if proficiency == "expert" and duration == 0:
            count += 1
    return count


def _count_ai_skill_names(candidate: dict) -> int:
    """Count skills whose names match AI/ML keyword patterns."""
    count = 0
    for skill in data_access.get_skills(candidate):
        name = str(skill.get("name", ""))
        if _AI_SKILL_NAME_PATTERN.search(name):
            count += 1
    return count


def detect_trap_signals(
    candidate: dict,
    anchor: TimeAnchor,
    evidence_grade: float,
) -> TrapDetectionResult:
    """Detect honeypot and keyword-stuffer traps for a single candidate.

    Args:
        candidate: A raw candidate dictionary.
        anchor: The data-derived time anchor.
        evidence_grade: The candidate's aggregated evidence grade from
            the evidence scorer, used to identify keyword-stuffers (many AI
            skill names listed but no evidence in described work).

    Returns:
        A ``TrapDetectionResult`` indicating whether the candidate is a
        honeypot (to be excluded from the shortlist) and whether they exhibit
        keyword-stuffer patterns (soft penalty).
    """
    reasons: list[str] = []

    candidate_profile = data_access.get_profile(candidate)
    stated_years = candidate_profile.get("years_of_experience")
    stated_years = float(stated_years) if isinstance(stated_years, (int, float)) else 0.0

    career_span_years = _compute_career_span_years(candidate, anchor)
    experience_gap = stated_years - career_span_years

    if career_span_years > 0.0 and experience_gap > config.YOE_INFLATION_GAP_THRESHOLD:
        reasons.append(
            f"stated {stated_years:.1f}y experience vs only "
            f"~{career_span_years:.1f}y of career history"
        )

    phantom_experts = _count_phantom_expert_skills(candidate)
    if phantom_experts >= config.PHANTOM_EXPERT_MINIMUM_COUNT:
        reasons.append(
            f"{phantom_experts} skills claimed 'expert' with 0 months of use"
        )

    ai_skill_count = _count_ai_skill_names(candidate)
    is_keyword_stuffer = (
        ai_skill_count >= config.AI_SKILL_STUFFER_THRESHOLD
        and evidence_grade < config.KEYWORD_STUFFER_EVIDENCE_CEILING
    )

    return TrapDetectionResult(
        is_honeypot=bool(reasons),
        honeypot_reasons=reasons,
        is_keyword_stuffer=is_keyword_stuffer,
        ai_skill_count=ai_skill_count,
        years_of_experience_gap=round(experience_gap, 2),
        phantom_expert_count=phantom_experts,
    )

"""The evidence scorer — the core of the ranking pipeline.

This module answers the question: "Did this candidate actually do the kind of
work the JD describes?" It reads free-text career descriptions (plus headline
and summary as minor signals) and produces a 0-1 evidence grade based on how
closely the work matches IR/ranking/recsys/NLP concepts.

Key design principles:
  1. Evidence is read from career *descriptions* — never the skills list,
     which is the keyword-stuffer trap surface the JD warns about.
  2. Matching uses word-boundary regex, not substring search, so short tokens
     like "rag" or "ltr" don't match inside unrelated words ("storage", "filter").
  3. The text scorer is memoized: the pool contains only ~44 distinct description
     templates, so caching collapses ~300k scoring calls into a few thousand.
  4. Aggregation at the candidate level uses the RECENCY-ADJUSTED best role
     (evidence is about strongest demonstrated work), with a small bonus for
     a second independent strong role.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field
from typing import Pattern

from . import configuration as config
from . import data_access
from .time_anchor import TimeAnchor


# ============================================================================
# Regex compilation (word-boundary matching)
# ============================================================================

def _compile_word_boundary_alternation(phrases: list[str]) -> Pattern[str]:
    """Compile a regex that matches any of the phrases at word boundaries.

    Phrases are sorted longest-first so the alternation prefers the most
    specific match (e.g. "learning-to-rank" before "rank").
    """
    ordered = sorted(set(phrases), key=len, reverse=True)
    body = "|".join(re.escape(phrase) for phrase in ordered)
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")


_POSITIVE_FAMILY_PATTERNS: dict[str, tuple[float, Pattern[str]]] = {
    name: (weight, _compile_word_boundary_alternation(phrases))
    for name, (weight, phrases) in config.POSITIVE_SIGNAL_FAMILIES.items()
}
_NONTECH_PATTERN = _compile_word_boundary_alternation(config.NONTECH_KEYWORD_MARKERS)
_CV_PRIMARY_PATTERN = _compile_word_boundary_alternation(config.CV_PRIMARY_KEYWORD_MARKERS)
_DISCLAIMER_PATTERN = _compile_word_boundary_alternation(config.HONESTY_DISCLAIMER_MARKERS)


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for robust pattern matching."""
    return " ".join(text.lower().split())


# ============================================================================
# Text scoring (memoized)
# ============================================================================

@dataclass(frozen=True)
class TextEvidenceGrade:
    """Scoring result for a single free-text block."""

    grade: float
    matched_families: tuple[str, ...]
    nontech_phrase: str | None = None
    is_cv_primary: bool = False
    has_disclaimer: bool = False


@functools.cache
def grade_text_block(text: str) -> TextEvidenceGrade:
    """Grade a single free-text block for JD evidence relevance.

    This is a pure function and is memoized because the pool's ~300k role
    descriptions collapse to only ~44 unique strings (plus a few thousand
    unique headlines/summaries), making caching extremely effective.

    Args:
        text: A free-text block (headline, summary, or career description).

    Returns:
        A ``TextEvidenceGrade`` with the computed evidence grade [0, 1] and
        metadata about which concept families were matched.
    """
    if not text or not isinstance(text, str):
        return TextEvidenceGrade(grade=0.0, matched_families=())

    normalized = _normalize_text(text)

    matched_families: dict[str, float] = {}
    for family_name, (weight, pattern) in _POSITIVE_FAMILY_PATTERNS.items():
        if pattern.search(normalized):
            matched_families[family_name] = weight

    floor_only = matched_families.get("general_engineering") is not None
    raw_primary = sum(
        weight
        for name, weight in matched_families.items()
        if name not in config.FLOOR_ONLY_FAMILIES
    )
    grade = 1.0 - math.exp(-config.EVIDENCE_SQUASH_K * raw_primary)
    if floor_only:
        grade = max(grade, config.GENERAL_ENGINEERING_FLOOR)

    disclaimer_match = _DISCLAIMER_PATTERN.search(normalized)
    if disclaimer_match:
        grade *= config.HONESTY_DISCLAIMER_MULTIPLIER

    nontech_match = _NONTECH_PATTERN.search(normalized)
    nontech_phrase = nontech_match.group(0) if nontech_match else None
    if nontech_match:
        grade *= config.NONTECH_KEYWORD_MULTIPLIER

    cv_match = _CV_PRIMARY_PATTERN.search(normalized)
    is_cv_primary = cv_match is not None
    has_core = any(f in matched_families for f in config.CORE_EVIDENCE_FAMILIES)
    if is_cv_primary and not has_core:
        grade = min(grade, config.CV_PRIMARY_GRADE_CAP)

    return TextEvidenceGrade(
        grade=round(grade, 6),
        matched_families=tuple(sorted(matched_families.keys())),
        nontech_phrase=nontech_phrase,
        is_cv_primary=is_cv_primary,
        has_disclaimer=disclaimer_match is not None,
    )


# ============================================================================
# Recency discount per role
# ============================================================================

def _compute_role_recency_discount(role: dict, anchor: TimeAnchor) -> float:
    """Compute a recency discount factor in [0.6, 1.0] for a single role.

    Roles that ended recently count fully; older roles are discounted. Current
    roles (no end date) always score 1.0. The discount ramps from 1.0 at 3
    years ago down to 0.6 at 10+ years ago.

    This directly addresses the JD's concern about candidates who "haven't
    written production code in 18 months."
    """
    if role.get("is_current"):
        return 1.0
    end_month = data_access.month_index_from_date(role.get("end_date"))
    if end_month is None:
        return 0.9
    years_since_last_active = max(0.0, (anchor.month_index - end_month) / 12.0)
    return max(0.6, min(1.0, 1.0 - 0.04 * max(0.0, years_since_last_active - 3.0)))


# ============================================================================
# Candidate-level evidence aggregation
# ============================================================================

@dataclass
class AggregatedCandidateEvidence:
    """Aggregated evidence across all roles for a single candidate."""

    grade: float
    best_role_index: int
    best_role_raw_grade: float
    matched_families: tuple[str, ...]
    per_role_grades: list[float] = field(default_factory=list)
    strong_role_count: int = 0
    is_cv_primary_only: bool = False


def aggregate_candidate_evidence(candidate: dict, anchor: TimeAnchor) -> AggregatedCandidateEvidence:
    """Score and aggregate evidence across all roles for a candidate.

    Aggregation strategy:
      - The best (recency-adjusted) role sets the candidate's grade.
      - A second strong role (grade >= 0.58) adds a small bonus (+0.03).
      - Headline/summary can lift (never lower) the grade slightly.
      - The final grade is clamped to [0, 1].

    Args:
        candidate: A raw candidate dictionary.
        anchor: The data-derived time anchor for recency calculations.

    Returns:
        An ``AggregatedCandidateEvidence`` with the candidate's final evidence
        grade and supporting metadata.
    """
    roles = data_access.get_career_history(candidate)
    per_role_grades: list[float] = []
    best_families: tuple[str, ...] = ()
    best_role_idx = -1
    best_adjusted_grade = 0.0
    best_raw_grade = 0.0
    strong_role_count = 0
    cv_flag_count = 0
    core_or_ml_role_count = 0

    for idx, role in enumerate(roles):
        result = grade_text_block(role.get("description", ""))
        per_role_grades.append(result.grade)
        recency_discount = _compute_role_recency_discount(role, anchor)
        adjusted_grade = result.grade * recency_discount
        if result.grade >= config.STRONG_ROLE_GRADE_THRESHOLD:
            strong_role_count += 1
        if result.grade >= config.ML_ROLE_GRADE_THRESHOLD:
            core_or_ml_role_count += 1
        if result.is_cv_primary:
            cv_flag_count += 1
        if adjusted_grade > best_adjusted_grade:
            best_adjusted_grade = adjusted_grade
            best_raw_grade = result.grade
            best_families = result.matched_families
            best_role_idx = idx

    candidate_profile = data_access.get_profile(candidate)
    auxiliary_grade = 0.0
    for block in (candidate_profile.get("headline"), candidate_profile.get("summary")):
        if isinstance(block, str):
            auxiliary_grade = max(auxiliary_grade, grade_text_block(block).grade)

    base_grade = best_adjusted_grade
    if strong_role_count >= 2:
        base_grade = min(1.0, base_grade + 0.03)

    final_grade = max(
        base_grade,
        min(base_grade + 0.05, auxiliary_grade * 0.5),
    )
    final_grade = round(min(1.0, final_grade), 6)

    return AggregatedCandidateEvidence(
        grade=final_grade,
        best_role_index=best_role_idx,
        best_role_raw_grade=round(best_raw_grade, 6),
        matched_families=best_families,
        per_role_grades=[round(g, 6) for g in per_role_grades],
        strong_role_count=strong_role_count,
        is_cv_primary_only=(cv_flag_count > 0 and core_or_ml_role_count == 0),
    )

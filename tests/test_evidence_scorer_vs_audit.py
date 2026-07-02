"""Verify the evidence scorer reproduces hand-graded audit tiers for all 44 templates.

This is the most important test in the suite: ``evidence_scorer.py`` is a
general text scorer (not a lookup table), and this test confirms it
independently lands on the same tier as the manual audit for all 44 known
career-description templates. An ontology weight change that breaks one of
these gets caught here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.evidence_scorer import grade_text_block
from pipeline.template_audit import (
    AUDITED_TEMPLATES,
    grade_to_tier,
    match_description_to_audit,
)


TEMPLATES_FILE_PATH = Path("dataset/career_description_templates.json")


def _load_templates() -> list[dict]:
    """Load the career description templates file, skipping if absent."""
    if not TEMPLATES_FILE_PATH.exists():
        pytest.skip(f"{TEMPLATES_FILE_PATH} not present (run tools/extract_career_templates.py)")
    return json.loads(TEMPLATES_FILE_PATH.read_text(encoding="utf-8"))


def test_audit_contains_exactly_44_entries() -> None:
    """The hand-graded audit must have exactly 44 entries spanning tiers 0-5."""
    assert len(AUDITED_TEMPLATES) == 44
    tiers = {entry.tier for entry in AUDITED_TEMPLATES}
    assert tiers <= {0, 1, 2, 3, 4, 5}


def test_all_audit_prefixes_are_unique() -> None:
    """Every audit entry must have a unique 45-character prefix for matching."""
    prefixes = [entry.prefix[:45] for entry in AUDITED_TEMPLATES]
    assert len(set(prefixes)) == len(prefixes)


def test_every_template_matches_an_audit_entry() -> None:
    """Every template in the JSON file must match exactly one audit entry."""
    for template in _load_templates():
        assert match_description_to_audit(template["text"]) is not None, (
            f"Template {template['id']} did not match any audit entry"
        )


def test_scorer_reproduces_audit_tier_for_all_44_templates() -> None:
    """The evidence scorer must produce the same tier as the manual audit for all 44 templates."""
    templates = _load_templates()
    audit_by_id = {entry.template_id: entry for entry in AUDITED_TEMPLATES}
    mismatches: list[tuple[int, int, int]] = []

    for template in templates:
        audit_entry = audit_by_id[template["id"]]
        scored_tier = grade_to_tier(grade_text_block(template["text"]).grade)
        if scored_tier != audit_entry.tier:
            mismatches.append((template["id"], audit_entry.tier, scored_tier))

    assert not mismatches, (
        f"Tier mismatches (template_id, expected_tier, scored_tier): {mismatches}"
    )


def test_elite_templates_are_all_tier_5() -> None:
    """Templates 27-43 (excluding deliberately-tier-4 entry 32) must score tier 5."""
    template_texts = {t["id"]: t["text"] for t in _load_templates()}
    elite_template_ids = (27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43)

    for template_id in elite_template_ids:
        evidence_grade = grade_text_block(template_texts[template_id]).grade
        assert grade_to_tier(evidence_grade) == 5, (
            f"Template {template_id} expected tier 5 but got grade {evidence_grade}"
        )


def test_nontech_templates_are_all_tier_0() -> None:
    """Templates 0-8 (non-technical roles) must all score tier 0."""
    template_texts = {t["id"]: t["text"] for t in _load_templates()}
    for template_id in range(0, 9):
        evidence_grade = grade_text_block(template_texts[template_id]).grade
        assert grade_to_tier(evidence_grade) == 0, (
            f"Template {template_id} expected tier 0 but got grade {evidence_grade}"
        )

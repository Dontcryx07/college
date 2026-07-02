"""Side-by-side comparison of the evidence scorer output vs the hand-graded audit.

Used during weight tuning in ``pipeline/evidence_scorer.py``: adjust an
ontology weight, rerun this script, and see which of the 44 templates moved
tier. Exits with a non-zero status on any mismatch so it doubles as a quick
validation check.

Usage:
    python tools/calibrate_scorer_vs_audit.py \
        --templates ./dataset/career_description_templates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.evidence_scorer import grade_text_block
from pipeline.template_audit import (
    AUDITED_TEMPLATES,
    grade_to_tier,
    match_description_to_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--templates",
        default="./dataset/career_description_templates.json",
        type=Path,
    )
    args = parser.parse_args(argv)

    templates = json.loads(args.templates.read_text(encoding="utf-8"))
    audit_by_id = {entry.template_id: entry for entry in AUDITED_TEMPLATES}

    mismatch_count = 0
    header = f"{'T':>3} {'aud_tier':>8} {'scr_tier':>8} {'aud_grd':>8} {'scr_grd':>8}  families"
    print(header)
    print("-" * 88)

    for template in sorted(templates, key=lambda t: t["id"]):
        text = template["text"]
        audit_entry = audit_by_id.get(template["id"])
        scoring_result = grade_text_block(text)
        scored_tier = grade_to_tier(scoring_result.grade)
        is_match = audit_entry is not None and scored_tier == audit_entry.tier
        flag = "" if is_match else "  <-- MISMATCH"
        if not is_match:
            mismatch_count += 1

        audited_tier = audit_entry.tier if audit_entry else -1
        audited_grade = audit_entry.target_grade if audit_entry else -1.0
        families = ",".join(scoring_result.matched_families)
        extra_flags = []
        if scoring_result.nontech_phrase:
            extra_flags.append(f"NONTECH:{scoring_result.nontech_phrase}")
        if scoring_result.is_cv_primary:
            extra_flags.append("CV")
        if scoring_result.has_disclaimer:
            extra_flags.append("DISC")

        print(
            f"T{template['id']:02d} {audited_tier:>8} {scored_tier:>8} "
            f"{audited_grade:>8.2f} {scoring_result.grade:>8.3f}  "
            f"{families} {' '.join(extra_flags)}{flag}"
        )

    print("-" * 88)
    print(f"mismatches: {mismatch_count} / {len(templates)}")

    unmatched = [
        t["id"] for t in templates if match_description_to_audit(t["text"]) is None
    ]
    if unmatched:
        print(f"PREFIX MATCH FAILURES for template ids: {unmatched}")
        mismatch_count += len(unmatched)

    return 1 if mismatch_count else 0


if __name__ == "__main__":
    sys.exit(main())

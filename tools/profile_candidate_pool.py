"""Sanity-check the candidate pool: grade distribution, honeypot examples, clean strong candidates.

Prints evidence grade distribution, timing, and a handful of honeypot + clean
examples so you can eyeball whether the numbers look right before trusting the
ranking output.

Usage:
    python tools/profile_candidate_pool.py --candidates ./dataset/candidates.jsonl
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

from pipeline import data_access, evidence_scorer, trap_detector
from pipeline.time_anchor import compute_time_anchor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="./dataset/candidates.jsonl",
        type=Path,
    )
    args = parser.parse_args(argv)

    wall_clock = time.time()
    all_candidates = data_access.load_all_candidates(args.candidates)
    load_duration = time.time() - wall_clock

    anchor = compute_time_anchor(all_candidates)

    scoring_start = time.time()
    grade_histogram: collections.Counter = collections.Counter()
    honeypot_count = 0
    strong_honeypot_count = 0
    keyword_stuffer_count = 0
    strong_candidate_count = 0
    tier_4_plus_count = 0
    honeypot_examples: list[tuple] = []
    strong_clean_examples: list[tuple] = []

    for candidate in all_candidates:
        evidence = evidence_scorer.aggregate_candidate_evidence(candidate, anchor)
        trap_result = trap_detector.detect_trap_signals(
            candidate, anchor, evidence.grade
        )
        bucket = round(evidence.grade, 1)
        grade_histogram[bucket] += 1

        is_strong = evidence.grade >= 0.78
        if is_strong:
            strong_candidate_count += 1
        if evidence.grade >= 0.58:
            tier_4_plus_count += 1

        if trap_result.is_honeypot:
            honeypot_count += 1
            if is_strong and len(honeypot_examples) < 12:
                honeypot_examples.append((
                    data_access.extract_candidate_id(candidate),
                    round(evidence.grade, 3),
                    trap_result.honeypot_reasons,
                ))
            if is_strong:
                strong_honeypot_count += 1

        if trap_result.is_keyword_stuffer:
            keyword_stuffer_count += 1

        if is_strong and not trap_result.is_honeypot and len(strong_clean_examples) < 8:
            candidate_profile = data_access.get_profile(candidate)
            strong_clean_examples.append((
                data_access.extract_candidate_id(candidate),
                round(evidence.grade, 3),
                candidate_profile.get("current_title"),
                candidate_profile.get("location"),
            ))

    scoring_duration = time.time() - scoring_start

    print(f"loaded {len(all_candidates)} candidates in {load_duration:.1f}s; "
          f"scored in {scoring_duration:.1f}s")
    print(f"time anchor: {anchor.date}")
    print(f"strong (evidence >= 0.78, ~tier 5): {strong_candidate_count}")
    print(f"tier 4+ (evidence >= 0.58): {tier_4_plus_count}")
    print(f"honeypots flagged: {honeypot_count} "
          f"(of which strong-pool: {strong_honeypot_count})")
    print(f"keyword stuffers flagged: {keyword_stuffer_count}")
    print("\nevidence grade histogram (rounded to 0.1):")
    for bucket in sorted(grade_histogram):
        print(f"  {bucket:>4}: {grade_histogram[bucket]}")
    print("\nstrong-pool honeypot examples (excluded from top-100):")
    for candidate_id, grade, reasons in honeypot_examples:
        print(f"  {candidate_id} grade={grade}: {reasons}")
    print("\nclean strong-pool examples (eligible):")
    for candidate_id, grade, title, location in strong_clean_examples:
        print(f"  {candidate_id} grade={grade} | {title} | {location}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

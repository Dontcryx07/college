#!/usr/bin/env python3
"""Candidate ranking entrypoint — produces the top-100 submission CSV.

Usage:
    python rank.py --candidates ./dataset/candidates.jsonl --out ./submission.csv

Runs on CPU only, with no network access and no precomputed artifacts, in
well under the 5-minute competition budget. The output CSV matches the
submission specification exactly with columns: candidate_id, rank, score,
reasoning.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from pipeline import composite_scorer, data_access, reasoning_generator, time_anchor


def build_output_rows(
    scored_candidates: list[composite_scorer.ScoredCandidateResult],
) -> list[dict]:
    """Convert scored candidates into CSV-ready row dicts with reasoning."""
    rows: list[dict] = []
    for position, scored in enumerate(scored_candidates, start=1):
        rows.append({
            "candidate_id": scored.candidate_id,
            "rank": position,
            "score": f"{scored.composite_score:.6f}",
            "reasoning": reasoning_generator.generate_reasoning_text(scored, position),
        })
    return rows


def write_submission_csv(rows: list[dict], output_path: Path) -> None:
    """Write the ranked rows to a CSV file at ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_command_line_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="Path to candidates.jsonl (plain text or gzipped).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to write the submission CSV.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of candidates to include in the ranking (default 100).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load, score, rank, and export the top-N candidates.

    Returns 0 on success, 1 on failure.
    """
    arguments = parse_command_line_arguments(argv)

    wall_clock_start = time.time()

    candidates = data_access.load_all_candidates(arguments.candidates)
    anchor = time_anchor.compute_time_anchor(candidates)
    load_duration = time.time() - wall_clock_start

    ranking_start = time.time()
    scored = composite_scorer.rank_candidates(
        candidates, anchor, top_n=arguments.top, exclude_honeypots=True
    )
    rows = build_output_rows(scored)
    write_submission_csv(rows, arguments.out)
    ranking_duration = time.time() - ranking_start

    honeypot_count = sum(1 for s in scored if s.trap.is_honeypot)
    evidence_tiers = [s.evidence_tier for s in scored]

    print(
        f"[rank] loaded {len(candidates)} candidates "
        f"(anchor {anchor.date}) in {load_duration:.1f}s"
    )
    print(
        f"[rank] ranked top-{len(scored)} in {ranking_duration:.1f}s "
        f"(total {time.time() - wall_clock_start:.1f}s)"
    )
    print(f"[rank] honeypots in output: {honeypot_count} (must be 0)")
    if scored:
        print(
            f"[rank] top score {scored[0].composite_score:.4f} -> "
            f"bottom score {scored[-1].composite_score:.4f}"
        )
        print(
            f"[rank] evidence tiers in top-{len(scored)}: "
            f"min={min(evidence_tiers)} max={max(evidence_tiers)}"
        )
    print(f"[rank] wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

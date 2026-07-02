"""Run the proxy evaluation with ablations and sensitivity sweeps.

Used to verify that honeypot exclusion, JD-fit weights, and behavioral
bounds actually affect the metrics before committing to production settings.

Usage:
    python tools/run_evaluation_sweep.py --candidates ./dataset/candidates.jsonl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pipeline import (
    behavior_analyzer,
    composite_scorer,
    data_access,
    evaluation_harness,
    job_fit_scorer,
)
from pipeline.time_anchor import compute_time_anchor


def _get_ranked_candidate_ids(candidates, anchor, **kwargs):
    scored = composite_scorer.rank_candidates(candidates, anchor, top_n=100, **kwargs)
    return [s.candidate_id for s in scored]


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
    anchor = compute_time_anchor(all_candidates)
    ground_truth = evaluation_harness.build_proxy_ground_truth(all_candidates, anchor)
    relevant_count = sum(
        1 for t in ground_truth.values()
        if t >= evaluation_harness.RELEVANT_TIER_THRESHOLD
    )
    tier_5_count = sum(1 for t in ground_truth.values() if t == 5)
    print(
        f"loaded {len(all_candidates)} in {time.time() - wall_clock:.1f}s | "
        f"proxy relevant(>=3): {relevant_count} | tier5: {tier_5_count}"
    )

    print("\n== Full model ==")
    metrics = evaluation_harness.evaluate_ranking(
        _get_ranked_candidate_ids(all_candidates, anchor), ground_truth
    )
    print(
        f"  NDCG@10={metrics.ndcg_at_10} NDCG@50={metrics.ndcg_at_50} "
        f"MAP={metrics.mean_average_precision} P@10={metrics.precision_at_10} "
        f"P@5={metrics.precision_at_5} => composite={metrics.composite_score}"
    )

    print("\n== Ablation: honeypots NOT excluded ==")
    metrics_with_honeypots = evaluation_harness.evaluate_ranking(
        _get_ranked_candidate_ids(all_candidates, anchor, exclude_honeypots=False),
        ground_truth,
    )
    print(
        f"  NDCG@10={metrics_with_honeypots.ndcg_at_10} "
        f"NDCG@50={metrics_with_honeypots.ndcg_at_50} "
        f"MAP={metrics_with_honeypots.mean_average_precision} "
        f"P@10={metrics_with_honeypots.precision_at_10} "
        f"=> composite={metrics_with_honeypots.composite_score}"
    )

    print("\n== Sensitivity: JD Fit ALPHA ==")
    for alpha in (0.0, 0.15, 0.30, 0.45):
        job_fit_scorer.config.JD_FIT_ALPHA = alpha
        sweep_metrics = evaluation_harness.evaluate_ranking(
            _get_ranked_candidate_ids(all_candidates, anchor), ground_truth
        )
        print(
            f"  ALPHA={alpha:>4}: composite={sweep_metrics.composite_score} "
            f"(NDCG@10={sweep_metrics.ndcg_at_10}, NDCG@50={sweep_metrics.ndcg_at_50})"
        )
    job_fit_scorer.config.JD_FIT_ALPHA = 0.30

    print("\n== Sensitivity: behavior MAX_MULT ==")
    for max_mult in (1.0, 1.10, 1.25):
        behavior_analyzer.config.BEHAVIOR_MAXIMUM_MULTIPLIER = max_mult
        sweep_metrics = evaluation_harness.evaluate_ranking(
            _get_ranked_candidate_ids(all_candidates, anchor), ground_truth
        )
        print(
            f"  MAX_MULT={max_mult:>4}: composite={sweep_metrics.composite_score} "
            f"(NDCG@10={sweep_metrics.ndcg_at_10})"
        )
    behavior_analyzer.config.BEHAVIOR_MAXIMUM_MULTIPLIER = 1.10

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

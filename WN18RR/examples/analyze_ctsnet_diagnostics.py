#!/usr/bin/env python3
"""Audit a CTS-Net query-diagnostics JSON for tie-robust gains."""

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_delta(base, fused, draws, seed):
    base = np.asarray(base, dtype=np.float64)
    fused = np.asarray(fused, dtype=np.float64)
    if base.shape != fused.shape or base.size == 0:
        raise ValueError("rank arrays must have the same nonzero length")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, base.size, size=(draws, base.size))
    deltas = (fused[indices] - base[indices]).mean(axis=1)
    return np.percentile(deltas, [2.5, 50.0, 97.5])


def metric(report, prefix, rank_key):
    return np.asarray(
        [record[prefix + rank_key] for record in report["records"]],
        dtype=np.float64,
    )


def audit_robust_gain(
    base_average_ranks,
    fused_average_ranks,
    base_optimistic_ranks,
    fused_optimistic_ranks,
    base_ties,
    fused_ties,
    draws=2000,
    seed=23,
    min_average_tie_mrr_delta=0.005,
    max_optimistic_degradation_fraction=0.50,
):
    """Apply the strict, tie-aware validation gate.

    A tie-count decrease is deliberately not a criterion: the gain must be
    visible in average-tie MRR, with a positive paired bootstrap lower bound.
    """
    base_mrr = 1.0 / np.asarray(base_average_ranks, dtype=np.float64)
    fused_mrr = 1.0 / np.asarray(fused_average_ranks, dtype=np.float64)
    delta = float(np.mean(fused_mrr) - np.mean(base_mrr))
    ci_low, ci_mid, ci_high = bootstrap_delta(
        base_mrr, fused_mrr, draws, seed
    )
    base_opt = np.asarray(base_optimistic_ranks, dtype=np.float64)
    fused_opt = np.asarray(fused_optimistic_ranks, dtype=np.float64)
    if base_opt.shape != fused_opt.shape or base_opt.shape != base_mrr.shape:
        raise ValueError("all rank arrays must have the same shape")
    degraded_fraction = float(np.mean(fused_opt > base_opt))
    base_ties = np.asarray(base_ties, dtype=np.float64)
    fused_ties = np.asarray(fused_ties, dtype=np.float64)
    if base_ties.shape != fused_ties.shape or base_ties.shape != base_mrr.shape:
        raise ValueError("tie arrays must have the same shape as rank arrays")
    result = {
        "average_tie_mrr_delta": delta,
        "average_tie_mrr_bootstrap_ci_low": float(ci_low),
        "average_tie_mrr_bootstrap_ci_mid": float(ci_mid),
        "average_tie_mrr_bootstrap_ci_high": float(ci_high),
        "optimistic_rank_degraded_fraction": degraded_fraction,
        "base_mean_target_tie_count": float(np.mean(base_ties)),
        "fused_mean_target_tie_count": float(np.mean(fused_ties)),
        "average_tie_mrr_delta_pass": bool(
            delta >= min_average_tie_mrr_delta
        ),
        "bootstrap_lower_bound_pass": bool(ci_low > 0.0),
        "optimistic_degradation_pass": bool(
            degraded_fraction <= max_optimistic_degradation_fraction
        ),
    }
    result["robust_gain"] = all(
        (
            result["average_tie_mrr_delta_pass"],
            result["bootstrap_lower_bound_pass"],
            result["optimistic_degradation_pass"],
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Audit optimistic, average-tie, and pessimistic CTS ranks"
    )
    parser.add_argument("diagnostics_json", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError("draws must be at least 100")

    report = json.loads(args.diagnostics_json.read_text(encoding="utf-8"))
    records = report.get("records", [])
    if not records:
        raise ValueError("diagnostics JSON contains no records")

    print("Checkpoint:", report.get("checkpoint"))
    print("Split:", report.get("split"), "query sides:", len(records))
    print("multiplier:", report.get("multiplier"))
    print()
    print("metric                     base          fused       delta")
    for label, rank_key in (
        ("optimistic MRR", "_rank"),
        ("average-tie MRR", "_average_rank"),
        ("pessimistic MRR", "_pessimistic_rank"),
    ):
        base_rank = metric(report, "base", rank_key)
        fused_rank = metric(report, "fused", rank_key)
        base_mrr = np.mean(1.0 / base_rank)
        fused_mrr = np.mean(1.0 / fused_rank)
        print(
            "{:<25} {:>10.6f}  {:>10.6f}  {:+10.6f}".format(
                label, base_mrr, fused_mrr, fused_mrr - base_mrr
            )
        )

    base_ties = np.asarray(
        [record["base_target_tie_count"] for record in records]
    )
    fused_ties = np.asarray(
        [record["fused_target_tie_count"] for record in records]
    )
    base_unique = np.asarray(
        [record["base_score_unique_ratio"] for record in records]
    )
    fused_unique = np.asarray(
        [record["fused_score_unique_ratio"] for record in records]
    )
    feature_unique = np.asarray(
        [record.get("feature_unique_ratio", np.nan) for record in records],
        dtype=np.float64,
    )
    feature_zero = np.asarray(
        [record.get("feature_zero_fraction", np.nan) for record in records],
        dtype=np.float64,
    )
    score_zero = np.asarray(
        [record.get("base_score_near_zero_fraction", np.nan) for record in records],
        dtype=np.float64,
    )
    score_one = np.asarray(
        [record.get("base_score_near_one_fraction", np.nan) for record in records],
        dtype=np.float64,
    )
    score_round6_unique = np.asarray(
        [record.get("base_score_round6_unique_ratio", np.nan) for record in records],
        dtype=np.float64,
    )
    target_feature_ties = np.asarray(
        [record.get("target_feature_tie_count", np.nan) for record in records],
        dtype=np.float64,
    )
    target_feature_tie_fraction = np.asarray(
        [record.get("target_feature_tie_fraction", np.nan) for record in records],
        dtype=np.float64,
    )
    audit = audit_robust_gain(
        metric(report, "base", "_average_rank"),
        metric(report, "fused", "_average_rank"),
        metric(report, "base", "_rank"),
        metric(report, "fused", "_rank"),
        base_ties,
        fused_ties,
        draws=args.draws,
        seed=args.seed,
    )
    print()
    print("tie/uniqueness audit")
    print(
        "target tie median (base/fused): {:.1f} / {:.1f}".format(
            np.median(base_ties), np.median(fused_ties)
        )
    )
    print(
        "target tie mean   (base/fused): {:.1f} / {:.1f}".format(
            np.mean(base_ties), np.mean(fused_ties)
        )
    )
    print(
        "score unique ratio mean (base/fused): {:.6f} / {:.6f}".format(
            np.mean(base_unique), np.mean(fused_unique)
        )
    )
    if np.isfinite(feature_unique).any():
        print(
            "feature unique ratio mean: {:.6f} (median {:.6f})".format(
                np.nanmean(feature_unique), np.nanmedian(feature_unique)
            )
        )
        print(
            "all-zero feature row fraction mean: {:.6f}".format(
                np.nanmean(feature_zero)
            )
        )
        print(
            "base score near 0/1 fraction mean: {:.6f} / {:.6f}".format(
                np.nanmean(score_zero), np.nanmean(score_one)
            )
        )
        print(
            "base score unique ratio after round(6) mean: {:.6f}".format(
                np.nanmean(score_round6_unique)
            )
        )
        if np.isfinite(target_feature_ties).any():
            print(
                "target same-feature class size mean/median: {:.1f} / {:.1f}".format(
                    np.nanmean(target_feature_ties),
                    np.nanmedian(target_feature_ties),
                )
            )
            print(
                "target same-feature class fraction mean: {:.6f}".format(
                    np.nanmean(target_feature_tie_fraction)
                )
            )
    print(
        "average-tie MRR paired bootstrap 95% CI: [{:+.6f}, {:+.6f}]".format(
            audit["average_tie_mrr_bootstrap_ci_low"],
            audit["average_tie_mrr_bootstrap_ci_high"],
        )
    )

    improved = np.mean(
        [record["fused_rank"] < record["base_rank"] for record in records]
    )
    degraded = np.mean(
        [record["fused_rank"] > record["base_rank"] for record in records]
    )
    print(
        "optimistic rank improved/degraded: {:.3f} / {:.3f}".format(
            improved, degraded
        )
    )
    print(
        "average-tie MRR delta (required >= +0.005): {:+.6f}".format(
            audit["average_tie_mrr_delta"]
        )
    )
    print(
        "optimistic rank degradation (required <= 0.500): {:.3f}".format(
            audit["optimistic_rank_degraded_fraction"]
        )
    )
    print(
        "ROBUST_GAIN:", "YES" if audit["robust_gain"] else "NO"
    )


if __name__ == "__main__":
    main()

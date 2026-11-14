#!/usr/bin/env python3
"""Audit train-only counterfactual matching invariants for WN18RR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow direct execution from ``examples/`` while importing the local package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openke.module.counterfactual.popularity import read_openke_triples
from openke.module.counterfactual.offline_matcher import fingerprint_triples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("counterfactual_cache/WN18RR_popularity_h1.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_evidence/wn18rr_cf_cache_audit.json"),
    )
    return parser.parse_args()


def ratio(mask):
    return float(np.mean(mask)) if len(mask) else 0.0


def main():
    args = parse_args()
    root = args.project_root.resolve()
    cache_path = args.cache if args.cache.is_absolute() else root / args.cache
    output_path = args.output if args.output.is_absolute() else root / args.output

    try:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(cache_path, map_location="cpu")

    train = read_openke_triples(root / "benchmarks/WN18RR/train2id.txt")
    triples = cache["triples"].cpu().numpy().astype(np.int64, copy=False)
    treatment = cache["treatment"].cpu().numpy().astype(np.int64, copy=False)
    cf_index = cache["cf_index"].cpu().numpy().astype(np.int64, copy=False)
    cf_triples = cache["cf_triples"].cpu().numpy().astype(np.int64, copy=False)
    cf_treatment = cache["cf_treatment"].cpu().numpy().astype(np.int64, copy=False)
    valid = cache["cf_valid_mask"].cpu().numpy().astype(bool, copy=False)
    distances = cache["cf_match_distance"].cpu().numpy().astype(np.float64, copy=False)
    features = cache["subgraph_feature"].cpu().numpy()
    cf_features = cache["cf_subgraph_feature"].cpu().numpy()

    matched = cf_index >= 0
    valid_indices = cf_index[valid]
    valid_distances = distances[valid]
    unique_cf, usage = np.unique(valid_indices, return_counts=True)
    fingerprint = fingerprint_triples(train)

    checks = {
        "cache_triples_equal_train_file": bool(np.array_equal(triples, train)),
        "recorded_fingerprint_matches_train_file": bool(
            cache["triple_fingerprint"] == fingerprint
        ),
        "all_valid_matches_have_index": bool(np.all(matched[valid])),
        "all_valid_cf_rows_are_indexed_train_rows": bool(
            np.array_equal(cf_triples[valid], triples[valid_indices])
        ),
        "all_valid_cf_features_are_indexed_train_features": bool(
            np.array_equal(cf_features[valid], features[valid_indices])
        ),
        "same_relation_rate": ratio(
            triples[valid, 2] == cf_triples[valid, 2]
        ),
        "opposite_treatment_rate": ratio(
            treatment[valid] != cf_treatment[valid]
        ),
    }

    if not all(
        value is True or (isinstance(value, float) and value == 1.0)
        for value in checks.values()
    ):
        raise AssertionError(f"Counterfactual cache audit failed: {checks}")

    report = {
        "cache_path": str(cache_path),
        "cache_version": int(cache["version"]),
        "triple_fingerprint": fingerprint,
        "total_train_triples": int(len(train)),
        "feature_dimension": int(features.shape[1]),
        "matched_count": int(matched.sum()),
        "valid_count": int(valid.sum()),
        "matched_ratio": ratio(matched),
        "valid_ratio": ratio(valid),
        "treatment_high_ratio": ratio(treatment == 1),
        "unique_valid_cf_count": int(len(unique_cf)),
        "unique_valid_cf_ratio": float(len(unique_cf) / max(1, int(valid.sum()))),
        "max_valid_cf_reuse_count": int(usage.max()) if len(usage) else 0,
        "distance_threshold": float(cache["distance_threshold"]),
        "valid_distance_mean": float(valid_distances.mean()),
        "valid_distance_median": float(np.median(valid_distances)),
        "valid_distance_p95": float(np.percentile(valid_distances, 95.0)),
        "valid_distance_max": float(valid_distances.max()),
        "exact_feature_match_ratio": ratio(np.isclose(valid_distances, 0.0)),
        "max_distance_percentile": float(cache["max_distance_percentile"]),
        "match_neighbor_count": int(cache["match_neighbor_count"]),
        "treatment_fallback_relations": list(cache["treatment_fallback_relations"]),
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

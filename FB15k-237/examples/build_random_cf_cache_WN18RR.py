"""Build a train-only random counterfactual control cache.

The control preserves each triple's relation and flips its binary treatment,
but samples the counterfactual uniformly from the opposite-treatment triples
of that relation instead of using nearest-neighbour intersection features.
The original cache's valid mask is retained so coverage is matched to the
canonical CF experiment.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from openke.module.counterfactual import load_counterfactual_cache


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="./counterfactual_cache/WN18RR_popularity_triple_h1.pt",
    )
    parser.add_argument(
        "--output",
        default="./counterfactual_cache/WN18RR_random_cf_triple_h1.pt",
    )
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main():
    args = parse_args()
    cache = load_counterfactual_cache(args.input)
    triples = cache["triples"].cpu().numpy().astype(np.int64, copy=False)
    treatment = cache["treatment"].cpu().numpy().astype(np.int64, copy=False)
    features = cache["subgraph_feature"].cpu().numpy().astype(
        np.float32, copy=False
    )
    original_index = cache["cf_index"].cpu().numpy().astype(np.int64, copy=False)
    valid = cache["cf_valid_mask"].cpu().numpy().astype(bool, copy=True)
    rng = np.random.default_rng(args.seed)

    relations = triples[:, 2]
    random_index = np.full(len(triples), -1, dtype=np.int64)
    for relation in np.unique(relations):
        rel_indices = np.flatnonzero(relations == relation)
        for regime in (0, 1):
            source = rel_indices[treatment[rel_indices] == regime]
            candidates = rel_indices[treatment[rel_indices] != regime]
            if len(source) == 0 or len(candidates) == 0:
                continue
            # Random-CF is a control for matching quality. Sampling is with
            # replacement across source rows, as in the independent nearest
            # neighbour matches used by the canonical cache.
            random_index[source] = rng.choice(candidates, size=len(source))

    matched = random_index >= 0
    if not np.any(matched):
        raise RuntimeError("No relation has both treatment regimes")
    # Keep the canonical coverage mask, but never mark a row valid when the
    # random control cannot provide an opposite-treatment candidate.
    valid &= matched

    cf_triples = np.full_like(triples, -1)
    cf_features = np.zeros_like(features, dtype=np.float32)
    cf_treatment = np.full_like(treatment, -1)
    cf_triples[matched] = triples[random_index[matched]]
    cf_features[matched] = features[random_index[matched]]
    cf_treatment[matched] = treatment[random_index[matched]]
    if np.any(relations[valid] != cf_triples[valid, 2]):
        raise AssertionError("Random CF relation mismatch")
    if np.any(treatment[valid] == cf_treatment[valid]):
        raise AssertionError("Random CF treatment did not change")

    mean = cache["feature_mean"].cpu().numpy().astype(np.float32)
    std = cache["feature_std"].cpu().numpy().astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (features - mean) / std
    normalized_cf = (cf_features - mean) / std
    distance = np.full(len(triples), np.inf, dtype=np.float32)
    distance[matched] = np.linalg.norm(
        normalized[matched] - normalized_cf[matched], axis=1
    ).astype(np.float32)
    valid_distances = distance[valid]
    threshold = float(np.percentile(valid_distances, 95.0)) if len(valid_distances) else 0.0

    output_cache = dict(cache)
    output_cache.update(
        {
            "cf_index": torch.from_numpy(random_index),
            "cf_triples": torch.from_numpy(cf_triples),
            "cf_subgraph_feature": torch.from_numpy(cf_features),
            "cf_treatment": torch.from_numpy(cf_treatment),
            "cf_match_distance": torch.from_numpy(distance),
            "cf_valid_mask": torch.from_numpy(valid),
            "distance_threshold": threshold,
            "valid_count": int(valid.sum()),
            "matched_count": int(matched.sum()),
            "match_neighbor_count": 0,
            "random_cf": True,
            "random_cf_seed": int(args.seed),
            "random_cf_source": str(args.input),
        }
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_cache, args.output)
    print("Saved random-CF cache: {}".format(args.output))
    print(
        "Coverage: {}/{} ({:.2%}); matched={}; seed={}".format(
            int(valid.sum()), len(triples), valid.mean(), int(matched.sum()), args.seed
        )
    )
    print("Random-CF relation/treatment checks passed")


if __name__ == "__main__":
    main()

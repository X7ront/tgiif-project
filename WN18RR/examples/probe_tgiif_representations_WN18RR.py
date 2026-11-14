"""Probe CF invariance and treatment decodability in a frozen TGIIF checkpoint."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "examples"))

from evaluate_ctsnet_checkpoint_WN18RR import build_predictor  # noqa: E402
from openke.module.counterfactual import load_counterfactual_cache  # noqa: E402
from openke.utils import set_seeds_all  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--cache", default="./counterfactual_cache/WN18RR_popularity_h1.pt")
    parser.add_argument("--data-path", default="./benchmarks/WN18RR/")
    parser.add_argument("--pair-sample", type=int, default=10000)
    parser.add_argument("--probe-sample", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_total(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return int(handle.readline().strip())


def build_args(args):
    return SimpleNamespace(
        architecture="treatment-aware",
        checkpoint=args.checkpoint,
        hash_hops=1,
        feature_mode="triple",
        hidden_dim=128,
        branch_dim=64,
        branch_activation="relu",
        branch_negative_slope=0.01,
        delta_limit=None,
        delta_temperature=1.0,
        delta_normalization="auto",
        delta_target_rms=None,
        gate_temperature=1.0,
        fusion_mode="invariant-margin",
        invariant_margin_fraction=0.0001,
        normalization="batch",
        floor_residual_at_baseline_min=False,
        residual_max_top_gap_ratio=None,
        residual_multiplier=0.75,
        disable_invariant_branch=False,
        disable_bias_branch=False,
        invariant_branch_multiplier=1.0,
        bias_branch_multiplier=1.0,
        device=args.device,
    )


@torch.no_grad()
def encode_indices(model, cache, indices, batch_size):
    output = {name: [] for name in ("z_shared", "z_inv", "z_bias", "gate", "invariant_delta")}
    parameter = next(model.parameters())
    for start in range(0, len(indices), batch_size):
        chosen = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long)
        triples = cache["triples"][chosen]
        data = {
            "subgraph_feature": cache["subgraph_feature"][chosen].to(parameter.device),
            "batch_h": triples[:, 0].to(parameter.device),
            "batch_t": triples[:, 1].to(parameter.device),
            "batch_r": triples[:, 2].to(parameter.device),
            "mode": "normal",
        }
        components = model.forward_components(data)
        for name in output:
            output[name].append(components[name].detach().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in output.items()}


def nearest_alternative_controls(cache, factual_indices, neighbors):
    features = cache["subgraph_feature"].cpu().numpy().astype(np.float32)
    mean = cache["feature_mean"].cpu().numpy()
    std = cache["feature_std"].cpu().numpy()
    normalized = (features - mean) / std
    triples = cache["triples"].cpu().numpy()
    treatment = cache["treatment"].cpu().numpy()
    actual = cache["cf_index"].cpu().numpy()
    controls = np.full(len(factual_indices), -1, dtype=np.int64)
    control_distance = np.full(len(factual_indices), np.nan, dtype=np.float32)
    for relation in np.unique(triples[factual_indices, 2]):
        for factual_treatment in (0, 1):
            row_mask = (
                (triples[factual_indices, 2] == relation)
                & (treatment[factual_indices] == factual_treatment)
            )
            rows = np.flatnonzero(row_mask)
            if len(rows) == 0:
                continue
            candidates = np.flatnonzero(
                (triples[:, 2] == relation) & (treatment != factual_treatment)
            )
            if len(candidates) < 2:
                continue
            count = min(max(2, neighbors), len(candidates))
            search = NearestNeighbors(n_neighbors=count, metric="euclidean")
            search.fit(normalized[candidates])
            distances, local = search.kneighbors(normalized[factual_indices[rows]])
            for offset, row in enumerate(rows):
                candidate_ids = candidates[local[offset]]
                usable = candidate_ids != actual[factual_indices[row]]
                if np.any(usable):
                    first = int(np.flatnonzero(usable)[0])
                    controls[row] = int(candidate_ids[first])
                    control_distance[row] = float(distances[offset, first])
    return controls, control_distance


def paired_bootstrap_ratio(matched, control, repetitions, seed):
    matched = np.asarray(matched, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    observed = float(matched.mean() / max(control.mean(), 1e-12))
    if repetitions <= 0:
        return {"ratio": observed, "ci95": None}
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        chosen = rng.integers(0, len(matched), size=len(matched))
        values[index] = matched[chosen].mean() / max(control[chosen].mean(), 1e-12)
    low, high = np.quantile(values, [0.025, 0.975])
    return {"ratio": observed, "ci95": [float(low), float(high)]}


def treatment_probes(representations, labels, groups, folds, seed):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    output = {}
    for name, values in representations.items():
        if name not in ("input_feature", "z_shared", "z_inv", "z_bias"):
            continue
        fold_rows = []
        for fold, (train, test) in enumerate(splitter.split(values, labels, groups)):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed + fold),
            )
            model.fit(values[train], labels[train])
            probability = model.predict_proba(values[test])[:, 1]
            prediction = probability >= 0.5
            rng = np.random.default_rng(seed + 1000 + fold)
            shuffled = labels[train].copy()
            rng.shuffle(shuffled)
            null_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed + 100 + fold),
            )
            null_model.fit(values[train], shuffled)
            null_probability = null_model.predict_proba(values[test])[:, 1]
            fold_rows.append(
                {
                    "fold": fold,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "auroc": float(roc_auc_score(labels[test], probability)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels[test], prediction)),
                    "permuted_train_auroc": float(roc_auc_score(labels[test], null_probability)),
                }
            )
        output[name] = {
            "folds": fold_rows,
            "auroc_mean": float(np.mean([row["auroc"] for row in fold_rows])),
            "auroc_sd": float(np.std([row["auroc"] for row in fold_rows], ddof=1)),
            "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
            "permuted_train_auroc_mean": float(np.mean([row["permuted_train_auroc"] for row in fold_rows])),
        }
    return output


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_seeds_all(args.seed)
    rng = np.random.default_rng(args.seed)
    cache = load_counterfactual_cache(args.cache)
    data_path = Path(args.data_path)
    model = build_predictor(
        build_args(args),
        read_total(data_path / "entity2id.txt"),
        read_total(data_path / "relation2id.txt"),
    )
    model.eval()

    valid = np.flatnonzero(cache["cf_valid_mask"].cpu().numpy())
    pair_count = min(args.pair_sample, len(valid))
    factual = rng.choice(valid, size=pair_count, replace=False)
    matched = cache["cf_index"][torch.as_tensor(factual)].cpu().numpy()
    control, control_distance = nearest_alternative_controls(cache, factual, args.neighbors)
    usable = control >= 0
    factual, matched, control, control_distance = (
        factual[usable], matched[usable], control[usable], control_distance[usable]
    )
    combined = np.concatenate((factual, matched, control))
    encoded = encode_indices(model, cache, combined, args.batch_size)
    count = len(factual)
    invariance = {}
    for name in ("z_shared", "z_inv", "z_bias"):
        factual_z = encoded[name][:count]
        matched_z = encoded[name][count : 2 * count]
        control_z = encoded[name][2 * count :]
        matched_distance = np.linalg.norm(factual_z - matched_z, axis=1)
        alternative_distance = np.linalg.norm(factual_z - control_z, axis=1)
        invariance[name] = {
            "matched_l2_mean": float(matched_distance.mean()),
            "alternative_l2_mean": float(alternative_distance.mean()),
            **paired_bootstrap_ratio(
                matched_distance, alternative_distance, args.bootstrap, args.seed + len(invariance)
            ),
        }

    cached_distance = cache["cf_match_distance"][torch.as_tensor(factual)].cpu().numpy()
    invariance["input_feature_distance"] = {
        "matched_mean": float(cached_distance.mean()),
        "alternative_mean": float(control_distance.mean()),
        **paired_bootstrap_ratio(cached_distance, control_distance, args.bootstrap, args.seed + 50),
    }

    all_indices = np.arange(len(cache["triples"]), dtype=np.int64)
    labels_all = cache["treatment"].cpu().numpy().astype(np.int64)
    half = min(args.probe_sample // 2, int((labels_all == 0).sum()), int((labels_all == 1).sum()))
    probe_indices = np.concatenate(
        [
            rng.choice(all_indices[labels_all == label], size=half, replace=False)
            for label in (0, 1)
        ]
    )
    rng.shuffle(probe_indices)
    probe_encoded = encode_indices(model, cache, probe_indices, args.batch_size)
    labels = labels_all[probe_indices]
    feature_rows = cache["subgraph_feature"][torch.as_tensor(probe_indices)].cpu().numpy()
    probe_encoded["input_feature"] = feature_rows
    _, groups = np.unique(feature_rows, axis=0, return_inverse=True)
    probes = treatment_probes(probe_encoded, labels, groups, args.folds, args.seed)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "cache": str(Path(args.cache).resolve()),
        "pair_count": int(count),
        "probe_count": int(len(probe_indices)),
        "control": "nearest non-matched same-relation opposite-treatment triple",
        "grouped_cv": "exact intersection-feature rows never cross folds",
        "invariance": invariance,
        "treatment_linear_probes": probes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Saved representation probe: {}".format(output))


if __name__ == "__main__":
    main()

"""Probe factual/counterfactual invariant geometry without updating a model."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.diagnose_tgiif_gradient_conflict_NELL995 import (
    make_predictor,
    make_strategy,
    prepare_data,
    sha256,
)
from openke.data import TrainDataLoader
from openke.module.counterfactual import (
    CounterfactualBatchEnricher,
    load_counterfactual_cache,
    read_openke_triples,
    validate_counterfactual_cache,
)
from openke.utils import set_seeds_all


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched factual/counterfactual invariant representations "
            "against relation-controlled random mismatches."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    return parser.parse_args()


def relation_controlled_permutation(relations):
    permutation = torch.arange(relations.numel(), device=relations.device)
    usable = torch.zeros(relations.numel(), dtype=torch.bool, device=relations.device)
    for relation in torch.unique(relations):
        indices = torch.nonzero(relations == relation, as_tuple=False).view(-1)
        if indices.numel() < 2:
            continue
        permutation[indices] = indices.roll(1)
        usable[indices] = True
    return permutation, usable


def scalar_summary(values):
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("batches must be positive")
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    device_name = args.device or config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)

    set_seeds_all(config.get("seed", 23))
    cache = load_counterfactual_cache(config["cf_cache"])
    train_triples = read_openke_triples(
        str(Path(config["data_path"]) / "train2id.txt")
    )
    cache_fingerprint = validate_counterfactual_cache(
        cache,
        train_triples,
        dataset=config["dataset"],
        hash_hops=config["hash_hops"],
        feature_mode="triple",
        feature_dim=config["hash_hops"] * (config["hash_hops"] ** 2 + 3),
    )
    official_loader = TrainDataLoader(
        in_path=config["data_path"],
        nbatches=config.get("nbatches", 100),
        threads=config.get("threads", 8),
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=config.get("neg_ent", 25),
        neg_rel=0,
        device=device_name,
        max_hash_hops=config["hash_hops"],
    )
    train_loader = CounterfactualBatchEnricher(official_loader, cache)
    predictor = make_predictor(config, train_loader, cache)
    predictor.load_checkpoint(args.checkpoint)
    predictor.set_residual_multiplier(1.0)
    predictor.to(device)
    predictor.eval()
    strategy = make_strategy(config, predictor, train_loader.get_batch_size())
    strategy.to(device)
    strategy.set_epoch(max(config.get("epochs", 1000), 1000000))

    rows = []
    with torch.no_grad():
        for batch_index, data in enumerate(train_loader):
            if batch_index >= args.batches:
                break
            prepared = prepare_data(data, device)
            (
                factual_components,
                cf_unique_components,
                valid_mask,
                inverse,
                _representatives,
                valid_ratio,
            ) = strategy._forward_components_once(
                prepared, strategy.auxiliary_weight()
            )
            if cf_unique_components is None or int(valid_mask.sum()) < 2:
                continue
            factual_positive = strategy._slice_components(
                factual_components, 0, train_loader.get_batch_size()
            )
            factual = factual_positive["z_inv"][valid_mask]
            matched = cf_unique_components["z_inv"][inverse]
            relations = prepared["batch_r"].view(-1)[: train_loader.get_batch_size()]
            relations = relations[valid_mask]
            permutation, usable = relation_controlled_permutation(relations)
            if int(usable.sum()) < 2:
                continue
            factual_usable = factual[usable]
            matched_usable = matched[usable]
            random_usable = matched[permutation][usable]
            matched_mse = (factual_usable - matched_usable).square().mean()
            random_mse = (factual_usable - random_usable).square().mean()
            matched_cosine = F.cosine_similarity(
                factual_usable, matched_usable, dim=-1, eps=1e-8
            ).mean()
            random_cosine = F.cosine_similarity(
                factual_usable, random_usable, dim=-1, eps=1e-8
            ).mean()
            representation_std = factual.std(dim=0, unbiased=False).mean()
            inv_delta_std = factual_positive["invariant_delta"].std(
                unbiased=False
            )
            row = {
                "batch": batch_index,
                "cf_valid_ratio": float(valid_ratio),
                "relation_controlled_pairs": int(usable.sum().item()),
                "matched_mse": float(matched_mse.item()),
                "random_mse": float(random_mse.item()),
                "mse_separation": float((random_mse - matched_mse).item()),
                "matched_cosine": float(matched_cosine.item()),
                "random_cosine": float(random_cosine.item()),
                "cosine_separation": float(
                    (matched_cosine - random_cosine).item()
                ),
                "invariant_representation_std_mean": float(
                    representation_std.item()
                ),
                "invariant_delta_std": float(inv_delta_std.item()),
            }
            rows.append(row)
            print(
                "batch={:02d} matched_mse={:.6g} random_mse={:.6g} "
                "z_std={:.6g}".format(
                    batch_index,
                    row["matched_mse"],
                    row["random_mse"],
                    row["invariant_representation_std_mean"],
                )
            )

    if len(rows) != args.batches:
        raise RuntimeError(
            "Requested {} batches but measured {}".format(args.batches, len(rows))
        )
    summary_keys = (
        "cf_valid_ratio",
        "relation_controlled_pairs",
        "matched_mse",
        "random_mse",
        "mse_separation",
        "matched_cosine",
        "random_cosine",
        "cosine_separation",
        "invariant_representation_std_mean",
        "invariant_delta_std",
    )
    result = {
        "purpose": "train-only factual/counterfactual invariant geometry probe",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "cache_fingerprint": cache_fingerprint,
        "torch_version": torch.__version__,
        "device": device_name,
        "batches": args.batches,
        "summary": {
            key: scalar_summary([row[key] for row in rows])
            for key in summary_keys
        },
        "per_batch": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Saved representation probe: {}".format(output_path))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

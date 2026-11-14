"""Measure loss-to-module gradient routing in a frozen TGIIF checkpoint.

No optimizer step is executed and the predictor stays in eval mode, so neither
parameters nor BatchNorm running statistics are modified.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "examples"))

from evaluate_ctsnet_checkpoint_WN18RR import build_predictor  # noqa: E402
from openke.data import TrainDataLoader  # noqa: E402
from openke.module.counterfactual import (  # noqa: E402
    CounterfactualBatchEnricher,
    load_counterfactual_cache,
)
from openke.module.loss import MarginLoss  # noqa: E402
from openke.module.strategy import CTSNegativeSampling  # noqa: E402
from openke.utils import set_seeds_all  # noqa: E402


LOSS_NAMES = ("official_rank", "residual_rank", "residual_preserve", "cf", "treatment", "orthogonality")
GROUP_PREFIXES = {
    "baseline": ("sf_lin_layer.", "lin."),
    "shared": ("residual_shared_encoder.",),
    "invariant": ("invariant_encoder.", "invariant_delta_head."),
    "bias": ("bias_encoder.", "bias_delta_head.", "treatment_head."),
    "gate": ("gate_head.",),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", default="./counterfactual_cache/WN18RR_popularity_h1.pt")
    parser.add_argument("--data-path", default="./benchmarks/WN18RR/")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--nbatches", type=int, default=100)
    parser.add_argument("--neg-ent", type=int, default=25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def build_args(args, config):
    return SimpleNamespace(
        architecture="treatment-aware",
        checkpoint=args.checkpoint,
        hash_hops=1,
        feature_mode="triple",
        hidden_dim=int(config.get("hidden_dim", 128)),
        branch_dim=int(config.get("branch_dim", 64)),
        branch_activation=config.get("branch_activation", "relu"),
        branch_negative_slope=float(config.get("branch_negative_slope", 0.01)),
        delta_limit=config.get("delta_limit"),
        delta_temperature=float(config.get("delta_temperature", 1.0)),
        delta_normalization="auto",
        delta_target_rms=config.get("delta_target_rms"),
        gate_temperature=float(config.get("gate_temperature", 1.0)),
        fusion_mode=config.get("fusion_mode", "invariant-margin"),
        invariant_margin_fraction=float(config.get("invariant_margin_fraction", 0.25)),
        normalization="batch",
        floor_residual_at_baseline_min=bool(config.get("floor_residual_at_baseline_min", False)),
        residual_max_top_gap_ratio=config.get("residual_max_top_gap_ratio"),
        residual_multiplier=1.0,
        disable_invariant_branch=False,
        disable_bias_branch=False,
        invariant_branch_multiplier=float(config.get("invariant_branch_multiplier", 1.0)),
        bias_branch_multiplier=float(config.get("bias_branch_multiplier", 1.0)),
        device=args.device,
    )


def prepare_data(data, device):
    result = {
        "batch_h": torch.as_tensor(data["batch_h"], dtype=torch.long, device=device),
        "batch_t": torch.as_tensor(data["batch_t"], dtype=torch.long, device=device),
        "batch_r": torch.as_tensor(data["batch_r"], dtype=torch.long, device=device),
        "batch_y": torch.as_tensor(data["batch_y"], dtype=torch.float32, device=device),
        "subgraph_feature": data["subgraph_feature"],
        "mode": data["mode"],
    }
    for key in (
        "cf_subgraph_feature", "factual_treatment", "cf_treatment",
        "cf_valid_mask", "cf_match_distance", "cf_target_index",
        "cf_batch_h", "cf_batch_t", "cf_batch_r",
    ):
        if key in data:
            result[key] = data[key]
    return result


def group_parameters(model):
    groups = {name: [] for name in GROUP_PREFIXES}
    names = {name: [] for name in GROUP_PREFIXES}
    for parameter_name, parameter in model.named_parameters():
        for group, prefixes in GROUP_PREFIXES.items():
            if parameter_name.startswith(prefixes):
                groups[group].append(parameter)
                names[group].append(parameter_name)
                break
    return groups, names


def gradients_by_group(loss, group_params, retain_graph):
    all_params = []
    slices = {}
    for group, parameters in group_params.items():
        start = len(all_params)
        all_params.extend(parameters)
        slices[group] = (start, len(all_params))
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return {group: [None] * len(parameters) for group, parameters in group_params.items()}
    grads = torch.autograd.grad(loss, all_params, retain_graph=retain_graph, allow_unused=True)
    return {group: list(grads[start:end]) for group, (start, end) in slices.items()}


def grad_norm(grads):
    total = None
    for grad in grads:
        if grad is None:
            continue
        value = grad.detach().double().square().sum()
        total = value if total is None else total + value
    return 0.0 if total is None else float(torch.sqrt(total).item())


def grad_cosine(left, right):
    dot = None
    left_sq = None
    right_sq = None
    for grad_left, grad_right in zip(left, right):
        if grad_left is None or grad_right is None:
            continue
        grad_left = grad_left.detach().double()
        grad_right = grad_right.detach().double()
        value = (grad_left * grad_right).sum()
        dot = value if dot is None else dot + value
        value_left = grad_left.square().sum()
        value_right = grad_right.square().sum()
        left_sq = value_left if left_sq is None else left_sq + value_left
        right_sq = value_right if right_sq is None else right_sq + value_right
    if dot is None or float(left_sq.item()) == 0.0 or float(right_sq.item()) == 0.0:
        return None
    return float((dot / torch.sqrt(left_sq * right_sq)).item())


def make_strategy(predictor, train_loader, config):
    return CTSNegativeSampling(
        model=predictor,
        loss=MarginLoss(margin=float(config.get("margin", 5.0))),
        batch_size=train_loader.get_batch_size(),
        lambda_cf=float(config.get("lambda_cf", 0.0)),
        lambda_treat=float(config.get("lambda_treat", 0.0)),
        lambda_orth=float(config.get("lambda_orth", 0.0)),
        lambda_gate=float(config.get("lambda_gate", 0.0)),
        lambda_residual_rank=float(config.get("lambda_residual_rank", 0.0)),
        lambda_invariant_rank=float(config.get("lambda_invariant_rank", 0.0)),
        lambda_bias_rank=float(config.get("lambda_bias_rank", 0.0)),
        lambda_residual_preserve=float(config.get("lambda_residual_preserve", 0.0)),
        residual_hard_negatives=int(config.get("residual_hard_negatives", 0)),
        residual_near_tie_negatives=int(config.get("residual_near_tie_negatives", 0)),
        center_residual=bool(config.get("center_residual", False)),
        preserve_base_ties=bool(config.get("preserve_base_ties", False)),
        break_base_ties=bool(config.get("break_base_ties", False)),
        floor_residual_at_baseline_min=bool(config.get("floor_residual_at_baseline_min", False)),
        isolate_base_rank=True,
        warmup_start_epoch=0,
        warmup_epochs=0,
        residual_start_epoch=0,
        residual_warmup_epochs=0,
        residual_loss_schedule=config.get("residual_loss_schedule", "auxiliary"),
        treatment_supervision=config.get("treatment_supervision", "factual"),
        shuffle_treatment_labels=False,
        lambda_invariant_variance=float(config.get("lambda_invariant_variance", 0.0)),
        lambda_bias_variance=float(config.get("lambda_bias_variance", 0.0)),
        representation_std_floor=float(config.get("representation_std_floor", 0.5)),
        gate_regularizer=config.get("gate_regularizer", "sample-kl"),
        gate_target=float(config.get("gate_target", 0.5)),
        lambda_gate_variance=float(config.get("lambda_gate_variance", 0.0)),
        gate_std_floor=float(config.get("gate_std_floor", 0.05)),
        treatment_label_smoothing=float(config.get("treatment_label_smoothing", 0.0)),
        treatment_logit_temperature=float(config.get("treatment_logit_temperature", 1.0)),
        lambda_bias_counterfactual=float(config.get("lambda_bias_counterfactual", 0.0)),
        bias_counterfactual_margin=float(config.get("bias_counterfactual_margin", 0.1)),
        lambda_branch_balance=float(config.get("lambda_branch_balance", 0.0)),
        branch_balance_target=float(config.get("branch_balance_target", 0.5)),
        residual_tie_band=float(config.get("residual_tie_band", 0.0)),
    )


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.batches < 1:
        raise ValueError("--batches must be positive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    set_seeds_all(args.seed)
    cache = load_counterfactual_cache(args.cache)
    train_loader = TrainDataLoader(
        in_path=args.data_path,
        nbatches=args.nbatches,
        threads=args.threads,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_ent,
        neg_rel=0,
        device=args.device,
        max_hash_hops=1,
        feature_mode="triple",
    )
    enriched = CounterfactualBatchEnricher(train_loader, cache)
    predictor = build_predictor(
        build_args(args, config), train_loader.get_ent_tot(), train_loader.get_rel_tot()
    )
    predictor.eval()
    strategy = make_strategy(predictor, train_loader, config)
    strategy.to(next(predictor.parameters()).device)
    group_params, parameter_names = group_parameters(predictor)
    weights = {
        "official_rank": 1.0,
        "residual_rank": float(config.get("lambda_residual_rank", 0.0)),
        "residual_preserve": float(config.get("lambda_residual_preserve", 0.0)),
        "cf": float(config.get("lambda_cf", 0.0)),
        "treatment": float(config.get("lambda_treat", 0.0)),
        "orthogonality": float(config.get("lambda_orth", 0.0)),
    }
    norm_rows = []
    cosine_rows = []
    device = next(predictor.parameters()).device

    for batch_index, raw_data in enumerate(enriched):
        if batch_index >= args.batches:
            break
        data = prepare_data(raw_data, device)
        (
            components, cf_components, valid_mask, inverse, representatives, _valid_ratio
        ) = strategy._forward_components_once(data, 1.0)
        positive = strategy._get_positive_score(components["base_score"])
        negative = strategy._get_negative_score(components["base_score"])
        rank_loss = strategy.loss(positive, negative)
        residual_rank, residual_preserve, _diagnostics = strategy._residual_losses(
            components, rank_loss, fusion_multiplier=1.0
        )
        auxiliary, _aux_diagnostics = strategy._auxiliary_losses(
            data, components, cf_components, valid_mask, inverse,
            representatives, rank_loss, 1.0
        )
        losses = {
            "official_rank": rank_loss,
            "residual_rank": residual_rank,
            "residual_preserve": residual_preserve,
            "cf": auxiliary["cf"],
            "treatment": auxiliary["treat"],
            "orthogonality": auxiliary["orth"],
        }
        gradients = {}
        for loss_index, loss_name in enumerate(LOSS_NAMES):
            weighted = weights[loss_name] * losses[loss_name]
            gradients[loss_name] = gradients_by_group(
                weighted, group_params, retain_graph=loss_index < len(LOSS_NAMES) - 1
            )
            for group in GROUP_PREFIXES:
                norm_rows.append(
                    {
                        "batch": batch_index,
                        "loss": loss_name,
                        "weight": weights[loss_name],
                        "raw_loss": float(losses[loss_name].detach().item()),
                        "module": group,
                        "gradient_norm": grad_norm(gradients[loss_name][group]),
                    }
                )
        pairs = (
            ("residual_rank", "cf"),
            ("residual_rank", "treatment"),
            ("residual_rank", "orthogonality"),
            ("cf", "treatment"),
            ("cf", "orthogonality"),
            ("treatment", "orthogonality"),
        )
        for left, right in pairs:
            for group in GROUP_PREFIXES:
                cosine = grad_cosine(gradients[left][group], gradients[right][group])
                if cosine is not None:
                    cosine_rows.append(
                        {"batch": batch_index, "left": left, "right": right, "module": group, "cosine": cosine}
                    )
        print("gradient probe batch {}/{}".format(batch_index + 1, args.batches))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "gradient_norms.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(norm_rows[0].keys()))
        writer.writeheader()
        writer.writerows(norm_rows)
    with (output_dir / "gradient_cosines.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cosine_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cosine_rows)

    norm_aggregate = {}
    for loss_name in LOSS_NAMES:
        norm_aggregate[loss_name] = {}
        for group in GROUP_PREFIXES:
            values = [row["gradient_norm"] for row in norm_rows if row["loss"] == loss_name and row["module"] == group]
            norm_aggregate[loss_name][group] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "nonzero_fraction": float(np.mean(np.asarray(values) > 0.0)),
            }
    cosine_aggregate = defaultdict(dict)
    for row in cosine_rows:
        key = row["left"] + "__" + row["right"]
        cosine_aggregate[key].setdefault(row["module"], []).append(row["cosine"])
    cosine_aggregate = {
        pair: {
            group: {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "count": len(values),
            }
            for group, values in groups.items()
        }
        for pair, groups in cosine_aggregate.items()
    }
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "batches": args.batches,
        "optimizer_steps": 0,
        "predictor_eval_mode": True,
        "loss_weights": weights,
        "parameter_names": parameter_names,
        "gradient_norms": norm_aggregate,
        "gradient_cosines": cosine_aggregate,
    }
    output_json = output_dir / "loss_gradient_probe.json"
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Saved gradient probe: {}".format(output_json))


if __name__ == "__main__":
    main()

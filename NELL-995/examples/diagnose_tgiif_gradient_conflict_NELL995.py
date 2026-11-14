import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from openke.data import TrainDataLoader
from openke.module.counterfactual import (
    CounterfactualBatchEnricher,
    load_counterfactual_cache,
    read_openke_triples,
    validate_counterfactual_cache,
)
from openke.module.loss import MarginLoss
from openke.module.model import CTSNetTreatmentAwareResidualPredictor
from openke.module.strategy import CTSNegativeSampling
from openke.utils import set_seeds_all


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure train-only gradient agreement between the TGIIF structural "
            "objective and treatment supervision on the shared residual stem."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameters_with_prefix(model, prefixes):
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes) and parameter.requires_grad
    ]


def gradients(loss, parameters, retain_graph=True):
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return [None for _ in parameters]
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def grad_norm(values):
    total = 0.0
    for value in values:
        if value is not None:
            total += float(value.detach().double().square().sum().item())
    return math.sqrt(total)


def grad_cosine(left, right):
    dot = 0.0
    left_square = 0.0
    right_square = 0.0
    for left_value, right_value in zip(left, right):
        if left_value is None or right_value is None:
            continue
        left_value = left_value.detach().double()
        right_value = right_value.detach().double()
        dot += float((left_value * right_value).sum().item())
        left_square += float(left_value.square().sum().item())
        right_square += float(right_value.square().sum().item())
    denominator = math.sqrt(left_square * right_square)
    if denominator == 0.0:
        return None
    return dot / denominator


def summarize(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "negative_fraction": (
            sum(value < 0.0 for value in values) / float(len(values))
            if "cosine" in key
            else None
        ),
    }


def make_predictor(config, train_loader, cache):
    predictor = CTSNetTreatmentAwareResidualPredictor(
        ent_tot=train_loader.get_ent_tot(),
        rel_tot=train_loader.get_rel_tot(),
        max_hash_hops=config["hash_hops"],
        hidden_dim=config["hidden_dim"],
        branch_dim=config["branch_dim"],
        residual_scale=config["residual_scale"],
        residual_multiplier=1.0,
        center_residual=config.get("center_residual", False),
        preserve_base_ties=config.get("preserve_base_ties", False),
        break_base_ties=config.get("break_base_ties", False),
        floor_residual_at_baseline_min=config.get(
            "floor_residual_at_baseline_min", False
        ),
        residual_max_top_gap_ratio=config.get("residual_max_top_gap_ratio"),
        feature_mode="triple",
        branch_activation=config.get("branch_activation", "relu"),
        branch_negative_slope=config.get("branch_negative_slope", 0.01),
        delta_limit=config.get("delta_limit"),
        delta_temperature=config.get("delta_temperature", 1.0),
        gate_temperature=config.get("gate_temperature", 1.0),
        delta_normalization=config.get("delta_normalization", "none"),
        delta_target_rms=config.get("delta_target_rms", 0.25),
        fusion_mode=config.get("fusion_mode", "additive"),
        invariant_margin_fraction=config.get("invariant_margin_fraction", 0.25),
        invariant_margin_ungated=config.get("invariant_margin_ungated", False),
        structural_abstention=config.get("structural_abstention", False),
        structural_abstention_min_span_ratio=config.get(
            "structural_abstention_min_span_ratio", 0.05
        ),
        structural_abstention_require_gate_agreement=not config.get(
            "no_structural_abstention_gate_agreement", False
        ),
        isolate_treatment_gradients=config.get(
            "isolate_treatment_gradients", False
        ),
        treatment_shared_gradient_scale=config.get(
            "treatment_shared_gradient_scale", 1.0
        ),
    )
    predictor.set_branch_multipliers(
        invariant=config.get("invariant_branch_multiplier", 1.0),
        bias=config.get("bias_branch_multiplier", 1.0),
    )
    predictor.set_score_space(config.get("score_space", "probability"))
    predictor.set_train_triples(cache["triples"].cpu().numpy())
    return predictor


def make_strategy(config, predictor, batch_size):
    return CTSNegativeSampling(
        model=predictor,
        loss=MarginLoss(margin=config["margin"]),
        batch_size=batch_size,
        lambda_cf=config.get("lambda_cf", 0.0),
        cf_distance_temperature=config.get("cf_distance_temperature", 0.0),
        lambda_treat=config.get("lambda_treat", 0.0),
        lambda_orth=config.get("lambda_orth", 0.0),
        lambda_gate=config.get("lambda_gate", 0.0),
        lambda_residual_rank=config.get("lambda_residual_rank", 0.0),
        lambda_invariant_rank=config.get("lambda_invariant_rank", 0.0),
        lambda_bias_rank=config.get("lambda_bias_rank", 0.0),
        lambda_residual_preserve=config.get("lambda_residual_preserve", 0.0),
        residual_hard_negatives=config.get("residual_hard_negatives", 0),
        residual_near_tie_negatives=config.get("residual_near_tie_negatives", 0),
        residual_min_treatment_gap=config.get("residual_min_treatment_gap", 0.0),
        residual_exclude_exact_ties=config.get(
            "residual_exclude_exact_ties", False
        ),
        center_residual=config.get("center_residual", False),
        preserve_base_ties=config.get("preserve_base_ties", False),
        break_base_ties=config.get("break_base_ties", False),
        floor_residual_at_baseline_min=config.get(
            "floor_residual_at_baseline_min", False
        ),
        isolate_base_rank=True,
        baseline_rank_score_space=config.get(
            "baseline_rank_score_space", "probability"
        ),
        warmup_start_epoch=config.get("warmup_start_epoch", 5),
        warmup_epochs=config.get("warmup_epochs", 10),
        residual_start_epoch=config.get("residual_start_epoch", 0),
        residual_warmup_epochs=config.get("residual_warmup_epochs", 0),
        residual_loss_schedule=config.get("residual_loss_schedule", "fusion"),
        treatment_supervision=config.get("treatment_supervision", "both"),
        shuffle_treatment_labels=config.get("shuffle_treatment_labels", False),
        lambda_invariant_variance=config.get("lambda_invariant_variance", 0.0),
        lambda_bias_variance=config.get("lambda_bias_variance", 0.0),
        representation_std_floor=config.get("representation_std_floor", 0.5),
        gate_regularizer=config.get("gate_regularizer", "sample-kl"),
        gate_target=config.get("gate_target", 0.5),
        lambda_gate_variance=config.get("lambda_gate_variance", 0.0),
        gate_std_floor=config.get("gate_std_floor", 0.05),
        treatment_label_smoothing=config.get("treatment_label_smoothing", 0.0),
        treatment_logit_temperature=config.get(
            "treatment_logit_temperature", 1.0
        ),
        lambda_bias_counterfactual=config.get("lambda_bias_counterfactual", 0.0),
        bias_counterfactual_margin=config.get("bias_counterfactual_margin", 0.1),
        lambda_branch_balance=config.get("lambda_branch_balance", 0.0),
        branch_balance_target=config.get("branch_balance_target", 0.5),
        residual_tie_band=config.get("residual_tie_band", 0.0),
        lambda_delta_magnitude=config.get("lambda_delta_magnitude", 0.0),
    )


def prepare_data(data, device):
    prepared = {
        "batch_h": torch.as_tensor(data["batch_h"], device=device),
        "batch_t": torch.as_tensor(data["batch_t"], device=device),
        "batch_r": torch.as_tensor(data["batch_r"], device=device),
        "batch_y": torch.as_tensor(data["batch_y"], device=device),
        "subgraph_feature": data["subgraph_feature"],
        "mode": data["mode"],
    }
    for key in (
        "cf_subgraph_feature",
        "factual_treatment",
        "cf_treatment",
        "cf_valid_mask",
        "cf_match_distance",
        "cf_target_index",
        "cf_batch_h",
        "cf_batch_t",
        "cf_batch_r",
    ):
        if key in data:
            prepared[key] = data[key]
    return prepared


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    device = args.device or config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.batches <= 0:
        raise ValueError("batches must be positive")

    set_seeds_all(config.get("seed", 23))
    cache = load_counterfactual_cache(config["cf_cache"])
    train_triples = read_openke_triples(
        os.path.join(config["data_path"], "train2id.txt")
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
        device=device,
        max_hash_hops=config["hash_hops"],
    )
    train_loader = CounterfactualBatchEnricher(official_loader, cache)
    predictor = make_predictor(config, train_loader, cache)
    predictor.load_checkpoint(args.checkpoint)
    predictor.set_residual_multiplier(1.0)
    predictor.to(torch.device(device))
    predictor.train()
    strategy = make_strategy(config, predictor, train_loader.get_batch_size())
    strategy.to(torch.device(device))
    strategy.set_epoch(max(config.get("epochs", 1000), 1000000))

    shared_parameters = parameters_with_prefix(
        predictor, ("residual_shared_encoder.",)
    )
    bias_parameters = parameters_with_prefix(
        predictor, ("bias_encoder.", "treatment_head.")
    )
    rows = []
    for batch_index, data in enumerate(train_loader):
        if batch_index >= args.batches:
            break
        data = prepare_data(data, torch.device(device))
        warmup = strategy.auxiliary_weight()
        (
            components,
            cf_unique_components,
            valid_mask,
            inverse,
            representatives,
            valid_ratio,
        ) = strategy._forward_components_once(data, warmup)
        rank_score = components[
            "base_logit"
            if strategy.baseline_rank_score_space == "logit"
            else "base_score"
        ]
        positive_score = strategy._get_positive_score(rank_score)
        negative_score = strategy._get_negative_score(rank_score)
        rank_loss = strategy.loss(positive_score, negative_score)
        residual_rank, residual_preserve, residual_diagnostics = (
            strategy._residual_losses(
                components,
                rank_loss,
                fusion_multiplier=1.0,
            )
        )
        auxiliary, auxiliary_diagnostics = strategy._auxiliary_losses(
            data,
            components,
            cf_unique_components,
            valid_mask,
            inverse,
            representatives,
            rank_loss,
            warmup,
        )
        ranking_objective = (
            strategy.lambda_residual_rank * residual_rank
            + strategy.lambda_residual_preserve * residual_preserve
        )
        cf_objective = strategy.lambda_cf * auxiliary["cf"]
        structural_objective = ranking_objective + cf_objective
        treatment_objective = strategy.lambda_treat * auxiliary["treat"]

        rank_shared = gradients(ranking_objective, shared_parameters)
        cf_shared = gradients(cf_objective, shared_parameters)
        structural_shared = gradients(structural_objective, shared_parameters)
        treatment_shared = gradients(treatment_objective, shared_parameters)
        treatment_bias = gradients(treatment_objective, bias_parameters)

        row = {
            "batch": batch_index,
            "cf_valid_ratio": valid_ratio,
            "ranking_loss": float(ranking_objective.detach().item()),
            "cf_loss": float(cf_objective.detach().item()),
            "treatment_loss": float(treatment_objective.detach().item()),
            "ranking_shared_grad_norm": grad_norm(rank_shared),
            "cf_shared_grad_norm": grad_norm(cf_shared),
            "structural_shared_grad_norm": grad_norm(structural_shared),
            "treatment_shared_grad_norm": grad_norm(treatment_shared),
            "treatment_bias_grad_norm": grad_norm(treatment_bias),
            "treatment_vs_ranking_cosine": grad_cosine(
                treatment_shared, rank_shared
            ),
            "treatment_vs_cf_cosine": grad_cosine(treatment_shared, cf_shared),
            "treatment_vs_structural_cosine": grad_cosine(
                treatment_shared, structural_shared
            ),
            "cf_vs_ranking_cosine": grad_cosine(cf_shared, rank_shared),
            "residual_margin_improved_ratio": residual_diagnostics.get(
                "residual_margin_improved_ratio", 0.0
            ),
            "treat_accuracy_f": auxiliary_diagnostics.get(
                "treat_accuracy_f", 0.0
            ),
        }
        rows.append(row)
        print(
            "batch={:02d} treat_vs_struct={:+.6f} "
            "treat_norm={:.6g} struct_norm={:.6g}".format(
                batch_index,
                row["treatment_vs_structural_cosine"] or 0.0,
                row["treatment_shared_grad_norm"],
                row["structural_shared_grad_norm"],
            )
        )

    if len(rows) != args.batches:
        raise RuntimeError(
            "Requested {} batches but measured {}".format(args.batches, len(rows))
        )
    summary_keys = (
        "ranking_shared_grad_norm",
        "cf_shared_grad_norm",
        "structural_shared_grad_norm",
        "treatment_shared_grad_norm",
        "treatment_bias_grad_norm",
        "treatment_vs_ranking_cosine",
        "treatment_vs_cf_cosine",
        "treatment_vs_structural_cosine",
        "cf_vs_ranking_cosine",
        "residual_margin_improved_ratio",
        "treat_accuracy_f",
    )
    result = {
        "purpose": "train-only shared-gradient conflict diagnostic",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "cache_fingerprint": cache_fingerprint,
        "torch_version": torch.__version__,
        "device": device,
        "batches": args.batches,
        "summary": {key: summarize(rows, key) for key in summary_keys},
        "per_batch": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("Saved gradient diagnostic: {}".format(output_path))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

import argparse
import json
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from openke.config import Tester
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.model import (
    CTSNetDualBranchPredictor,
    CTSNetIndependentResidualPredictor,
    CTSNetResidualDualBranchPredictor,
    CTSNetTreatmentAwareResidualPredictor,
)
from openke.utils import set_seeds_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate or diagnose a TGIIF NELL-995 checkpoint without retraining"
    )
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--architecture",
        choices=(
            "auto",
            "legacy",
            "residual",
            "independent",
            "treatment-aware",
        ),
        default="auto",
    )
    parser.add_argument("--data-path", default="./benchmarks/NELL-995/")
    parser.add_argument(
        "--split", choices=("test", "valid"), default="valid"
    )
    parser.add_argument("--hash-hops", type=int, default=1)
    parser.add_argument(
        "--feature-mode", choices=("triple", "pairwise"), default="triple",
        help="Feature family used to train the checkpoint",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--branch-dim", type=int, default=64)
    parser.add_argument(
        "--branch-activation",
        choices=("relu", "leaky_relu", "silu"),
        default="relu",
    )
    parser.add_argument("--branch-negative-slope", type=float, default=0.01)
    parser.add_argument("--delta-limit", type=float, default=None)
    parser.add_argument("--delta-temperature", type=float, default=1.0)
    parser.add_argument(
        "--delta-normalization",
        choices=("auto", "none", "query-rms"),
        default="auto",
        help="Auto-detect query-RMS calibration from the checkpoint",
    )
    parser.add_argument("--delta-target-rms", type=float, default=None)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument(
        "--fusion-mode",
        choices=("additive", "invariant-margin"),
        default="additive",
    )
    parser.add_argument("--invariant-margin-fraction", type=float, default=0.25)
    parser.add_argument(
        "--invariant-margin-ungated",
        action="store_true",
        help="Use the invariant branch directly in invariant-margin fusion",
    )
    parser.add_argument(
        "--structural-abstention",
        action="store_true",
        help="Apply residuals only to structurally identifiable top collisions",
    )
    parser.add_argument(
        "--structural-abstention-min-span-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--no-structural-abstention-gate-agreement",
        action="store_true",
    )
    parser.add_argument(
        "--normalization",
        choices=("batch", "layer"),
        default="batch",
        help="Use layer only for checkpoints trained with the LayerNorm ablation",
    )
    parser.add_argument("--nbatches", type=int, default=100)
    parser.add_argument("--neg-ent", type=int, default=25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--diagnostic-batches", type=int, default=5)
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip train/eval ordering probes and run only filtered ranking",
    )
    parser.add_argument(
        "--residual-multiplier",
        type=float,
        default=None,
        help=(
            "Override the saved independent-residual multiplier for a "
            "controlled base-only or scaled-residual ablation"
        ),
    )
    parser.add_argument(
        "--floor-residual-at-baseline-min",
        action="store_true",
        help=(
            "At evaluation, prevent a residual-adjusted candidate from "
            "beating the query's minimum official baseline score"
        ),
    )
    parser.add_argument(
        "--residual-max-top-gap-ratio",
        type=float,
        default=None,
        help=(
            "Enable residual scoring only when the normalized gap between "
            "the first two distinct baseline score groups is no larger than "
            "this value"
        ),
    )
    parser.add_argument(
        "--calibrate-batchnorm",
        action="store_true",
        help="Recompute legacy BatchNorm running statistics before diagnostics",
    )
    parser.add_argument("--calibration-batches", type=int, default=100)
    parser.add_argument(
        "--calibrated-checkpoint",
        default=None,
        help="Optional path for saving the calibrated predictor state",
    )
    parser.add_argument(
        "--filtered-ranking",
        action="store_true",
        help="Run the complete official Base.so filtered-ranking evaluator",
    )
    parser.add_argument(
        "--score-space",
        choices=("probability", "logit"),
        default="probability",
        help=(
            "Ranking score representation. 'probability' is the historical "
            "default; 'logit' is a diagnostic/adaptation mode for datasets "
            "whose sigmoid outputs numerically saturate."
        ),
    )
    parser.add_argument(
        "--disable-invariant-branch",
        action="store_true",
        help="Ablate the invariant residual contribution at evaluation",
    )
    parser.add_argument(
        "--disable-bias-branch",
        action="store_true",
        help="Ablate the bias residual contribution at evaluation",
    )
    parser.add_argument(
        "--invariant-branch-multiplier",
        type=float,
        default=1.0,
        help="Scale the invariant residual contribution at evaluation",
    )
    parser.add_argument(
        "--bias-branch-multiplier",
        type=float,
        default=1.0,
        help="Scale the bias residual contribution at evaluation",
    )
    parser.add_argument(
        "--relation-reliability-calibration-batches",
        type=int,
        default=0,
        help=(
            "Use this many train-only batches to estimate relation-level "
            "residual reliability; zero disables the guard"
        ),
    )
    parser.add_argument(
        "--relation-reliability-confirmation-batches",
        type=int,
        default=10,
        help="Independent train-only batches used to confirm relation direction",
    )
    parser.add_argument(
        "--relation-reliability-min-queries",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--relation-reliability-confidence-z",
        type=float,
        default=1.96,
        help="Normal quantile used by the Wilson lower confidence bound",
    )
    parser.add_argument(
        "--relation-reliability-success-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--relation-reliability-min-effect",
        type=float,
        default=0.0,
        help="Minimum mean train margin improvement required in both splits",
    )
    parser.add_argument(
        "--relation-reliability-effect-epsilon",
        type=float,
        default=1e-12,
        help="Absolute margin change treated as neutral",
    )
    parser.add_argument(
        "--relation-reliability-mode",
        choices=("binary", "continuous"),
        default="binary",
    )
    parser.add_argument(
        "--relation-reliability-output",
        default=None,
        help="Optional JSON path for the train-only reliability audit",
    )
    parser.add_argument(
        "--tie-policy-override",
        choices=("checkpoint", "free", "preserve", "break"),
        default="checkpoint",
        help=(
            "Override the checkpoint tie policy for evaluation only. "
            "'free' disables both preservation and explicit tie breaking; "
            "the default keeps the checkpoint policy unchanged."
        ),
    )
    return parser.parse_args()


def build_predictor(args, ent_tot, rel_tot):
    architecture = args.architecture
    state = torch.load(args.checkpoint, map_location="cpu")
    delta_normalization = args.delta_normalization
    if delta_normalization == "auto":
        delta_normalization = (
            "query-rms" if "delta_query_rms_enabled" in state else "none"
        )
    delta_target_rms = args.delta_target_rms
    if delta_target_rms is None:
        saved_target = state.get("delta_target_rms_value")
        delta_target_rms = (
            float(saved_target.item()) if saved_target is not None else 0.25
        )
    if architecture == "auto":
        if "entity_log_degree" in state:
            architecture = "treatment-aware"
        elif "residual_shared_encoder.dense1.weight" in state:
            architecture = "independent"
        elif "invariant_delta_head.weight" in state:
            architecture = "residual"
        else:
            architecture = "legacy"
    if architecture == "treatment-aware":
        predictor = CTSNetTreatmentAwareResidualPredictor(
            ent_tot=ent_tot,
            rel_tot=rel_tot,
            max_hash_hops=args.hash_hops,
            hidden_dim=args.hidden_dim,
            branch_dim=args.branch_dim,
            feature_mode=args.feature_mode,
            branch_activation=args.branch_activation,
            branch_negative_slope=args.branch_negative_slope,
            delta_limit=args.delta_limit,
            delta_temperature=args.delta_temperature,
            delta_normalization=delta_normalization,
            delta_target_rms=delta_target_rms,
            gate_temperature=args.gate_temperature,
            fusion_mode=args.fusion_mode,
            invariant_margin_fraction=args.invariant_margin_fraction,
            invariant_margin_ungated=args.invariant_margin_ungated,
            structural_abstention=args.structural_abstention,
            structural_abstention_min_span_ratio=(
                args.structural_abstention_min_span_ratio
            ),
            structural_abstention_require_gate_agreement=(
                not args.no_structural_abstention_gate_agreement
            ),
            center_residual="center_residual_enabled" in state,
            preserve_base_ties="preserve_base_ties_enabled" in state,
            break_base_ties="break_base_ties_enabled" in state,
            floor_residual_at_baseline_min=(
                "floor_residual_at_baseline_min_enabled" in state
            ),
        )
    elif architecture == "independent":
        predictor = CTSNetIndependentResidualPredictor(
            ent_tot=ent_tot,
            rel_tot=rel_tot,
            max_hash_hops=args.hash_hops,
            hidden_dim=args.hidden_dim,
            branch_dim=args.branch_dim,
            feature_mode=args.feature_mode,
            branch_activation=args.branch_activation,
            branch_negative_slope=args.branch_negative_slope,
            delta_limit=args.delta_limit,
            delta_temperature=args.delta_temperature,
            delta_normalization=delta_normalization,
            delta_target_rms=delta_target_rms,
            gate_temperature=args.gate_temperature,
        )
    elif architecture == "residual":
        predictor = CTSNetResidualDualBranchPredictor(
            ent_tot=ent_tot,
            rel_tot=rel_tot,
            max_hash_hops=args.hash_hops,
            hidden_dim=args.hidden_dim,
            branch_dim=args.branch_dim,
            feature_mode=args.feature_mode,
            freeze_baseline=True,
        )
    else:
        predictor = CTSNetDualBranchPredictor(
            ent_tot=ent_tot,
            rel_tot=rel_tot,
            max_hash_hops=args.hash_hops,
            hidden_dim=args.hidden_dim,
            branch_dim=args.branch_dim,
            normalization=args.normalization,
            feature_mode=args.feature_mode,
        )
    predictor.load_checkpoint(args.checkpoint)
    if args.tie_policy_override != "checkpoint":
        if not hasattr(predictor, "preserve_base_ties"):
            raise ValueError(
                "--tie-policy-override requires a residual predictor"
            )
        predictor.preserve_base_ties = args.tie_policy_override == "preserve"
        predictor.break_base_ties = args.tie_policy_override == "break"
        print(
            "Evaluation tie policy override: {}".format(
                args.tie_policy_override
            )
        )
    if args.score_space != "probability":
        if not hasattr(predictor, "set_score_space"):
            raise ValueError("--score-space logit requires a residual predictor")
        predictor.set_score_space(args.score_space)
    if args.floor_residual_at_baseline_min:
        if architecture != "treatment-aware":
            raise ValueError(
                "--floor-residual-at-baseline-min requires a treatment-aware checkpoint"
            )
        predictor.floor_residual_at_baseline_min = True
    if args.residual_max_top_gap_ratio is not None:
        if not hasattr(predictor, "set_residual_max_top_gap_ratio"):
            raise ValueError(
                "--residual-max-top-gap-ratio requires a treatment-aware checkpoint"
            )
        predictor.set_residual_max_top_gap_ratio(
            args.residual_max_top_gap_ratio
        )
    if args.residual_multiplier is not None:
        if not hasattr(predictor, "set_residual_multiplier"):
            raise ValueError(
                "--residual-multiplier requires an independent residual checkpoint"
            )
        predictor.set_residual_multiplier(args.residual_multiplier)
    if (
        args.disable_invariant_branch
        or args.disable_bias_branch
        or args.invariant_branch_multiplier != 1.0
        or args.bias_branch_multiplier != 1.0
    ):
        if not hasattr(predictor, "set_branch_multipliers"):
            raise ValueError("branch ablation requires a residual checkpoint")
        predictor.set_branch_multipliers(
            invariant=(
                0.0
                if args.disable_invariant_branch
                else args.invariant_branch_multiplier
            ),
            bias=(
                0.0
                if args.disable_bias_branch
                else args.bias_branch_multiplier
            ),
        )
    if args.device == "cuda":
        predictor.cuda()
    return predictor


@torch.no_grad()
def score_ordering(predictor, batches, batch_size, training):
    predictor.train(training)
    strict_correct = []
    non_worse = []
    tied = []
    separation = []
    positive = []
    negative = []

    for batch in batches:
        score = predictor(batch).view(-1)
        p_score = score[:batch_size].view(-1, batch_size).permute(1, 0)
        n_score = score[batch_size:].view(-1, batch_size).permute(1, 0)
        strict_correct.append((p_score < n_score).float().mean())
        non_worse.append((p_score <= n_score).float().mean())
        tied.append((p_score == n_score).float().mean())
        separation.append((n_score - p_score).mean())
        positive.append(p_score.mean())
        negative.append(n_score.mean())

    return {
        "strict_correct_ratio": torch.stack(strict_correct).mean().item(),
        "non_worse_ratio": torch.stack(non_worse).mean().item(),
        "tie_ratio": torch.stack(tied).mean().item(),
        "mean_separation": torch.stack(separation).mean().item(),
        "positive_mean": torch.stack(positive).mean().item(),
        "negative_mean": torch.stack(negative).mean().item(),
    }


def _wilson_lower_bound(successes, trials, z):
    if trials <= 0:
        return 0.0
    probability = successes / float(trials)
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = probability + z2 / (2.0 * trials)
    spread = z * math.sqrt(
        probability * (1.0 - probability) / trials
        + z2 / (4.0 * trials * trials)
    )
    return (center - spread) / denominator


def _relation_names(data_path, relation_total):
    names = [str(index) for index in range(relation_total)]
    path = Path(data_path) / "relation2id.txt"
    if not path.is_file():
        return names
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\n").rsplit("\t", 1)
            if len(parts) != 2:
                parts = line.strip().rsplit(maxsplit=1)
            if len(parts) != 2:
                continue
            try:
                relation_id = int(parts[1])
            except ValueError:
                continue
            if 0 <= relation_id < relation_total:
                names[relation_id] = parts[0]
    return names


@torch.no_grad()
def calibrate_relation_reliability(predictor, train_loader, args):
    """Estimate a conservative relation mask using train data only.

    Each positive triple is one observation.  Its effect is the mean change in
    positive-vs-negative separation induced by the residual.  Relations must
    pass the same directional test in disjoint calibration and confirmation
    batches; validation and test triples never enter this estimate.
    """
    if not hasattr(predictor, "set_relation_residual_reliability"):
        raise ValueError(
            "relation reliability requires a treatment-aware predictor"
        )
    if args.residual_multiplier is None or args.residual_multiplier <= 0.0:
        raise ValueError(
            "relation reliability requires an explicit positive "
            "--residual-multiplier"
        )
    if args.relation_reliability_confirmation_batches < 1:
        raise ValueError(
            "relation reliability confirmation batches must be at least 1"
        )
    if args.relation_reliability_min_queries < 1:
        raise ValueError("relation reliability min queries must be at least 1")
    if args.relation_reliability_confidence_z < 0.0:
        raise ValueError("relation reliability confidence z must be non-negative")
    if not 0.0 <= args.relation_reliability_success_threshold < 1.0:
        raise ValueError(
            "relation reliability success threshold must be in [0, 1)"
        )
    if args.relation_reliability_effect_epsilon < 0.0:
        raise ValueError("relation reliability effect epsilon must be non-negative")

    relation_total = predictor.rel_tot
    phase_names = ("calibration", "confirmation")
    phase_batches = (
        args.relation_reliability_calibration_batches,
        args.relation_reliability_confirmation_batches,
    )
    statistics = []
    for _ in phase_names:
        statistics.append(
            {
                key: torch.zeros(relation_total, dtype=torch.float64)
                for key in (
                    "queries",
                    "improved",
                    "worsened",
                    "neutral",
                    "effect_sum",
                )
            }
        )

    predictor.eval()
    predictor.set_relation_residual_reliability(None)
    batch_size = train_loader.get_batch_size()
    total_batches = sum(phase_batches)
    processed = 0
    for batch_index, batch in enumerate(train_loader):
        if batch_index >= total_batches:
            break
        phase_index = 0 if batch_index < phase_batches[0] else 1
        components = predictor.forward_components(batch)
        base_score = components["residual_anchor"].reshape(-1)
        final_score = components["score"].reshape(-1)
        if base_score.numel() <= batch_size:
            raise ValueError("relation calibration batch has no negatives")
        negative_count, remainder = divmod(
            base_score.numel() - batch_size, batch_size
        )
        if remainder or negative_count < 1:
            raise ValueError("unexpected positive/negative batch layout")

        positive_base = base_score[:batch_size].view(batch_size, 1)
        negative_base = base_score[batch_size:].view(
            negative_count, batch_size
        ).transpose(0, 1)
        positive_final = final_score[:batch_size].view(batch_size, 1)
        negative_final = final_score[batch_size:].view(
            negative_count, batch_size
        ).transpose(0, 1)
        effect = (
            (negative_final - positive_final)
            - (negative_base - positive_base)
        ).mean(dim=1)
        relation_ids = torch.as_tensor(
            batch["batch_r"], dtype=torch.long, device=effect.device
        ).reshape(-1)[:batch_size]
        relation_ids = relation_ids.detach().cpu()
        effect = effect.detach().double().cpu()
        phase = statistics[phase_index]
        ones = torch.ones_like(effect)
        epsilon = args.relation_reliability_effect_epsilon
        phase["queries"].scatter_add_(0, relation_ids, ones)
        phase["improved"].scatter_add_(
            0, relation_ids, (effect > epsilon).double()
        )
        phase["worsened"].scatter_add_(
            0, relation_ids, (effect < -epsilon).double()
        )
        phase["neutral"].scatter_add_(
            0, relation_ids, (effect.abs() <= epsilon).double()
        )
        phase["effect_sum"].scatter_add_(0, relation_ids, effect)
        processed += 1

    if processed != total_batches:
        raise ValueError(
            "requested {} reliability batches but loader yielded {}".format(
                total_batches, processed
            )
        )

    weights = torch.zeros(relation_total, dtype=torch.float32)
    relation_records = []
    relation_names = _relation_names(args.data_path, relation_total)
    for relation_id in range(relation_total):
        split_records = {}
        split_scores = []
        split_passes = []
        for phase_name, phase in zip(phase_names, statistics):
            queries = int(phase["queries"][relation_id].item())
            improved = int(phase["improved"][relation_id].item())
            worsened = int(phase["worsened"][relation_id].item())
            neutral = int(phase["neutral"][relation_id].item())
            evidence = improved + worsened
            mean_effect = (
                phase["effect_sum"][relation_id].item() / queries
                if queries
                else 0.0
            )
            lower = _wilson_lower_bound(
                improved,
                evidence,
                args.relation_reliability_confidence_z,
            )
            passed = (
                queries >= args.relation_reliability_min_queries
                and evidence >= args.relation_reliability_min_queries
                and lower > args.relation_reliability_success_threshold
                and mean_effect > args.relation_reliability_min_effect
            )
            normalized_score = max(
                0.0,
                (lower - args.relation_reliability_success_threshold)
                / (1.0 - args.relation_reliability_success_threshold),
            )
            split_scores.append(normalized_score)
            split_passes.append(passed)
            split_records[phase_name] = {
                "queries": queries,
                "evidence": evidence,
                "improved": improved,
                "worsened": worsened,
                "neutral": neutral,
                "mean_effect": mean_effect,
                "wilson_lower": lower,
                "passed": passed,
            }
        if all(split_passes):
            weights[relation_id] = (
                1.0
                if args.relation_reliability_mode == "binary"
                else min(split_scores)
            )
        relation_records.append(
            {
                "relation_id": relation_id,
                "relation_name": relation_names[relation_id],
                "weight": float(weights[relation_id].item()),
                **split_records,
            }
        )

    combined_queries = statistics[0]["queries"] + statistics[1]["queries"]
    weighted_query_coverage = float(
        (combined_queries * weights.double()).sum().item()
        / combined_queries.sum().clamp_min(1.0).item()
    )
    predictor.set_relation_residual_reliability(weights)
    report = {
        "schema_version": 1,
        "method": "train-only two-split Wilson relation reliability",
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "residual_multiplier": args.residual_multiplier,
        "mode": args.relation_reliability_mode,
        "calibration_batches": phase_batches[0],
        "confirmation_batches": phase_batches[1],
        "min_queries": args.relation_reliability_min_queries,
        "confidence_z": args.relation_reliability_confidence_z,
        "success_threshold": args.relation_reliability_success_threshold,
        "min_effect": args.relation_reliability_min_effect,
        "effect_epsilon": args.relation_reliability_effect_epsilon,
        "active_relations": int((weights > 0.0).sum().item()),
        "relation_total": relation_total,
        "weighted_query_coverage": weighted_query_coverage,
        "relations": relation_records,
    }
    return report


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.hash_hops != 1:
        raise ValueError("This V7a/V8b main line requires --hash-hops 1")
    if not args.skip_diagnostics and args.diagnostic_batches < 1:
        raise ValueError("diagnostic-batches must be at least 1")

    set_seeds_all(args.seed)
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
        max_hash_hops=args.hash_hops,
        feature_mode=args.feature_mode,
    )
    eval_predictor = build_predictor(
        args, train_loader.get_ent_tot(), train_loader.get_rel_tot()
    )
    if args.calibrate_batchnorm:
        if not hasattr(eval_predictor, "calibrate_batch_norm"):
            raise ValueError(
                "BatchNorm calibration is supported only for legacy scorers"
            )
        processed = eval_predictor.calibrate_batch_norm(
            train_loader, args.calibration_batches
        )
        print("BatchNorm calibration batches: {}".format(processed))
        if args.calibrated_checkpoint:
            Path(args.calibrated_checkpoint).parent.mkdir(
                parents=True, exist_ok=True
            )
            eval_predictor.save_checkpoint(args.calibrated_checkpoint)
            print(
                "Saved calibrated checkpoint: {}".format(
                    args.calibrated_checkpoint
                )
            )

    if args.relation_reliability_calibration_batches < 0:
        raise ValueError(
            "relation reliability calibration batches must be non-negative"
        )
    if args.relation_reliability_calibration_batches:
        reliability_report = calibrate_relation_reliability(
            eval_predictor, train_loader, args
        )
        print(
            "Relation reliability: active={}/{}, weighted_train_coverage="
            "{:.6f}, mode={}".format(
                reliability_report["active_relations"],
                reliability_report["relation_total"],
                reliability_report["weighted_query_coverage"],
                reliability_report["mode"],
            )
        )
        if args.relation_reliability_output:
            output_path = Path(args.relation_reliability_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(reliability_report, handle, indent=2, sort_keys=True)
                handle.write("\n")
            print("Saved relation reliability: {}".format(output_path))

    print("Checkpoint: {}".format(args.checkpoint))
    if not args.skip_diagnostics:
        batches = []
        for index, batch in enumerate(train_loader):
            batches.append(batch)
            if index + 1 >= args.diagnostic_batches:
                break

        eval_result = score_ordering(
            eval_predictor, batches, train_loader.get_batch_size(), training=False
        )

        # Reload before the train-mode pass so BatchNorm updates cannot leak
        # from one diagnostic mode into the other.
        train_predictor = build_predictor(
            args, train_loader.get_ent_tot(), train_loader.get_rel_tot()
        )
        if args.calibrate_batchnorm:
            train_predictor.load_state_dict(eval_predictor.state_dict())
        train_result = score_ordering(
            train_predictor, batches, train_loader.get_batch_size(), training=True
        )

        for mode, result in (("eval", eval_result), ("train", train_result)):
            print(
                "{} ordering: strict_correct={:.6f}, non_worse={:.6f}, "
                "ties={:.6f}, mean_separation={:.6f}, positive_mean={:.6f}, "
                "negative_mean={:.6f}".format(
                    mode,
                    result["strict_correct_ratio"],
                    result["non_worse_ratio"],
                    result["tie_ratio"],
                    result["mean_separation"],
                    result["positive_mean"],
                    result["negative_mean"],
                )
            )
        print(
            "train/eval strict-correct gap: {:.6f}".format(
                abs(
                    train_result["strict_correct_ratio"]
                    - eval_result["strict_correct_ratio"]
                )
            )
        )

    if args.filtered_ranking:
        test_loader = TestDataLoader(
            args.data_path,
            "link",
            device=args.device,
            max_hash_hops=args.hash_hops,
            feature_mode=args.feature_mode,
            split=args.split,
        )
        eval_predictor.eval()
        tester = Tester(
            model=eval_predictor,
            data_loader=test_loader,
            use_gpu=args.device == "cuda",
        )
        mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(False)
        if args.structural_abstention:
            stats = eval_predictor.structural_abstention_stats()
            print(
                "Structural abstention: queries={}, allowed={:.6f}, "
                "gate_agreement={:.6f}, mean_span_ratio={:.6f}".format(
                    stats["query_count"],
                    stats["allowed_fraction"],
                    stats["gate_agreement_fraction"],
                    stats["mean_invariant_span_ratio"],
                )
            )
        print(
            "Filtered ranking ({}): MRR={:.6f}, MR={:.6f}, Hits@10={:.6f}, "
            "Hits@3={:.6f}, Hits@1={:.6f}".format(
                args.split, mrr, mr, hit10, hit3, hit1
            )
        )


if __name__ == "__main__":
    main()

import argparse
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
        description="Evaluate or diagnose a CTS-Net FB15k-237 checkpoint without retraining"
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
    parser.add_argument("--data-path", default="./benchmarks/FB15k-237/")
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
        print(
            "Filtered ranking ({}): MRR={:.6f}, MR={:.6f}, Hits@10={:.6f}, "
            "Hits@3={:.6f}, Hits@1={:.6f}".format(
                args.split, mrr, mr, hit10, hit3, hit1
            )
        )


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from openke.config import CTSTrainer, Tester
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.counterfactual import (
    CounterfactualBatchEnricher,
    load_counterfactual_cache,
    read_openke_triples,
    validate_counterfactual_cache,
)
from openke.module.loss import MarginLoss
from openke.module.model import (
    CTSNetIndependentResidualPredictor,
    CTSNetTreatmentAwareResidualPredictor,
)
from openke.module.strategy import CTSNegativeSampling
from openke.utils import set_seeds_all


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTS-Net training from random initialization on an OpenKE dataset"
    )
    parser.add_argument(
        "--dataset",
        default="NELL-995",
        help="Dataset name recorded in the cache/checkpoint metadata",
    )
    parser.add_argument("--data-path", default="./benchmarks/NELL-995/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--margin", type=float, default=5.0)
    parser.add_argument("--hash-hops", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--branch-dim", type=int, default=64)
    parser.add_argument("--residual-scale", type=float, default=0.1)
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
        choices=("none", "query-rms"),
        default="none",
        help=(
            "Optionally shrink each query's residual RMS before tanh; "
            "small residuals are never amplified"
        ),
    )
    parser.add_argument(
        "--delta-target-rms",
        type=float,
        default=0.25,
        help="Maximum pre-tanh RMS used by query-rms calibration",
    )
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument(
        "--score-space",
        choices=("probability", "logit"),
        default="probability",
        help="Score space used for residual fusion and residual ranking",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("additive", "invariant-margin"),
        default="additive",
        help="Use baseline-preserving invariant margin fusion",
    )
    parser.add_argument("--invariant-margin-fraction", type=float, default=0.25)
    parser.add_argument(
        "--invariant-margin-ungated",
        action="store_true",
        help=(
            "In invariant-margin fusion, let the invariant branch rank "
            "directly; keep the treatment gate auxiliary to avoid gate collapse"
        ),
    )
    parser.add_argument(
        "--structural-abstention",
        action="store_true",
        help=(
            "Apply residuals only to baseline top-score collisions with a "
            "stable, identifiable invariant separator"
        ),
    )
    parser.add_argument(
        "--structural-abstention-min-span-ratio",
        type=float,
        default=0.05,
        help="Minimum top-collision invariant span divided by full-query span",
    )
    parser.add_argument(
        "--no-structural-abstention-gate-agreement",
        action="store_true",
        help="Do not require gated and ungated invariant top choices to agree",
    )
    parser.add_argument("--invariant-branch-multiplier", type=float, default=1.0)
    parser.add_argument("--bias-branch-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--treatment-aware",
        action="store_true",
        help="Provide train-only endpoint popularity to the Bias branch",
    )
    parser.add_argument(
        "--isolate-treatment-gradients",
        action="store_true",
        help=(
            "Detach the treatment/bias branch input from the shared residual "
            "stem while preserving its forward features and branch updates"
        ),
    )
    parser.add_argument(
        "--treatment-shared-gradient-scale",
        type=float,
        default=1.0,
        help=(
            "Scale only the treatment/bias backward signal entering the shared "
            "residual stem; forward scores and branch gradients are unchanged"
        ),
    )
    parser.add_argument(
        "--residual-hard-negatives",
        type=int,
        default=0,
        help="Use this many baseline-hard negatives per positive; 0 uses all",
    )
    parser.add_argument(
        "--residual-near-tie-negatives",
        type=int,
        default=0,
        help=(
            "Mine this many candidates with the smallest absolute baseline "
            "margin for residual ranking; 0 uses all"
        ),
    )
    parser.add_argument(
        "--residual-min-treatment-gap",
        type=float,
        default=0.0,
        help=(
            "Require residual-mined negatives to differ from the positive "
            "by at least this amount in the train-only treatment feature; "
            "0 preserves legacy mining"
        ),
    )
    parser.add_argument(
        "--residual-exclude-exact-ties",
        action="store_true",
        help=(
            "Exclude exact official-score collisions from residual negative "
            "mining and learn only from non-zero near margins"
        ),
    )
    parser.add_argument(
        "--center-residual",
        action="store_true",
        help="Remove query-wise constant residual shifts",
    )
    parser.add_argument(
        "--preserve-base-ties",
        action="store_true",
        help=(
            "Give candidates with an identical official baseline score the "
            "same residual correction"
        ),
    )
    parser.add_argument(
        "--break-base-ties",
        action="store_true",
        help="Learn residual tie-breaks only inside exact baseline score groups",
    )
    parser.add_argument(
        "--floor-residual-at-baseline-min",
        action="store_true",
        help="Prevent residual scores from beating each query's baseline minimum",
    )
    parser.add_argument(
        "--residual-start-epoch",
        type=int,
        default=None,
        help=(
            "Epoch that enables CTS score fusion. The safe default is the "
            "total epoch count, which trains auxiliary representations but "
            "keeps scoring identical to the official branch."
        ),
    )
    parser.add_argument("--residual-warmup-epochs", type=int, default=20)
    parser.add_argument(
        "--residual-loss-schedule",
        choices=("fusion", "auxiliary"),
        default="fusion",
        help=(
            "Weight residual ranking by the score-fusion schedule or by the "
            "auxiliary-loss warmup; V5c used auxiliary"
        ),
    )
    parser.add_argument(
        "--treatment-supervision",
        choices=("both", "factual"),
        default="both",
        help="Apply treatment classification to both domains or factual only",
    )
    parser.add_argument(
        "--shuffle-treatment-labels",
        action="store_true",
        help="Shuffle treatment labels for the auxiliary loss while keeping graph/CF inputs unchanged",
    )
    parser.add_argument("--nbatches", type=int, default=100)
    parser.add_argument("--neg-ent", type=int, default=25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--lambda-cf", type=float, default=1.0)
    parser.add_argument(
        "--cf-distance-temperature",
        type=float,
        default=0.0,
        help=(
            "If positive, weight train-only CF invariance pairs by "
            "exp(-match_distance / temperature); 0 keeps uniform weighting"
        ),
    )
    parser.add_argument(
        "--cf-objective",
        choices=("legacy", "bounded-vicreg"),
        default="legacy",
        help=(
            "Use the historical raw CF alignment or a bounded VICReg-style "
            "alignment that supports explicit anti-collapse penalties"
        ),
    )
    parser.add_argument(
        "--cf-representation-temperature", type=float, default=1.0
    )
    parser.add_argument("--lambda-cf-variance", type=float, default=0.0)
    parser.add_argument("--lambda-cf-covariance", type=float, default=0.0)
    parser.add_argument("--cf-std-floor", type=float, default=0.1)
    parser.add_argument("--lambda-treat", type=float, default=0.1)
    parser.add_argument("--lambda-orth", type=float, default=0.05)
    parser.add_argument("--lambda-gate", type=float, default=0.1)
    parser.add_argument(
        "--gate-regularizer",
        choices=("sample-kl", "batch-balance"),
        default="sample-kl",
    )
    parser.add_argument("--gate-target", type=float, default=0.5)
    parser.add_argument("--lambda-gate-variance", type=float, default=0.0)
    parser.add_argument("--gate-std-floor", type=float, default=0.05)
    parser.add_argument(
        "--lambda-invariant-variance", type=float, default=0.0
    )
    parser.add_argument("--lambda-bias-variance", type=float, default=0.0)
    parser.add_argument("--representation-std-floor", type=float, default=0.5)
    parser.add_argument(
        "--treatment-label-smoothing", type=float, default=0.0
    )
    parser.add_argument(
        "--treatment-logit-temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--lambda-bias-counterfactual", type=float, default=0.0
    )
    parser.add_argument(
        "--bias-counterfactual-margin", type=float, default=0.1
    )
    parser.add_argument("--lambda-branch-balance", type=float, default=0.0)
    parser.add_argument("--branch-balance-target", type=float, default=0.5)
    parser.add_argument(
        "--lambda-delta-magnitude",
        type=float,
        default=0.0,
        help="L2 penalty on raw invariant and bias score deltas",
    )
    parser.add_argument("--lambda-residual-rank", type=float, default=1.0)
    parser.add_argument("--residual-tie-band", type=float, default=0.0, help="Exponentially focus residual ranking on baseline-near ties; 0 keeps the legacy loss")
    parser.add_argument("--lambda-invariant-rank", type=float, default=0.0)
    parser.add_argument("--lambda-bias-rank", type=float, default=0.0)
    parser.add_argument(
        "--lambda-residual-preserve", type=float, default=1.0
    )
    parser.add_argument("--warmup-start-epoch", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--cf-cache", default=None)
    parser.add_argument("--checkpoint-dir", default="./checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--init-official-checkpoint",
        default=None,
        help=(
            "Initialize only the official sf_lin_layer/lin parameters from "
            "a compatible checkpoint; residual, gate, and treatment modules "
            "remain freshly initialized"
        ),
    )
    parser.add_argument("--resume-training-state", default=None)
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--history-jsonl", default=None)
    parser.add_argument("--config-manifest", default=None)
    parser.add_argument("--baseline-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--shared-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--invariant-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--bias-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--gate-lr-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--baseline-rank-score-space",
        choices=("probability", "logit"),
        default="probability",
        help=(
            "Score space used only by the isolated official rank loss; "
            "evaluation remains unchanged"
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument(
        "--lr-scheduler", choices=("none", "cosine"), default="none"
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument(
        "--architecture-diagnostic-interval",
        type=int,
        default=100,
        help="Write a low-frequency architecture contribution probe every N epochs; 0 disables it",
    )
    parser.add_argument("--architecture-diagnostic-log", default=None)
    parser.add_argument("--run-tag", default="scratch_v1")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip the post-training one-batch diagnostics snapshot",
    )
    parser.add_argument(
        "--eval-split", choices=("valid", "test"), default="valid"
    )
    parser.add_argument(
        "--eval-residual-multiplier", type=float, default=0.03
    )
    parser.add_argument(
        "--residual-max-top-gap-ratio", type=float, default=None
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.dataset or "/" in args.dataset or "\\" in args.dataset:
        raise ValueError("dataset must be a non-empty name without path separators")
    if args.residual_start_epoch is None:
        args.residual_start_epoch = args.epochs
    if args.hash_hops != 1:
        raise ValueError("This V7a/V8b main line requires --hash-hops 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.residual_start_epoch < 0:
        raise ValueError("residual-start-epoch cannot be negative")
    if args.residual_start_epoch > args.epochs:
        raise ValueError("residual-start-epoch cannot exceed epochs")
    if args.architecture_diagnostic_interval < 0:
        raise ValueError("architecture-diagnostic-interval cannot be negative")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval cannot be negative")
    use_gpu = args.device == "cuda"
    set_seeds_all(args.seed)

    cache_path = args.cf_cache or (
        "./counterfactual_cache/{}_popularity_h{}.pt".format(
            args.dataset, args.hash_hops
        )
    )
    cache = load_counterfactual_cache(cache_path)
    expected_dim = args.hash_hops * (args.hash_hops ** 2 + 3)
    train_triples = read_openke_triples(
        os.path.join(args.data_path, "train2id.txt")
    )
    train_fingerprint = validate_counterfactual_cache(
        cache,
        train_triples,
        dataset=args.dataset,
        hash_hops=args.hash_hops,
        feature_mode="triple",
        feature_dim=expected_dim,
    )
    print("Validated train-only CF cache fingerprint: {}".format(
        train_fingerprint
    ))

    official_train_dataloader = TrainDataLoader(
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
    )
    train_dataloader = CounterfactualBatchEnricher(
        official_train_dataloader, cache
    )

    # Keep the official baseline initialization order exactly: both data
    # loaders are constructed before the model.  --skip-test controls only the
    # final evaluator and must not change the seeded model initialization.
    test_dataloader = TestDataLoader(
        args.data_path,
        "link",
        device=args.device,
        max_hash_hops=args.hash_hops,
        split=args.eval_split,
    )

    predictor_class = (
        CTSNetTreatmentAwareResidualPredictor
        if args.treatment_aware
        else CTSNetIndependentResidualPredictor
    )
    predictor = predictor_class(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        max_hash_hops=args.hash_hops,
        hidden_dim=args.hidden_dim,
        branch_dim=args.branch_dim,
        residual_scale=args.residual_scale,
        residual_multiplier=0.0,
        branch_activation=args.branch_activation,
        branch_negative_slope=args.branch_negative_slope,
        delta_limit=args.delta_limit,
        delta_temperature=args.delta_temperature,
        delta_normalization=args.delta_normalization,
        delta_target_rms=args.delta_target_rms,
        gate_temperature=args.gate_temperature,
        fusion_mode=args.fusion_mode,
        invariant_margin_fraction=args.invariant_margin_fraction,
        **(
            {
                "center_residual": args.center_residual,
                "preserve_base_ties": args.preserve_base_ties,
                "break_base_ties": args.break_base_ties,
                "floor_residual_at_baseline_min": (
                    args.floor_residual_at_baseline_min
                ),
                "residual_max_top_gap_ratio": (
                    args.residual_max_top_gap_ratio
                ),
                "invariant_margin_ungated": args.invariant_margin_ungated,
                "structural_abstention": args.structural_abstention,
                "structural_abstention_min_span_ratio": (
                    args.structural_abstention_min_span_ratio
                ),
                "structural_abstention_require_gate_agreement": (
                    not args.no_structural_abstention_gate_agreement
                ),
                "isolate_treatment_gradients": (
                    args.isolate_treatment_gradients
                ),
                "treatment_shared_gradient_scale": (
                    args.treatment_shared_gradient_scale
                ),
            }
            if args.treatment_aware
            else {}
        ),
    )
    predictor.set_branch_multipliers(
        invariant=args.invariant_branch_multiplier,
        bias=args.bias_branch_multiplier,
    )
    predictor.set_score_space(args.score_space)
    if args.treatment_aware:
        predictor.set_train_triples(cache["triples"].cpu().numpy())
    if args.resume_checkpoint and args.init_official_checkpoint:
        raise ValueError(
            "--resume-checkpoint and --init-official-checkpoint are mutually exclusive"
        )
    if args.init_official_checkpoint:
        source_state = torch.load(
            args.init_official_checkpoint, map_location="cpu"
        )
        if not isinstance(source_state, dict):
            raise ValueError("official initialization checkpoint must be a state dict")
        target_state = predictor.state_dict()
        official_prefixes = ("sf_lin_layer.", "lin.")
        official_state = {
            key: value
            for key, value in source_state.items()
            if key.startswith(official_prefixes)
        }
        expected_keys = {
            key for key in target_state if key.startswith(official_prefixes)
        }
        missing = sorted(expected_keys.difference(official_state))
        unexpected = sorted(set(official_state).difference(expected_keys))
        mismatched = sorted(
            key
            for key in expected_keys.intersection(official_state)
            if tuple(official_state[key].shape) != tuple(target_state[key].shape)
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "incompatible official initialization checkpoint: "
                "missing={}, unexpected={}, shape_mismatch={}".format(
                    missing, unexpected, mismatched
                )
            )
        predictor.load_state_dict(official_state, strict=False)
        predictor.train()
        print(
            "Initialized {} official tensors from {}; residual adapter reset".format(
                len(official_state), args.init_official_checkpoint
            )
        )
    if args.resume_checkpoint:
        predictor.load_checkpoint(args.resume_checkpoint)
        predictor.train()
        print("Loaded resume checkpoint: {}".format(args.resume_checkpoint))
    model = CTSNegativeSampling(
        model=predictor,
        loss=MarginLoss(margin=args.margin),
        batch_size=train_dataloader.get_batch_size(),
        lambda_cf=args.lambda_cf,
        cf_distance_temperature=args.cf_distance_temperature,
        cf_objective=args.cf_objective,
        cf_representation_temperature=(
            args.cf_representation_temperature
        ),
        lambda_cf_variance=args.lambda_cf_variance,
        lambda_cf_covariance=args.lambda_cf_covariance,
        cf_std_floor=args.cf_std_floor,
        lambda_treat=args.lambda_treat,
        lambda_orth=args.lambda_orth,
        lambda_gate=args.lambda_gate,
        lambda_residual_rank=args.lambda_residual_rank,
        lambda_invariant_rank=args.lambda_invariant_rank,
        lambda_bias_rank=args.lambda_bias_rank,
        lambda_residual_preserve=args.lambda_residual_preserve,
        residual_hard_negatives=args.residual_hard_negatives,
        residual_near_tie_negatives=args.residual_near_tie_negatives,
        residual_min_treatment_gap=args.residual_min_treatment_gap,
        residual_exclude_exact_ties=args.residual_exclude_exact_ties,
        center_residual=args.center_residual,
        preserve_base_ties=args.preserve_base_ties,
        break_base_ties=args.break_base_ties,
        floor_residual_at_baseline_min=args.floor_residual_at_baseline_min,
        isolate_base_rank=True,
        baseline_rank_score_space=args.baseline_rank_score_space,
        warmup_start_epoch=args.warmup_start_epoch,
        warmup_epochs=args.warmup_epochs,
        residual_start_epoch=args.residual_start_epoch,
        residual_warmup_epochs=args.residual_warmup_epochs,
        residual_loss_schedule=args.residual_loss_schedule,
        treatment_supervision=args.treatment_supervision,
        shuffle_treatment_labels=args.shuffle_treatment_labels,
        lambda_invariant_variance=args.lambda_invariant_variance,
        lambda_bias_variance=args.lambda_bias_variance,
        representation_std_floor=args.representation_std_floor,
        gate_regularizer=args.gate_regularizer,
        gate_target=args.gate_target,
        lambda_gate_variance=args.lambda_gate_variance,
        gate_std_floor=args.gate_std_floor,
        treatment_label_smoothing=args.treatment_label_smoothing,
        treatment_logit_temperature=args.treatment_logit_temperature,
        lambda_bias_counterfactual=args.lambda_bias_counterfactual,
        bias_counterfactual_margin=args.bias_counterfactual_margin,
        lambda_branch_balance=args.lambda_branch_balance,
        branch_balance_target=args.branch_balance_target,
        residual_tie_band=args.residual_tie_band,
        lambda_delta_magnitude=args.lambda_delta_magnitude,
    )

    print(
        "CTS-Net from-scratch config: epochs={}, lr={}, optimizer={}, "
        "dataset={}, margin={}, hash_hops={}, nbatches={}, neg_ent={}, seed={}, "
        "residual_scale={}, score_space={}, delta_normalization={}, delta_target_rms={}, "
        "treatment_aware={}, residual_hard_negatives={}, "
        "residual_near_tie_negatives={}, "
        "residual_min_treatment_gap={}, "
        "residual_exclude_exact_ties={}, "
        "center_residual={}, preserve_base_ties={}, break_base_ties={}, floor_at_base_min={}, "
        "residual_start={}, residual_warmup={}, residual_loss_schedule={}, "
        "treatment_supervision={}, "
        "lambda_cf={}, lambda_treat={}, lambda_orth={}, lambda_gate={}, "
        "lambda_residual_rank={}, lambda_invariant_rank={}, "
        "lambda_bias_rank={}, lambda_residual_preserve={}, "
        "isolate_base_rank=True, baseline_rank_score_space={}, "
        "lambda_delta_magnitude={}, "
        "aux_start={}, aux_warmup={}".format(
            args.epochs,
            args.lr,
            args.optimizer,
            args.dataset,
            args.margin,
            args.hash_hops,
            args.nbatches,
            args.neg_ent,
            args.seed,
            args.residual_scale,
            args.score_space,
            args.delta_normalization,
            args.delta_target_rms,
            args.treatment_aware,
            args.residual_hard_negatives,
            args.residual_near_tie_negatives,
            args.residual_min_treatment_gap,
            args.residual_exclude_exact_ties,
            args.center_residual,
            args.preserve_base_ties,
            args.break_base_ties,
            args.floor_residual_at_baseline_min,
            args.residual_start_epoch,
            args.residual_warmup_epochs,
            args.residual_loss_schedule,
            args.treatment_supervision,
            args.lambda_cf,
            args.lambda_treat,
            args.lambda_orth,
            args.lambda_gate,
            args.lambda_residual_rank,
            args.lambda_invariant_rank,
            args.lambda_bias_rank,
            args.lambda_residual_preserve,
            args.baseline_rank_score_space,
            args.lambda_delta_magnitude,
            args.warmup_start_epoch,
            args.warmup_epochs,
        )
    )
    print(
        "Counterfactual objective: mode={}, distance_temperature={}, "
        "representation_temperature={}, lambda_variance={}, "
        "lambda_covariance={}, std_floor={}".format(
            args.cf_objective,
            args.cf_distance_temperature,
            args.cf_representation_temperature,
            args.lambda_cf_variance,
            args.lambda_cf_covariance,
            args.cf_std_floor,
        )
    )
    os.makedirs(args.log_dir, exist_ok=True)
    architecture_diagnostic_log = args.architecture_diagnostic_log or os.path.join(
        args.log_dir,
        "ctsnet_{}_architecture_diagnostics.jsonl".format(args.run_tag),
    )
    history_jsonl = args.history_jsonl or os.path.join(
        args.log_dir, "ctsnet_{}_epochs.jsonl".format(args.run_tag)
    )
    config_manifest = args.config_manifest or os.path.join(
        args.log_dir, "ctsnet_{}_config.json".format(args.run_tag)
    )
    with open(config_manifest, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    print("Saved run config manifest: {}".format(config_manifest))
    trainer = CTSTrainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.epochs,
        opt_method=args.optimizer,
        alpha=args.lr,
        use_gpu=use_gpu,
        architecture_diagnostic_interval=args.architecture_diagnostic_interval,
        architecture_diagnostic_path=architecture_diagnostic_log,
        history_jsonl_path=history_jsonl,
        save_steps=args.checkpoint_interval or None,
        checkpoint_dir=os.path.join(args.checkpoint_dir, args.run_tag),
        baseline_lr_multiplier=args.baseline_lr_multiplier,
        shared_lr_multiplier=args.shared_lr_multiplier,
        invariant_lr_multiplier=args.invariant_lr_multiplier,
        bias_lr_multiplier=args.bias_lr_multiplier,
        gate_lr_multiplier=args.gate_lr_multiplier,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler=args.lr_scheduler,
        min_lr_ratio=args.min_lr_ratio,
        resume_training_state=args.resume_training_state,
    )
    history = trainer.run()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "ctsnet_{}_{}_h{}_{}_{}_ep{}.ckpt".format(
            args.run_tag,
            args.dataset,
            args.hash_hops,
            args.lr,
            args.optimizer,
            args.epochs,
        ),
    )
    predictor.save_checkpoint(checkpoint_path)
    print("Saved from-scratch checkpoint: {}".format(checkpoint_path))

    os.makedirs(args.log_dir, exist_ok=True)
    history_path = os.path.join(
        args.log_dir,
        "ctsnet_{}_{}_h{}_ep{}_history.json".format(
            args.run_tag, args.dataset, args.hash_hops, args.epochs
        ),
    )
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print("Saved decomposed loss history: {}".format(history_path))
    if args.architecture_diagnostic_interval > 0:
        print("Saved architecture diagnostics: {}".format(architecture_diagnostic_log))

    if not args.skip_diagnostics:
        predictor.load_checkpoint(checkpoint_path)
        predictor.eval()
        diagnostic_batch = next(iter(train_dataloader))
        diagnostics = predictor.diagnostics({
            "subgraph_feature": diagnostic_batch["subgraph_feature"],
            "batch_h": diagnostic_batch["batch_h"],
            "batch_t": diagnostic_batch["batch_t"],
            "batch_r": diagnostic_batch["batch_r"],
        })
        print(
            "Final from-scratch diagnostics: "
            + ", ".join(
                "{}={:.6f}".format(key, value)
                for key, value in diagnostics.items()
            )
        )

    if not args.skip_test:
        if args.skip_diagnostics:
            predictor.load_checkpoint(checkpoint_path)
            predictor.eval()
        predictor.set_residual_multiplier(args.eval_residual_multiplier)
        tester = Tester(
            model=predictor,
            data_loader=test_dataloader,
            use_gpu=use_gpu,
        )
        mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(False)
        print(
            "Filtered ranking ({}): MRR={:.6f}, MR={:.6f}, Hits@10={:.6f}, "
            "Hits@3={:.6f}, Hits@1={:.6f}".format(
                args.eval_split, mrr, mr, hit10, hit3, hit1
            )
        )


if __name__ == "__main__":
    main()

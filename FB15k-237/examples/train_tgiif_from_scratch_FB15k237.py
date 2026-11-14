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
        default="FB15k-237",
        help="Dataset name recorded in the cache/checkpoint metadata",
    )
    parser.add_argument("--data-path", default="./benchmarks/FB15k-237/")
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
        "--fusion-mode",
        choices=("additive", "invariant-margin"),
        default="additive",
        help="Use baseline-preserving invariant margin fusion",
    )
    parser.add_argument("--invariant-margin-fraction", type=float, default=0.25)
    parser.add_argument("--invariant-branch-multiplier", type=float, default=1.0)
    parser.add_argument("--bias-branch-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--treatment-aware",
        action="store_true",
        help="Provide train-only endpoint popularity to the Bias branch",
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
    parser.add_argument("--resume-training-state", default=None)
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--history-jsonl", default=None)
    parser.add_argument("--config-manifest", default=None)
    parser.add_argument("--baseline-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--shared-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--invariant-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--bias-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--gate-lr-multiplier", type=float, default=1.0)
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
            }
            if args.treatment_aware
            else {}
        ),
    )
    predictor.set_branch_multipliers(
        invariant=args.invariant_branch_multiplier,
        bias=args.bias_branch_multiplier,
    )
    if args.treatment_aware:
        predictor.set_train_triples(cache["triples"].cpu().numpy())
    if args.resume_checkpoint:
        predictor.load_checkpoint(args.resume_checkpoint)
        predictor.train()
        print("Loaded resume checkpoint: {}".format(args.resume_checkpoint))
    model = CTSNegativeSampling(
        model=predictor,
        loss=MarginLoss(margin=args.margin),
        batch_size=train_dataloader.get_batch_size(),
        lambda_cf=args.lambda_cf,
        lambda_treat=args.lambda_treat,
        lambda_orth=args.lambda_orth,
        lambda_gate=args.lambda_gate,
        lambda_residual_rank=args.lambda_residual_rank,
        lambda_invariant_rank=args.lambda_invariant_rank,
        lambda_bias_rank=args.lambda_bias_rank,
        lambda_residual_preserve=args.lambda_residual_preserve,
        residual_hard_negatives=args.residual_hard_negatives,
        residual_near_tie_negatives=args.residual_near_tie_negatives,
        center_residual=args.center_residual,
        preserve_base_ties=args.preserve_base_ties,
        break_base_ties=args.break_base_ties,
        floor_residual_at_baseline_min=args.floor_residual_at_baseline_min,
        isolate_base_rank=True,
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
    )

    print(
        "CTS-Net from-scratch config: epochs={}, lr={}, optimizer={}, "
        "dataset={}, margin={}, hash_hops={}, nbatches={}, neg_ent={}, seed={}, "
        "residual_scale={}, delta_normalization={}, delta_target_rms={}, "
        "treatment_aware={}, residual_hard_negatives={}, "
        "residual_near_tie_negatives={}, "
        "center_residual={}, preserve_base_ties={}, break_base_ties={}, floor_at_base_min={}, "
        "residual_start={}, residual_warmup={}, residual_loss_schedule={}, "
        "treatment_supervision={}, "
        "lambda_cf={}, lambda_treat={}, lambda_orth={}, lambda_gate={}, "
        "lambda_residual_rank={}, lambda_invariant_rank={}, "
        "lambda_bias_rank={}, lambda_residual_preserve={}, "
        "isolate_base_rank=True, "
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
            args.delta_normalization,
            args.delta_target_rms,
            args.treatment_aware,
            args.residual_hard_negatives,
            args.residual_near_tie_negatives,
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
            args.warmup_start_epoch,
            args.warmup_epochs,
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

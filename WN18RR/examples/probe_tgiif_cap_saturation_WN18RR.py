"""Read-only diagnostic for invariant-margin clipping and gate activity.

This script loads an existing checkpoint, performs no optimizer step, and
writes only aggregate statistics below the requested output path.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "examples"))

from evaluate_ctsnet_checkpoint_WN18RR import build_predictor  # noqa: E402
from openke.data import TestDataLoader, TrainDataLoader  # noqa: E402
from openke.utils import set_seeds_all  # noqa: E402
from probe_tgiif_score_pathways_WN18RR import prepare_batch, read_openke  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--data-path", default="./benchmarks/WN18RR/")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--hash-hops", type=int, default=1)
    parser.add_argument("--feature-mode", choices=("triple", "pairwise"), default="triple")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--branch-dim", type=int, default=64)
    parser.add_argument("--branch-activation", choices=("relu", "leaky_relu", "silu"), default="relu")
    parser.add_argument("--branch-negative-slope", type=float, default=0.01)
    parser.add_argument("--delta-limit", type=float, default=None)
    parser.add_argument("--delta-temperature", type=float, default=1.0)
    parser.add_argument("--delta-normalization", choices=("auto", "none", "query-rms"), default="auto")
    parser.add_argument("--delta-target-rms", type=float, default=None)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--invariant-margin-fraction", type=float, default=0.0001)
    parser.add_argument("--residual-multiplier", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--nbatches", type=int, default=100)
    parser.add_argument("--neg-ent", type=int, default=25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-triples", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_args(args):
    return SimpleNamespace(
        architecture="treatment-aware",
        checkpoint=args.checkpoint,
        hash_hops=args.hash_hops,
        feature_mode=args.feature_mode,
        hidden_dim=args.hidden_dim,
        branch_dim=args.branch_dim,
        branch_activation=args.branch_activation,
        branch_negative_slope=args.branch_negative_slope,
        delta_limit=args.delta_limit,
        delta_temperature=args.delta_temperature,
        delta_normalization=args.delta_normalization,
        delta_target_rms=args.delta_target_rms,
        gate_temperature=args.gate_temperature,
        fusion_mode="invariant-margin",
        invariant_margin_fraction=args.invariant_margin_fraction,
        normalization="batch",
        floor_residual_at_baseline_min=False,
        residual_max_top_gap_ratio=None,
        residual_multiplier=args.residual_multiplier,
        disable_invariant_branch=False,
        disable_bias_branch=False,
        invariant_branch_multiplier=1.0,
        bias_branch_multiplier=1.0,
        device=args.device,
    )


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.hash_hops != 1 or args.feature_mode != "triple":
        raise ValueError("The canonical TGIIF probe requires hash_hops=1 and feature_mode=triple")
    set_seeds_all(args.seed)
    data_path = Path(args.data_path)
    train_loader = TrainDataLoader(
        in_path=str(data_path) + "/",
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
    predictor = build_predictor(build_args(args), train_loader.get_ent_tot(), train_loader.get_rel_tot())
    predictor.eval()
    device = next(predictor.parameters()).device
    loader = TestDataLoader(
        str(data_path) + "/",
        "link",
        device=args.device,
        max_hash_hops=args.hash_hops,
        feature_mode=args.feature_mode,
        split=args.split,
    )
    triple_count = len(read_openke(data_path / (args.split + "2id.txt")))
    if args.max_triples > 0:
        triple_count = min(triple_count, args.max_triples)

    query_gate_mean = []
    query_gate_std = []
    query_saturation = []
    query_cap = []
    query_pre_cap_abs = []
    candidate_count = 0
    saturated_count = 0
    positive_cap_candidate_count = 0
    positive_cap_saturated_count = 0
    native_gate_one_equal_count = 0
    native_gate_one_exact_equal_count = 0
    same_invariant_sign_count = 0
    residual_positive = 0
    residual_negative = 0
    residual_zero = 0
    residual_exact_positive = 0
    residual_exact_negative = 0
    residual_exact_zero = 0
    residual_abs_sum = 0.0
    residual_abs_max = 0.0
    gate_sum = 0.0
    gate_sq_sum = 0.0
    gate_min = float("inf")
    gate_max = float("-inf")

    with torch.no_grad():
        for triple_index, pair in enumerate(tqdm(iter(loader), total=triple_count, desc="cap probe")):
            if triple_index >= triple_count:
                break
            for data in pair:
                components = predictor.forward_components(prepare_batch(data, device))
                gate = components["gate"]
                inv = components["invariant_delta"]
                base = components["base_score"]
                mixed = predictor.invariant_branch_multiplier * gate * inv
                unit = predictor._prepare_residual_unit(mixed, base, True)
                pre_cap = float(predictor._residual_multiplier.item()) * unit
                cap = predictor.invariant_margin_fraction * (base.max() - base.min()).detach()
                bounded = pre_cap.clamp(-cap, cap)
                one_unit = predictor._prepare_residual_unit(inv, base, True)
                gate_one = (float(predictor._residual_multiplier.item()) * one_unit).clamp(-cap, cap)

                tolerance = torch.maximum(cap.abs() * 1e-6, torch.as_tensor(1e-12, device=device))
                saturated = pre_cap.abs() >= (cap.abs() - tolerance)
                count = gate.numel()
                candidate_count += count
                saturated_count += int(saturated.sum().item())
                if float(cap.item()) > 0.0:
                    positive_cap_candidate_count += count
                    positive_cap_saturated_count += int(saturated.sum().item())
                native_gate_one_equal_count += int(torch.isclose(bounded, gate_one, rtol=0.0, atol=1e-12).sum().item())
                native_gate_one_exact_equal_count += int((bounded == gate_one).sum().item())
                same_invariant_sign_count += int(bool(torch.all(inv >= 0) or torch.all(inv <= 0)))
                residual_positive += int((bounded > 1e-12).sum().item())
                residual_negative += int((bounded < -1e-12).sum().item())
                residual_zero += int((bounded.abs() <= 1e-12).sum().item())
                residual_exact_positive += int((bounded > 0).sum().item())
                residual_exact_negative += int((bounded < 0).sum().item())
                residual_exact_zero += int((bounded == 0).sum().item())
                residual_abs_sum += float(bounded.abs().sum().item())
                residual_abs_max = max(residual_abs_max, float(bounded.abs().max().item()))
                gate_sum += float(gate.sum().item())
                gate_sq_sum += float((gate * gate).sum().item())
                gate_min = min(gate_min, float(gate.min().item()))
                gate_max = max(gate_max, float(gate.max().item()))
                query_gate_mean.append(float(gate.mean().item()))
                query_gate_std.append(float(gate.std(unbiased=False).item()))
                query_saturation.append(float(saturated.float().mean().item()))
                query_cap.append(float(cap.item()))
                query_pre_cap_abs.append(float(pre_cap.abs().mean().item()))

    query_count = len(query_gate_mean)
    gate_mean = gate_sum / candidate_count
    gate_variance = max(gate_sq_sum / candidate_count - gate_mean * gate_mean, 0.0)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "seed": args.seed,
        "residual_multiplier": args.residual_multiplier,
        "invariant_margin_fraction": args.invariant_margin_fraction,
        "triple_count": triple_count,
        "query_count": query_count,
        "candidate_count": candidate_count,
        "candidate_saturation_fraction_including_zero_cap": saturated_count / candidate_count,
        "candidate_saturation_fraction_positive_cap": (
            positive_cap_saturated_count / positive_cap_candidate_count
            if positive_cap_candidate_count else None
        ),
        "positive_cap_candidate_fraction": positive_cap_candidate_count / candidate_count,
        "candidate_native_equals_gate_one_fraction": native_gate_one_equal_count / candidate_count,
        "candidate_native_exactly_equals_gate_one_fraction": native_gate_one_exact_equal_count / candidate_count,
        "query_all_invariant_same_sign_fraction": same_invariant_sign_count / query_count,
        "bounded_residual_sign_fraction": {
            "negative": residual_negative / candidate_count,
            "zero": residual_zero / candidate_count,
            "positive": residual_positive / candidate_count,
        },
        "bounded_residual_exact_sign_fraction": {
            "negative": residual_exact_negative / candidate_count,
            "zero": residual_exact_zero / candidate_count,
            "positive": residual_exact_positive / candidate_count,
        },
        "bounded_residual_abs": {
            "mean": residual_abs_sum / candidate_count,
            "max": residual_abs_max,
        },
        "candidate_gate": {
            "mean": gate_mean,
            "std": gate_variance ** 0.5,
            "min": gate_min,
            "max": gate_max,
        },
        "query_gate_mean": describe(query_gate_mean),
        "query_gate_std": describe(query_gate_std),
        "query_saturation_fraction": describe(query_saturation),
        "query_cap": describe(query_cap),
        "query_pre_cap_abs_mean": describe(query_pre_cap_abs),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Saved cap saturation probe: {}".format(output))


if __name__ == "__main__":
    main()

"""Low-cost, checkpoint-only pathway interventions for TGIIF on WN18RR.

The script never mutates or saves model parameters.  It evaluates several
counterfactual score constructions from one shared forward pass and implements
the same strict, filtered rank used by OpenKE's ``Test.h``.  The ``native``
row must be checked against the official evaluator before interpreting any
intervention.
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
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "examples"))

from evaluate_ctsnet_checkpoint_WN18RR import build_predictor  # noqa: E402
from openke.data import TestDataLoader, TrainDataLoader  # noqa: E402
from openke.utils import set_seeds_all  # noqa: E402


VARIANTS = (
    "baseline_m0",
    "native",
    "gate_one",
    "gate_query_mean",
    "gate_shuffled",
    "popularity_zero",
    "invariant_shuffled",
    "invariant_sign_flip",
)


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
    parser.add_argument("--fusion-mode", choices=("additive", "invariant-margin"), default="invariant-margin")
    parser.add_argument("--invariant-margin-fraction", type=float, default=0.0001)
    parser.add_argument("--residual-multiplier", type=float, default=0.75)
    parser.add_argument("--residual-max-top-gap-ratio", type=float, default=None)
    parser.add_argument("--floor-residual-at-baseline-min", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--nbatches", type=int, default=100)
    parser.add_argument("--neg-ent", type=int, default=25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-triples", type=int, default=0, help="Debug only; zero evaluates the full split")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_openke(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        expected = int(handle.readline().strip())
        triples = np.asarray(
            [[int(value) for value in line.split()] for line in handle if line.strip()],
            dtype=np.int64,
        )
    if triples.shape != (expected, 3):
        raise ValueError("Malformed OpenKE triple file: {}".format(path))
    return triples


def read_total(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return int(handle.readline().strip())


def make_build_args(args):
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
        fusion_mode=args.fusion_mode,
        invariant_margin_fraction=args.invariant_margin_fraction,
        normalization="batch",
        floor_residual_at_baseline_min=args.floor_residual_at_baseline_min,
        residual_max_top_gap_ratio=args.residual_max_top_gap_ratio,
        residual_multiplier=args.residual_multiplier,
        disable_invariant_branch=False,
        disable_bias_branch=False,
        invariant_branch_multiplier=1.0,
        bias_branch_multiplier=1.0,
        device=args.device,
    )


def prepare_batch(data, device):
    result = {
        "batch_h": torch.as_tensor(data["batch_h"], dtype=torch.long, device=device),
        "batch_t": torch.as_tensor(data["batch_t"], dtype=torch.long, device=device),
        "batch_r": torch.as_tensor(data["batch_r"], dtype=torch.long, device=device),
        "subgraph_feature": data["subgraph_feature"],
        "mode": data["mode"],
    }
    return result


def rebuild_score(model, components, gate, inv_delta):
    base_score = components["base_score"]
    mixed = model.invariant_branch_multiplier * gate * inv_delta
    unit = model._prepare_residual_unit(mixed, base_score, True)
    bounded = float(model._residual_multiplier.item()) * unit
    if model.fusion_mode == "invariant-margin" and base_score.shape[0] > 1:
        score_range = (base_score.max() - base_score.min()).detach()
        cap = model.invariant_margin_fraction * score_range
        bounded = bounded.clamp(-cap, cap)
    if not model._query_allows_residual(base_score):
        bounded = torch.zeros_like(bounded)
    score = base_score + bounded
    if model.floor_residual_at_baseline_min and base_score.shape[0] > 1:
        baseline_min = base_score.min().detach()
        score = torch.where(
            base_score == baseline_min,
            base_score,
            torch.where(score < baseline_min, base_score, score),
        )
    return score


def variant_scores(model, components, generator):
    gate = components["gate"]
    inv = components["invariant_delta"]
    count = gate.shape[0]
    perm_gate = torch.randperm(count, generator=generator, device="cpu").to(gate.device)
    perm_inv = torch.randperm(count, generator=generator, device="cpu").to(inv.device)
    zero_popularity = torch.zeros_like(components["treatment_feature"])
    zero_gate_input = torch.cat((components["z_shared"], zero_popularity), dim=-1)
    gate_zero_popularity = torch.sigmoid(
        model.gate_head(zero_gate_input) / model.gate_temperature
    )
    return {
        "baseline_m0": components["base_score"],
        "native": components["score"],
        "gate_one": rebuild_score(model, components, torch.ones_like(gate), inv),
        "gate_query_mean": rebuild_score(model, components, gate.mean().expand_as(gate), inv),
        "gate_shuffled": rebuild_score(model, components, gate[perm_gate], inv),
        "popularity_zero": rebuild_score(model, components, gate_zero_popularity, inv),
        "invariant_shuffled": rebuild_score(model, components, gate, inv[perm_inv]),
        "invariant_sign_flip": rebuild_score(model, components, gate, -inv),
    }


def metric_summary(ranks):
    values = np.asarray(ranks, dtype=np.float64)
    return {
        "count": int(values.size),
        "mrr": float(np.mean(1.0 / values)),
        "mr": float(np.mean(values)),
        "hits10": float(np.mean(values <= 10)),
        "hits3": float(np.mean(values <= 3)),
        "hits1": float(np.mean(values <= 1)),
    }


def percentile_label(values):
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    return np.searchsorted(edges, values, side="right") + 1, edges


def bootstrap_differences(rank_by_variant, repetitions, seed):
    if repetitions <= 0:
        return {}
    native = np.asarray(rank_by_variant["native"], dtype=np.float64).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    output = {}
    for variant in VARIANTS:
        if variant == "native":
            continue
        other = np.asarray(rank_by_variant[variant], dtype=np.float64).reshape(-1, 2)
        observed = float(np.mean(1.0 / native) - np.mean(1.0 / other))
        samples = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            chosen = rng.integers(0, native.shape[0], size=native.shape[0])
            samples[index] = np.mean(1.0 / native[chosen]) - np.mean(1.0 / other[chosen])
        low, high = np.quantile(samples, [0.025, 0.975])
        output[variant] = {
            "native_minus_variant_mrr": observed,
            "cluster_bootstrap_ci95": [float(low), float(high)],
            "bootstrap_repetitions": repetitions,
        }
    return output


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.hash_hops != 1 or args.feature_mode != "triple":
        raise ValueError("The canonical TGIIF probe requires hash_hops=1 and feature_mode=triple")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)
    set_seeds_all(args.seed)

    # Preserve the proven OpenKE initialization order.  Base.so owns process-
    # global dataset state; constructing the official training loader first is
    # required even though this probe never iterates it.
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
    ent_tot = train_loader.get_ent_tot()
    rel_tot = train_loader.get_rel_tot()
    predictor = build_predictor(make_build_args(args), ent_tot, rel_tot)
    predictor.eval()
    parameter = next(predictor.parameters())
    device = parameter.device

    split_count = len(read_openke(data_path / (args.split + "2id.txt")))
    if args.max_triples > 0:
        split_count = min(split_count, args.max_triples)
    all_triples = np.concatenate(
        [read_openke(data_path / name) for name in ("train2id.txt", "valid2id.txt", "test2id.txt")],
        axis=0,
    )
    true_heads = defaultdict(set)
    true_tails = defaultdict(set)
    for head, tail, relation in all_triples:
        true_heads[(int(tail), int(relation))].add(int(head))
        true_tails[(int(head), int(relation))].add(int(tail))

    loader = TestDataLoader(
        str(data_path) + "/",
        "link",
        device=args.device,
        max_hash_hops=args.hash_hops,
        feature_mode=args.feature_mode,
        split=args.split,
    )
    rank_by_variant = {name: [] for name in VARIANTS}
    query_rows = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1701)

    with torch.no_grad():
        iterator = iter(loader)
        for triple_index, pair in enumerate(
            tqdm(iterator, total=split_count, desc="pathway probe")
        ):
            if triple_index >= split_count:
                break
            # Base.so sorts its internal testList, so file order is not a
            # valid target mapping.  Recover the true triple from the paired
            # candidate batches themselves.
            head = int(np.asarray(pair[1]["batch_h"]).reshape(-1)[0])
            tail = int(np.asarray(pair[0]["batch_t"]).reshape(-1)[0])
            relation = int(np.asarray(pair[0]["batch_r"]).reshape(-1)[0])
            popularity = float(
                predictor.entity_log_degree[head].item()
                + predictor.entity_log_degree[tail].item()
            )
            for direction, data, target, filtered in (
                ("head", pair[0], head, true_heads[(tail, relation)]),
                ("tail", pair[1], tail, true_tails[(head, relation)]),
            ):
                prepared = prepare_batch(data, device)
                components = predictor.forward_components(prepared)
                scores = variant_scores(predictor, components, generator)
                keep = torch.ones(ent_tot, dtype=torch.bool, device=device)
                if filtered:
                    keep[torch.as_tensor(sorted(filtered), dtype=torch.long, device=device)] = False
                target_gate = float(components["gate"][target].item())
                target_inv = float(components["invariant_delta"][target].item())
                for variant, score in scores.items():
                    target_score = score[target]
                    rank = int(((score.reshape(-1) < target_score) & keep).sum().item()) + 1
                    rank_by_variant[variant].append(rank)
                    query_rows.append(
                        {
                            "triple_index": triple_index,
                            "direction": direction,
                            "head": head,
                            "tail": tail,
                            "relation": relation,
                            "target": target,
                            "popularity": popularity,
                            "variant": variant,
                            "rank": rank,
                            "reciprocal_rank": 1.0 / rank,
                            "target_gate": target_gate,
                            "target_invariant_delta": target_inv,
                        }
                    )

    summary = {name: metric_summary(rank_by_variant[name]) for name in VARIANTS}
    baseline_mrr = summary["baseline_m0"]["mrr"]
    for name in VARIANTS:
        summary[name]["delta_mrr_vs_baseline_m0"] = summary[name]["mrr"] - baseline_mrr

    base_rows = [row for row in query_rows if row["variant"] == "baseline_m0"]
    popularity_labels, popularity_edges = percentile_label([row["popularity"] for row in base_rows])
    baseline_rank_bins = []
    for row in base_rows:
        rank = row["rank"]
        baseline_rank_bins.append("1" if rank == 1 else "2-10" if rank <= 10 else "11-100" if rank <= 100 else ">100")
    group_lookup = {
        (row["triple_index"], row["direction"]): (int(popularity_labels[index]), baseline_rank_bins[index])
        for index, row in enumerate(base_rows)
    }
    grouped_ranks = defaultdict(list)
    for row in query_rows:
        quartile, rank_bin = group_lookup[(row["triple_index"], row["direction"])]
        row["popularity_quartile"] = quartile
        row["baseline_rank_bin"] = rank_bin
        grouped_ranks[(row["variant"], "popularity_quartile", str(quartile))].append(row["rank"])
        grouped_ranks[(row["variant"], "baseline_rank_bin", rank_bin)].append(row["rank"])
        grouped_ranks[(row["variant"], "relation", str(row["relation"]))].append(row["rank"])
    group_summary = []
    for (variant, group_type, group), ranks in sorted(grouped_ranks.items()):
        result = metric_summary(ranks)
        result.update({"variant": variant, "group_type": group_type, "group": group})
        group_summary.append(result)

    query_csv = output_dir / "query_ranks.csv"
    with query_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(query_rows)
    group_csv = output_dir / "group_metrics.csv"
    with group_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_summary[0].keys()))
        writer.writeheader()
        writer.writerows(group_summary)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "seed": args.seed,
        "residual_multiplier": args.residual_multiplier,
        "invariant_margin_fraction": args.invariant_margin_fraction,
        "strict_filtered_rank": True,
        "triple_count": split_count,
        "metrics": summary,
        "popularity_quartile_edges": [float(value) for value in popularity_edges],
        "paired_cluster_bootstrap": bootstrap_differences(rank_by_variant, args.bootstrap, args.seed + 99),
        "artifacts": {"query_ranks": str(query_csv), "group_metrics": str(group_csv)},
    }
    output_json = output_dir / "score_pathway_probe.json"
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Saved score pathway probe: {}".format(output_json))


if __name__ == "__main__":
    main()

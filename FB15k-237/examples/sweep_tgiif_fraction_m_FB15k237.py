#!/usr/bin/env python3
"""Single-pass filtered-ranking sweep for a fixed FB15k-237 checkpoint.

The normal OpenKE evaluator recomputes graph features and model outputs for
every (fraction, residual multiplier) pair.  This diagnostic computes the
baseline score and prepared residual once per query, then applies all bounded
fusion settings together.  Ranking matches Base.so's optimistic filtered
rule: only candidates with a strictly lower score count as better.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from tqdm import tqdm

from examples.evaluate_tgiif_checkpoint_FB15k237 import build_predictor
from openke.data import TestDataLoader, TrainDataLoader
from openke.utils import set_seeds_all


def comma_floats(value):
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one number")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data-path", default="./benchmarks/FB15k-237/")
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument(
        "--fractions",
        type=comma_floats,
        default=comma_floats("0.001,0.005,0.01,0.015,0.0175,0.02,0.025,0.03,0.05,0.1"),
    )
    parser.add_argument(
        "--multipliers",
        type=comma_floats,
        default=comma_floats("0.25,0.5,0.75,1.0,1.25"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_triples(path):
    with open(path, "r", encoding="utf-8") as handle:
        expected = int(handle.readline().strip())
        triples = []
        for line in handle:
            fields = line.split()
            if fields:
                triples.append(tuple(map(int, fields[:3])))
    if len(triples) != expected:
        raise ValueError(
            "{} declares {} triples but contains {}".format(
                path, expected, len(triples)
            )
        )
    return triples


def load_filter_maps(data_path):
    data_path = Path(data_path)
    all_triples = []
    by_file = {}
    for name in ("train2id.txt", "valid2id.txt", "test2id.txt"):
        triples = read_triples(data_path / name)
        by_file[name] = triples
        all_triples.extend(triples)
    heads = defaultdict(set)
    tails = defaultdict(set)
    for head, tail, relation in all_triples:
        heads[(tail, relation)].add(head)
        tails[(head, relation)].add(tail)
    return by_file, heads, tails


def predictor_args(args):
    return SimpleNamespace(
        architecture="treatment-aware",
        checkpoint=args.checkpoint,
        delta_normalization="auto",
        delta_target_rms=None,
        hash_hops=1,
        hidden_dim=128,
        branch_dim=64,
        feature_mode="triple",
        branch_activation="relu",
        branch_negative_slope=0.01,
        delta_limit=None,
        delta_temperature=1.0,
        gate_temperature=1.0,
        fusion_mode="invariant-margin",
        invariant_margin_fraction=0.02,
        normalization="batch",
        floor_residual_at_baseline_min=False,
        residual_max_top_gap_ratio=None,
        residual_multiplier=0.0,
        disable_invariant_branch=False,
        disable_bias_branch=False,
        invariant_branch_multiplier=1.0,
        bias_branch_multiplier=1.0,
        device=args.device,
    )


def tensor_batch(data, device):
    return {
        "batch_h": torch.from_numpy(data["batch_h"]).to(device),
        "batch_t": torch.from_numpy(data["batch_t"]).to(device),
        "batch_r": torch.from_numpy(data["batch_r"]).to(device),
        "subgraph_feature": data["subgraph_feature"],
        "mode": data["mode"],
    }


@torch.no_grad()
def main():
    args = parse_args()
    invalid_multipliers = [
        value for value in args.multipliers if value < 0.0 or value > 1.0
    ]
    if invalid_multipliers:
        raise ValueError(
            "residual multipliers must be in [0, 1], matching "
            "CTSNetIndependentResidualPredictor.set_residual_multiplier; "
            "got {}".format(invalid_multipliers)
        )
    invalid_fractions = [
        value for value in args.fractions if value <= 0.0 or value > 1.0
    ]
    if invalid_fractions:
        raise ValueError(
            "invariant-margin fractions must be in (0, 1]; got {}".format(
                invalid_fractions
            )
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_seeds_all(args.seed)
    device = torch.device(args.device)

    # Match the production evaluator's initialization order.  Base.so keeps
    # process-global dataset state, so the training loader must establish the
    # entity/relation totals before the held-out loader swaps valid/test files.
    train_loader = TrainDataLoader(
        in_path=args.data_path,
        nbatches=100,
        threads=args.threads,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0,
        device=args.device,
        max_hash_hops=1,
        feature_mode="triple",
    )
    predictor = build_predictor(
        predictor_args(args),
        train_loader.get_ent_tot(),
        train_loader.get_rel_tot(),
    )
    predictor.eval()

    loader = TestDataLoader(
        args.data_path,
        "link",
        device=args.device,
        max_hash_hops=1,
        feature_mode="triple",
        split=args.split,
    )
    by_file, filter_heads, filter_tails = load_filter_maps(args.data_path)
    split_name = "valid2id.txt" if args.split == "valid" else "test2id.txt"
    targets = sorted(by_file[split_name], key=lambda item: (item[2], item[0], item[1]))
    if len(targets) != loader.get_triple_tot():
        raise ValueError("Python and Base.so split sizes differ")

    configs = [(0.0, 0.0)]
    configs.extend(
        (fraction, multiplier)
        for multiplier in args.multipliers
        for fraction in args.fractions
    )
    fractions = torch.tensor(
        [item[0] for item in configs], dtype=torch.float32, device=device
    ).view(-1, 1)
    multipliers = torch.tensor(
        [item[1] for item in configs], dtype=torch.float32, device=device
    ).view(-1, 1)
    count = len(configs)
    rank_sum = torch.zeros(count, dtype=torch.float64, device=device)
    reciprocal_sum = torch.zeros_like(rank_sum)
    hit1 = torch.zeros_like(rank_sum)
    hit3 = torch.zeros_like(rank_sum)
    hit10 = torch.zeros_like(rank_sum)

    def accumulate(data, target, filtered_entities):
        components = predictor.forward_components(tensor_batch(data, device))
        base = components["base_score"].reshape(-1)
        residual = components["residual_unit_delta"].reshape(-1)
        score_range = (base.max() - base.min()).detach()
        cap = fractions * score_range
        delta = torch.clamp(multipliers * residual.unsqueeze(0), -cap, cap)
        if not predictor._query_allows_residual(base):
            delta.zero_()
        scores = base.unsqueeze(0) + delta
        if predictor.floor_residual_at_baseline_min:
            baseline_min = base.min().detach()
            scores = torch.where(
                (base == baseline_min).unsqueeze(0),
                base.unsqueeze(0),
                torch.where(scores < baseline_min, base.unsqueeze(0), scores),
            )
        target_scores = scores[:, target].unsqueeze(1)
        better = scores < target_scores
        filtered = [entity for entity in filtered_entities if entity != target]
        if filtered:
            filtered_index = torch.tensor(filtered, dtype=torch.long, device=device)
            better[:, filtered_index] = False
        ranks = 1 + better.sum(dim=1)
        ranks_f64 = ranks.to(torch.float64)
        rank_sum.add_(ranks_f64)
        reciprocal_sum.add_(ranks_f64.reciprocal())
        hit1.add_(ranks <= 1)
        hit3.add_(ranks <= 3)
        hit10.add_(ranks <= 10)

    loader.set_sampling_mode("link")
    for index, (data_head, data_tail) in enumerate(tqdm(loader)):
        head, tail, relation = targets[index]
        if int(data_head["batch_t"][0]) != tail or int(data_head["batch_r"][0]) != relation:
            raise ValueError("head-query order differs from sorted split order")
        if int(data_tail["batch_h"][0]) != head or int(data_tail["batch_r"][0]) != relation:
            raise ValueError("tail-query order differs from sorted split order")
        accumulate(data_head, head, filter_heads[(tail, relation)])
        accumulate(data_tail, tail, filter_tails[(head, relation)])

    denominator = float(2 * len(targets))
    results = []
    for index, (fraction, multiplier) in enumerate(configs):
        results.append(
            {
                "split": args.split,
                "fraction": fraction,
                "residual_multiplier": multiplier,
                "MRR": float((reciprocal_sum[index] / denominator).item()),
                "MR": float((rank_sum[index] / denominator).item()),
                "Hits@1": float((hit1[index] / denominator).item()),
                "Hits@3": float((hit3[index] / denominator).item()),
                "Hits@10": float((hit10[index] / denominator).item()),
            }
        )

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "split": args.split,
                "query_sides": int(denominator),
                "results": results,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    baseline = results[0]
    best = max(results, key=lambda item: item["MRR"])
    print("baseline=" + json.dumps(baseline, sort_keys=True))
    print("best=" + json.dumps(best, sort_keys=True))
    print("saved_json={}".format(json_path))
    print("saved_csv={}".format(csv_path))


if __name__ == "__main__":
    main()

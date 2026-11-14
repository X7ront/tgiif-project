import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from examples.ctsnet_ranking import (
    filtered_rank_stats,
    official_filtered_rank,
    sort_triples_like_base_so,
    summarize,
)
from openke.data import TestDataLoader, TrainDataLoader
from openke.module.model import (
    CTSNetIndependentResidualPredictor,
    CTSNetTreatmentAwareResidualPredictor,
    LinkPredictor,
)
from openke.utils import set_seeds_all


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare base and CTS residual rankings on held-out queries "
            "without invoking Base.so metrics"
        )
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--data-path", default="./benchmarks/WN18RR/")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many triples; 0 disables progress",
    )
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument(
        "--floor-residual-at-baseline-min", action="store_true"
    )
    parser.add_argument("--residual-max-top-gap-ratio", type=float, default=None)
    parser.add_argument("--hash-hops", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--branch-dim", type=int, default=64)
    parser.add_argument("--nbatches", type=int, default=100)
    parser.add_argument("--neg-ent", type=int, default=25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--include-records",
        action="store_true",
        help="Include every query-side record for paired statistical analysis",
    )
    return parser.parse_args()


def load_triples(path):
    with open(path, "r", encoding="utf-8") as handle:
        expected = int(handle.readline().strip())
        triples = [tuple(map(int, line.split())) for line in handle if line.strip()]
    if len(triples) != expected:
        raise ValueError(
            "{} declares {} triples but contains {}".format(
                path, expected, len(triples)
            )
        )
    return triples


def build_filter_maps(data_path):
    heads_by_rt = defaultdict(set)
    tails_by_hr = defaultdict(set)
    for name in ("train2id.txt", "valid2id.txt", "test2id.txt"):
        for head, tail, relation in load_triples(data_path / name):
            heads_by_rt[(relation, tail)].add(head)
            tails_by_hr[(head, relation)].add(tail)
    return heads_by_rt, tails_by_hr


def build_model(args, ent_tot, rel_tot):
    state = torch.load(args.checkpoint, map_location="cpu")
    if "entity_log_degree" in state:
        architecture = "treatment-aware"
        model_class = CTSNetTreatmentAwareResidualPredictor
    elif "residual_shared_encoder.dense1.weight" in state:
        architecture = "independent"
        model_class = CTSNetIndependentResidualPredictor
    else:
        architecture = "official"
        model_class = LinkPredictor
    model_kwargs = {
        "ent_tot": ent_tot,
        "rel_tot": rel_tot,
        "max_hash_hops": args.hash_hops,
        "hidden_dim": args.hidden_dim,
    }
    if architecture != "official":
        model_kwargs["branch_dim"] = args.branch_dim
    if architecture == "treatment-aware":
        model_kwargs.update({
            "center_residual": "center_residual_enabled" in state,
            "preserve_base_ties": "preserve_base_ties_enabled" in state,
            "break_base_ties": "break_base_ties_enabled" in state,
            "floor_residual_at_baseline_min": (
                "floor_residual_at_baseline_min_enabled" in state
            ),
        })
    model = model_class(**model_kwargs)
    model.load_state_dict(state)
    if args.floor_residual_at_baseline_min:
        if architecture != "treatment-aware":
            raise ValueError(
                "--floor-residual-at-baseline-min requires a treatment-aware checkpoint"
            )
        model.floor_residual_at_baseline_min = True
    if args.residual_max_top_gap_ratio is not None:
        if not hasattr(model, "set_residual_max_top_gap_ratio"):
            raise ValueError(
                "--residual-max-top-gap-ratio requires a treatment-aware checkpoint"
            )
        model.set_residual_max_top_gap_ratio(
            args.residual_max_top_gap_ratio
        )
    if architecture != "official":
        model.set_residual_multiplier(args.multiplier)
    return model, architecture


def signed_scores(base_components, residual_components, multiplier, sign):
    """Evaluate the same residual in either ranking direction.

    Independent checkpoints add the residual in logit space; treatment-aware
    checkpoints add it to the sigmoid score anchor. Keeping this distinction
    makes the sign audit faithful to each checkpoint's forward path.
    """
    base_logit = base_components["base_logit"]
    unit = residual_components["residual_unit_delta"]
    if "final_logit" in base_components:
        return torch.sigmoid(base_logit + float(sign) * multiplier * unit)
    return base_components["base_score"] + float(sign) * multiplier * unit


def baseline_tie_residual_stats(
    base_scores, candidates, target, filtered_entities, residual
):
    """Measure residual direction inside the exact filtered baseline tie group."""
    base_scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
    candidates = np.asarray(candidates).reshape(-1)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    target_positions = np.flatnonzero(candidates == target)
    if len(target_positions) != 1:
        raise ValueError("expected exactly one target candidate")
    target_position = int(target_positions[0])
    filtered_scores = base_scores.copy()
    for entity in filtered_entities:
        if entity != target:
            filtered_scores[candidates == entity] = np.inf
    target_score = filtered_scores[target_position]
    tie_positions = np.flatnonzero(filtered_scores == target_score)
    tie_residual = residual[tie_positions]
    target_residual = float(residual[target_position])
    peers = tie_residual[tie_positions != target_position]
    if peers.size:
        plus_wins = np.count_nonzero(target_residual > peers)
        plus_ties = np.count_nonzero(target_residual == peers)
        minus_wins = np.count_nonzero(target_residual < peers)
        minus_ties = plus_ties
        plus_win_rate = (plus_wins + 0.5 * plus_ties) / peers.size
        minus_win_rate = (minus_wins + 0.5 * minus_ties) / peers.size
        plus_percentile = (
            np.count_nonzero(peers < target_residual)
            + 0.5 * plus_ties
        ) / peers.size
        minus_percentile = (
            np.count_nonzero(peers > target_residual)
            + 0.5 * minus_ties
        ) / peers.size
    else:
        # A singleton tie group supplies no directional evidence; use the
        # neutral midpoint rather than counting it as a win for both signs.
        plus_win_rate = minus_win_rate = 0.5
        plus_percentile = minus_percentile = 0.5
    return {
        "baseline_tie_group_count": int(tie_positions.size),
        "baseline_tie_residual_target": target_residual,
        "baseline_tie_residual_plus_win_rate": float(plus_win_rate),
        "baseline_tie_residual_minus_win_rate": float(minus_win_rate),
        "baseline_tie_residual_plus_percentile": float(plus_percentile),
        "baseline_tie_residual_minus_percentile": float(minus_percentile),
    }


def summarize_sign(records, prefix):
    """Summarize optimistic and average-tie ranks for one residual sign."""
    if not records:
        return {}
    ranks = np.asarray([record[prefix + "_rank"] for record in records])
    average_ranks = np.asarray(
        [record[prefix + "_average_rank"] for record in records]
    )
    pessimistic_ranks = np.asarray(
        [record[prefix + "_pessimistic_rank"] for record in records]
    )
    return {
        "optimistic_mrr": float(np.mean(1.0 / ranks)),
        "average_tie_mrr": float(np.mean(1.0 / average_ranks)),
        "pessimistic_mrr": float(np.mean(1.0 / pessimistic_ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mean_target_tie_count": float(
            np.mean(
                [record[prefix + "_target_tie_count"] for record in records]
            )
        ),
    }


def summarize_tie_residual_audit(records):
    if not records:
        return {}
    return {
        "mean_baseline_tie_group_count": float(
            np.mean([record["baseline_tie_group_count"] for record in records])
        ),
        "non_singleton_tie_group_fraction": float(
            np.mean(
                [record["baseline_tie_group_count"] > 1 for record in records]
            )
        ),
        "median_baseline_tie_group_count": float(
            np.median(
                [record["baseline_tie_group_count"] for record in records]
            )
        ),
        "mean_plus_win_rate": float(
            np.mean(
                [
                    record["baseline_tie_residual_plus_win_rate"]
                    for record in records
                ]
            )
        ),
        "mean_minus_win_rate": float(
            np.mean(
                [
                    record["baseline_tie_residual_minus_win_rate"]
                    for record in records
                ]
            )
        ),
        "mean_plus_percentile": float(
            np.mean(
                [
                    record["baseline_tie_residual_plus_percentile"]
                    for record in records
                ]
            )
        ),
        "mean_minus_percentile": float(
            np.mean(
                [
                    record["baseline_tie_residual_minus_percentile"]
                    for record in records
                ]
            )
        ),
    }


def main():
    args = parse_args()
    if args.queries < 1:
        raise ValueError("queries must be at least 1")
    if args.progress_every < 0:
        raise ValueError("progress-every cannot be negative")
    if args.hash_hops != 1:
        raise ValueError("this WN18RR audit is fixed to hash_hops=1")
    if not 0.0 <= args.multiplier <= 1.0:
        raise ValueError("multiplier must be in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    set_seeds_all(args.seed)
    data_path = Path(args.data_path)
    # importTestFiles() sorts testList with Triple::cmp_rel2 before batches are
    # generated. Keep Python targets in that exact relation/head/tail order.
    held_out = sort_triples_like_base_so(
        load_triples(data_path / (args.split + "2id.txt"))
    )
    heads_by_rt, tails_by_hr = build_filter_maps(data_path)

    # Base.so owns process-global training graph state. TestDataLoader reads
    # that graph through getEdgeIndices(), so initialize it in the same order
    # as the official checkpoint evaluator before constructing the held-out
    # loader. Skipping this step can dereference an uninitialized trainList.
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
    )
    loader = TestDataLoader(
        str(data_path) + "/",
        "link",
        device=args.device,
        max_hash_hops=args.hash_hops,
        split=args.split,
    )
    del train_loader
    model, architecture = build_model(
        args, loader.get_ent_tot(), loader.get_rel_tot()
    )
    if args.device == "cuda":
        model.cuda()
    model.eval()

    records = []
    with torch.no_grad():
        for query_index, (head_data, tail_data) in enumerate(loader):
            if query_index >= min(args.queries, len(held_out)):
                break
            if (
                args.progress_every
                and query_index > 0
                and query_index % args.progress_every == 0
            ):
                print(
                    "Processed {}/{} validation triples ({:.1%})".format(
                        query_index,
                        min(args.queries, len(held_out)),
                        query_index / float(min(args.queries, len(held_out))),
                    ),
                    flush=True,
                )
            true_head, true_tail, relation = held_out[query_index]
            head_fixed_tail = int(np.asarray(head_data["batch_t"]).reshape(-1)[0])
            head_fixed_relation = int(
                np.asarray(head_data["batch_r"]).reshape(-1)[0]
            )
            tail_fixed_head = int(np.asarray(tail_data["batch_h"]).reshape(-1)[0])
            tail_fixed_relation = int(
                np.asarray(tail_data["batch_r"]).reshape(-1)[0]
            )
            if (
                head_fixed_tail != true_tail
                or head_fixed_relation != relation
                or tail_fixed_head != true_head
                or tail_fixed_relation != relation
            ):
                raise RuntimeError(
                    "Base.so/Python triple order mismatch at query {}: "
                    "expected ({}, {}, {}), head fixed=({}, {}), "
                    "tail fixed=({}, {})".format(
                        query_index,
                        true_head,
                        true_tail,
                        relation,
                        head_fixed_tail,
                        head_fixed_relation,
                        tail_fixed_head,
                        tail_fixed_relation,
                    )
                )
            for side, data, target, filtered in (
                (
                    "head",
                    head_data,
                    true_head,
                    heads_by_rt[(relation, true_tail)],
                ),
                (
                    "tail",
                    tail_data,
                    true_tail,
                    tails_by_hr[(true_head, relation)],
                ),
            ):
                payload = {
                    "subgraph_feature": data["subgraph_feature"],
                    "batch_h": data["batch_h"],
                    "batch_t": data["batch_t"],
                    "batch_r": data["batch_r"],
                    "mode": data["mode"],
                }
                if architecture == "official":
                    base_tensor = model(payload).view(-1)
                    base_components = {
                        "score": base_tensor,
                        "base_score": base_tensor,
                        "base_logit": torch.logit(
                            base_tensor.clamp(1e-7, 1.0 - 1e-7)
                        ),
                    }
                    fused_components = base_components
                    base = base_tensor.cpu().numpy()
                    fused = base.copy()
                    minus = base.copy()
                    residual_unit = torch.zeros_like(base_tensor)
                else:
                    model.set_residual_multiplier(0.0)
                    base_components = model.forward_components(payload)
                    model.set_residual_multiplier(args.multiplier)
                    fused_components = model.forward_components(payload)

                    base = base_components["score"].view(-1).cpu().numpy()
                    fused = fused_components["score"].view(-1).cpu().numpy()
                    residual_unit = (
                        fused_components["residual_unit_delta"].view(-1)
                    )
                    minus_tensor = signed_scores(
                        base_components,
                        fused_components,
                        args.multiplier,
                        -1,
                    )
                    minus = minus_tensor.view(-1).cpu().numpy()
                candidates = data[
                    "batch_h" if side == "head" else "batch_t"
                ]
                if torch.is_tensor(candidates):
                    candidates = candidates.detach().cpu().numpy()
                base_stats = filtered_rank_stats(
                    base, candidates, target, filtered
                )
                fused_stats = filtered_rank_stats(
                    fused, candidates, target, filtered
                )
                minus_stats = filtered_rank_stats(
                    minus, candidates, target, filtered
                )
                base_rank = base_stats["optimistic_rank"]
                fused_rank = fused_stats["optimistic_rank"]
                base_order = np.argsort(-base, kind="stable")
                fused_order = np.argsort(-fused, kind="stable")
                top10_overlap = len(
                    set(base_order[:10]).intersection(fused_order[:10])
                ) / 10.0
                score_change = np.abs(fused - base)
                features = data["subgraph_feature"].detach().cpu().numpy()
                residual = residual_unit.detach().cpu().numpy()
                feature_unique_count = len(np.unique(features, axis=0))
                base_unique_count = len(np.unique(base))
                fused_unique_count = len(np.unique(fused))
                target_positions = np.flatnonzero(candidates == target)
                if len(target_positions) != 1:
                    raise ValueError("expected exactly one target candidate")
                target_feature = features[int(target_positions[0])]
                active_mask = np.ones(len(candidates), dtype=bool)
                for entity in filtered:
                    if entity != target:
                        active_mask[candidates == entity] = False
                target_feature_tie_count = int(
                    np.count_nonzero(
                        active_mask
                        & np.all(features == target_feature, axis=1)
                    )
                )
                tie_stats = baseline_tie_residual_stats(
                    base, candidates, target, filtered, residual
                )
                records.append({
                    "query": query_index,
                    "side": side,
                    "base_rank": float(base_rank),
                    "fused_rank": float(fused_rank),
                    "base_average_rank": base_stats["average_rank"],
                    "fused_average_rank": fused_stats["average_rank"],
                    "base_pessimistic_rank": base_stats[
                        "pessimistic_rank"
                    ],
                    "fused_pessimistic_rank": fused_stats[
                        "pessimistic_rank"
                    ],
                    "plus_rank": fused_stats["optimistic_rank"],
                    "plus_average_rank": fused_stats["average_rank"],
                    "plus_pessimistic_rank": fused_stats[
                        "pessimistic_rank"
                    ],
                    "plus_target_tie_count": fused_stats[
                        "target_tie_count"
                    ],
                    "base_minus_rank": minus_stats["optimistic_rank"],
                    "base_minus_average_rank": minus_stats["average_rank"],
                    "base_minus_pessimistic_rank": minus_stats[
                        "pessimistic_rank"
                    ],
                    "base_minus_target_tie_count": minus_stats[
                        "target_tie_count"
                    ],
                    "minus_rank": minus_stats["optimistic_rank"],
                    "minus_average_rank": minus_stats["average_rank"],
                    "minus_pessimistic_rank": minus_stats[
                        "pessimistic_rank"
                    ],
                    "minus_target_tie_count": minus_stats[
                        "target_tie_count"
                    ],
                    "base_target_tie_count": base_stats[
                        "target_tie_count"
                    ],
                    "fused_target_tie_count": fused_stats[
                        "target_tie_count"
                    ],
                    "base_score_unique_ratio": float(
                        base_unique_count / len(base)
                    ),
                    "fused_score_unique_ratio": float(
                        fused_unique_count / len(fused)
                    ),
                    "base_score_unique_count": int(base_unique_count),
                    "fused_score_unique_count": int(fused_unique_count),
                    "ordering_changed": bool(
                        not np.array_equal(base_order, fused_order)
                    ),
                    "top1_changed": bool(base_order[0] != fused_order[0]),
                    "top10_overlap": float(top10_overlap),
                    "mean_abs_score_change": float(score_change.mean()),
                    "max_abs_score_change": float(score_change.max()),
                    "feature_unique_ratio": float(
                        feature_unique_count / len(features)
                    ),
                    "feature_unique_count": int(feature_unique_count),
                    "target_feature_tie_count": target_feature_tie_count,
                    "target_feature_tie_fraction": float(
                        target_feature_tie_count / np.count_nonzero(active_mask)
                    ),
                    "feature_zero_fraction": float(
                        np.mean(np.all(features == 0.0, axis=1))
                    ),
                    "base_score_near_zero_fraction": float(
                        np.mean(base <= 1e-6)
                    ),
                    "base_score_near_one_fraction": float(
                        np.mean(base >= 1.0 - 1e-6)
                    ),
                    "base_score_round6_unique_ratio": float(
                        len(np.unique(np.round(base, decimals=6))) / len(base)
                    ),
                    "residual_unique_ratio": float(
                        len(np.unique(np.round(residual, decimals=7)))
                        / len(residual)
                    ),
                    **tie_stats,
                })

    if architecture != "official":
        model.set_residual_multiplier(0.0)
    report = {
        "checkpoint": args.checkpoint,
        "architecture": architecture,
        "split": args.split,
        "queries": min(args.queries, len(held_out)),
        "multiplier": args.multiplier,
        "summary": summarize(records),
        "largest_rank_changes": sorted(
            records,
            key=lambda item: abs(item["fused_rank"] - item["base_rank"]),
            reverse=True,
        )[:10],
        "sign_audit": {
            "plus": summarize_sign(records, "plus"),
            "minus": summarize_sign(records, "minus"),
        },
        "tie_group_residual_audit": summarize_tie_residual_audit(records),
    }
    if args.include_records:
        report["records"] = records
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

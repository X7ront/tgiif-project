import numpy as np


def sort_triples_like_base_so(triples):
    """Match Triple::cmp_rel2: relation, then head, then tail."""
    return sorted(triples, key=lambda triple: (triple[2], triple[0], triple[1]))


def filtered_rank_stats(scores, candidate_ids, target, filtered_entities):
    """Return optimistic, average, and pessimistic filtered tie ranks."""
    scores = np.asarray(scores).reshape(-1).copy()
    candidate_ids = np.asarray(candidate_ids).reshape(-1)
    target_positions = np.flatnonzero(candidate_ids == target)
    if len(target_positions) != 1:
        raise ValueError(
            "Expected one target candidate {}, found {}".format(
                target, len(target_positions)
            )
        )
    target_position = int(target_positions[0])
    for entity in filtered_entities:
        if entity == target:
            continue
        positions = np.flatnonzero(candidate_ids == entity)
        scores[positions] = np.inf
    target_score = scores[target_position]
    better = int(np.count_nonzero(scores < target_score))
    tied = int(np.count_nonzero(scores == target_score))
    return {
        "optimistic_rank": float(1 + better),
        "average_rank": float(1 + better + 0.5 * max(0, tied - 1)),
        "pessimistic_rank": float(better + tied),
        "target_tie_count": tied,
    }


def official_filtered_rank(scores, candidate_ids, target, filtered_entities):
    """Match Base.so: rank only candidates with a strictly lower score."""
    return filtered_rank_stats(
        scores, candidate_ids, target, filtered_entities
    )["optimistic_rank"]


def summarize(records):
    base_ranks = np.asarray([record["base_rank"] for record in records])
    fused_ranks = np.asarray([record["fused_rank"] for record in records])
    summary = {
        "query_sides": len(records),
        "base_sample_mrr": float(np.mean(1.0 / base_ranks)),
        "fused_sample_mrr": float(np.mean(1.0 / fused_ranks)),
        "base_sample_mr": float(np.mean(base_ranks)),
        "fused_sample_mr": float(np.mean(fused_ranks)),
        "rank_improved_fraction": float(np.mean(fused_ranks < base_ranks)),
        "rank_degraded_fraction": float(np.mean(fused_ranks > base_ranks)),
        "rank_unchanged_fraction": float(np.mean(fused_ranks == base_ranks)),
        "ordering_changed_fraction": float(
            np.mean([record["ordering_changed"] for record in records])
        ),
        "top1_changed_fraction": float(
            np.mean([record["top1_changed"] for record in records])
        ),
        "mean_top10_overlap": float(
            np.mean([record["top10_overlap"] for record in records])
        ),
        "mean_abs_score_change": float(
            np.mean([record["mean_abs_score_change"] for record in records])
        ),
        "max_abs_score_change": float(
            np.max([record["max_abs_score_change"] for record in records])
        ),
        "mean_feature_unique_ratio": float(
            np.mean([record["feature_unique_ratio"] for record in records])
        ),
        "mean_residual_unique_ratio": float(
            np.mean([record["residual_unique_ratio"] for record in records])
        ),
    }
    for threshold in (10, 3, 1):
        summary["base_sample_hits@{}".format(threshold)] = float(
            np.mean(base_ranks <= threshold)
        )
        summary["fused_sample_hits@{}".format(threshold)] = float(
            np.mean(fused_ranks <= threshold)
        )
    if records and "base_average_rank" in records[0]:
        for prefix in ("base", "fused"):
            average_ranks = np.asarray(
                [record[prefix + "_average_rank"] for record in records]
            )
            pessimistic_ranks = np.asarray(
                [record[prefix + "_pessimistic_rank"] for record in records]
            )
            tie_counts = np.asarray(
                [record[prefix + "_target_tie_count"] for record in records]
            )
            summary[prefix + "_average_tie_mrr"] = float(
                np.mean(1.0 / average_ranks)
            )
            summary[prefix + "_pessimistic_tie_mrr"] = float(
                np.mean(1.0 / pessimistic_ranks)
            )
            summary[prefix + "_tied_query_fraction"] = float(
                np.mean(tie_counts > 1)
            )
            summary[prefix + "_mean_target_tie_count"] = float(
                np.mean(tie_counts)
            )
            summary[prefix + "_median_target_tie_count"] = float(
                np.median(tie_counts)
            )
            summary[prefix + "_max_target_tie_count"] = int(
                np.max(tie_counts)
            )
            summary[prefix + "_mean_score_unique_ratio"] = float(
                np.mean(
                    [
                        record[prefix + "_score_unique_ratio"]
                        for record in records
                    ]
                )
            )
    return summary

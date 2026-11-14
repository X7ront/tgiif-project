import unittest

import numpy as np

from examples.analyze_ctsnet_diagnostics import audit_robust_gain
from examples.ctsnet_ranking import (
    filtered_rank_stats,
    official_filtered_rank,
    sort_triples_like_base_so,
    summarize,
)
try:
    from examples.diagnose_ctsnet_residual_queries_WN18RR import (
        baseline_tie_residual_stats,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    baseline_tie_residual_stats = None


class CTSNetDiagnosticsTest(unittest.TestCase):
    def test_triples_follow_base_so_relation_head_tail_order(self):
        triples = [
            (5, 1, 2),
            (3, 7, 1),
            (2, 9, 1),
            (2, 4, 1),
        ]

        self.assertEqual(
            sort_triples_like_base_so(triples),
            [(2, 4, 1), (2, 9, 1), (3, 7, 1), (5, 1, 2)],
        )

    def test_official_filtered_rank_uses_optimistic_tie_policy(self):
        scores = np.asarray([0.5, 0.5, 0.5, 0.75])
        candidates = np.arange(4)

        self.assertEqual(
            official_filtered_rank(scores, candidates, 1, set()), 1.0
        )
        stats = filtered_rank_stats(scores, candidates, 1, set())
        self.assertEqual(stats["average_rank"], 2.0)
        self.assertEqual(stats["pessimistic_rank"], 3.0)
        self.assertEqual(stats["target_tie_count"], 3)

    def test_official_filtered_rank_excludes_known_true_entities(self):
        scores = np.asarray([0.1, 0.2, 0.3, 0.4])
        candidates = np.arange(4)

        self.assertEqual(
            official_filtered_rank(scores, candidates, 3, {0, 2, 3}), 2.0
        )

    def test_summary_reports_mrr_mr_and_hits(self):
        records = []
        for base_rank, fused_rank in ((1.0, 1.0), (4.0, 3.0), (11.0, 10.0)):
            records.append(
                {
                    "base_rank": base_rank,
                    "fused_rank": fused_rank,
                    "ordering_changed": base_rank != fused_rank,
                    "top1_changed": False,
                    "top10_overlap": 1.0,
                    "mean_abs_score_change": 0.0,
                    "max_abs_score_change": 0.0,
                    "feature_unique_ratio": 1.0,
                    "residual_unique_ratio": 1.0,
                }
            )

        result = summarize(records)

        self.assertEqual(result["base_sample_hits@1"], 1.0 / 3.0)
        self.assertEqual(result["fused_sample_hits@1"], 1.0 / 3.0)
        self.assertEqual(result["base_sample_hits@3"], 1.0 / 3.0)
        self.assertEqual(result["fused_sample_hits@3"], 2.0 / 3.0)
        self.assertEqual(result["base_sample_hits@10"], 2.0 / 3.0)
        self.assertEqual(result["fused_sample_hits@10"], 1.0)

    def test_strict_audit_rejects_tie_count_only_change(self):
        result = audit_robust_gain(
            [100.0] * 20,
            [99.9] * 20,
            [100.0] * 20,
            [1.0] * 20,
            [1000.0] * 20,
            [1.0] * 20,
            draws=200,
            seed=7,
        )
        self.assertFalse(result["average_tie_mrr_delta_pass"])
        self.assertFalse(result["robust_gain"])

    def test_strict_audit_rejects_excessive_optimistic_degradation(self):
        result = audit_robust_gain(
            [10.0] * 40,
            [9.0] * 40,
            [10.0] * 40,
            [11.0] * 21 + [10.0] * 19,
            [2.0] * 40,
            [2.0] * 40,
            draws=200,
            seed=11,
        )
        self.assertTrue(result["average_tie_mrr_delta_pass"])
        self.assertFalse(result["optimistic_degradation_pass"])
        self.assertFalse(result["robust_gain"])

    @unittest.skipIf(
        baseline_tie_residual_stats is None,
        "residual diagnostic requires torch",
    )
    def test_tie_group_residual_audit_reports_both_directions(self):
        result = baseline_tie_residual_stats(
            np.asarray([0.5, 0.5, 0.2]),
            np.asarray([0, 1, 2]),
            1,
            set(),
            np.asarray([0.2, 0.8, 0.1]),
        )
        self.assertEqual(result["baseline_tie_group_count"], 2)
        self.assertEqual(result["baseline_tie_residual_plus_win_rate"], 1.0)
        self.assertEqual(result["baseline_tie_residual_minus_win_rate"], 0.0)
        self.assertEqual(result["baseline_tie_residual_plus_percentile"], 1.0)
        self.assertEqual(result["baseline_tie_residual_minus_percentile"], 0.0)


if __name__ == "__main__":
    unittest.main()

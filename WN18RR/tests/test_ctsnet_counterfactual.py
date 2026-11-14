import unittest

import numpy as np
import torch

from openke.module.counterfactual import (
    CounterfactualBatchEnricher,
    build_counterfactual_cache,
    build_popularity_treatment,
    validate_counterfactual_cache,
)
from openke.module.loss import MarginLoss
from openke.module.model import CTSNetDualBranchPredictor
from openke.module.strategy import CTSNegativeSampling


class _OneBatchLoader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        yield dict(self.data)

    def __len__(self):
        return 1

    def get_batch_size(self):
        return self.batch_size


class CounterfactualConstructionTest(unittest.TestCase):
    def setUp(self):
        self.triples = np.asarray([
            [0, 1, 0],
            [0, 2, 0],
            [3, 4, 0],
            [4, 5, 0],
            [0, 3, 1],
            [1, 2, 1],
            [4, 6, 1],
            [5, 7, 1],
        ], dtype=np.int64)
        self.features = np.asarray([
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.1, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.1, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.1, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

    def test_popularity_is_train_only_and_relation_conditioned(self):
        info = build_popularity_treatment(self.triples, ent_tot=8, rel_tot=2)
        self.assertEqual(info["entity_degree"].shape, (8,))
        for relation in (0, 1):
            labels = info["treatment"][self.triples[:, 2] == relation]
            self.assertEqual(set(labels.tolist()), {0, 1})

    def test_matches_keep_relation_and_flip_treatment(self):
        treatment = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        info = {
            "treatment": treatment,
            "popularity_score": np.arange(8, dtype=np.float32),
            "fallback_relations": [],
        }
        cache = build_counterfactual_cache(
            self.triples,
            self.features,
            info,
            max_distance_percentile=100.0,
        )
        valid = cache["cf_valid_mask"].numpy()
        cf_triples = cache["cf_triples"].numpy()
        cf_treatment = cache["cf_treatment"].numpy()
        self.assertTrue(valid.all())
        self.assertTrue(
            np.array_equal(self.triples[:, 2], cf_triples[:, 2])
        )
        self.assertTrue(np.all(treatment != cf_treatment))

    def test_batch_enricher_uses_positive_pairs_only(self):
        treatment = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        cache = build_counterfactual_cache(
            self.triples,
            self.features,
            {
                "treatment": treatment,
                "popularity_score": np.arange(8, dtype=np.float32),
            },
            max_distance_percentile=100.0,
        )
        data = {
            "batch_h": np.asarray([0, 3, 7, 7, 7, 7]),
            "batch_t": np.asarray([1, 4, 7, 7, 7, 7]),
            "batch_r": np.asarray([0, 0, 0, 0, 0, 0]),
            "batch_y": np.zeros(6, dtype=np.float32),
            "subgraph_feature": torch.randn(6, 4),
            "mode": "normal",
        }
        loader = CounterfactualBatchEnricher(
            _OneBatchLoader(data, batch_size=2), cache
        )
        enriched = next(iter(loader))
        self.assertEqual(enriched["cf_subgraph_feature"].shape, (2, 4))
        self.assertEqual(enriched["factual_treatment"].shape, (2, 1))
        self.assertEqual(enriched["cf_valid_mask"].numel(), 2)
        self.assertEqual(enriched["cf_batch_h"].shape, (2,))
        self.assertEqual(enriched["cf_batch_t"].shape, (2,))
        self.assertEqual(enriched["cf_batch_r"].shape, (2,))

    def test_cache_validation_rejects_a_different_train_split(self):
        treatment = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        cache = build_counterfactual_cache(
            self.triples,
            self.features,
            {
                "treatment": treatment,
                "popularity_score": np.arange(8, dtype=np.float32),
            },
            max_distance_percentile=100.0,
        )
        cache.update({
            "dataset": "WN18RR",
            "hash_hops": 1,
            "feature_mode": "triple",
            "feature_dim": 4,
        })
        validate_counterfactual_cache(
            cache,
            self.triples,
            dataset="WN18RR",
            hash_hops=1,
            feature_mode="triple",
            feature_dim=4,
        )

        different_split = self.triples.copy()
        different_split[0, 1] = 7
        with self.assertRaisesRegex(ValueError, "train split fingerprint"):
            validate_counterfactual_cache(cache, different_split)


class CTSAuxiliaryLossTest(unittest.TestCase):
    def _make_strategy(self, negative_count=2):
        batch_size = 4
        model = CTSNetDualBranchPredictor(
            ent_tot=20,
            rel_tot=3,
            max_hash_hops=1,
            hidden_dim=16,
            branch_dim=8,
        )
        strategy = CTSNegativeSampling(
            model=model,
            loss=MarginLoss(margin=5.0),
            batch_size=batch_size,
            lambda_cf=1.0,
            lambda_treat=0.1,
            lambda_orth=0.05,
            lambda_gate=0.1,
            warmup_start_epoch=5,
            warmup_epochs=10,
        )
        data = {
            "subgraph_feature": torch.randn(
                batch_size * (1 + negative_count), 4
            ),
            "cf_subgraph_feature": torch.randn(batch_size, 4),
            "factual_treatment": torch.tensor(
                [[0.0], [1.0], [0.0], [1.0]]
            ),
            "cf_treatment": torch.tensor(
                [[1.0], [0.0], [1.0], [0.0]]
            ),
            "cf_valid_mask": torch.ones(batch_size, dtype=torch.bool),
            "cf_target_index": torch.tensor([10, 11, 12, 13]),
        }
        return model, strategy, data

    def test_warmup_and_all_auxiliary_gradients(self):
        model, strategy, data = self._make_strategy()
        strategy.set_epoch(0)
        rank_only = strategy(data)
        metrics = strategy.get_last_metrics()
        self.assertEqual(metrics["warmup"], 0.0)
        self.assertAlmostEqual(
            metrics["total_loss"], metrics["rank_loss"], places=6
        )

        strategy.zero_grad()
        strategy.set_epoch(5)
        total = strategy(data)
        total.backward()
        metrics = strategy.get_last_metrics()
        self.assertAlmostEqual(metrics["warmup"], 0.1, places=6)
        self.assertGreater(metrics["weighted_treat"], 0.0)
        self.assertIsNotNone(model.treatment_head.weight.grad)
        self.assertIsNotNone(model.invariant_head.weight.grad)
        self.assertIsNotNone(model.bias_head.weight.grad)
        self.assertIsNotNone(model.gate_head.weight.grad)

    def test_auxiliary_means_do_not_scale_with_negative_count(self):
        torch.manual_seed(7)
        model, strategy_two, data_two = self._make_strategy(negative_count=2)
        model.eval()
        strategy_two.set_epoch(5)
        strategy_two(data_two)
        metrics_two = strategy_two.get_last_metrics()

        strategy_five = CTSNegativeSampling(
            model=model,
            loss=MarginLoss(margin=5.0),
            batch_size=4,
            lambda_cf=1.0,
            lambda_treat=0.1,
            lambda_orth=0.05,
            lambda_gate=0.1,
            warmup_start_epoch=5,
            warmup_epochs=10,
        )
        data_five = dict(data_two)
        data_five["subgraph_feature"] = torch.cat(
            (
                data_two["subgraph_feature"][:4],
                data_two["subgraph_feature"][4:],
                torch.randn(12, 4),
            ),
            dim=0,
        )
        strategy_five.set_epoch(5)
        strategy_five(data_five)
        metrics_five = strategy_five.get_last_metrics()

        for key in ("cf_loss", "treat_loss", "orth_loss", "gate_loss"):
            self.assertAlmostEqual(metrics_two[key], metrics_five[key], places=6)

    def test_each_step_uses_one_shared_encoder_forward(self):
        model, strategy, data = self._make_strategy()
        calls = []
        handle = model.shared_encoder.register_forward_hook(
            lambda module, inputs, output: calls.append(1)
        )
        try:
            strategy.set_epoch(0)
            strategy(data)
            self.assertEqual(len(calls), 1)

            strategy.set_epoch(5)
            strategy(data)
            self.assertEqual(len(calls), 2)
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()

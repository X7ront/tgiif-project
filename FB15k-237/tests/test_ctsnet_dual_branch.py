import os
import tempfile
import unittest

import torch

from openke.module.loss import MarginLoss
from openke.module.model import CTSNetDualBranchPredictor
from openke.module.strategy import NegativeSampling


class CTSNetDualBranchPredictorTest(unittest.TestCase):
    def _make_model(self, hash_hops=2):
        return CTSNetDualBranchPredictor(
            ent_tot=100,
            rel_tot=10,
            max_hash_hops=hash_hops,
            hidden_dim=32,
            branch_dim=16,
        )

    def test_feature_dimensions_and_score_range(self):
        for hash_hops, feature_dim in ((1, 4), (2, 14)):
            model = self._make_model(hash_hops=hash_hops)
            model.eval()
            features = torch.randn(12, feature_dim)
            components = model.forward_components({"subgraph_feature": features})

            self.assertEqual(components["score"].shape, (12, 1))
            self.assertEqual(components["z_inv"].shape, (12, 16))
            self.assertEqual(components["z_bias"].shape, (12, 16))
            for key in ("score", "s_inv", "s_bias", "gate"):
                self.assertTrue(torch.all(components[key] >= 0.0))
                self.assertTrue(torch.all(components[key] <= 1.0))

            expected = (
                components["gate"] * components["s_inv"]
                + (1.0 - components["gate"]) * components["s_bias"]
            )
            self.assertTrue(torch.allclose(components["score"], expected))

    def test_pairwise_feature_dimensions(self):
        for hash_hops, feature_dim in ((1, 3), (2, 8), (3, 15)):
            model = CTSNetDualBranchPredictor(
                ent_tot=100,
                rel_tot=10,
                max_hash_hops=hash_hops,
                hidden_dim=32,
                branch_dim=16,
                feature_mode="pairwise",
                normalization="layer",
            )
            model.eval()
            score = model({"subgraph_feature": torch.randn(7, feature_dim)})
            self.assertEqual(score.shape, (7, 1))
            self.assertEqual(model.dim, feature_dim)

    def test_pairwise_rejects_triple_dimension(self):
        model = CTSNetDualBranchPredictor(
            ent_tot=100,
            rel_tot=10,
            max_hash_hops=2,
            feature_mode="pairwise",
        )
        with self.assertRaisesRegex(ValueError, "Expected subgraph_feature"):
            model({"subgraph_feature": torch.randn(4, 14)})

    def test_official_negative_sampling_backward(self):
        batch_size = 4
        negative_count = 2
        model = self._make_model(hash_hops=2)
        strategy = NegativeSampling(
            model=model,
            loss=MarginLoss(margin=5.0),
            batch_size=batch_size,
        )
        features = torch.randn(batch_size * (1 + negative_count), 14)
        loss = strategy({"subgraph_feature": features})
        loss.backward()

        # The official MarginLoss keeps margin as a [1] parameter, so its
        # result is a one-element tensor rather than a zero-dimensional one.
        self.assertEqual(loss.numel(), 1)
        self.assertIsNotNone(model.invariant_head.weight.grad)
        self.assertIsNotNone(model.bias_head.weight.grad)
        self.assertIsNotNone(model.gate_head.weight.grad)
        self.assertIsNotNone(model.shared_encoder.dense1.weight.grad)

    def test_checkpoint_round_trip(self):
        model = self._make_model(hash_hops=1)
        model.eval()
        data = {"subgraph_feature": torch.randn(8, 4)}
        expected = model(data).detach()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = os.path.join(temp_dir, "stage1.ckpt")
            model.save_checkpoint(checkpoint)

            restored = self._make_model(hash_hops=1)
            restored.load_checkpoint(checkpoint)
            actual = restored(data).detach()

        self.assertTrue(torch.allclose(expected, actual))

    def test_rejects_wrong_feature_dimension(self):
        model = self._make_model(hash_hops=2)
        with self.assertRaisesRegex(ValueError, "Expected subgraph_feature"):
            model({"subgraph_feature": torch.randn(8, 4)})

    def test_layer_norm_has_identical_train_and_eval_scores(self):
        model = CTSNetDualBranchPredictor(
            ent_tot=100,
            rel_tot=10,
            max_hash_hops=1,
            hidden_dim=32,
            branch_dim=16,
            normalization="layer",
        )
        features = torch.randn(32, 4)
        model.train()
        train_score = model({"subgraph_feature": features}).detach()
        model.eval()
        eval_score = model({"subgraph_feature": features}).detach()
        self.assertTrue(torch.equal(train_score, eval_score))

    def test_rejects_unknown_normalization(self):
        with self.assertRaisesRegex(ValueError, "normalization"):
            CTSNetDualBranchPredictor(
                ent_tot=100,
                rel_tot=10,
                max_hash_hops=1,
                normalization="instance",
            )

    def test_batch_norm_calibration_resets_and_counts_batches(self):
        model = self._make_model(hash_hops=1)
        batches = [
            {"subgraph_feature": torch.randn(24, 4) + shift}
            for shift in (0.0, 1.0, 2.0)
        ]
        processed = model.calibrate_batch_norm(batches, num_batches=2)
        self.assertEqual(processed, 2)
        batch_norms = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.BatchNorm1d)
        ]
        self.assertTrue(batch_norms)
        for module in batch_norms:
            self.assertEqual(module.num_batches_tracked.item(), 2)

        layer_model = CTSNetDualBranchPredictor(
            ent_tot=100,
            rel_tot=10,
            max_hash_hops=1,
            normalization="layer",
        )
        self.assertEqual(layer_model.calibrate_batch_norm(batches, 2), 0)


if __name__ == "__main__":
    unittest.main()

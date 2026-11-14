import os
import tempfile
import unittest

import torch

from openke.module.model import (
    CTSNetResidualDualBranchPredictor,
    LinkPredictor,
)


class CTSNetResidualPredictorTest(unittest.TestCase):
    def _models(self):
        baseline = LinkPredictor(
            ent_tot=100,
            rel_tot=10,
            max_hash_hops=1,
            hidden_dim=32,
        )
        residual = CTSNetResidualDualBranchPredictor(
            ent_tot=100,
            rel_tot=10,
            max_hash_hops=1,
            hidden_dim=32,
            branch_dim=16,
            residual_scale=0.1,
            freeze_baseline=True,
        )
        return baseline, residual

    def test_zero_residual_exactly_preserves_official_scores(self):
        baseline, residual = self._models()
        features = torch.randn(24, 4)
        baseline.eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = os.path.join(temp_dir, "baseline.ckpt")
            baseline.save_checkpoint(checkpoint)
            residual.load_baseline_checkpoint(checkpoint)

        residual.eval()
        expected = baseline({"subgraph_feature": features}).detach()
        actual = residual({"subgraph_feature": features}).detach()
        self.assertTrue(torch.equal(expected, actual))

        residual.gate_head.bias.data.fill_(100.0)
        high_gate = residual({"subgraph_feature": features}).detach()
        residual.gate_head.bias.data.fill_(-100.0)
        low_gate = residual({"subgraph_feature": features}).detach()
        self.assertTrue(torch.equal(expected, high_gate))
        self.assertTrue(torch.equal(expected, low_gate))

    def test_residual_logit_correction_is_bounded(self):
        _, residual = self._models()
        residual.invariant_delta_head.bias.data.fill_(100.0)
        residual.bias_delta_head.bias.data.fill_(100.0)
        components = residual.forward_components({
            "subgraph_feature": torch.randn(24, 4)
        })
        self.assertLessEqual(
            components["residual_delta"].abs().max().item(), 0.100001
        )

    def test_frozen_baseline_keeps_batch_norm_and_gradients_fixed(self):
        _, residual = self._models()
        before = residual.sf_lin_layer.norm1.running_mean.detach().clone()
        residual.train()
        output = residual({"subgraph_feature": torch.randn(24, 4)})
        output.mean().backward()
        after = residual.sf_lin_layer.norm1.running_mean.detach().clone()
        self.assertTrue(torch.equal(before, after))
        for parameter in residual.sf_lin_layer.parameters():
            self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()

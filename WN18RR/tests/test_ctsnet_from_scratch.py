import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

import torch

from openke.module.model import (
    CTSNetIndependentResidualPredictor,
    CTSNetTreatmentAwareResidualPredictor,
    LinkPredictor,
)
from openke.module.loss import MarginLoss
from openke.module.strategy import CTSNegativeSampling
from openke.config import CTSTrainer


class CTSNetFromScratchPredictorTest(unittest.TestCase):
    def test_leaky_branch_and_bounded_delta_remain_differentiable(self):
        predictor = CTSNetIndependentResidualPredictor(
            ent_tot=20,
            rel_tot=3,
            max_hash_hops=1,
            hidden_dim=16,
            branch_dim=8,
            branch_activation="leaky_relu",
            branch_negative_slope=0.1,
            delta_limit=2.0,
            delta_temperature=2.0,
        )
        activation = predictor._make_activation()
        value = torch.tensor([-1.0], requires_grad=True)
        activation(value).sum().backward()
        self.assertAlmostEqual(value.grad.item(), 0.1, places=6)

        raw = torch.tensor([[-100.0], [0.5], [100.0]], requires_grad=True)
        bounded = predictor._transform_delta(raw)
        self.assertLessEqual(bounded.abs().max().item(), 2.0)
        bounded[1].backward()
        self.assertGreater(raw.grad[1].item(), 0.0)

    def test_query_rms_calibration_shrinks_without_amplifying(self):
        predictor = CTSNetIndependentResidualPredictor(
            ent_tot=20,
            rel_tot=3,
            max_hash_hops=1,
            hidden_dim=16,
            branch_dim=8,
            delta_normalization="query-rms",
            delta_target_rms=0.25,
        )
        values = torch.tensor([[3.0, 4.0], [0.1, -0.1]])

        calibrated = predictor._calibrate_residual_raw(values, dim=1)

        self.assertAlmostEqual(
            calibrated[0].pow(2).mean().sqrt().item(), 0.25, places=6
        )
        self.assertTrue(torch.equal(calibrated[1], values[1]))

        zeros = torch.zeros(2, 3, requires_grad=True)
        predictor._calibrate_residual_raw(zeros, dim=1).sum().backward()
        self.assertTrue(torch.isfinite(zeros.grad).all())

    def test_residual_training_uses_query_rms_calibration(self):
        predictor = CTSNetIndependentResidualPredictor(
            ent_tot=20,
            rel_tot=3,
            max_hash_hops=1,
            hidden_dim=16,
            branch_dim=8,
            delta_normalization="query-rms",
            delta_target_rms=0.25,
        )
        strategy = CTSNegativeSampling(
            model=predictor,
            loss=MarginLoss(margin=5.0),
            batch_size=1,
            center_residual=True,
        )
        raw = torch.tensor(
            [[100.0], [20.0], [-100.0]], requires_grad=True
        )
        rank_loss, preserve_loss, diagnostics = strategy._residual_losses(
            {
                "base_logit": torch.tensor([[0.0], [0.01], [0.02]]),
                "residual_unit_delta": torch.zeros_like(raw),
                "residual_raw": raw,
            },
            raw.sum(),
        )
        (rank_loss + preserve_loss).backward()

        self.assertLess(diagnostics["residual_unit_abs_mean"], 0.03)
        self.assertEqual(diagnostics["residual_saturation_fraction"], 0.0)
        self.assertGreater(raw.grad.abs().sum().item(), 0.0)

    def test_batch_gate_and_variance_losses_detect_constant_collapse(self):
        predictor = self._cts()
        predictor.gate_head.weight.data.zero_()
        predictor.gate_head.bias.data.zero_()
        for encoder in (
            predictor.invariant_encoder,
            predictor.bias_encoder,
        ):
            for parameter in encoder.parameters():
                parameter.data.zero_()
        strategy = CTSNegativeSampling(
            model=predictor,
            loss=MarginLoss(margin=5.0),
            batch_size=4,
            lambda_gate=1.0,
            gate_regularizer="batch-balance",
            gate_target=0.5,
            lambda_gate_variance=1.0,
            gate_std_floor=0.1,
            lambda_invariant_variance=1.0,
            lambda_bias_variance=1.0,
            representation_std_floor=0.5,
            warmup_start_epoch=0,
            warmup_epochs=0,
        )
        strategy.set_epoch(0)
        strategy({"subgraph_feature": torch.randn(12, 4)})
        metrics = strategy.get_last_metrics()
        self.assertAlmostEqual(metrics["gate_loss"], 0.0, places=6)
        self.assertGreater(metrics["gate_variance_loss"], 0.0)
        self.assertGreater(metrics["invariant_variance_loss"], 0.0)
        self.assertGreater(metrics["bias_variance_loss"], 0.0)

    def test_optimizer_group_lr_multipliers_map_to_architecture(self):
        strategy = CTSNegativeSampling(
            model=self._cts(),
            loss=MarginLoss(margin=5.0),
            batch_size=2,
        )
        trainer = CTSTrainer(
            model=strategy,
            data_loader=[],
            train_times=2,
            alpha=0.01,
            use_gpu=False,
            baseline_lr_multiplier=0.5,
            shared_lr_multiplier=0.75,
            invariant_lr_multiplier=1.25,
            bias_lr_multiplier=1.5,
            gate_lr_multiplier=2.0,
        )
        trainer._build_optimizer()
        actual = {
            group["group_name"]: group["lr"]
            for group in trainer.optimizer.param_groups
        }
        self.assertAlmostEqual(actual["baseline"], 0.005)
        self.assertAlmostEqual(actual["shared"], 0.0075)
        self.assertAlmostEqual(actual["invariant"], 0.0125)
        self.assertAlmostEqual(actual["bias"], 0.015)
        self.assertAlmostEqual(actual["gate"], 0.02)

    def test_bias_counterfactual_loss_enables_pair_forward_by_itself(self):
        strategy = CTSNegativeSampling(
            model=self._cts(),
            loss=MarginLoss(margin=5.0),
            batch_size=2,
            lambda_bias_counterfactual=1.0,
        )
        self.assertTrue(strategy._pair_auxiliary_enabled())

    def test_low_frequency_architecture_probe_records_branch_contributions(self):
        predictor = self._cts()
        strategy = CTSNegativeSampling(
            model=predictor,
            loss=MarginLoss(margin=5.0),
            batch_size=2,
        )
        data = {
            "batch_h": np.asarray([1, 2, 3, 4], dtype=np.int64),
            "batch_t": np.asarray([2, 3, 4, 5], dtype=np.int64),
            "batch_r": np.asarray([1, 1, 2, 2], dtype=np.int64),
            "batch_y": np.zeros(4, dtype=np.float32),
            "subgraph_feature": torch.randn(4, 4),
            "mode": "normal",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "architecture.jsonl"
            trainer = CTSTrainer(
                model=strategy,
                data_loader=[],
                train_times=1,
                use_gpu=False,
                architecture_diagnostic_interval=100,
                architecture_diagnostic_path=str(path),
            )
            probe = trainer._architecture_probe(data)
            trainer._write_architecture_probe(99, probe)
            self.assertIn("invariant_contribution_abs_mean", probe)
            self.assertIn("bias_contribution_abs_mean", probe)
            self.assertIn("invariant_contribution_share", probe)
            self.assertIn("bias_contribution_share", probe)
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["epoch"], 99)
    def _official(self):
        return LinkPredictor(
            ent_tot=100,
            rel_tot=10,
            max_hash_hops=1,
            hidden_dim=32,
        )

    def _cts(self):
        return CTSNetIndependentResidualPredictor(
            ent_tot=100,
            rel_tot=10,
            max_hash_hops=1,
            hidden_dim=32,
            branch_dim=16,
            residual_scale=0.1,
            residual_multiplier=0.0,
        )

    def _treatment_aware(
        self, center_residual=False, preserve_base_ties=False
    ):
        model = CTSNetTreatmentAwareResidualPredictor(
            ent_tot=100,
            rel_tot=10,
            max_hash_hops=1,
            hidden_dim=32,
            branch_dim=16,
            residual_scale=0.1,
            residual_multiplier=0.0,
            center_residual=center_residual,
            preserve_base_ties=preserve_base_ties,
        )
        model.set_train_triples(
            [
                [0, 1, 0],
                [0, 2, 0],
                [0, 3, 0],
                [4, 5, 0],
            ]
        )
        return model

    def test_zero_multiplier_matches_official_random_initialization(self):
        torch.manual_seed(23)
        official = self._official()
        torch.manual_seed(23)
        cts = self._cts()
        features = torch.randn(24, 4)
        official.eval()
        cts.eval()
        expected = official({"subgraph_feature": features}).detach()
        actual = cts({"subgraph_feature": features}).detach()
        self.assertTrue(torch.equal(expected, actual))

    def test_treatment_aware_zero_multiplier_matches_official(self):
        torch.manual_seed(23)
        official = self._official()
        torch.manual_seed(23)
        cts = self._treatment_aware()
        features = torch.randn(2, 4)
        data = {
            "subgraph_feature": features,
            "batch_h": torch.tensor([0, 4]),
            "batch_t": torch.tensor([1, 5]),
            "batch_r": torch.tensor([0, 0]),
        }
        official.eval()
        cts.eval()

        expected = official({"subgraph_feature": features}).detach()
        actual = cts(data).detach()

        self.assertTrue(torch.equal(expected, actual))

    def test_popularity_breaks_same_sketch_bias_tie(self):
        cts = self._treatment_aware()
        cts.eval()
        components = cts.forward_components(
            {
                "subgraph_feature": torch.zeros(2, 4),
                "batch_h": torch.tensor([0, 4]),
                "batch_t": torch.tensor([1, 5]),
                "batch_r": torch.tensor([0, 0]),
            }
        )

        self.assertNotEqual(
            components["treatment_feature"][0].item(),
            components["treatment_feature"][1].item(),
        )

    def test_query_centering_removes_constant_residual_shift(self):
        cts = self._treatment_aware(center_residual=True)
        cts.set_residual_multiplier(1.0)
        cts.invariant_delta_head.weight.data.zero_()
        cts.invariant_delta_head.bias.data.fill_(10.0)
        cts.bias_delta_head.weight.data.zero_()
        cts.bias_delta_head.bias.data.fill_(10.0)
        cts.eval()
        features = torch.zeros(3, 4)
        data = {
            "subgraph_feature": features,
            "batch_h": torch.tensor([0, 1, 2]),
            "batch_t": torch.tensor([3]),
            "batch_r": torch.tensor([0]),
            "mode": "head_batch",
        }

        components = cts.forward_components(data)

        self.assertTrue(
            torch.equal(
                components["residual_delta"],
                torch.zeros_like(components["residual_delta"]),
            )
        )
        self.assertTrue(
            torch.equal(components["score"], components["base_score"])
        )

    def test_query_residual_preserves_exact_baseline_ties(self):
        cts = self._treatment_aware(preserve_base_ties=True)
        cts.set_residual_multiplier(1.0)
        cts.eval()
        features = torch.zeros(3, 4)
        components = cts.forward_components({
            "subgraph_feature": features,
            "batch_h": torch.tensor([0, 1, 4]),
            "batch_t": torch.tensor([5]),
            "batch_r": torch.tensor([0]),
            "mode": "head_batch",
        })

        self.assertTrue(
            torch.equal(
                components["residual_delta"],
                components["residual_delta"][0].expand_as(
                    components["residual_delta"]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                components["score"],
                components["score"][0].expand_as(components["score"]),
            )
        )

    def test_query_residual_floor_never_beats_baseline_min(self):
        cts = self._treatment_aware(
            preserve_base_ties=True,
        )
        cts.floor_residual_at_baseline_min = True
        cts.set_residual_multiplier(1.0)
        cts.eval()
        components = cts.forward_components({
            "subgraph_feature": torch.randn(4, 4),
            "batch_h": torch.tensor([0, 1, 2, 4]),
            "batch_t": torch.tensor([5]),
            "batch_r": torch.tensor([0]),
            "mode": "head_batch",
        })
        baseline_min = components["base_score"].min()
        self.assertGreaterEqual(
            components["score"].min().item(), baseline_min.item()
        )
        best_mask = components["base_score"] == baseline_min
        self.assertTrue(torch.equal(
            components["score"][best_mask],
            components["base_score"][best_mask],
        ))

    def test_query_confidence_gate_uses_normalized_top_gap(self):
        cts = self._treatment_aware()
        scores = torch.tensor([[0.0], [0.05], [1.0]])

        cts.set_residual_max_top_gap_ratio(0.1)
        self.assertTrue(cts._query_allows_residual(scores))
        cts.set_residual_max_top_gap_ratio(0.01)
        self.assertFalse(cts._query_allows_residual(scores))
        cts.set_residual_max_top_gap_ratio(None)
        self.assertTrue(cts._query_allows_residual(scores))

    def test_tie_break_centering_preserves_group_means_and_cross_group_bounds(self):
        cts = self._treatment_aware()
        values = torch.tensor([[1.0], [3.0], [8.0], [10.0]])
        scores = torch.tensor([[0.0], [0.0], [1.0], [2.0]])
        centered = cts._center_within_equal_scores(values, scores)
        bounds = cts._tie_group_bounds(scores)
        self.assertTrue(torch.equal(centered, torch.tensor([[-1.0], [1.0], [0.0], [0.0]])))
        self.assertEqual(bounds[0].item(), bounds[1].item())
        self.assertGreater(bounds[0].item(), 0.0)
        self.assertEqual(bounds[2].item(), 0.0)
        self.assertEqual(bounds[3].item(), 0.0)

    def test_row_tie_mean_is_differentiable_and_group_constant(self):
        values = torch.tensor(
            [[1.0, 3.0, 5.0], [7.0, 11.0, 13.0]],
            requires_grad=True,
        )
        scores = torch.tensor([[0.0, 0.0, 1.0], [2.0, 3.0, 2.0]])

        grouped = CTSNegativeSampling._mean_within_row_ties(values, scores)

        self.assertTrue(torch.equal(
            grouped,
            torch.tensor([[2.0, 2.0, 5.0], [10.0, 11.0, 10.0]]),
        ))
        grouped.sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.ones_like(values)))

    def test_residual_rank_can_focus_on_baseline_hard_negative(self):
        strategy = CTSNegativeSampling(
            model=self._cts(),
            loss=MarginLoss(margin=5.0),
            batch_size=1,
            residual_hard_negatives=1,
        )
        residual = torch.zeros(4, 1, requires_grad=True)
        rank_loss, _, diagnostics = strategy._residual_losses(
            {
                "base_logit": torch.tensor([[0.0], [-1.0], [1.0], [3.0]]),
                "residual_unit_delta": residual,
            },
            residual.sum(),
        )
        rank_loss.backward()

        self.assertGreater(residual.grad[1].abs().item(), 0.0)
        self.assertEqual(residual.grad[2].item(), 0.0)
        self.assertEqual(residual.grad[3].item(), 0.0)
        self.assertAlmostEqual(
            diagnostics["hard_negative_fraction"], 1.0 / 3.0
        )

    def test_residual_rank_can_mine_near_tie_negative(self):
        strategy = CTSNegativeSampling(
            model=self._cts(),
            loss=MarginLoss(margin=5.0),
            batch_size=1,
            residual_near_tie_negatives=1,
        )
        residual = torch.zeros(4, 1, requires_grad=True)
        rank_loss, _, diagnostics = strategy._residual_losses(
            {
                "base_logit": torch.tensor([[0.0], [0.01], [1.0], [3.0]]),
                "residual_unit_delta": residual,
            },
            residual.sum(),
        )
        rank_loss.backward()

        self.assertGreater(residual.grad[1].abs().item(), 0.0)
        self.assertEqual(residual.grad[2].item(), 0.0)
        self.assertEqual(residual.grad[3].item(), 0.0)
        self.assertAlmostEqual(
            diagnostics["near_tie_negative_fraction"], 1.0 / 3.0
        )
        self.assertAlmostEqual(
            diagnostics["selected_abs_base_margin_mean"], 0.01
        )

    def test_counterfactual_features_do_not_change_official_bn_stats(self):
        torch.manual_seed(23)
        official = self._official()
        torch.manual_seed(23)
        cts = self._cts()
        rank_features = torch.randn(24, 4)
        cf_features = torch.randn(11, 4) + 5.0
        official.train()
        cts.train()
        official({"subgraph_feature": rank_features})
        cts.forward_components({
            "subgraph_feature": torch.cat(
                (rank_features, cf_features), dim=0
            ),
            "rank_feature_count": len(rank_features),
        })
        for name in ("norm1", "norm2", "norm3"):
            official_bn = getattr(official.sf_lin_layer, name)
            cts_bn = getattr(cts.sf_lin_layer, name)
            self.assertTrue(
                torch.equal(official_bn.running_mean, cts_bn.running_mean)
            )
            self.assertTrue(
                torch.equal(official_bn.running_var, cts_bn.running_var)
            )

    def test_residual_multiplier_is_clamped_and_persisted(self):
        cts = self._cts()
        cts.set_residual_multiplier(2.0)
        self.assertEqual(cts._residual_multiplier.item(), 1.0)
        cts.set_residual_multiplier(-1.0)
        self.assertEqual(cts._residual_multiplier.item(), 0.0)

    def test_unit_residual_is_trainable_when_fusion_is_disabled(self):
        cts = self._cts()
        features = torch.randn(8, 4)
        components = cts.forward_components({"subgraph_feature": features})
        self.assertTrue(
            torch.equal(
                components["residual_delta"],
                torch.zeros_like(components["residual_delta"]),
            )
        )
        self.assertGreater(
            components["residual_unit_delta"].abs().sum().item(), 0.0
        )

    def test_branch_rank_losses_bypass_a_saturated_gate(self):
        cts = self._cts()
        cts.gate_head.weight.data.zero_()
        cts.gate_head.bias.data.fill_(100.0)
        strategy = CTSNegativeSampling(
            model=cts,
            loss=MarginLoss(margin=5.0),
            batch_size=2,
            lambda_residual_rank=0.0,
            lambda_residual_preserve=0.0,
            lambda_invariant_rank=1.0,
            lambda_bias_rank=1.0,
            center_residual=True,
            isolate_base_rank=True,
            residual_start_epoch=0,
            residual_warmup_epochs=0,
        )
        loss = strategy({"subgraph_feature": torch.randn(6, 4)})
        loss.backward()

        self.assertGreater(
            cts.invariant_delta_head.weight.grad.abs().sum().item(), 0.0
        )
        self.assertGreater(
            cts.bias_delta_head.weight.grad.abs().sum().item(), 0.0
        )
        metrics = strategy.get_last_metrics()
        self.assertGreater(metrics["weighted_invariant_rank"], 0.0)
        self.assertGreater(metrics["weighted_bias_rank"], 0.0)

    def test_residual_loss_uses_detached_baseline_and_preserves_margin(self):
        strategy = CTSNegativeSampling(
            model=self._cts(),
            loss=MarginLoss(margin=5.0),
            batch_size=1,
            lambda_residual_rank=1.0,
            lambda_residual_preserve=1.0,
            isolate_base_rank=True,
        )
        base_logit = torch.tensor([[0.0], [1.0]], requires_grad=True)
        harmful_delta = torch.tensor(
            [[0.2], [-0.2]], requires_grad=True
        )
        rank_loss, preserve_loss, diagnostics = strategy._residual_losses(
            {
                "base_logit": base_logit,
                "residual_unit_delta": harmful_delta,
            },
            base_logit.sum(),
        )
        (rank_loss + preserve_loss).backward()
        self.assertIsNone(base_logit.grad)
        self.assertGreater(harmful_delta.grad.abs().sum().item(), 0.0)
        self.assertAlmostEqual(preserve_loss.item(), 0.4, places=6)
        self.assertLess(
            diagnostics["candidate_margin_mean"],
            diagnostics["base_margin_mean"],
        )

    def test_residual_preserve_protects_baseline_ties(self):
        strategy = CTSNegativeSampling(
            model=self._cts(),
            loss=MarginLoss(margin=5.0),
            batch_size=1,
        )
        residual = torch.tensor([[0.2], [0.0]], requires_grad=True)
        _, preserve_loss, _ = strategy._residual_losses(
            {
                "base_logit": torch.tensor([[0.0], [0.0]]),
                "residual_unit_delta": residual,
            },
            residual.sum(),
        )
        self.assertAlmostEqual(preserve_loss.item(), 0.2, places=6)

    def test_residual_losses_follow_the_late_residual_schedule(self):
        cts = self._cts()
        strategy = CTSNegativeSampling(
            model=cts,
            loss=MarginLoss(margin=5.0),
            batch_size=1,
            lambda_residual_rank=1.0,
            lambda_residual_preserve=1.0,
            isolate_base_rank=True,
            warmup_start_epoch=0,
            warmup_epochs=0,
            residual_start_epoch=8,
            residual_warmup_epochs=2,
        )
        features = torch.randn(3, 4)

        strategy.set_epoch(7)
        strategy({"subgraph_feature": features})
        before = strategy.get_last_metrics()
        self.assertEqual(before["residual_weight"], 0.0)
        self.assertEqual(before["weighted_residual_rank"], 0.0)

        strategy.set_epoch(8)
        strategy({"subgraph_feature": features})
        after = strategy.get_last_metrics()
        self.assertEqual(after["residual_weight"], 0.5)
        self.assertGreater(after["weighted_residual_rank"], 0.0)
        self.assertAlmostEqual(cts._residual_multiplier.item(), 0.5)

    def test_isolated_residual_rank_keeps_official_gradients_exact(self):
        torch.manual_seed(23)
        official = self._official()
        torch.manual_seed(23)
        cts = self._cts()
        features = torch.randn(6, 4)

        official.train()
        official_score = official({"subgraph_feature": features})
        official_positive = official_score[:2].view(-1, 2).permute(1, 0)
        official_negative = official_score[2:].view(-1, 2).permute(1, 0)
        MarginLoss(margin=5.0)(
            official_positive, official_negative
        ).backward()

        strategy = CTSNegativeSampling(
            model=cts,
            loss=MarginLoss(margin=5.0),
            batch_size=2,
            lambda_residual_rank=1.0,
            lambda_residual_preserve=1.0,
            isolate_base_rank=True,
            warmup_start_epoch=0,
            warmup_epochs=0,
            residual_start_epoch=0,
            residual_warmup_epochs=0,
        )
        cts.train()
        strategy({"subgraph_feature": features}).backward()

        official_parameters = dict(official.named_parameters())
        cts_parameters = dict(cts.named_parameters())
        for name, parameter in official_parameters.items():
            cts_gradient = cts_parameters[name].grad
            if parameter.grad is None or cts_gradient is None:
                self.assertIsNone(parameter.grad)
                self.assertIsNone(cts_gradient)
                continue
            self.assertTrue(
                torch.equal(parameter.grad, cts_gradient),
                msg="official gradient changed for {}".format(name),
            )
        self.assertGreater(
            cts.invariant_delta_head.weight.grad.abs().sum().item(), 0.0
        )
        self.assertGreater(
            cts.bias_delta_head.weight.grad.abs().sum().item(), 0.0
        )


if __name__ == "__main__":
    unittest.main()

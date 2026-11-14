from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear

from .CTSNetIndependentResidualPredictor import (
    CTSNetIndependentResidualPredictor,
)


class CTSNetTreatmentAwareResidualPredictor(
    CTSNetIndependentResidualPredictor
):
    """Residual scorer whose Bias branch observes train-only popularity."""

    def __init__(
        self,
        ent_tot,
        rel_tot,
        max_hash_hops=1,
        hidden_dim=128,
        branch_dim=64,
        residual_scale=0.1,
        residual_multiplier=0.0,
        center_residual=False,
        preserve_base_ties=False,
        break_base_ties=False,
        floor_residual_at_baseline_min=False,
        residual_max_top_gap_ratio=None,
        feature_mode="triple",
        branch_activation="relu",
        branch_negative_slope=0.01,
        delta_limit=None,
        delta_temperature=1.0,
        gate_temperature=1.0,
        delta_normalization="none",
        delta_target_rms=0.25,
        fusion_mode="additive",
        invariant_margin_fraction=0.25,
        invariant_margin_ungated=False,
        structural_abstention=False,
        structural_abstention_min_span_ratio=0.05,
        structural_abstention_require_gate_agreement=True,
        isolate_treatment_gradients=False,
        treatment_shared_gradient_scale=1.0,
    ):
        super(CTSNetTreatmentAwareResidualPredictor, self).__init__(
            ent_tot=ent_tot,
            rel_tot=rel_tot,
            max_hash_hops=max_hash_hops,
            hidden_dim=hidden_dim,
            branch_dim=branch_dim,
            residual_scale=residual_scale,
            residual_multiplier=residual_multiplier,
            feature_mode=feature_mode,
            branch_activation=branch_activation,
            branch_negative_slope=branch_negative_slope,
            delta_limit=delta_limit,
            delta_temperature=delta_temperature,
            gate_temperature=gate_temperature,
            delta_normalization=delta_normalization,
            delta_target_rms=delta_target_rms,
        )
        self.register_buffer("entity_log_degree", torch.zeros(ent_tot))
        self.register_buffer("popularity_mean", torch.zeros(1))
        self.register_buffer("popularity_std", torch.ones(1))
        self.center_residual = bool(center_residual)
        if self.center_residual:
            self.register_buffer("center_residual_enabled", torch.ones(1))
        self.preserve_base_ties = bool(preserve_base_ties)
        self.break_base_ties = bool(break_base_ties)
        if self.preserve_base_ties and self.break_base_ties:
            raise ValueError(
                "preserve_base_ties and break_base_ties are mutually exclusive"
            )
        if self.break_base_ties:
            self.register_buffer("break_base_ties_enabled", torch.ones(1))
        if self.preserve_base_ties:
            self.register_buffer("preserve_base_ties_enabled", torch.ones(1))
        self.floor_residual_at_baseline_min = bool(
            floor_residual_at_baseline_min
        )
        if self.floor_residual_at_baseline_min:
            self.register_buffer(
                "floor_residual_at_baseline_min_enabled", torch.ones(1)
            )
        self.set_residual_max_top_gap_ratio(residual_max_top_gap_ratio)
        if fusion_mode not in ("additive", "invariant-margin"):
            raise ValueError("fusion_mode must be 'additive' or 'invariant-margin'")
        if invariant_margin_fraction <= 0.0 or invariant_margin_fraction > 1.0:
            raise ValueError("invariant_margin_fraction must be in (0, 1]")
        self.fusion_mode = fusion_mode
        self.invariant_margin_fraction = float(invariant_margin_fraction)
        # In invariant-margin mode the Bias/treatment gate is auxiliary.  On
        # sparse datasets it can collapse to either 0 or 1 and accidentally
        # switch off the invariant ranking branch.  This runtime/config flag
        # lets the invariant branch rank directly while retaining the gate for
        # diagnostics and treatment supervision.
        self.invariant_margin_ungated = bool(invariant_margin_ungated)
        self.structural_abstention = bool(structural_abstention)
        self.structural_abstention_min_span_ratio = float(
            structural_abstention_min_span_ratio
        )
        if not 0.0 <= self.structural_abstention_min_span_ratio <= 1.0:
            raise ValueError(
                "structural_abstention_min_span_ratio must be in [0, 1]"
            )
        self.structural_abstention_require_gate_agreement = bool(
            structural_abstention_require_gate_agreement
        )
        # Optional train-time isolation: the treatment/bias branch still learns
        # from the shared structural features, but cannot rewrite the shared
        # residual stem through its substantially larger classification
        # gradients.  Detach is runtime-only and does not change checkpoint
        # tensors or evaluation scores.
        self.isolate_treatment_gradients = bool(isolate_treatment_gradients)
        self.treatment_shared_gradient_scale = float(
            treatment_shared_gradient_scale
        )
        if not 0.0 <= self.treatment_shared_gradient_scale <= 1.0:
            raise ValueError(
                "treatment_shared_gradient_scale must be in [0, 1]"
            )
        self.reset_structural_abstention_stats()
        # Runtime-only train-calibrated reliability weights.  Keeping this
        # buffer non-persistent preserves strict compatibility with historical
        # checkpoints while allowing relation-level residual abstention at
        # evaluation time.
        self.register_buffer(
            "relation_residual_reliability",
            torch.ones(rel_tot),
            persistent=False,
        )

        self.bias_encoder = self._make_branch(
            hidden_dim + 1, branch_dim, hidden_dim=hidden_dim
        )
        self.gate_head = Linear(hidden_dim + 1, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def set_relation_residual_reliability(self, reliability=None):
        """Set runtime relation weights in [0, 1]; ``None`` disables guard."""
        if reliability is None:
            self.relation_residual_reliability.fill_(1.0)
            return
        value = torch.as_tensor(
            reliability,
            dtype=self.relation_residual_reliability.dtype,
            device=self.relation_residual_reliability.device,
        ).reshape(-1)
        if value.numel() != self.rel_tot:
            raise ValueError(
                "Expected {} relation reliability weights, got {}".format(
                    self.rel_tot, value.numel()
                )
            )
        if not torch.isfinite(value).all():
            raise ValueError("relation reliability weights must be finite")
        if bool(((value < 0.0) | (value > 1.0)).any().item()):
            raise ValueError("relation reliability weights must be in [0, 1]")
        self.relation_residual_reliability.copy_(value)

    def _relation_residual_scale(self, data, count, device, dtype):
        if "batch_r" not in data:
            return torch.ones((count, 1), device=device, dtype=dtype)
        relation_ids = self._candidate_ids(data["batch_r"], count, device)
        return self.relation_residual_reliability[relation_ids].to(
            dtype=dtype
        ).view(-1, 1)

    def set_train_triples(self, triples):
        triples = np.asarray(triples, dtype=np.int64)
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("triples must have shape [num_triples, 3]")
        endpoints = np.concatenate((triples[:, 0], triples[:, 1]))
        degree = np.bincount(endpoints, minlength=self.ent_tot)
        log_degree = np.log1p(degree).astype(np.float32)
        popularity = log_degree[triples[:, 0]] + log_degree[triples[:, 1]]
        std = max(float(popularity.std()), 1e-6)
        self.entity_log_degree.copy_(torch.from_numpy(log_degree))
        self.popularity_mean.fill_(float(popularity.mean()))
        self.popularity_std.fill_(std)

    @staticmethod
    def _candidate_ids(value, count, device):
        ids = torch.as_tensor(value, dtype=torch.long, device=device).view(-1)
        if ids.numel() == 1:
            return ids.expand(count)
        if ids.numel() != count:
            raise ValueError(
                "Expected one or {} candidate ids, got {}".format(
                    count, ids.numel()
                )
            )
        return ids

    def _popularity_feature(self, data, count, device, dtype):
        required = {"batch_h", "batch_t", "batch_r"}
        if not required.issubset(data):
            missing = sorted(required.difference(data))
            raise KeyError(
                "treatment-aware scorer requires candidate ids: {}".format(
                    missing
                )
            )
        heads = self._candidate_ids(data["batch_h"], count, device)
        tails = self._candidate_ids(data["batch_t"], count, device)
        self._candidate_ids(data["batch_r"], count, device)
        raw = self.entity_log_degree[heads] + self.entity_log_degree[tails]
        normalized = (raw - self.popularity_mean) / self.popularity_std
        return normalized.clamp(min=-5.0, max=5.0).to(dtype=dtype).view(-1, 1)

    @staticmethod
    def _mean_within_equal_scores(values, scores):
        flat_values = values.reshape(-1)
        flat_scores = scores.detach().reshape(-1)
        unique_scores, inverse = torch.unique(
            flat_scores, sorted=False, return_inverse=True
        )
        group_count = unique_scores.numel()
        sums = values.new_zeros(group_count)
        counts = values.new_zeros(group_count)
        sums.scatter_add_(0, inverse, flat_values)
        counts.scatter_add_(0, inverse, torch.ones_like(flat_values))
        return (sums / counts.clamp_min(1.0))[inverse].view_as(values)

    @staticmethod
    def _center_within_equal_scores(values, scores):
        return values - CTSNetTreatmentAwareResidualPredictor._mean_within_equal_scores(
            values, scores
        )

    @staticmethod
    def _tie_group_bounds(scores):
        flat_scores = scores.detach().reshape(-1)
        unique_scores, inverse = torch.unique(
            flat_scores, sorted=True, return_inverse=True
        )
        counts = scores.new_zeros(unique_scores.numel())
        counts.scatter_add_(0, inverse, torch.ones_like(flat_scores))
        bounds = scores.new_zeros(unique_scores.numel())
        if unique_scores.numel() > 1:
            gaps = unique_scores[1:] - unique_scores[:-1]
            bounds[:-1] = gaps
            bounds[1:] = torch.minimum(bounds[1:], gaps)
            bounds[0] = gaps[0]
            bounds[-1] = gaps[-1]
        bounds = torch.where(counts > 1.0, 0.49 * bounds, bounds.new_zeros(()))
        return bounds[inverse].view_as(scores)

    def set_residual_max_top_gap_ratio(self, value):
        if value is None:
            self.residual_max_top_gap_ratio = None
            return
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("residual max top-gap ratio must be in [0, 1]")
        self.residual_max_top_gap_ratio = value

    def _query_allows_residual(self, base_score):
        if self.residual_max_top_gap_ratio is None:
            return True
        unique_scores = torch.unique(base_score.detach().reshape(-1)).sort().values
        if unique_scores.numel() < 2:
            return False
        score_range = unique_scores[-1] - unique_scores[0]
        if float(score_range.item()) <= 0.0:
            return False
        top_gap_ratio = (unique_scores[1] - unique_scores[0]) / score_range
        return float(top_gap_ratio.item()) <= self.residual_max_top_gap_ratio

    def _query_allows_structural_residual(
        self, fusion_anchor, invariant_delta, gate
    ):
        """Abstain when a top collision has no stable invariant separator."""
        if not self.structural_abstention:
            return True, 0.0, True
        flat_anchor = fusion_anchor.detach().reshape(-1)
        flat_invariant = invariant_delta.detach().reshape(-1)
        flat_gate = gate.detach().reshape(-1)
        if flat_anchor.numel() < 2:
            return False, 0.0, False
        top_mask = flat_anchor == flat_anchor.min()
        if int(top_mask.sum().item()) < 2:
            # This guard is designed only for baseline collisions.  Preserve a
            # unique official winner instead of allowing an unnecessary edit.
            return False, 0.0, False
        full_span = flat_invariant.max() - flat_invariant.min()
        if float(full_span.item()) <= 1e-12:
            return False, 0.0, False
        top_invariant = flat_invariant[top_mask]
        top_span = top_invariant.max() - top_invariant.min()
        span_ratio = float((top_span / full_span.clamp_min(1e-12)).item())
        identifiable = (
            float(top_span.item()) > 1e-12
            and span_ratio >= self.structural_abstention_min_span_ratio
        )
        agreement = True
        if self.structural_abstention_require_gate_agreement:
            gated_invariant = (flat_gate * flat_invariant)[top_mask]
            agreement = bool(
                torch.argmin(top_invariant).item()
                == torch.argmin(gated_invariant).item()
            )
        return identifiable and agreement, span_ratio, agreement

    def reset_structural_abstention_stats(self):
        self._structural_query_count = 0
        self._structural_allowed_count = 0
        self._structural_gate_agreement_count = 0
        self._structural_span_ratio_sum = 0.0

    def structural_abstention_stats(self):
        count = max(self._structural_query_count, 1)
        return {
            "query_count": self._structural_query_count,
            "allowed_fraction": self._structural_allowed_count / float(count),
            "gate_agreement_fraction": (
                self._structural_gate_agreement_count / float(count)
            ),
            "mean_invariant_span_ratio": (
                self._structural_span_ratio_sum / float(count)
            ),
        }

    def _prepare_residual_unit(self, raw_delta, base_score, query_mode):
        prepared = raw_delta
        if self.break_base_ties and query_mode and prepared.shape[0] > 1:
            prepared = self._center_within_equal_scores(prepared, base_score)
        elif self.preserve_base_ties and query_mode and prepared.shape[0] > 1:
            prepared = self._mean_within_equal_scores(prepared, base_score)
        if self.center_residual and query_mode and prepared.shape[0] > 1:
            prepared = prepared - prepared.mean(dim=0, keepdim=True)
        if query_mode and prepared.shape[0] > 1:
            prepared = self._calibrate_residual_raw(prepared, dim=0)
        unit_delta = self._residual_scale * torch.tanh(prepared)
        if self.break_base_ties and query_mode and prepared.shape[0] > 1:
            unit_delta = unit_delta.clamp(
                -self._tie_group_bounds(base_score),
                self._tie_group_bounds(base_score),
            )
        return unit_delta

    def forward_components(self, data):
        feature = self._get_subgraph_feature(data)
        official_shared, base_logit = self._official_forward(
            feature, data.get("rank_feature_count")
        )
        residual_shared = self.residual_shared_encoder(feature)
        popularity = self._popularity_feature(
            data, feature.shape[0], feature.device, feature.dtype
        )
        treatment_gradient_scale = (
            0.0
            if self.isolate_treatment_gradients
            else self.treatment_shared_gradient_scale
        )
        bias_shared = residual_shared.detach() + treatment_gradient_scale * (
            residual_shared - residual_shared.detach()
        )
        bias_input = torch.cat((bias_shared, popularity), dim=-1)

        z_inv = self.invariant_encoder(residual_shared)
        z_bias = self.bias_encoder(bias_input)
        inv_delta_raw = self.invariant_delta_head(z_inv)
        bias_delta_raw = self.bias_delta_head(z_bias)
        inv_delta = self._transform_delta(inv_delta_raw)
        bias_delta = self._transform_delta(bias_delta_raw)
        gate_logit = self.gate_head(bias_input)
        gate = torch.sigmoid(gate_logit / self.gate_temperature)
        if self.fusion_mode == "invariant-margin" and self.invariant_margin_ungated:
            invariant_contribution = (
                self.invariant_branch_multiplier * inv_delta
            )
        else:
            invariant_contribution = (
                self.invariant_branch_multiplier * gate * inv_delta
            )
        bias_contribution = (
            self.bias_branch_multiplier * (1.0 - gate) * bias_delta
        )
        if self.fusion_mode == "invariant-margin":
            # Keep the treatment-sensitive branch auxiliary; only invariant
            # structure can alter the baseline ranking in this mode.
            mixed_delta = invariant_contribution
        else:
            mixed_delta = invariant_contribution + bias_contribution
        base_score = torch.sigmoid(base_logit)
        fusion_anchor = (
            base_logit
            if self.output_score_space == "logit"
            else base_score
        )
        query_mode = data.get("mode") in ("head_batch", "tail_batch")
        residual_unit_delta = self._prepare_residual_unit(
            mixed_delta, fusion_anchor, query_mode
        )
        invariant_unit_delta = self._prepare_residual_unit(
            self.invariant_branch_multiplier * inv_delta,
            fusion_anchor,
            query_mode,
        )
        bias_unit_delta = self._prepare_residual_unit(
            self.bias_branch_multiplier * bias_delta,
            fusion_anchor,
            query_mode,
        )
        residual_unit_delta_raw = self._residual_scale * torch.tanh(
            mixed_delta
        )
        bounded_delta = self._residual_multiplier * residual_unit_delta
        if self.fusion_mode == "invariant-margin" and query_mode and fusion_anchor.shape[0] > 1:
            score_range = (
                fusion_anchor.max() - fusion_anchor.min()
            ).detach()
            cap = self.invariant_margin_fraction * score_range
            bounded_delta = bounded_delta.clamp(-cap, cap)
        bounded_delta = bounded_delta * self._relation_residual_scale(
            data,
            bounded_delta.shape[0],
            bounded_delta.device,
            bounded_delta.dtype,
        )
        structural_allowed = True
        structural_span_ratio = 0.0
        structural_gate_agreement = True
        if query_mode:
            (
                structural_allowed,
                structural_span_ratio,
                structural_gate_agreement,
            ) = self._query_allows_structural_residual(
                fusion_anchor, inv_delta, gate
            )
            if self.structural_abstention:
                self._structural_query_count += 1
                self._structural_allowed_count += int(structural_allowed)
                self._structural_gate_agreement_count += int(
                    structural_gate_agreement
                )
                self._structural_span_ratio_sum += structural_span_ratio
            if not structural_allowed:
                bounded_delta = torch.zeros_like(bounded_delta)
        if query_mode and not self._query_allows_residual(base_score):
            bounded_delta = torch.zeros_like(bounded_delta)
        # In probability mode this preserves the historical TGIIF path.  In
        # logit mode we keep the same bounded residual but rank by the latent
        # official logit, avoiding NELL-995's 0/1 sigmoid collapse.  The mode
        # is runtime-only and therefore does not alter checkpoint structure.
        final_score = fusion_anchor + bounded_delta
        if (
            self.floor_residual_at_baseline_min
            and query_mode
            and fusion_anchor.shape[0] > 1
        ):
            baseline_min = fusion_anchor.min().detach()
            final_score = torch.where(
                fusion_anchor == baseline_min,
                fusion_anchor,
                torch.where(
                    final_score < baseline_min,
                    fusion_anchor,
                    final_score,
                ),
            )

        return {
            "score": final_score,
            "base_score": base_score,
            "base_logit": base_logit,
            "fusion_mode": self.fusion_mode,
            "residual_anchor": fusion_anchor,
            "residual_delta": bounded_delta,
            "residual_unit_delta": residual_unit_delta,
            "invariant_unit_delta": invariant_unit_delta,
            "bias_unit_delta": bias_unit_delta,
            "residual_unit_delta_raw": residual_unit_delta_raw,
            "residual_raw": mixed_delta,
            "invariant_residual_raw": (
                self.invariant_branch_multiplier * inv_delta
            ),
            "bias_residual_raw": self.bias_branch_multiplier * bias_delta,
            "invariant_delta": inv_delta,
            "bias_delta": bias_delta,
            "invariant_delta_raw": inv_delta_raw,
            "bias_delta_raw": bias_delta_raw,
            "invariant_contribution": invariant_contribution,
            "bias_contribution": bias_contribution,
            "s_inv": torch.sigmoid(inv_delta),
            "s_bias": torch.sigmoid(bias_delta),
            "treatment_logit": self.treatment_head(z_bias),
            "treatment_feature": popularity,
            "gate": gate,
            "gate_logit": gate_logit,
            "z_shared": residual_shared,
            "official_shared": official_shared,
            "z_inv": z_inv,
            "z_bias": z_bias,
        }

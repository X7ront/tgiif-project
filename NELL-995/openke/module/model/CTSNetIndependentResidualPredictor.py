from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Linear

from .Model import Model


class CTSNetIndependentResidualPredictor(Model):
    """Fair from-scratch CTS scorer with an official main path.

    The main path is architecture-identical to the official LinkPredictor.
    CTS representations are produced by an independent LayerNorm encoder, so
    factual/counterfactual batches cannot alter the official BatchNorm running
    statistics.  A scheduled multiplier keeps the residual disabled during
    baseline learning and introduces it only near the end of the same total
    epoch budget.
    """

    def __init__(
        self,
        ent_tot,
        rel_tot,
        max_hash_hops=1,
        hidden_dim=128,
        branch_dim=64,
        residual_scale=0.1,
        residual_multiplier=0.0,
        feature_mode="triple",
        branch_activation="relu",
        branch_negative_slope=0.01,
        delta_limit=None,
        delta_temperature=1.0,
        gate_temperature=1.0,
        delta_normalization="none",
        delta_target_rms=0.25,
    ):
        super(CTSNetIndependentResidualPredictor, self).__init__(
            ent_tot, rel_tot
        )
        if max_hash_hops < 1:
            raise ValueError("max_hash_hops must be at least 1")
        if residual_scale < 0.0:
            raise ValueError("residual_scale cannot be negative")
        if branch_activation not in ("relu", "leaky_relu", "silu"):
            raise ValueError(
                "branch_activation must be 'relu', 'leaky_relu', or 'silu'"
            )
        if branch_negative_slope < 0.0:
            raise ValueError("branch_negative_slope cannot be negative")
        if delta_limit is not None and float(delta_limit) <= 0.0:
            raise ValueError("delta_limit must be positive or None")
        if delta_temperature <= 0.0:
            raise ValueError("delta_temperature must be positive")
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive")
        if delta_normalization not in ("none", "query-rms"):
            raise ValueError(
                "delta_normalization must be 'none' or 'query-rms'"
            )
        if delta_target_rms <= 0.0:
            raise ValueError("delta_target_rms must be positive")

        self.max_hash_hops = int(max_hash_hops)
        if feature_mode not in ("triple", "pairwise"):
            raise ValueError("feature_mode must be 'triple' or 'pairwise'")
        self.feature_mode = feature_mode
        self.dim = (
            max_hash_hops * (max_hash_hops + 2)
            if feature_mode == "pairwise"
            else max_hash_hops * (max_hash_hops ** 2 + 3)
        )
        self.hidden_dim = int(hidden_dim)
        self.branch_dim = int(branch_dim)
        self.branch_activation = branch_activation
        self.branch_negative_slope = float(branch_negative_slope)
        self.delta_limit = (
            None if delta_limit is None else float(delta_limit)
        )
        self.delta_temperature = float(delta_temperature)
        self.gate_temperature = float(gate_temperature)
        self.delta_normalization = delta_normalization
        self.delta_target_rms = float(delta_target_rms)
        # The historical evaluator consumes sigmoid probabilities.  NELL-995
        # can drive the official logits into floating-point saturation, so we
        # expose an explicit, runtime-only logit ranking mode for diagnosis
        # and controlled dataset-adaptation experiments.  It is deliberately
        # not part of state_dict, preserving compatibility with old ckpts.
        self.output_score_space = "probability"
        if self.delta_normalization == "query-rms":
            self.register_buffer("delta_query_rms_enabled", torch.ones(1))
            self.register_buffer(
                "delta_target_rms_value",
                torch.tensor(self.delta_target_rms),
            )
        self.register_buffer(
            "_residual_scale", torch.tensor(float(residual_scale))
        )
        self.register_buffer(
            "_residual_multiplier",
            torch.tensor(float(residual_multiplier)),
        )

        # Exact official LinkPredictor main path, including module names.
        self.sf_lin_layer = nn.Sequential(OrderedDict([
            ("dense1", Linear(self.dim, hidden_dim)),
            ("norm1", nn.BatchNorm1d(hidden_dim)),
            ("relu1", nn.ReLU()),
            ("dense2", Linear(hidden_dim, hidden_dim)),
            ("norm2", nn.BatchNorm1d(hidden_dim)),
            ("relu2", nn.ReLU()),
            ("dense3", Linear(hidden_dim, hidden_dim)),
            ("norm3", nn.BatchNorm1d(hidden_dim)),
            ("relu3", nn.ReLU()),
        ]))
        self.lin = Linear(hidden_dim, 1)

        # CTS path is independent from the official BatchNorm encoder.
        self.residual_shared_encoder = nn.Sequential(OrderedDict([
            ("dense1", Linear(self.dim, hidden_dim)),
            ("norm1", nn.LayerNorm(hidden_dim)),
            ("relu1", nn.ReLU()),
            ("dense2", Linear(hidden_dim, hidden_dim)),
            ("norm2", nn.LayerNorm(hidden_dim)),
            ("relu2", nn.ReLU()),
        ]))
        self.invariant_encoder = self._make_branch(hidden_dim, branch_dim)
        self.bias_encoder = self._make_branch(hidden_dim, branch_dim)
        self.invariant_delta_head = Linear(branch_dim, 1)
        self.bias_delta_head = Linear(branch_dim, 1)
        self.treatment_head = Linear(branch_dim, 1)
        self.gate_head = Linear(hidden_dim, 1)

        nn.init.normal_(self.invariant_delta_head.weight, std=1e-3)
        nn.init.zeros_(self.invariant_delta_head.bias)
        nn.init.normal_(self.bias_delta_head.weight, std=1e-3)
        nn.init.zeros_(self.bias_delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        # Runtime-only ablation controls deliberately stay out of state_dict so
        # historical checkpoints remain strictly loadable.
        self.invariant_branch_multiplier = 1.0
        self.bias_branch_multiplier = 1.0

    def _make_activation(self):
        if self.branch_activation == "leaky_relu":
            return nn.LeakyReLU(self.branch_negative_slope)
        if self.branch_activation == "silu":
            return nn.SiLU()
        return nn.ReLU()

    def _make_branch(self, input_dim, branch_dim, hidden_dim=None):
        hidden_dim = int(hidden_dim or input_dim)
        return nn.Sequential(OrderedDict([
            ("dense1", Linear(input_dim, hidden_dim)),
            ("norm1", nn.LayerNorm(hidden_dim)),
            ("relu1", self._make_activation()),
            ("dense2", Linear(hidden_dim, branch_dim)),
            ("norm2", nn.LayerNorm(branch_dim)),
            ("relu2", self._make_activation()),
        ]))

    def _transform_delta(self, raw_delta):
        if self.delta_limit is None:
            return raw_delta
        return self.delta_limit * torch.tanh(
            raw_delta / self.delta_temperature
        )

    def _calibrate_residual_raw(self, values, dim):
        """Shrink oversized residuals without amplifying small signals."""
        if self.delta_normalization != "query-rms":
            return values
        rms = (values.pow(2).mean(dim=dim, keepdim=True) + 1e-12).sqrt()
        divisor = (rms / self.delta_target_rms).clamp_min(1.0)
        return values / divisor

    def set_residual_multiplier(self, value):
        value = min(1.0, max(0.0, float(value)))
        self._residual_multiplier.fill_(value)

    def set_score_space(self, value):
        value = str(value).lower()
        if value not in ("probability", "logit"):
            raise ValueError("score space must be 'probability' or 'logit'")
        self.output_score_space = value

    def _score_from_logit(self, logit):
        if self.output_score_space == "logit":
            return logit
        return torch.sigmoid(logit)

    def set_branch_multipliers(self, invariant=1.0, bias=1.0):
        self.invariant_branch_multiplier = float(invariant)
        self.bias_branch_multiplier = float(bias)

    def _get_subgraph_feature(self, data):
        if "subgraph_feature" not in data:
            raise KeyError("data must contain 'subgraph_feature'")
        feature = data["subgraph_feature"]
        if feature.ndim != 2 or feature.shape[1] != self.dim:
            raise ValueError(
                "Expected subgraph_feature shape [N, {}], got {}".format(
                    self.dim, tuple(feature.shape)
                )
            )
        parameter = next(self.parameters())
        return feature.to(device=parameter.device, dtype=parameter.dtype)

    def _official_forward(self, feature, rank_feature_count=None):
        count = feature.shape[0]
        if (
            not self.training
            or rank_feature_count is None
            or int(rank_feature_count) >= count
        ):
            shared = self.sf_lin_layer(feature)
            return shared, self.lin(shared)

        rank_count = int(rank_feature_count)
        if rank_count < 1:
            raise ValueError("rank_feature_count must be positive")

        # Only the official factual+negative rank batch updates BatchNorm.
        rank_shared = self.sf_lin_layer(feature[:rank_count])
        rank_logit = self.lin(rank_shared)

        was_training = self.sf_lin_layer.training
        self.sf_lin_layer.eval()
        try:
            with torch.no_grad():
                cf_shared = self.sf_lin_layer(feature[rank_count:])
                cf_logit = self.lin(cf_shared)
        finally:
            self.sf_lin_layer.train(was_training)
        return (
            torch.cat((rank_shared, cf_shared.detach()), dim=0),
            torch.cat((rank_logit, cf_logit.detach()), dim=0),
        )

    def forward_components(self, data):
        feature = self._get_subgraph_feature(data)
        official_shared, base_logit = self._official_forward(
            feature, data.get("rank_feature_count")
        )
        residual_shared = self.residual_shared_encoder(feature)
        z_inv = self.invariant_encoder(residual_shared)
        z_bias = self.bias_encoder(residual_shared)

        inv_delta_raw = self.invariant_delta_head(z_inv)
        bias_delta_raw = self.bias_delta_head(z_bias)
        inv_delta = self._transform_delta(inv_delta_raw)
        bias_delta = self._transform_delta(bias_delta_raw)
        gate_logit = self.gate_head(residual_shared)
        gate = torch.sigmoid(gate_logit / self.gate_temperature)
        invariant_contribution = (
            self.invariant_branch_multiplier * gate * inv_delta
        )
        bias_contribution = (
            self.bias_branch_multiplier * (1.0 - gate) * bias_delta
        )
        query_mode = data.get("mode") in ("head_batch", "tail_batch")
        invariant_prepared = (
            self._calibrate_residual_raw(
                self.invariant_branch_multiplier * inv_delta, dim=0
            )
            if query_mode
            else self.invariant_branch_multiplier * inv_delta
        )
        bias_prepared = (
            self._calibrate_residual_raw(
                self.bias_branch_multiplier * bias_delta, dim=0
            )
            if query_mode
            else self.bias_branch_multiplier * bias_delta
        )
        invariant_unit_delta = self._residual_scale * torch.tanh(
            invariant_prepared
        )
        bias_unit_delta = self._residual_scale * torch.tanh(
            bias_prepared
        )
        mixed_delta = invariant_contribution + bias_contribution
        mixed_delta_prepared = (
            self._calibrate_residual_raw(mixed_delta, dim=0)
            if query_mode
            else mixed_delta
        )
        residual_unit_delta = self._residual_scale * torch.tanh(
            mixed_delta_prepared
        )
        bounded_delta = self._residual_multiplier * residual_unit_delta
        final_logit = base_logit + bounded_delta

        return {
            "score": self._score_from_logit(final_logit),
            "base_score": torch.sigmoid(base_logit),
            "base_logit": base_logit,
            "final_logit": final_logit,
            "residual_delta": bounded_delta,
            "residual_unit_delta": residual_unit_delta,
            "invariant_unit_delta": invariant_unit_delta,
            "bias_unit_delta": bias_unit_delta,
            "residual_raw": mixed_delta,
            "residual_calibrated_raw": mixed_delta_prepared,
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
            "gate": gate,
            "gate_logit": gate_logit,
            "z_shared": residual_shared,
            "official_shared": official_shared,
            "z_inv": z_inv,
            "z_bias": z_bias,
        }

    def forward(self, data):
        return self.forward_components(data)["score"]

    @torch.no_grad()
    def diagnostics(self, data):
        components = self.forward_components(data)
        residual = components["residual_delta"]
        residual_unit = components["residual_unit_delta"]
        scale = max(float(self._residual_scale.item()), 1e-12)
        return {
            "residual_multiplier": self._residual_multiplier.item(),
            "gate_mean": components["gate"].mean().item(),
            "gate_min": components["gate"].min().item(),
            "gate_max": components["gate"].max().item(),
            "base_score_mean": components["base_score"].mean().item(),
            "s_inv_mean": components["s_inv"].mean().item(),
            "s_bias_mean": components["s_bias"].mean().item(),
            "s_final_mean": components["score"].mean().item(),
            "residual_abs_mean": residual.abs().mean().item(),
            "residual_abs_max": residual.abs().max().item(),
            "residual_unit_abs_mean": residual_unit.abs().mean().item(),
            "residual_saturation_fraction": (
                residual_unit.abs() >= 0.95 * scale
            ).float().mean().item(),
            "treatment_probability_mean": torch.sigmoid(
                components["treatment_logit"]
            ).mean().item(),
        }

    @torch.no_grad()
    def predict(self, data):
        self.eval()
        return self.forward(data).detach().cpu().numpy()

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Linear

from .Model import Model


class CTSNetResidualDualBranchPredictor(Model):
    """Baseline-preserving CTS scorer over the official Triple Sketch.

    The official three-layer LinkPredictor is retained as ``sf_lin_layer`` +
    ``lin`` so its checkpoint can be loaded without key translation.  CTS
    branches add only a bounded logit correction.  Both correction heads are
    zero-initialized, therefore a newly constructed CTS scorer is exactly the
    loaded official scorer regardless of the gate value.
    """

    def __init__(
        self,
        ent_tot,
        rel_tot,
        max_hash_hops=1,
        hidden_dim=128,
        branch_dim=64,
        residual_scale=0.1,
        freeze_baseline=True,
        feature_mode="triple",
    ):
        super(CTSNetResidualDualBranchPredictor, self).__init__(
            ent_tot, rel_tot
        )
        if max_hash_hops < 1:
            raise ValueError("max_hash_hops must be at least 1")
        if residual_scale < 0.0:
            raise ValueError("residual_scale cannot be negative")

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
        self.register_buffer(
            "_residual_scale", torch.tensor(float(residual_scale))
        )
        self.baseline_frozen = False

        # Names and shapes intentionally match the official LinkPredictor.
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

        # Residual branches use per-triple normalization. Their scores cannot
        # depend on the composition of a training or evaluator batch.
        self.invariant_encoder = self._make_branch(hidden_dim, branch_dim)
        self.bias_encoder = self._make_branch(hidden_dim, branch_dim)
        self.invariant_delta_head = Linear(branch_dim, 1)
        self.bias_delta_head = Linear(branch_dim, 1)
        self.treatment_head = Linear(branch_dim, 1)
        self.gate_head = Linear(hidden_dim, 1)

        nn.init.zeros_(self.invariant_delta_head.weight)
        nn.init.zeros_(self.invariant_delta_head.bias)
        nn.init.zeros_(self.bias_delta_head.weight)
        nn.init.zeros_(self.bias_delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

        self.freeze_baseline(freeze_baseline)

    @staticmethod
    def _make_branch(hidden_dim, branch_dim):
        return nn.Sequential(OrderedDict([
            ("dense1", Linear(hidden_dim, hidden_dim)),
            ("norm1", nn.LayerNorm(hidden_dim)),
            ("relu1", nn.ReLU()),
            ("dense2", Linear(hidden_dim, branch_dim)),
            ("norm2", nn.LayerNorm(branch_dim)),
            ("relu2", nn.ReLU()),
        ]))

    def freeze_baseline(self, frozen=True):
        self.baseline_frozen = bool(frozen)
        for module in (self.sf_lin_layer, self.lin):
            for parameter in module.parameters():
                parameter.requires_grad = not self.baseline_frozen
        if self.baseline_frozen:
            self.sf_lin_layer.eval()
            self.lin.eval()
        return self

    def train(self, mode=True):
        super(CTSNetResidualDualBranchPredictor, self).train(mode)
        if self.baseline_frozen:
            self.sf_lin_layer.eval()
            self.lin.eval()
        return self

    def load_baseline_checkpoint(self, path):
        state = torch.load(path, map_location="cpu")
        baseline_keys = set(self.sf_lin_layer.state_dict().keys())
        baseline_keys = {"sf_lin_layer." + key for key in baseline_keys}
        baseline_keys.update({"lin.weight", "lin.bias"})
        baseline_keys.update({"zero_const", "pi_const"})
        missing = sorted(baseline_keys.difference(state))
        unexpected = sorted(set(state).difference(baseline_keys))
        if missing or unexpected:
            raise ValueError(
                "Checkpoint is not an official compatible LinkPredictor state; "
                "missing={}, unexpected={}".format(missing, unexpected)
            )
        load_result = self.load_state_dict(state, strict=False)
        loaded_missing = set(load_result.missing_keys)
        if loaded_missing.intersection(baseline_keys):
            raise RuntimeError("Failed to load all official baseline parameters")
        self.freeze_baseline(self.baseline_frozen)
        return self

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

    def _baseline_forward(self, feature):
        if self.baseline_frozen:
            self.sf_lin_layer.eval()
            with torch.no_grad():
                z_shared = self.sf_lin_layer(feature)
                base_logit = self.lin(z_shared)
            return z_shared.detach(), base_logit.detach()
        z_shared = self.sf_lin_layer(feature)
        return z_shared, self.lin(z_shared)

    def forward_components(self, data):
        feature = self._get_subgraph_feature(data)
        z_shared, base_logit = self._baseline_forward(feature)
        z_inv = self.invariant_encoder(z_shared)
        z_bias = self.bias_encoder(z_shared)

        inv_delta = self.invariant_delta_head(z_inv)
        bias_delta = self.bias_delta_head(z_bias)
        gate = torch.sigmoid(self.gate_head(z_shared))
        mixed_delta = gate * inv_delta + (1.0 - gate) * bias_delta
        bounded_delta = self._residual_scale * torch.tanh(mixed_delta)
        final_logit = base_logit + bounded_delta

        return {
            "score": torch.sigmoid(final_logit),
            "base_score": torch.sigmoid(base_logit),
            "base_logit": base_logit,
            "residual_delta": bounded_delta,
            "s_inv": torch.sigmoid(inv_delta),
            "s_bias": torch.sigmoid(bias_delta),
            "treatment_logit": self.treatment_head(z_bias),
            "gate": gate,
            "z_shared": z_shared,
            "z_inv": z_inv,
            "z_bias": z_bias,
        }

    def forward(self, data):
        return self.forward_components(data)["score"]

    @torch.no_grad()
    def diagnostics(self, data):
        components = self.forward_components(data)
        residual = components["residual_delta"]
        return {
            "gate_mean": components["gate"].mean().item(),
            "gate_min": components["gate"].min().item(),
            "gate_max": components["gate"].max().item(),
            "base_score_mean": components["base_score"].mean().item(),
            "s_inv_mean": components["s_inv"].mean().item(),
            "s_bias_mean": components["s_bias"].mean().item(),
            "s_final_mean": components["score"].mean().item(),
            "residual_abs_mean": residual.abs().mean().item(),
            "residual_abs_max": residual.abs().max().item(),
            "treatment_probability_mean": torch.sigmoid(
                components["treatment_logit"]
            ).mean().item(),
        }

    @torch.no_grad()
    def predict(self, data):
        self.eval()
        return self.forward(data).detach().cpu().numpy()

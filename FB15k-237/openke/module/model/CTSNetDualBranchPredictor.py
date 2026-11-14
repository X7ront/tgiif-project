from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import Linear

from .Model import Model


class CTSNetDualBranchPredictor(Model):
    """Stage-1 CTS-Net scorer over triple or ELPH pairwise sketches.

    The model intentionally keeps the same public contract as LinkPredictor:
    ``forward(data)`` returns one score in ``[0, 1]`` for every candidate
    triple.  This allows the official NegativeSampling, Trainer and Tester to
    be reused without modification.

    ``forward_components`` exposes the branch representations and scores that
    later CTS-Net stages need for treatment and counterfactual losses.  Stage 1
    does not apply any of those auxiliary objectives.
    """

    def __init__(
        self,
        ent_tot,
        rel_tot,
        max_hash_hops=1,
        hidden_dim=128,
        branch_dim=64,
        normalization="batch",
        feature_mode="triple",
    ):
        super(CTSNetDualBranchPredictor, self).__init__(ent_tot, rel_tot)

        if max_hash_hops < 1:
            raise ValueError("max_hash_hops must be at least 1")

        self.max_hash_hops = max_hash_hops
        if feature_mode not in ("triple", "pairwise"):
            raise ValueError("feature_mode must be 'triple' or 'pairwise'")
        self.feature_mode = feature_mode
        self.dim = (
            max_hash_hops * (max_hash_hops + 2)
            if feature_mode == "pairwise"
            else max_hash_hops * (max_hash_hops ** 2 + 3)
        )
        self.hidden_dim = hidden_dim
        self.branch_dim = branch_dim
        if normalization not in ("batch", "layer"):
            raise ValueError("normalization must be 'batch' or 'layer'")
        self.normalization = normalization

        self.shared_encoder = nn.Sequential(OrderedDict([
            ("dense1", Linear(self.dim, hidden_dim)),
            ("norm1", self._make_norm(hidden_dim)),
            ("relu1", nn.ReLU()),
            ("dense2", Linear(hidden_dim, hidden_dim)),
            ("norm2", self._make_norm(hidden_dim)),
            ("relu2", nn.ReLU()),
        ]))

        self.invariant_encoder = self._make_branch(hidden_dim, branch_dim)
        self.bias_encoder = self._make_branch(hidden_dim, branch_dim)

        self.invariant_head = Linear(branch_dim, 1)
        self.bias_head = Linear(branch_dim, 1)
        self.treatment_head = Linear(branch_dim, 1)
        self.gate_head = Linear(hidden_dim, 1)

    def _make_norm(self, dimension):
        if self.normalization == "layer":
            return nn.LayerNorm(dimension)
        return nn.BatchNorm1d(dimension)

    def _make_branch(self, hidden_dim, branch_dim):
        return nn.Sequential(OrderedDict([
            ("dense1", Linear(hidden_dim, hidden_dim)),
            ("norm1", self._make_norm(hidden_dim)),
            ("relu1", nn.ReLU()),
            ("dense2", Linear(hidden_dim, branch_dim)),
            ("norm2", self._make_norm(branch_dim)),
            ("relu2", nn.ReLU()),
        ]))

    def _get_subgraph_feature(self, data):
        if "subgraph_feature" not in data:
            raise KeyError("data must contain 'subgraph_feature'")

        subgraph_feature = data["subgraph_feature"]
        if subgraph_feature.ndim != 2:
            raise ValueError(
                "subgraph_feature must have shape [num_triples, feature_dim]"
            )
        if subgraph_feature.shape[1] != self.dim:
            raise ValueError(
                "Expected subgraph_feature dimension {}, got {}".format(
                    self.dim, subgraph_feature.shape[1]
                )
            )

        parameter = next(self.parameters())
        return subgraph_feature.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def forward_components(self, data):
        """Return the final score and all tensors needed by later CTS stages."""
        subgraph_feature = self._get_subgraph_feature(data)
        z_shared = self.shared_encoder(subgraph_feature)

        z_inv = self.invariant_encoder(z_shared)
        z_bias = self.bias_encoder(z_shared)

        s_inv = torch.sigmoid(self.invariant_head(z_inv))
        s_bias = torch.sigmoid(self.bias_head(z_bias))
        treatment_logit = self.treatment_head(z_bias)
        gate = torch.sigmoid(self.gate_head(z_shared))
        final_score = gate * s_inv + (1.0 - gate) * s_bias

        return {
            "score": final_score,
            "s_inv": s_inv,
            "s_bias": s_bias,
            "treatment_logit": treatment_logit,
            "gate": gate,
            "z_shared": z_shared,
            "z_inv": z_inv,
            "z_bias": z_bias,
        }

    def forward(self, data):
        return self.forward_components(data)["score"]

    @torch.no_grad()
    def calibrate_batch_norm(self, data_loader, num_batches=None):
        """Recompute running statistics from rank-batch Triple Sketches.

        Parameters are not updated.  This is only meaningful for legacy
        BatchNorm scorers; LayerNorm scorers return zero immediately.
        """
        batch_norms = [
            module
            for module in self.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        if not batch_norms:
            return 0
        if num_batches is not None and num_batches < 1:
            raise ValueError("num_batches must be at least 1 or None")

        was_training = self.training
        momentums = [module.momentum for module in batch_norms]
        for module in batch_norms:
            module.reset_running_stats()
            module.momentum = None

        self.train()
        processed = 0
        try:
            for batch in data_loader:
                self({"subgraph_feature": batch["subgraph_feature"]})
                processed += 1
                if num_batches is not None and processed >= num_batches:
                    break
        finally:
            for module, momentum in zip(batch_norms, momentums):
                module.momentum = momentum
            self.train(was_training)
        return processed

    @torch.no_grad()
    def diagnostics(self, data):
        """Return compact branch statistics without changing the score path."""
        components = self.forward_components(data)
        return {
            "gate_mean": components["gate"].mean().item(),
            "gate_min": components["gate"].min().item(),
            "gate_max": components["gate"].max().item(),
            "s_inv_mean": components["s_inv"].mean().item(),
            "s_bias_mean": components["s_bias"].mean().item(),
            "s_final_mean": components["score"].mean().item(),
            "treatment_probability_mean": torch.sigmoid(
                components["treatment_logit"]
            ).mean().item(),
        }

    @torch.no_grad()
    def predict(self, data):
        self.eval()
        score = self.forward(data)
        return score.detach().cpu().numpy()

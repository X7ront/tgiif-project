import torch
import torch.nn.functional as F

from .Strategy import Strategy


class CTSNegativeSampling(Strategy):
    """Official rank loss plus CTS-Net pair-level auxiliary objectives."""

    def __init__(
        self,
        model,
        loss,
        batch_size,
        lambda_cf=0.0,
        lambda_treat=0.0,
        lambda_orth=0.0,
        lambda_gate=0.0,
        lambda_residual_rank=0.0,
        lambda_invariant_rank=0.0,
        lambda_bias_rank=0.0,
        lambda_residual_preserve=0.0,
        residual_hard_negatives=0,
        residual_near_tie_negatives=0,
        residual_min_treatment_gap=0.0,
        residual_exclude_exact_ties=False,
        center_residual=False,
        preserve_base_ties=False,
        break_base_ties=False,
        floor_residual_at_baseline_min=False,
        isolate_base_rank=False,
        baseline_rank_score_space="probability",
        warmup_start_epoch=5,
        warmup_epochs=10,
        residual_start_epoch=0,
        residual_warmup_epochs=0,
        residual_loss_schedule="fusion",
        treatment_supervision="both",
        shuffle_treatment_labels=False,
        regul_rate=0.0,
        l3_regul_rate=0.0,
        lambda_invariant_variance=0.0,
        lambda_bias_variance=0.0,
        representation_std_floor=0.5,
        gate_regularizer="sample-kl",
        gate_target=0.5,
        lambda_gate_variance=0.0,
        gate_std_floor=0.05,
        treatment_label_smoothing=0.0,
        treatment_logit_temperature=1.0,
        lambda_bias_counterfactual=0.0,
        bias_counterfactual_margin=0.1,
        lambda_branch_balance=0.0,
        branch_balance_target=0.5,
        residual_tie_band=0.0,
        lambda_delta_magnitude=0.0,
        cf_distance_temperature=0.0,
        cf_objective="legacy",
        cf_representation_temperature=1.0,
        lambda_cf_variance=0.0,
        lambda_cf_covariance=0.0,
        cf_std_floor=0.1,
    ):
        super(CTSNegativeSampling, self).__init__()
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        self.lambda_cf = float(lambda_cf)
        self.lambda_treat = float(lambda_treat)
        self.lambda_orth = float(lambda_orth)
        self.lambda_gate = float(lambda_gate)
        self.lambda_residual_rank = float(lambda_residual_rank)
        self.lambda_invariant_rank = float(lambda_invariant_rank)
        self.lambda_bias_rank = float(lambda_bias_rank)
        self.lambda_residual_preserve = float(lambda_residual_preserve)
        self.lambda_invariant_variance = float(lambda_invariant_variance)
        self.lambda_bias_variance = float(lambda_bias_variance)
        self.representation_std_floor = float(representation_std_floor)
        self.gate_regularizer = str(gate_regularizer)
        self.gate_target = float(gate_target)
        self.lambda_gate_variance = float(lambda_gate_variance)
        self.gate_std_floor = float(gate_std_floor)
        self.treatment_label_smoothing = float(
            treatment_label_smoothing
        )
        self.treatment_logit_temperature = float(
            treatment_logit_temperature
        )
        self.lambda_bias_counterfactual = float(
            lambda_bias_counterfactual
        )
        self.bias_counterfactual_margin = float(
            bias_counterfactual_margin
        )
        self.lambda_branch_balance = float(lambda_branch_balance)
        self.branch_balance_target = float(branch_balance_target)
        self.residual_tie_band = float(residual_tie_band)
        self.lambda_delta_magnitude = float(lambda_delta_magnitude)
        self.cf_distance_temperature = float(cf_distance_temperature)
        self.cf_objective = str(cf_objective)
        self.cf_representation_temperature = float(
            cf_representation_temperature
        )
        self.lambda_cf_variance = float(lambda_cf_variance)
        self.lambda_cf_covariance = float(lambda_cf_covariance)
        self.cf_std_floor = float(cf_std_floor)
        if self.representation_std_floor < 0.0:
            raise ValueError("representation_std_floor cannot be negative")
        if self.gate_regularizer not in ("sample-kl", "batch-balance"):
            raise ValueError(
                "gate_regularizer must be 'sample-kl' or 'batch-balance'"
            )
        if not 0.0 <= self.gate_target <= 1.0:
            raise ValueError("gate_target must be in [0, 1]")
        if self.gate_std_floor < 0.0:
            raise ValueError("gate_std_floor cannot be negative")
        if not 0.0 <= self.treatment_label_smoothing < 1.0:
            raise ValueError(
                "treatment_label_smoothing must be in [0, 1)"
            )
        if self.treatment_logit_temperature <= 0.0:
            raise ValueError("treatment_logit_temperature must be positive")
        if self.bias_counterfactual_margin < 0.0:
            raise ValueError("bias_counterfactual_margin cannot be negative")
        if not 0.0 <= self.branch_balance_target <= 1.0:
            raise ValueError("branch_balance_target must be in [0, 1]")
        if self.residual_tie_band < 0.0:
            raise ValueError("residual_tie_band cannot be negative")
        if self.lambda_delta_magnitude < 0.0:
            raise ValueError("lambda_delta_magnitude cannot be negative")
        if self.cf_distance_temperature < 0.0:
            raise ValueError("cf_distance_temperature cannot be negative")
        if self.cf_objective not in ("legacy", "bounded-vicreg"):
            raise ValueError(
                "cf_objective must be 'legacy' or 'bounded-vicreg'"
            )
        if self.cf_representation_temperature <= 0.0:
            raise ValueError(
                "cf_representation_temperature must be positive"
            )
        if self.lambda_cf_variance < 0.0:
            raise ValueError("lambda_cf_variance cannot be negative")
        if self.lambda_cf_covariance < 0.0:
            raise ValueError("lambda_cf_covariance cannot be negative")
        if self.cf_std_floor < 0.0 or self.cf_std_floor > 1.0:
            raise ValueError("cf_std_floor must be in [0, 1]")
        self.residual_hard_negatives = int(residual_hard_negatives)
        if self.residual_hard_negatives < 0:
            raise ValueError("residual_hard_negatives cannot be negative")
        self.residual_near_tie_negatives = int(
            residual_near_tie_negatives
        )
        if self.residual_near_tie_negatives < 0:
            raise ValueError("residual_near_tie_negatives cannot be negative")
        self.residual_min_treatment_gap = float(
            residual_min_treatment_gap
        )
        if self.residual_min_treatment_gap < 0.0:
            raise ValueError("residual_min_treatment_gap cannot be negative")
        self.residual_exclude_exact_ties = bool(
            residual_exclude_exact_ties
        )
        if self.residual_hard_negatives and self.residual_near_tie_negatives:
            raise ValueError(
                "hard-negative and near-tie mining are mutually exclusive"
            )
        self.center_residual = bool(center_residual)
        self.preserve_base_ties = bool(preserve_base_ties)
        self.break_base_ties = bool(break_base_ties)
        self.floor_residual_at_baseline_min = bool(
            floor_residual_at_baseline_min
        )
        self.isolate_base_rank = bool(isolate_base_rank)
        self.baseline_rank_score_space = str(
            baseline_rank_score_space
        ).lower()
        if self.baseline_rank_score_space not in ("probability", "logit"):
            raise ValueError(
                "baseline_rank_score_space must be 'probability' or 'logit'"
            )
        self.warmup_start_epoch = int(warmup_start_epoch)
        self.warmup_epochs = int(warmup_epochs)
        self.residual_start_epoch = int(residual_start_epoch)
        self.residual_warmup_epochs = int(residual_warmup_epochs)
        self.residual_loss_schedule = str(residual_loss_schedule)
        if self.residual_loss_schedule not in ("fusion", "auxiliary"):
            raise ValueError(
                "residual_loss_schedule must be 'fusion' or 'auxiliary'"
            )
        self.treatment_supervision = str(treatment_supervision)
        if self.treatment_supervision not in ("both", "factual"):
            raise ValueError(
                "treatment_supervision must be 'both' or 'factual'"
            )
        self.shuffle_treatment_labels = bool(shuffle_treatment_labels)
        self.regul_rate = float(regul_rate)
        self.l3_regul_rate = float(l3_regul_rate)
        self.epoch = 0
        self.last_metrics = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def auxiliary_weight(self):
        if self.epoch < self.warmup_start_epoch:
            return 0.0
        if self.warmup_epochs <= 0:
            return 1.0
        progress = self.epoch - self.warmup_start_epoch + 1
        return min(1.0, max(0.0, progress / float(self.warmup_epochs)))

    def residual_weight(self):
        if self.epoch < self.residual_start_epoch:
            return 0.0
        if self.residual_warmup_epochs <= 0:
            return 1.0
        progress = self.epoch - self.residual_start_epoch + 1
        return min(
            1.0,
            max(0.0, progress / float(self.residual_warmup_epochs)),
        )

    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size]
        return positive_score.view(-1, self.batch_size).permute(1, 0)

    def _get_negative_score(self, score):
        negative_score = score[self.batch_size:]
        return negative_score.view(-1, self.batch_size).permute(1, 0)

    def _zero(self, reference):
        return reference.new_zeros(())

    @staticmethod
    def _mean_within_row_ties(values, scores):
        if values.shape != scores.shape or values.ndim != 2:
            raise ValueError("values and scores must have the same 2D shape")
        flat_values = values.reshape(-1)
        flat_scores = scores.detach().reshape(-1)
        unique_scores, score_inverse = torch.unique(
            flat_scores, sorted=False, return_inverse=True
        )
        row_ids = torch.arange(
            values.shape[0], device=values.device, dtype=torch.long
        ).repeat_interleave(values.shape[1])
        composite = row_ids * unique_scores.numel() + score_inverse
        unique_groups, inverse = torch.unique(
            composite, sorted=False, return_inverse=True
        )
        sums = values.new_zeros(unique_groups.numel())
        counts = values.new_zeros(unique_groups.numel())
        sums.scatter_add_(0, inverse, flat_values)
        counts.scatter_add_(0, inverse, torch.ones_like(flat_values))
        return (sums / counts.clamp_min(1.0))[inverse].view_as(values)

    def _residual_losses(
        self,
        components,
        reference,
        fusion_multiplier=1.0,
        residual_key="residual_unit_delta",
        residual_raw_key="residual_raw",
    ):
        required = {residual_key}
        if not required.issubset(components) or not (
            "residual_anchor" in components or "base_logit" in components
        ):
            missing = sorted(required.difference(components))
            if "residual_anchor" not in components and "base_logit" not in components:
                missing.append("residual_anchor or base_logit")
            raise KeyError(
                "baseline-aware residual ranking requires components {}".format(
                    missing
                )
            )

        residual_anchor = components.get(
            "residual_anchor", components.get("base_logit")
        )
        base_positive = self._get_positive_score(
            residual_anchor
        ).detach()
        base_negative = self._get_negative_score(
            residual_anchor
        ).detach()
        full_treatment_gap = None
        full_treatment_eligible = None
        if "treatment_feature" in components:
            treatment_positive = self._get_positive_score(
                components["treatment_feature"]
            ).detach()
            treatment_negative = self._get_negative_score(
                components["treatment_feature"]
            ).detach()
            full_treatment_gap = (
                treatment_negative - treatment_positive
            ).abs()
            full_treatment_eligible = (
                full_treatment_gap >= self.residual_min_treatment_gap
            )
        elif self.residual_min_treatment_gap > 0.0:
            raise KeyError(
                "residual treatment-gap mining requires treatment_feature"
            )
        prepared_residual = None
        if (
            (self.center_residual or self.preserve_base_ties)
            and not self.break_base_ties
            and residual_raw_key in components
        ):
            raw_positive = self._get_positive_score(
                components[residual_raw_key]
            )
            raw_negative = self._get_negative_score(
                components[residual_raw_key]
            )
            grouped_raw = torch.cat((raw_positive, raw_negative), dim=1)
            grouped_base = torch.cat((base_positive, base_negative), dim=1)
            if self.preserve_base_ties:
                grouped_raw = self._mean_within_row_ties(
                    grouped_raw, grouped_base
                )
            if self.center_residual:
                grouped_raw = grouped_raw - grouped_raw.mean(
                    dim=1, keepdim=True
                )
            calibrate = getattr(
                self.model, "_calibrate_residual_raw", None
            )
            prepared_residual = (
                calibrate(grouped_raw, dim=1)
                if calibrate is not None
                else grouped_raw
            )
            scale = getattr(self.model, "_residual_scale", 1.0)
            grouped_residual = (
                float(fusion_multiplier)
                * scale
                * torch.tanh(prepared_residual)
            )
            residual_positive = grouped_residual[:, :1]
            residual_negative = grouped_residual[:, 1:]
        else:
            residual_positive = self._get_positive_score(
                components[residual_key]
            )
            residual_negative = self._get_negative_score(
                components[residual_key]
            )
            residual_positive = float(fusion_multiplier) * residual_positive
            residual_negative = float(fusion_multiplier) * residual_negative
            grouped_residual = torch.cat(
                (residual_positive, residual_negative), dim=1
            )

        full_base_margin = base_negative - base_positive
        eligibility_filter_enabled = (
            self.residual_min_treatment_gap > 0.0
            or self.residual_exclude_exact_ties
        )
        full_residual_eligible = torch.ones_like(
            full_base_margin, dtype=torch.bool
        )
        if self.residual_min_treatment_gap > 0.0:
            full_residual_eligible = (
                full_residual_eligible & full_treatment_eligible
            )
        if self.residual_exclude_exact_ties:
            full_residual_eligible = (
                full_residual_eligible & (full_base_margin != 0.0)
            )
        full_candidate_positive = base_positive + residual_positive
        full_candidate_negative = base_negative + residual_negative
        if self.floor_residual_at_baseline_min:
            grouped_base = torch.cat(
                (base_positive, base_negative), dim=1
            )
            grouped_candidate = torch.cat(
                (full_candidate_positive, full_candidate_negative), dim=1
            )
            baseline_limit = grouped_base.min(dim=1, keepdim=True).values
            grouped_candidate = torch.where(
                grouped_base == baseline_limit,
                grouped_base,
                torch.where(
                    grouped_candidate < baseline_limit,
                    grouped_base,
                    grouped_candidate,
                ),
            )
            full_candidate_positive = grouped_candidate[:, :1]
            full_candidate_negative = grouped_candidate[:, 1:]
        full_candidate_margin = (
            full_candidate_negative - full_candidate_positive
        )
        negative_count = base_negative.shape[1]
        hard_count = min(
            self.residual_hard_negatives, base_negative.shape[1]
        )
        near_tie_count = min(
            self.residual_near_tie_negatives, base_negative.shape[1]
        )
        selected_base_margin = full_base_margin
        selected_treatment_gap = full_treatment_gap
        selected_treatment_eligible = full_treatment_eligible
        selected_residual_eligible = full_residual_eligible
        if near_tie_count > 0 and near_tie_count < base_negative.shape[1]:
            # Evaluation asks the residual to order candidates whose baseline
            # scores are equal or nearly equal. Mine by absolute baseline gap
            # so exact ties are selected before unrelated random negatives.
            selection_distance = full_base_margin.abs()
            if eligibility_filter_enabled:
                selection_distance = selection_distance.masked_fill(
                    ~full_residual_eligible, float("inf")
                )
            selected_indices = torch.topk(
                selection_distance,
                near_tie_count,
                dim=1,
                largest=False,
            ).indices
            candidate_negative = torch.gather(
                full_candidate_negative, 1, selected_indices
            )
            selected_base_margin = torch.gather(
                full_base_margin, 1, selected_indices
            )
            if full_treatment_gap is not None:
                selected_treatment_gap = torch.gather(
                    full_treatment_gap, 1, selected_indices
                )
                selected_treatment_eligible = torch.gather(
                    full_treatment_eligible, 1, selected_indices
                )
            selected_residual_eligible = torch.gather(
                full_residual_eligible, 1, selected_indices
            )
        elif hard_count > 0 and hard_count < base_negative.shape[1]:
            hard_distance = full_base_margin
            if eligibility_filter_enabled:
                hard_distance = hard_distance.masked_fill(
                    ~full_residual_eligible, float("inf")
                )
            hard_indices = torch.topk(
                hard_distance, hard_count, dim=1, largest=False
            ).indices
            candidate_negative = torch.gather(
                full_candidate_negative, 1, hard_indices
            )
            selected_base_margin = torch.gather(
                full_base_margin, 1, hard_indices
            )
            if full_treatment_gap is not None:
                selected_treatment_gap = torch.gather(
                    full_treatment_gap, 1, hard_indices
                )
                selected_treatment_eligible = torch.gather(
                    full_treatment_eligible, 1, hard_indices
                )
            selected_residual_eligible = torch.gather(
                full_residual_eligible, 1, hard_indices
            )
        else:
            candidate_negative = full_candidate_negative

        candidate_positive = full_candidate_positive
        selected_candidate_margin = candidate_negative - candidate_positive
        rank_terms = F.softplus(candidate_positive - candidate_negative)
        if self.residual_tie_band > 0.0:
            # Focus residual learning on baseline-near ties. Far-separated
            # negatives are already ordered by the official scorer; pushing
            # every sampled negative above the positive collapses Hits@1.
            band = self.residual_tie_band
            tie_weight = torch.exp(-selected_base_margin.abs() / band)
            if eligibility_filter_enabled:
                tie_weight = tie_weight * selected_residual_eligible.to(
                    tie_weight.dtype
                )
            residual_rank_loss = (rank_terms * tie_weight).sum() / (
                tie_weight.sum().clamp_min(1e-6)
            )
        elif eligibility_filter_enabled:
            selection_weight = selected_residual_eligible.to(rank_terms.dtype)
            residual_rank_loss = (rank_terms * selection_weight).sum() / (
                selection_weight.sum().clamp_min(1e-6)
            )
        else:
            residual_rank_loss = rank_terms.mean()

        # Preserve the official optimistic result for both strict winners and
        # exact baseline ties. A tie may be broken in the positive direction,
        # but the residual must not push its target behind a tied negative.
        base_correct = selected_base_margin >= 0.0
        if eligibility_filter_enabled:
            base_correct = base_correct & selected_residual_eligible
        if bool(base_correct.any()):
            residual_preserve_loss = F.relu(
                selected_base_margin[base_correct]
                - selected_candidate_margin[base_correct]
            ).mean()
        else:
            residual_preserve_loss = self._zero(reference)

        diagnostics = {
            "base_margin_mean": float(
                full_base_margin.detach().mean().item()
            ),
            "candidate_margin_mean": float(
                selected_candidate_margin.detach().mean().item()
            ),
            "residual_margin_improved_ratio": float(
                (selected_candidate_margin > selected_base_margin)
                .float()
                .detach()
                .mean()
                .item()
            ),
            "hard_negative_fraction": float(
                (hard_count if hard_count > 0 else negative_count)
                / float(negative_count)
            ),
            "near_tie_negative_fraction": float(
                (near_tie_count if near_tie_count > 0 else negative_count)
                / float(negative_count)
            ),
            "selected_exact_tie_fraction": float(
                (selected_base_margin == 0.0)
                .float()
                .detach()
                .mean()
                .item()
            ),
            "selected_abs_base_margin_mean": float(
                selected_base_margin.abs().detach().mean().item()
            ),
            "selected_treatment_changed_fraction": (
                float(
                    (selected_treatment_gap > 0.0)
                    .float()
                    .detach()
                    .mean()
                    .item()
                )
                if selected_treatment_gap is not None
                else 0.0
            ),
            "selected_treatment_gap_mean": (
                float(selected_treatment_gap.detach().mean().item())
                if selected_treatment_gap is not None
                else 0.0
            ),
            "selected_treatment_eligible_fraction": (
                float(
                    selected_treatment_eligible.float()
                    .detach()
                    .mean()
                    .item()
                )
                if selected_treatment_eligible is not None
                else 0.0
            ),
            "selected_residual_eligible_fraction": float(
                selected_residual_eligible.float()
                .detach()
                .mean()
                .item()
            ),
            "residual_unit_abs_mean": float(
                grouped_residual.abs().detach().mean().item()
            ),
            "residual_saturation_fraction": float(
                (
                    grouped_residual.abs()
                    >= 0.95
                    * float(fusion_multiplier)
                    * float(getattr(self.model, "_residual_scale", 1.0))
                )
                .float()
                .detach()
                .mean()
                .item()
            ) if float(fusion_multiplier) > 0.0 else 0.0,
            "residual_prepared_abs_mean": (
                float(prepared_residual.abs().detach().mean().item())
                if prepared_residual is not None
                else 0.0
            ),
        }
        return residual_rank_loss, residual_preserve_loss, diagnostics

    def _pair_auxiliary_enabled(self):
        return any(
            weight != 0.0
            for weight in (
                self.lambda_cf,
                self.lambda_cf_variance,
                self.lambda_cf_covariance,
                self.lambda_treat,
                self.lambda_orth,
                self.lambda_bias_counterfactual,
            )
        )

    def _validate_enriched_batch(self, data):
        required = {
            "cf_subgraph_feature",
            "factual_treatment",
            "cf_treatment",
            "cf_valid_mask",
            "cf_target_index",
        }
        if not required.issubset(data):
            missing = sorted(required.difference(data))
            raise KeyError(
                "CTS auxiliary losses require enriched batches; missing {}".format(
                    missing
                )
            )

    @staticmethod
    def _deduplicate_counterfactuals(data, valid_mask):
        target_index = data["cf_target_index"][valid_mask].view(-1)
        sorted_target, order = torch.sort(target_index)
        is_new = torch.ones_like(sorted_target, dtype=torch.bool)
        is_new[1:] = sorted_target[1:] != sorted_target[:-1]
        group_sorted = torch.cumsum(is_new.to(torch.long), dim=0) - 1
        inverse = torch.empty_like(group_sorted)
        inverse[order] = group_sorted
        representatives = order[is_new]
        valid_features = data["cf_subgraph_feature"][valid_mask]
        return valid_features[representatives], inverse, representatives

    @staticmethod
    def _slice_components(components, start, end=None):
        return {
            key: value[start:end]
            for key, value in components.items()
            if torch.is_tensor(value)
        }

    def _counterfactual_invariance_loss(
        self, factual_invariant, counterfactual_invariant, match_distance=None
    ):
        per_pair = F.smooth_l1_loss(
            factual_invariant,
            counterfactual_invariant,
            reduction="none",
        ).reshape(factual_invariant.shape[0], -1).mean(dim=1)
        if self.cf_distance_temperature <= 0.0 or match_distance is None:
            return per_pair.mean(), per_pair.new_ones(()).detach()
        distance = match_distance.reshape(-1).to(
            device=per_pair.device, dtype=per_pair.dtype
        )
        if distance.numel() != per_pair.numel():
            raise ValueError(
                "CF match distance count must equal the valid pair count"
            )
        weights = torch.exp(
            -distance / self.cf_distance_temperature
        ).detach()
        weighted = (per_pair * weights).sum() / weights.sum().clamp_min(1e-8)
        return weighted, weights.mean()

    def _bounded_cf_representation(self, value):
        return torch.tanh(value / self.cf_representation_temperature)

    def _cf_vicreg_regularizers(self, factual, counterfactual):
        """Return bounded variance/covariance penalties for matched domains.

        The regularizer operates on a bounded view of the invariant
        representation.  It therefore cannot lower its objective merely by
        inflating the raw branch scale, while the variance term rules out the
        constant representation that makes the legacy CF loss trivially zero.
        """
        zero = factual.sum() * 0.0

        def variance_floor(value):
            if value.shape[0] < 2:
                return zero
            std = torch.sqrt(value.var(dim=0, unbiased=False) + 1e-4)
            return F.relu(value.new_tensor(self.cf_std_floor) - std).mean()

        def covariance(value):
            if value.shape[0] < 2 or value.shape[1] < 2:
                return zero
            centered = value - value.mean(dim=0, keepdim=True)
            cov = centered.transpose(0, 1).matmul(centered)
            cov = cov / float(value.shape[0] - 1)
            off_diagonal = cov - torch.diag_embed(torch.diagonal(cov))
            return off_diagonal.square().sum() / float(value.shape[1])

        variance = 0.5 * (
            variance_floor(factual) + variance_floor(counterfactual)
        )
        covariance_loss = 0.5 * (
            covariance(factual) + covariance(counterfactual)
        )
        std_mean = 0.5 * (
            torch.sqrt(factual.var(dim=0, unbiased=False) + 1e-4).mean()
            + torch.sqrt(
                counterfactual.var(dim=0, unbiased=False) + 1e-4
            ).mean()
        )
        return variance, covariance_loss, std_mean.detach()

    def _forward_components_once(self, data, warmup):
        pair_enabled = self._pair_auxiliary_enabled()
        valid_mask = None
        valid_ratio = 0.0
        inverse = None
        representatives = None
        cf_unique_components = None

        if pair_enabled:
            self._validate_enriched_batch(data)
            valid_mask = data["cf_valid_mask"].view(-1).bool()
            valid_count = int(valid_mask.sum().item())
            valid_ratio = valid_count / float(max(1, self.batch_size))

        # A zero warm-up weight must be a true rank-only pass. In particular,
        # it must not update BatchNorm running statistics with a second CF
        # domain forward.
        if warmup <= 0.0 or not pair_enabled or int(valid_mask.sum()) < 2:
            components = self.model.forward_components(data)
            return (
                components,
                cf_unique_components,
                valid_mask,
                inverse,
                representatives,
                valid_ratio,
            )

        cf_unique_features, inverse, representatives = (
            self._deduplicate_counterfactuals(data, valid_mask)
        )
        base_count = data["subgraph_feature"].shape[0]
        combined_features = torch.cat(
            (data["subgraph_feature"], cf_unique_features), dim=0
        )
        combined_data = {
            "subgraph_feature": combined_features,
            "rank_feature_count": base_count,
        }
        cf_id_keys = {"cf_batch_h", "cf_batch_t", "cf_batch_r"}
        if cf_id_keys.issubset(data):
            for name in ("h", "t", "r"):
                factual_ids = data["batch_" + name].view(-1)
                if factual_ids.numel() == 1:
                    factual_ids = factual_ids.expand(base_count)
                cf_ids = data["cf_batch_" + name][valid_mask].view(-1)
                cf_unique_ids = cf_ids[representatives]
                combined_data["batch_" + name] = torch.cat(
                    (factual_ids, cf_unique_ids), dim=0
                )
        combined = self.model.forward_components(combined_data)
        components = self._slice_components(combined, 0, base_count)
        cf_unique_components = self._slice_components(
            combined, base_count, None
        )
        return (
            components,
            cf_unique_components,
            valid_mask,
            inverse,
            representatives,
            valid_ratio,
        )

    def _auxiliary_losses(
        self,
        data,
        factual_components,
        cf_unique_components,
        valid_mask,
        inverse,
        representatives,
        rank_loss,
        warmup,
    ):
        zero = self._zero(rank_loss)
        losses = {
            "cf": zero,
            "cf_variance": zero,
            "cf_covariance": zero,
            "treat": zero,
            "orth": zero,
            "gate": zero,
            "gate_variance": zero,
            "invariant_variance": zero,
            "bias_variance": zero,
            "bias_counterfactual": zero,
            "branch_balance": zero,
        }
        diagnostics = {
            "treat_accuracy_f": 0.0,
            "treat_accuracy_cf": 0.0,
            "cf_unique_ratio": 0.0,
            "cf_effective_weight_mean": 1.0,
            "cf_representation_std_mean": 0.0,
        }
        aux_enabled = any(
            weight != 0.0
            for weight in (
                self.lambda_cf,
                self.lambda_cf_variance,
                self.lambda_cf_covariance,
                self.lambda_treat,
                self.lambda_orth,
                self.lambda_gate,
                self.lambda_gate_variance,
                self.lambda_invariant_variance,
                self.lambda_bias_variance,
                self.lambda_bias_counterfactual,
                self.lambda_branch_balance,
            )
        )
        if warmup <= 0.0 or not aux_enabled:
            return losses, diagnostics

        factual_positive = self._slice_components(
            factual_components, 0, self.batch_size
        )
        factual_gate = factual_positive["gate"].clamp(
            min=1e-6, max=1.0 - 1e-6
        )
        if self.gate_regularizer == "sample-kl":
            losses["gate"] = (
                factual_gate * torch.log(2.0 * factual_gate)
                + (1.0 - factual_gate)
                * torch.log(2.0 * (1.0 - factual_gate))
            ).mean()
        else:
            losses["gate"] = (
                factual_gate.mean() - self.gate_target
            ).square()
        gate_std = torch.sqrt(
            factual_gate.var(unbiased=False) + 1e-8
        )
        losses["gate_variance"] = F.relu(
            factual_gate.new_tensor(self.gate_std_floor) - gate_std
        )

        def variance_floor(value):
            if value.shape[0] < 2:
                return zero
            std = torch.sqrt(value.var(dim=0, unbiased=False) + 1e-8)
            return F.relu(
                value.new_tensor(self.representation_std_floor) - std
            ).mean()

        losses["invariant_variance"] = variance_floor(
            factual_positive["z_inv"]
        )
        losses["bias_variance"] = variance_floor(
            factual_positive["z_bias"]
        )
        invariant_share = zero
        if (
            "invariant_contribution" in factual_positive
            and "bias_contribution" in factual_positive
        ):
            inv_strength = factual_positive[
                "invariant_contribution"
            ].abs().mean()
            bias_strength = factual_positive[
                "bias_contribution"
            ].abs().mean()
            contribution_total = inv_strength + bias_strength + 1e-8
            invariant_share = inv_strength / contribution_total
            losses["branch_balance"] = (
                invariant_share - self.branch_balance_target
            ).square()
        diagnostics["gate_std"] = float(gate_std.detach().item())
        diagnostics["invariant_contribution_share"] = float(
            invariant_share.detach().item()
        )

        if not self._pair_auxiliary_enabled():
            return losses, diagnostics

        factual_labels = data["factual_treatment"].to(
            dtype=factual_positive["treatment_logit"].dtype
        )
        if self.shuffle_treatment_labels and factual_labels.numel() > 1:
            factual_labels = factual_labels.reshape(-1)[
                torch.randperm(factual_labels.numel(), device=factual_labels.device)
            ].view_as(factual_labels)
        smooth = self.treatment_label_smoothing
        factual_targets = factual_labels * (1.0 - smooth) + 0.5 * smooth
        factual_treat_loss = F.binary_cross_entropy_with_logits(
            factual_positive["treatment_logit"]
            / self.treatment_logit_temperature,
            factual_targets,
            reduction="mean",
        )
        diagnostics["treat_accuracy_f"] = float(
            (
                (
                    torch.sigmoid(
                        factual_positive["treatment_logit"]
                        / self.treatment_logit_temperature
                    )
                    >= 0.5
                )
                == (factual_labels >= 0.5)
            ).float().mean().detach().item()
        )

        factual_cosine = F.cosine_similarity(
            factual_positive["z_inv"],
            factual_positive["z_bias"],
            dim=-1,
            eps=1e-8,
        )
        factual_orth_loss = factual_cosine.square().mean()

        if cf_unique_components is None:
            losses["treat"] = factual_treat_loss
            losses["orth"] = factual_orth_loss
            return losses, diagnostics

        factual_valid = {
            key: value[valid_mask]
            for key, value in factual_positive.items()
        }
        cf_pair_components = {
            key: value[inverse]
            for key, value in cf_unique_components.items()
        }
        valid_match_distance = None
        if "cf_match_distance" in data:
            valid_match_distance = data["cf_match_distance"][valid_mask]
        factual_cf_representation = factual_valid["z_inv"]
        paired_cf_representation = cf_pair_components["z_inv"]
        unique_cf_representation = cf_unique_components["z_inv"]
        if self.cf_objective == "bounded-vicreg":
            factual_cf_representation = self._bounded_cf_representation(
                factual_cf_representation
            )
            paired_cf_representation = self._bounded_cf_representation(
                paired_cf_representation
            )
            unique_cf_representation = self._bounded_cf_representation(
                unique_cf_representation
            )
        losses["cf"], effective_weight = self._counterfactual_invariance_loss(
            factual_cf_representation,
            paired_cf_representation,
            valid_match_distance,
        )
        diagnostics["cf_effective_weight_mean"] = float(
            effective_weight.detach().item()
        )
        if self.cf_objective == "bounded-vicreg":
            (
                losses["cf_variance"],
                losses["cf_covariance"],
                cf_representation_std,
            ) = self._cf_vicreg_regularizers(
                factual_cf_representation,
                unique_cf_representation,
            )
            diagnostics["cf_representation_std_mean"] = float(
                cf_representation_std.item()
            )

        valid_cf_labels = data["cf_treatment"][valid_mask]
        if self.shuffle_treatment_labels and valid_cf_labels.numel() > 1:
            valid_cf_labels = valid_cf_labels.reshape(-1)[
                torch.randperm(valid_cf_labels.numel(), device=valid_cf_labels.device)
            ].view_as(valid_cf_labels)
        unique_cf_labels = valid_cf_labels[representatives].to(
            dtype=cf_unique_components["treatment_logit"].dtype
        )
        unique_cf_targets = (
            unique_cf_labels * (1.0 - smooth) + 0.5 * smooth
        )
        if (
            self.treatment_supervision == "both"
            and "treatment_feature" in cf_unique_components
        ):
            cf_treat_loss = F.binary_cross_entropy_with_logits(
                cf_unique_components["treatment_logit"]
                / self.treatment_logit_temperature,
                unique_cf_targets,
                reduction="mean",
            )
            losses["treat"] = 0.5 * (
                factual_treat_loss + cf_treat_loss
            )
        else:
            losses["treat"] = factual_treat_loss
        diagnostics["treat_accuracy_cf"] = float(
            (
                (
                    torch.sigmoid(
                        cf_unique_components["treatment_logit"]
                        / self.treatment_logit_temperature
                    )
                    >= 0.5
                )
                == (unique_cf_labels >= 0.5)
            ).float().mean().detach().item()
        )
        diagnostics["cf_unique_ratio"] = float(
            len(unique_cf_labels) / float(max(1, int(valid_mask.sum().item())))
        )
        if (
            "bias_delta" in factual_valid
            and "bias_delta" in cf_pair_components
        ):
            factual_bias = factual_valid["bias_delta"]
            counterfactual_bias = cf_pair_components["bias_delta"]
            losses["bias_counterfactual"] = F.relu(
                factual_bias.new_tensor(self.bias_counterfactual_margin)
                - (factual_bias - counterfactual_bias).abs()
            ).mean()

        losses["orth"] = factual_orth_loss
        return losses, diagnostics

    def forward(self, data):
        warmup = self.auxiliary_weight()
        residual_weight = self.residual_weight()
        residual_loss_weight = (
            warmup
            if self.residual_loss_schedule == "auxiliary"
            else residual_weight
        )
        residual_loss_multiplier = (
            1.0
            if self.residual_loss_schedule == "auxiliary"
            else residual_weight
        )
        if hasattr(self.model, "set_residual_multiplier"):
            self.model.set_residual_multiplier(residual_weight)
        (
            components,
            cf_unique_components,
            valid_mask,
            inverse,
            representatives,
            valid_ratio,
        ) = self._forward_components_once(data, warmup)
        if self.isolate_base_rank:
            rank_score = components[
                "base_logit"
                if self.baseline_rank_score_space == "logit"
                else "base_score"
            ]
        else:
            rank_score = components["score"]
        positive_score = self._get_positive_score(rank_score)
        negative_score = self._get_negative_score(rank_score)
        rank_loss = self.loss(positive_score, negative_score)

        residual_enabled = (
            self.lambda_residual_rank != 0.0
            or self.lambda_residual_preserve != 0.0
            or self.lambda_invariant_rank != 0.0
            or self.lambda_bias_rank != 0.0
        )
        if residual_enabled and residual_loss_weight > 0.0:
            (
                residual_rank_loss,
                residual_preserve_loss,
                residual_diagnostics,
            ) = self._residual_losses(
                components,
                rank_loss,
                fusion_multiplier=residual_loss_multiplier,
            )
        else:
            residual_rank_loss = self._zero(rank_loss)
            residual_preserve_loss = self._zero(rank_loss)
            residual_diagnostics = {
                "base_margin_mean": 0.0,
                "candidate_margin_mean": 0.0,
                "residual_margin_improved_ratio": 0.0,
            }

        if self.lambda_invariant_rank != 0.0 and residual_loss_weight > 0.0:
            invariant_rank_loss, _, _ = self._residual_losses(
                components,
                rank_loss,
                fusion_multiplier=residual_loss_multiplier,
                residual_key="invariant_unit_delta",
                residual_raw_key="invariant_residual_raw",
            )
        else:
            invariant_rank_loss = self._zero(rank_loss)
        if self.lambda_bias_rank != 0.0 and residual_loss_weight > 0.0:
            bias_rank_loss, _, _ = self._residual_losses(
                components,
                rank_loss,
                fusion_multiplier=residual_loss_multiplier,
                residual_key="bias_unit_delta",
                residual_raw_key="bias_residual_raw",
            )
        else:
            bias_rank_loss = self._zero(rank_loss)

        auxiliary, auxiliary_diagnostics = self._auxiliary_losses(
            data,
            components,
            cf_unique_components,
            valid_mask,
            inverse,
            representatives,
            rank_loss,
            warmup,
        )
        delta_magnitude_loss = self._zero(rank_loss)
        if self.lambda_delta_magnitude != 0.0:
            delta_terms = [
                components[key].square().mean()
                for key in ("invariant_delta_raw", "bias_delta_raw")
                if key in components
            ]
            if delta_terms:
                delta_magnitude_loss = sum(delta_terms[1:], delta_terms[0])
        weighted_cf = warmup * self.lambda_cf * auxiliary["cf"]
        weighted_cf_variance = (
            warmup
            * self.lambda_cf_variance
            * auxiliary["cf_variance"]
        )
        weighted_cf_covariance = (
            warmup
            * self.lambda_cf_covariance
            * auxiliary["cf_covariance"]
        )
        weighted_treat = warmup * self.lambda_treat * auxiliary["treat"]
        weighted_orth = warmup * self.lambda_orth * auxiliary["orth"]
        weighted_gate = warmup * self.lambda_gate * auxiliary["gate"]
        weighted_gate_variance = (
            warmup
            * self.lambda_gate_variance
            * auxiliary["gate_variance"]
        )
        weighted_invariant_variance = (
            warmup
            * self.lambda_invariant_variance
            * auxiliary["invariant_variance"]
        )
        weighted_bias_variance = (
            warmup
            * self.lambda_bias_variance
            * auxiliary["bias_variance"]
        )
        weighted_bias_counterfactual = (
            warmup
            * self.lambda_bias_counterfactual
            * auxiliary["bias_counterfactual"]
        )
        weighted_branch_balance = (
            warmup
            * self.lambda_branch_balance
            * auxiliary["branch_balance"]
        )
        weighted_delta_magnitude = (
            warmup
            * self.lambda_delta_magnitude
            * delta_magnitude_loss
        )
        weighted_residual_rank = (
            residual_loss_weight
            * self.lambda_residual_rank
            * residual_rank_loss
        )
        weighted_invariant_rank = (
            residual_loss_weight
            * self.lambda_invariant_rank
            * invariant_rank_loss
        )
        weighted_bias_rank = (
            residual_loss_weight
            * self.lambda_bias_rank
            * bias_rank_loss
        )
        weighted_residual_preserve = (
            residual_loss_weight
            * self.lambda_residual_preserve
            * residual_preserve_loss
        )
        total_loss = (
            rank_loss
            + weighted_residual_rank
            + weighted_invariant_rank
            + weighted_bias_rank
            + weighted_residual_preserve
            + weighted_cf
            + weighted_cf_variance
            + weighted_cf_covariance
            + weighted_treat
            + weighted_orth
            + weighted_gate
            + weighted_gate_variance
            + weighted_invariant_variance
            + weighted_bias_variance
            + weighted_bias_counterfactual
            + weighted_branch_balance
            + weighted_delta_magnitude
        )

        if self.regul_rate != 0.0:
            total_loss = total_loss + self.regul_rate * self.model.regularization(
                data
            )
        if self.l3_regul_rate != 0.0:
            total_loss = total_loss + self.l3_regul_rate * (
                self.model.l3_regularization()
            )

        def component_abs_mean(name):
            value = components.get(name)
            if value is None:
                return 0.0
            return float(
                value[:self.batch_size].detach().abs().mean().item()
            )

        self.last_metrics = {
            "rank_loss": float(rank_loss.detach().item()),
            "residual_rank_loss": float(residual_rank_loss.detach().item()),
            "invariant_rank_loss": float(invariant_rank_loss.detach().item()),
            "bias_rank_loss": float(bias_rank_loss.detach().item()),
            "residual_preserve_loss": float(
                residual_preserve_loss.detach().item()
            ),
            "cf_loss": float(auxiliary["cf"].detach().item()),
            "cf_variance_loss": float(
                auxiliary["cf_variance"].detach().item()
            ),
            "cf_covariance_loss": float(
                auxiliary["cf_covariance"].detach().item()
            ),
            "treat_loss": float(auxiliary["treat"].detach().item()),
            "orth_loss": float(auxiliary["orth"].detach().item()),
            "gate_loss": float(auxiliary["gate"].detach().item()),
            "gate_variance_loss": float(
                auxiliary["gate_variance"].detach().item()
            ),
            "invariant_variance_loss": float(
                auxiliary["invariant_variance"].detach().item()
            ),
            "bias_variance_loss": float(
                auxiliary["bias_variance"].detach().item()
            ),
            "bias_counterfactual_loss": float(
                auxiliary["bias_counterfactual"].detach().item()
            ),
            "branch_balance_loss": float(
                auxiliary["branch_balance"].detach().item()
            ),
            "delta_magnitude_loss": float(
                delta_magnitude_loss.detach().item()
            ),
            "weighted_cf": float(weighted_cf.detach().item()),
            "weighted_cf_variance": float(
                weighted_cf_variance.detach().item()
            ),
            "weighted_cf_covariance": float(
                weighted_cf_covariance.detach().item()
            ),
            "weighted_treat": float(weighted_treat.detach().item()),
            "weighted_orth": float(weighted_orth.detach().item()),
            "weighted_gate": float(weighted_gate.detach().item()),
            "weighted_gate_variance": float(
                weighted_gate_variance.detach().item()
            ),
            "weighted_invariant_variance": float(
                weighted_invariant_variance.detach().item()
            ),
            "weighted_bias_variance": float(
                weighted_bias_variance.detach().item()
            ),
            "weighted_bias_counterfactual": float(
                weighted_bias_counterfactual.detach().item()
            ),
            "weighted_branch_balance": float(
                weighted_branch_balance.detach().item()
            ),
            "weighted_delta_magnitude": float(
                weighted_delta_magnitude.detach().item()
            ),
            "weighted_residual_rank": float(
                weighted_residual_rank.detach().item()
            ),
            "weighted_invariant_rank": float(
                weighted_invariant_rank.detach().item()
            ),
            "weighted_bias_rank": float(weighted_bias_rank.detach().item()),
            "weighted_residual_preserve": float(
                weighted_residual_preserve.detach().item()
            ),
            "total_loss": float(total_loss.detach().item()),
            "warmup": float(warmup),
            "residual_weight": float(residual_weight),
            "gate_mean": float(
                components["gate"][:self.batch_size].detach().mean().item()
            ),
            "cf_representation_std_mean": auxiliary_diagnostics[
                "cf_representation_std_mean"
            ],
            "invariant_delta_abs_mean": component_abs_mean(
                "invariant_delta"
            ),
            "bias_delta_abs_mean": component_abs_mean("bias_delta"),
            "invariant_delta_raw_abs_mean": component_abs_mean(
                "invariant_delta_raw"
            ),
            "bias_delta_raw_abs_mean": component_abs_mean(
                "bias_delta_raw"
            ),
            "invariant_contribution_abs_mean": component_abs_mean(
                "invariant_contribution"
            ),
            "bias_contribution_abs_mean": component_abs_mean(
                "bias_contribution"
            ),
            "cf_valid_ratio": float(valid_ratio),
            **residual_diagnostics,
            **auxiliary_diagnostics,
        }
        return total_loss

    def get_last_metrics(self):
        return dict(self.last_metrics)

    def predict(self, data):
        return self.model.predict(data)

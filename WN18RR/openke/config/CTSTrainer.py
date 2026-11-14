# coding:utf-8
import json
import math
import os

import torch
import torch.optim as optim
from tqdm import tqdm

from .Trainer import Trainer


class CTSTrainer(Trainer):
    """Trainer extension that preserves the official optimization sequence.

    The original Trainer is not modified. This extension only passes enriched
    counterfactual fields, supplies the epoch for auxiliary-loss warm-up, and
    reports decomposed CTS loss diagnostics.
    """

    def __init__(
        self,
        *args,
        architecture_diagnostic_interval=0,
        architecture_diagnostic_path=None,
        history_jsonl_path=None,
        baseline_lr_multiplier=1.0,
        shared_lr_multiplier=1.0,
        invariant_lr_multiplier=1.0,
        bias_lr_multiplier=1.0,
        gate_lr_multiplier=1.0,
        max_grad_norm=0.0,
        lr_scheduler="none",
        min_lr_ratio=0.0,
        resume_training_state=None,
        **kwargs
    ):
        super(CTSTrainer, self).__init__(*args, **kwargs)
        self.history = []
        self.architecture_diagnostic_interval = int(
            architecture_diagnostic_interval
        )
        if self.architecture_diagnostic_interval < 0:
            raise ValueError("architecture_diagnostic_interval cannot be negative")
        self.architecture_diagnostic_path = architecture_diagnostic_path
        self.history_jsonl_path = history_jsonl_path
        self.lr_multipliers = {
            "baseline": float(baseline_lr_multiplier),
            "shared": float(shared_lr_multiplier),
            "invariant": float(invariant_lr_multiplier),
            "bias": float(bias_lr_multiplier),
            "gate": float(gate_lr_multiplier),
            "other": 1.0,
        }
        if any(value < 0.0 for value in self.lr_multipliers.values()):
            raise ValueError("learning-rate multipliers cannot be negative")
        self.max_grad_norm = float(max_grad_norm)
        if self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm cannot be negative")
        self.lr_scheduler_name = str(lr_scheduler)
        if self.lr_scheduler_name not in ("none", "cosine"):
            raise ValueError("lr_scheduler must be 'none' or 'cosine'")
        self.min_lr_ratio = float(min_lr_ratio)
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.scheduler = None
        self.resume_training_state = resume_training_state
        self.start_epoch = 0

    @staticmethod
    def _parameter_group_name(name):
        name = name[6:] if name.startswith("model.") else name
        if name.startswith(("sf_lin_layer.", "lin.")):
            return "baseline"
        if name.startswith("residual_shared_encoder."):
            return "shared"
        if name.startswith(("invariant_encoder.", "invariant_delta_head.")):
            return "invariant"
        if name.startswith(
            ("bias_encoder.", "bias_delta_head.", "treatment_head.")
        ):
            return "bias"
        if name.startswith("gate_head."):
            return "gate"
        return "other"

    def _optimizer_parameter_groups(self):
        grouped = {name: [] for name in self.lr_multipliers}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                grouped[self._parameter_group_name(name)].append(parameter)
        return [
            {
                "params": parameters,
                "lr": self.alpha * self.lr_multipliers[name],
                "group_name": name,
            }
            for name, parameters in grouped.items()
            if parameters
        ]

    def _build_optimizer(self):
        if self.optimizer is not None:
            return
        parameter_groups = self._optimizer_parameter_groups()
        if self.opt_method.lower() == "adagrad":
            self.optimizer = optim.Adagrad(
                parameter_groups,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method.lower() == "adadelta":
            self.optimizer = optim.Adadelta(
                parameter_groups,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method.lower() == "adam":
            self.optimizer = optim.Adam(
                parameter_groups,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                parameter_groups,
                weight_decay=self.weight_decay,
            )
        if self.lr_scheduler_name == "cosine":
            def multiplier(epoch):
                progress = min(1.0, epoch / float(max(1, self.train_times)))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return self.min_lr_ratio + (
                    1.0 - self.min_lr_ratio
                ) * cosine

            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=multiplier
            )

    def _prepare_data(self, data):
        prepared = {
            "batch_h": self.to_var(data["batch_h"], self.use_gpu),
            "batch_t": self.to_var(data["batch_t"], self.use_gpu),
            "batch_r": self.to_var(data["batch_r"], self.use_gpu),
            "batch_y": self.to_var(data["batch_y"], self.use_gpu),
            "subgraph_feature": data["subgraph_feature"],
            "mode": data["mode"],
        }
        for key in (
            "cf_subgraph_feature",
            "factual_treatment",
            "cf_treatment",
            "cf_valid_mask",
            "cf_match_distance",
            "cf_target_index",
            "cf_batch_h",
            "cf_batch_t",
            "cf_batch_r",
        ):
            if key in data:
                prepared[key] = data[key]
        return prepared

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss = self.model(self._prepare_data(data))
        loss.backward()
        if self.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
        metrics = self.model.get_last_metrics()
        metrics.update(self._gradient_metrics())
        self.optimizer.step()
        return loss.item(), metrics

    def _write_epoch_record(self, record):
        if not self.history_jsonl_path:
            return
        parent = os.path.dirname(os.path.abspath(self.history_jsonl_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.history_jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _gradient_metrics(self):
        predictor = self.model.model
        groups = {
            "baseline": ("sf_lin_layer.", "lin."),
            "invariant": (
                "invariant_encoder.",
                "invariant_head.",
                "invariant_delta_head.",
            ),
            "bias": (
                "bias_encoder.",
                "bias_head.",
                "bias_delta_head.",
                "treatment_head.",
            ),
            "gate": ("gate_head.",),
        }
        squared = {name: 0.0 for name in groups}
        for parameter_name, parameter in predictor.named_parameters():
            if parameter.grad is None:
                continue
            norm_square = float(parameter.grad.detach().square().sum().item())
            for group_name, prefixes in groups.items():
                if parameter_name.startswith(prefixes):
                    squared[group_name] += norm_square
                    break
        return {
            "{}_grad_norm".format(name): value ** 0.5
            for name, value in squared.items()
        }

    @staticmethod
    def _tensor_abs_mean(components, key):
        value = components.get(key)
        if value is None or not torch.is_tensor(value):
            return 0.0
        return float(value.detach().abs().mean().item())

    def _architecture_probe(self, data):
        """Collect a low-frequency, no-gradient contribution snapshot."""
        predictor = self.model.model
        was_training = predictor.training
        predictor.eval()
        try:
            prepared = self._prepare_data(data)
            with torch.no_grad():
                components = predictor.forward_components(prepared)
                probe = {
                    "baseline_score_abs_mean": self._tensor_abs_mean(
                        components, "base_score"
                    ),
                    "invariant_contribution_abs_mean": self._tensor_abs_mean(
                        components, "invariant_contribution"
                    ),
                    "bias_contribution_abs_mean": self._tensor_abs_mean(
                        components, "bias_contribution"
                    ),
                    "residual_contribution_abs_mean": self._tensor_abs_mean(
                        components, "residual_raw"
                    ),
                    "fused_residual_delta_abs_mean": self._tensor_abs_mean(
                        components, "residual_delta"
                    ),
                    "invariant_delta_abs_mean": self._tensor_abs_mean(
                        components, "invariant_delta"
                    ),
                    "bias_delta_abs_mean": self._tensor_abs_mean(
                        components, "bias_delta"
                    ),
                    "treatment_logit_abs_mean": self._tensor_abs_mean(
                        components, "treatment_logit"
                    ),
                    "z_inv_norm_mean": self._tensor_abs_mean(
                        {"value": components.get("z_inv")}, "value"
                    ),
                    "z_bias_norm_mean": self._tensor_abs_mean(
                        {"value": components.get("z_bias")}, "value"
                    ),
                    "gate_mean": self._tensor_abs_mean(
                        {"value": components.get("gate")}, "value"
                    ),
                }
                gate = components.get("gate")
                if torch.is_tensor(gate):
                    probe["gate_min"] = float(gate.min().item())
                    probe["gate_max"] = float(gate.max().item())
                    probe["invariant_gate_weight_mean"] = float(
                        gate.mean().item()
                    )
                    probe["bias_gate_weight_mean"] = float(
                        (1.0 - gate).mean().item()
                    )
                inv = components.get("invariant_contribution")
                bias = components.get("bias_contribution")
                if torch.is_tensor(inv) and torch.is_tensor(bias):
                    total = inv.detach().abs().mean() + bias.detach().abs().mean()
                    if float(total.item()) > 1e-12:
                        probe["invariant_contribution_share"] = float(
                            inv.detach().abs().mean().div(total).item()
                        )
                        probe["bias_contribution_share"] = float(
                            bias.detach().abs().mean().div(total).item()
                        )
                    else:
                        probe["invariant_contribution_share"] = 0.0
                        probe["bias_contribution_share"] = 0.0
                if hasattr(predictor, "diagnostics"):
                    for key, value in predictor.diagnostics(prepared).items():
                        if isinstance(value, (int, float)):
                            probe["model_{}".format(key)] = float(value)
            return probe
        finally:
            predictor.train(was_training)

    def _write_architecture_probe(self, epoch, probe):
        if not self.architecture_diagnostic_path:
            return
        parent = os.path.dirname(os.path.abspath(self.architecture_diagnostic_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        record = {"epoch": int(epoch), **probe}
        with open(self.architecture_diagnostic_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def run(self):
        if self.use_gpu:
            self.model.cuda()
        self._build_optimizer()
        if self.resume_training_state:
            state = torch.load(
                self.resume_training_state,
                map_location="cuda" if self.use_gpu else "cpu",
            )
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None and state.get("scheduler") is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            self.start_epoch = int(state["epoch"])
            if self.start_epoch < 0 or self.start_epoch > self.train_times:
                raise ValueError("resume epoch is outside the training range")
            print(
                "Resumed trainer state at epoch {} from {}".format(
                    self.start_epoch, self.resume_training_state
                )
            )
        print("Finish initializing CTS trainer...")

        training_range = tqdm(range(self.start_epoch, self.train_times))
        for epoch in training_range:
            if hasattr(self.model, "set_epoch"):
                self.model.set_epoch(epoch)

            epoch_total = 0.0
            metric_sums = {}
            batch_count = 0
            last_data = None
            for data in self.data_loader:
                last_data = data
                loss, metrics = self.train_one_step(data)
                epoch_total += loss
                batch_count += 1
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value

            divisor = float(max(1, batch_count))
            metric_means = {
                key: value / divisor for key, value in metric_sums.items()
            }
            weighted_aux_sum = sum(
                metric_sums.get(key, 0.0)
                for key in (
                    "weighted_residual_rank",
                    "weighted_invariant_rank",
                    "weighted_bias_rank",
                    "weighted_residual_preserve",
                    "weighted_cf",
                    "weighted_treat",
                    "weighted_orth",
                    "weighted_gate",
                    "weighted_gate_variance",
                    "weighted_invariant_variance",
                    "weighted_bias_variance",
                    "weighted_bias_counterfactual",
                    "weighted_branch_balance",
                )
            )
            weighted_cf_sum = metric_sums.get("weighted_cf", 0.0)
            weighted_residual_rank_sum = metric_sums.get(
                "weighted_residual_rank", 0.0
            )
            weighted_invariant_rank_sum = metric_sums.get(
                "weighted_invariant_rank", 0.0
            )
            weighted_bias_rank_sum = metric_sums.get(
                "weighted_bias_rank", 0.0
            )
            weighted_residual_preserve_sum = metric_sums.get(
                "weighted_residual_preserve", 0.0
            )
            weighted_treat_sum = metric_sums.get("weighted_treat", 0.0)
            weighted_orth_sum = metric_sums.get("weighted_orth", 0.0)
            weighted_gate_sum = metric_sums.get("weighted_gate", 0.0)
            weighted_gate_variance_sum = metric_sums.get(
                "weighted_gate_variance", 0.0
            )
            weighted_invariant_variance_sum = metric_sums.get(
                "weighted_invariant_variance", 0.0
            )
            weighted_bias_variance_sum = metric_sums.get(
                "weighted_bias_variance", 0.0
            )
            weighted_bias_counterfactual_sum = metric_sums.get(
                "weighted_bias_counterfactual", 0.0
            )
            weighted_branch_balance_sum = metric_sums.get(
                "weighted_branch_balance", 0.0
            )
            epoch_record = {
                "epoch": epoch,
                "total_loss_sum": epoch_total,
                "rank_loss_sum": metric_sums.get("rank_loss", 0.0),
                "weighted_aux_sum": weighted_aux_sum,
                "weighted_cf_sum": weighted_cf_sum,
                "weighted_residual_rank_sum": weighted_residual_rank_sum,
                "weighted_invariant_rank_sum": weighted_invariant_rank_sum,
                "weighted_bias_rank_sum": weighted_bias_rank_sum,
                "weighted_residual_preserve_sum": (
                    weighted_residual_preserve_sum
                ),
                "weighted_treat_sum": weighted_treat_sum,
                "weighted_orth_sum": weighted_orth_sum,
                "weighted_gate_sum": weighted_gate_sum,
                "weighted_gate_variance_sum": weighted_gate_variance_sum,
                "weighted_invariant_variance_sum": (
                    weighted_invariant_variance_sum
                ),
                "weighted_bias_variance_sum": weighted_bias_variance_sum,
                "weighted_bias_counterfactual_sum": (
                    weighted_bias_counterfactual_sum
                ),
                "weighted_branch_balance_sum": weighted_branch_balance_sum,
                **metric_means,
            }
            if (
                self.architecture_diagnostic_interval > 0
                and last_data is not None
                and (epoch + 1) % self.architecture_diagnostic_interval == 0
            ):
                architecture_probe = self._architecture_probe(last_data)
                for key in (
                    "baseline_grad_norm",
                    "invariant_grad_norm",
                    "bias_grad_norm",
                    "gate_grad_norm",
                    "rank_loss",
                    "cf_loss",
                    "treat_loss",
                    "orth_loss",
                    "gate_loss",
                    "residual_prepared_abs_mean",
                    "residual_unit_abs_mean",
                    "residual_saturation_fraction",
                    "weighted_residual_rank",
                    "weighted_invariant_rank",
                    "weighted_bias_rank",
                    "weighted_cf",
                    "weighted_treat",
                    "weighted_orth",
                    "weighted_gate",
                ):
                    if key in metric_means:
                        architecture_probe["train_{}".format(key)] = metric_means[key]
                epoch_record["architecture_probe"] = architecture_probe
                self._write_architecture_probe(epoch, architecture_probe)
            self.history.append(epoch_record)
            self._write_epoch_record(epoch_record)

            training_range.set_description(
                "Epoch {} | total_sum: {:.6f} | rank_sum: {:.6f} | "
                "res_rank: {:.6f} | w_res_rank_sum: {:.6f} | "
                "res_keep: {:.6f} | w_res_keep_sum: {:.6f} | "
                "base_margin: {:.6f} | candidate_margin: {:.6f} | "
                "cf: {:.6f} | treat: {:.6f} | orth: {:.6f} | gate_kl: {:.6f} | "
                "w_cf_sum: {:.6f} | w_treat_sum: {:.6f} | "
                "w_orth_sum: {:.6f} | w_gate_sum: {:.6f} | warmup: {:.3f} | "
                "residual_w: {:.3f} | "
                "gate: {:.4f} | treat_acc_f: {:.3f} | "
                "treat_acc_cf: {:.3f} | cf_unique: {:.3f} | cf_valid: {:.3f}".format(
                    epoch,
                    epoch_total,
                    metric_sums.get("rank_loss", 0.0),
                    metric_means.get("residual_rank_loss", 0.0),
                    weighted_residual_rank_sum,
                    metric_means.get("residual_preserve_loss", 0.0),
                    weighted_residual_preserve_sum,
                    metric_means.get("base_margin_mean", 0.0),
                    metric_means.get("candidate_margin_mean", 0.0),
                    metric_means.get("cf_loss", 0.0),
                    metric_means.get("treat_loss", 0.0),
                    metric_means.get("orth_loss", 0.0),
                    metric_means.get("gate_loss", 0.0),
                    weighted_cf_sum,
                    weighted_treat_sum,
                    weighted_orth_sum,
                    weighted_gate_sum,
                    metric_means.get("warmup", 0.0),
                    metric_means.get("residual_weight", 0.0),
                    metric_means.get("gate_mean", 0.0),
                    metric_means.get("treat_accuracy_f", 0.0),
                    metric_means.get("treat_accuracy_cf", 0.0),
                    metric_means.get("cf_unique_ratio", 0.0),
                    metric_means.get("cf_valid_ratio", 0.0),
                )
            )

            if self.scheduler is not None:
                self.scheduler.step()

            if (
                self.save_steps
                and self.checkpoint_dir
                and (epoch + 1) % self.save_steps == 0
            ):
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                checkpoint = os.path.join(
                    self.checkpoint_dir,
                    "epoch-{}.ckpt".format(epoch + 1),
                )
                predictor = getattr(self.model, "model", self.model)
                predictor.save_checkpoint(checkpoint)
                training_state = {
                    "epoch": epoch + 1,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": (
                        self.scheduler.state_dict()
                        if self.scheduler is not None
                        else None
                    ),
                }
                torch.save(
                    training_state,
                    os.path.join(
                        self.checkpoint_dir,
                        "epoch-{}.trainer.pt".format(epoch + 1),
                    ),
                )

        return self.history

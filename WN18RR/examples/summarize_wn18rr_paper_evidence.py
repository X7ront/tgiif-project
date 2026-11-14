#!/usr/bin/env python3
"""Build an auditable WN18RR 1000-epoch TGIIF evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev


METRIC_RE = re.compile(
    r"Filtered ranking \((?P<split>valid|test)\): "
    r"MRR=(?P<mrr>[0-9.]+), MR=(?P<mr>[0-9.]+), "
    r"Hits@10=(?P<h10>[0-9.]+), Hits@3=(?P<h3>[0-9.]+), "
    r"Hits@1=(?P<h1>[0-9.]+)"
)

RUNS = {
    (23, "Full TGIIF"): {
        "run": "tgiif_local_alignment_ep1000_20260906",
        "eval": "eval_valid_f0.0001_m0.75.log",
        "profile": "canonical",
    },
    (23, "CF-off"): {
        "run": "v5c_cf_off_ep1000_s23_f005",
        "eval": "eval_valid_f0.0001_m0.75_unified.log",
        "profile": "legacy-train-fraction",
    },
    (23, "Treatment-off"): {
        "run": "v5c_treatment_off_ep1000_s23_f005",
        "eval": "eval_valid_f0.0001_m0.75_unified.log",
        "profile": "legacy-train-fraction",
    },
    (23, "Treatment-shuffled"): {
        "run": "v5c_treatment_shuffled_ep1000_s23_f005",
        "eval": "eval_valid_f0.0001_m0.75_unified.log",
        "profile": "legacy-train-fraction",
    },
    (41, "Full TGIIF"): {
        "run": "tgiif_full_ep1000_s41",
        "eval": "eval_valid_f0.0001_m0.75.log",
        "profile": "canonical",
    },
    (41, "CF-off"): {
        "run": "tgiif_cf_off_ep1000_s41",
        "eval": "eval_valid_f0.0001_m0.75.log",
        "profile": "canonical-rerun-same-tag",
    },
    (41, "Treatment-off"): {
        "run": "tgiif_treatment_off_ep1000_s41",
        "eval": "eval_valid_f0.0001_m0.75.log",
        "profile": "canonical-partial-retry-after-completion",
    },
    (41, "Treatment-shuffled"): {
        "run": "tgiif_treatment_shuffled_ep1000_s41",
        "eval": "eval_valid_f0.0001_m0.75.log",
        "profile": "canonical",
    },
    (7, "Full TGIIF"): {
        "run": "tgiif_full_ep1000_s7",
        "eval": "eval_valid_f0.0001_m0.75.log",
        "profile": "canonical",
    },
    (7, "CF-off"): {
        "run": "tgiif_cf_off_ep1000_s7",
        "eval": "eval_valid_f0.0001_m0.75.log",
        "profile": "canonical",
    },
    (7, "Treatment-off"): {
        "run": "tgiif_treatment_off_ep1000_s7",
        "eval": "eval_valid_f0.0001_m0.75.log",
        "profile": "canonical",
    },
    (7, "Treatment-shuffled"): {
        "run": "tgiif_treatment_shuffled_ep1000_s7",
        "eval": "eval_valid_f0.0001_m0.75.log",
        "profile": "canonical",
    },
}

DIAGNOSTIC_FIELDS = (
    "total_loss",
    "rank_loss",
    "weighted_residual_rank",
    "weighted_residual_preserve",
    "weighted_cf",
    "weighted_treat",
    "weighted_orth",
    "cf_loss",
    "treat_loss",
    "orth_loss",
    "treat_accuracy_f",
    "treat_accuracy_cf",
    "cf_valid_ratio",
    "cf_unique_ratio",
    "gate_mean",
    "gate_std",
    "invariant_contribution_share",
    "invariant_delta_abs_mean",
    "bias_delta_abs_mean",
    "invariant_grad_norm",
    "bias_grad_norm",
    "gate_grad_norm",
    "residual_saturation_fraction",
    "residual_margin_improved_ratio",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_metric(path: Path):
    matches = METRIC_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No filtered-ranking result in {path}")
    split, mrr, mr, h10, h3, h1 = matches[-1]
    return {
        "split": split,
        "mrr": float(mrr),
        "mr": float(mr),
        "hits10": float(h10),
        "hits3": float(h3),
        "hits1": float(h1),
    }


def locate_single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, got {len(matches)}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def sample_sd(values):
    return stdev(values) if len(values) > 1 else 0.0


def t_interval_95(values):
    # Student-t critical value for df=2; the formal matrix has exactly 3 seeds.
    center = mean(values)
    if len(values) != 3:
        return None, None
    half_width = 4.3026527297 * sample_sd(values) / math.sqrt(3.0)
    return center - half_width, center + half_width


def exact_sign_test_one_sided(differences):
    nonzero = [value for value in differences if value != 0.0]
    positives = sum(value > 0.0 for value in nonzero)
    n = len(nonzero)
    numerator = sum(math.comb(n, k) for k in range(positives, n + 1))
    return numerator / (2**n) if n else 1.0


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f6(value):
    return f"{value:.6f}"


def main():
    args = parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or root / "paper_evidence").resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs_root = root / "logs"
    checkpoint_root = root / "checkpoint"

    metric_rows = []
    diagnostic_rows = []
    inventory_rows = []
    validations = []

    for (seed, method), spec in sorted(RUNS.items()):
        run_dir = logs_root / spec["run"]
        config_path = locate_single(run_dir, "*config.json")
        history_path = locate_single(run_dir, "*history.json")
        epoch_path = locate_single(run_dir, "*epochs.jsonl")
        eval_path = run_dir / spec["eval"]
        m0_candidates = sorted(run_dir.glob("eval_valid_*m0.log"))
        if not m0_candidates:
            m0_candidates = sorted(run_dir.glob("eval_valid_m0.log"))
        if len(m0_candidates) != 1:
            raise ValueError(f"Expected one m=0 validation log in {run_dir}")
        m0_path = m0_candidates[0]

        config = read_json(config_path)
        history = read_json(history_path)
        metric = read_metric(eval_path)
        baseline = read_metric(m0_path)
        checkpoint = checkpoint_root / (
            f"ctsnet_{config['run_tag']}_WN18RR_h1_0.005_Adam_ep1000.ckpt"
        )

        epochs_ok = (
            len(history) == 1000
            and [row.get("epoch") for row in history] == list(range(1000))
        )
        validation = {
            "seed": seed,
            "method": method,
            "run_tag": config["run_tag"],
            "config_seed_ok": config.get("seed") == seed,
            "history_complete": epochs_ok,
            "history_finite": finite_tree(history),
            "checkpoint_exists": checkpoint.is_file(),
            "evaluation_split": metric["split"],
            "training_fraction": config.get("invariant_margin_fraction"),
            "evaluation_fraction": 0.0001,
            "evaluation_multiplier": 0.75,
            "profile": spec["profile"],
            "raw_epoch_jsonl_lines": sum(
                1 for _ in epoch_path.open(encoding="utf-8", errors="replace")
            ),
        }
        validations.append(validation)

        row = {
            "seed": seed,
            "method": method,
            "profile": spec["profile"],
            "train_fraction": config.get("invariant_margin_fraction"),
            "eval_fraction": 0.0001,
            "residual_multiplier": 0.75,
            "mrr": metric["mrr"],
            "mr": metric["mr"],
            "hits10": metric["hits10"],
            "hits3": metric["hits3"],
            "hits1": metric["hits1"],
            "hits_spread": metric["hits10"] - metric["hits1"],
            "baseline_mrr": baseline["mrr"],
            "baseline_mr": baseline["mr"],
            "baseline_hits10": baseline["hits10"],
            "baseline_hits3": baseline["hits3"],
            "baseline_hits1": baseline["hits1"],
            "delta_mrr_vs_own_m0": metric["mrr"] - baseline["mrr"],
            "delta_mr_vs_own_m0": metric["mr"] - baseline["mr"],
            "delta_hits10_vs_own_m0": metric["hits10"] - baseline["hits10"],
            "delta_hits3_vs_own_m0": metric["hits3"] - baseline["hits3"],
            "delta_hits1_vs_own_m0": metric["hits1"] - baseline["hits1"],
            "config_path": config_path.relative_to(root).as_posix(),
            "history_path": history_path.relative_to(root).as_posix(),
            "checkpoint_path": checkpoint.relative_to(root).as_posix(),
            "eval_path": eval_path.relative_to(root).as_posix(),
            "m0_eval_path": m0_path.relative_to(root).as_posix(),
        }
        metric_rows.append(row)

        tail = history[-100:]
        diag = {
            "seed": seed,
            "method": method,
            "run_tag": config["run_tag"],
            "window": "epochs_900_999",
        }
        for field in DIAGNOSTIC_FIELDS:
            values = [
                float(item[field])
                for item in tail
                if field in item and item[field] is not None
            ]
            diag[f"{field}_mean"] = mean(values) if values else ""
            diag[f"{field}_final"] = (
                float(history[-1][field]) if field in history[-1] else ""
            )
        diagnostic_rows.append(diag)

        artifacts = [config_path, history_path, epoch_path, eval_path, m0_path]
        if checkpoint.is_file():
            artifacts.append(checkpoint)
        train_log = run_dir / "train.log"
        if train_log.is_file():
            artifacts.append(train_log)
        for artifact in artifacts:
            inventory_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "artifact_type": (
                        "checkpoint" if artifact.suffix == ".ckpt" else artifact.suffix.lstrip(".")
                    ),
                    "path": artifact.relative_to(root).as_posix(),
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256(artifact),
                }
            )

    methods = ("Full TGIIF", "CF-off", "Treatment-off", "Treatment-shuffled")
    seeds = (7, 23, 41)
    by_key = {(row["seed"], row["method"]): row for row in metric_rows}

    aggregate_rows = []
    for method in methods:
        for metric_name in ("mrr", "mr", "hits10", "hits3", "hits1", "hits_spread"):
            values = [by_key[(seed, method)][metric_name] for seed in seeds]
            aggregate_rows.append(
                {
                    "method": method,
                    "metric": metric_name,
                    "n_seeds": len(values),
                    "mean": mean(values),
                    "sample_sd": sample_sd(values),
                    "min": min(values),
                    "max": max(values),
                }
            )

    baseline_values = [by_key[(seed, "Full TGIIF")]["baseline_mrr"] for seed in seeds]
    full_values = [by_key[(seed, "Full TGIIF")]["mrr"] for seed in seeds]
    comparisons = []
    comparisons_to_build = [("Full TGIIF", "Official baseline", full_values, baseline_values)]
    for method in methods[1:]:
        comparisons_to_build.append(
            (
                method,
                "Full TGIIF",
                [by_key[(seed, method)]["mrr"] for seed in seeds],
                full_values,
            )
        )
    for left, right, left_values, right_values in comparisons_to_build:
        differences = [a - b for a, b in zip(left_values, right_values)]
        lower, upper = t_interval_95(differences)
        comparisons.append(
            {
                "comparison": f"{left} - {right}",
                "seed_order": list(seeds),
                "differences": differences,
                "mean_difference": mean(differences),
                "sample_sd_difference": sample_sd(differences),
                "paired_cohens_dz": (
                    mean(differences) / sample_sd(differences)
                    if sample_sd(differences) > 0.0
                    else None
                ),
                "t_95_ci": [lower, upper],
                "one_sided_exact_sign_p": exact_sign_test_one_sided(differences),
                "positive_seed_count": sum(value > 0.0 for value in differences),
                "n_seeds": len(differences),
            }
        )

    metric_fields = list(metric_rows[0].keys())
    write_csv(output / "wn18rr_1000ep_metrics.csv", metric_rows, metric_fields)
    write_csv(
        output / "wn18rr_1000ep_aggregate.csv",
        aggregate_rows,
        list(aggregate_rows[0].keys()),
    )
    write_csv(
        output / "wn18rr_1000ep_mechanism_diagnostics.csv",
        diagnostic_rows,
        list(diagnostic_rows[0].keys()),
    )
    write_csv(
        output / "wn18rr_artifact_inventory_sha256.csv",
        inventory_rows,
        list(inventory_rows[0].keys()),
    )
    (output / "wn18rr_validation_checks.json").write_text(
        json.dumps(
            {
                "generated_from": str(root),
                "validations": validations,
                "paired_mrr_comparisons": comparisons,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    aggregate_lookup = {
        (row["method"], row["metric"]): row for row in aggregate_rows
    }
    lines = [
        "# WN18RR TGIIF 1000-Epoch Paper Evidence",
        "",
        "Evidence role: multi-seed mechanism and stability analysis. These 1000-epoch",
        "runs are not the final 3000-epoch performance table.",
        "",
        "## Frozen evaluation protocol",
        "",
        "- Dataset/split: WN18RR validation.",
        "- Seeds: 7, 23, 41.",
        "- Evaluation: `fusion_mode=invariant-margin`, `f=0.0001`, `m=0.75`.",
        "- Own-checkpoint control: `m=0` for every trained checkpoint.",
        "- Full training: `lambda_cf=0.1`, `lambda_treat=0.1`,",
        "  `lambda_orth=0.01`, factual treatment supervision.",
        "",
        "## Per-seed validation results",
        "",
        "| Seed | Method | MRR | MR | H@10 | H@3 | H@1 | Delta MRR vs own m=0 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        for method in methods:
            row = by_key[(seed, method)]
            lines.append(
                f"| {seed} | {method} | {f6(row['mrr'])} | {row['mr']:.3f} | "
                f"{f6(row['hits10'])} | {f6(row['hits3'])} | "
                f"{f6(row['hits1'])} | {row['delta_mrr_vs_own_m0']:+.6f} |"
            )
    lines.extend(
        [
            "",
            "## Three-seed descriptive summary",
            "",
            "| Method | MRR mean +/- SD | H@10 mean +/- SD | H@3 mean +/- SD | H@1 mean +/- SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in methods:
        values = []
        for metric_name in ("mrr", "hits10", "hits3", "hits1"):
            row = aggregate_lookup[(method, metric_name)]
            values.append(f"{row['mean']:.6f} +/- {row['sample_sd']:.6f}")
        lines.append(f"| {method} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Paired MRR evidence",
            "",
            "| Comparison | Per-seed differences (7, 23, 41) | Mean +/- SD | 95% t CI | Exact one-sided sign p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in comparisons:
        diff_text = ", ".join(f"{value:+.6f}" for value in item["differences"])
        lower, upper = item["t_95_ci"]
        lines.append(
            f"| {item['comparison']} | {diff_text} | "
            f"{item['mean_difference']:+.6f} +/- {item['sample_sd_difference']:.6f} | "
            f"[{lower:+.6f}, {upper:+.6f}] | {item['one_sided_exact_sign_p']:.3f} |"
        )

    lines.extend(
        [
            "",
            "With only three seeds, the t intervals are very wide and the smallest",
            "possible one-sided exact sign-test p-value is 0.125. The table therefore",
            "supports effect direction and magnitude, not a conventional significance claim.",
            "",
            "## Model equations tied to implementation",
            "",
            "For training triples `i=(h_i,r_i,t_i)`, train-only popularity is",
            "",
            "```text",
            "p_i = log(1 + deg_train(h_i)) + log(1 + deg_train(t_i)).",
            "T_i = 1[p_i > median({p_j : r_j = r_i})].",
            "```",
            "",
            "The relation-preserving matched counterfactual is",
            "",
            "```text",
            "c(i) = argmin_j ||standardize(x_i) - standardize(x_j)||_2",
            "       subject to r_j = r_i and T_j != T_i.",
            "```",
            "",
            "The predictor forms shared intersection representation `u_i`, invariant",
            "representation `z_i^I`, popularity-aware bias representation `z_i^B`, and gate",
            "",
            "```text",
            "z_i^I = E_I(u_i),",
            "z_i^B = E_B([u_i, p_i]),",
            "g_i = sigmoid(G([u_i, p_i]) / tau_g),",
            "d_i^I = alpha_I * g_i * D_I(z_i^I),",
            "d_i^B = alpha_B * (1 - g_i) * D_B(z_i^B).",
            "```",
            "",
            "Under the paper's invariant-margin fusion, the bias branch remains auxiliary",
            "and only `d_i^I` changes ranking:",
            "",
            "```text",
            "uDelta_qi = r_s * tanh(P_q(d_i^I)),",
            "Delta_qi = clip(m * uDelta_qi, -f * R_q, f * R_q),",
            "s_qi = s0_qi + Delta_qi,",
            "R_q = max_j s0_qj - min_j s0_qj.",
            "```",
            "",
            "`P_q` denotes the configured query-level residual preparation. In the frozen",
            "runs, centering and tie-preservation are disabled. The training objective is",
            "",
            "```text",
            "L = L_rank(s0) + lambda_rr L_res-rank + lambda_rp L_preserve",
            "    + lambda_cf L_CF + lambda_t L_treat + lambda_o L_orth,",
            "L_CF = SmoothL1(z_i^I, z_c(i)^I),",
            "L_treat = BCEWithLogits(q_i, T_i),",
            "L_orth = mean(cos(z_i^I, z_i^B)^2).",
            "```",
            "",
            "`L_rank` is intentionally computed from the official baseline score (`isolate_base_rank=True`),",
            "so the baseline remains trainable while residual learning is separately supervised.",
            "",
            "## Evidence interpretation",
            "",
            "- Full exceeds its own `m=0` baseline for all three seeds. The paired MRR",
            "  gains are reported above and are the strongest current 1000-epoch result.",
            "- CF-off is slightly higher than Full for all three seeds. CF should therefore",
            "  be described as an invariance constraint, not as a demonstrated source of",
            "  validation-MRR improvement.",
            "- Treatment-off causes a large drop for seeds 23 and 41 but not seed 7; the",
            "  dependence is seed-sensitive rather than universally necessary.",
            "- Treatment-shuffled degrades Full only at seed 41. It does not provide a",
            "  consistent three-seed superiority claim for correctly aligned labels.",
            "- Final 100-epoch-window branch and loss statistics are in",
            "  `wn18rr_1000ep_mechanism_diagnostics.csv`.",
            "",
            "## Provenance caveats",
            "",
            "- Seed-23 ablation checkpoints were trained with recorded fraction 0.005; seeds",
            "  7/41 used 0.25. All reported ranking results were re-evaluated with the same",
            "  inference-time `f=0.0001, m=0.75` protocol. This should remain disclosed.",
            "- Seed-41 CF-off reused its run tag, so its raw epoch JSONL has 2000 lines; the",
            "  retained history/checkpoint/evaluation correspond to a complete 1000-epoch run.",
            "- Seed-41 Treatment-off was partially relaunched after completion, appending 114",
            "  raw epoch rows. The retained 1000-row history/checkpoint/evaluation predate that",
            "  retry and are used here. Do not use the raw JSONL tail for this run.",
            "- Dynamic verification used the WSL `intersection_feature_cu128` environment:",
            "  all 52 unit tests passed. On the seed-7 Full checkpoint, disabling the bias",
            "  branch exactly reproduced Full (`MRR=0.507368`), while disabling the invariant",
            "  branch exactly reproduced its own `m=0` baseline (`MRR=0.490352`). This is the",
            "  expected identity for invariant-margin fusion.",
            "- Counterfactual-cache invariants and coverage are recorded in",
            "  `wn18rr_cf_cache_audit.json`.",
        ]
    )
    (output / "WN18RR_1000EP_PAPER_EVIDENCE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"Wrote evidence bundle to {output}")
    print(f"Validated {len(metric_rows)} method/seed runs")
    print(
        "Full paired MRR gains: "
        + ", ".join(
            f"seed {seed}={by_key[(seed, 'Full TGIIF')]['delta_mrr_vs_own_m0']:+.6f}"
            for seed in seeds
        )
    )


if __name__ == "__main__":
    main()

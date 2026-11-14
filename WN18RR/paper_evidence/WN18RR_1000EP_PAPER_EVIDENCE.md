# WN18RR TGIIF 1000-Epoch Paper Evidence

Evidence role: multi-seed mechanism and stability analysis. These 1000-epoch
runs are not the final 3000-epoch performance table.

> Split correction (2026-11-14): the manuscript's formal WN18RR ranking
> results now use the **test split**. Validation performance in this document
> is retained only for mechanism, training-stability, and provenance analysis.
> The authoritative test results are under
> `paper_evidence/wn18rr_test_reevaluation_20261110/` and
> `paper_evidence/wn18rr_matched_f005_full_20261111/`.

## Frozen evaluation protocol

- Dataset/split: WN18RR validation.
- Seeds: 7, 23, 41.
- Evaluation: `fusion_mode=invariant-margin`, `f=0.0001`, `m=0.75`.
- Own-checkpoint control: `m=0` for every trained checkpoint.
- Full training: `lambda_cf=0.1`, `lambda_treat=0.1`,
  `lambda_orth=0.01`, factual treatment supervision.

## Per-seed validation results

| Seed | Method | MRR | MR | H@10 | H@3 | H@1 | Delta MRR vs own m=0 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | Full TGIIF | 0.507368 | 2185.101 | 0.502472 | 0.502472 | 0.502472 | +0.017016 |
| 7 | CF-off | 0.509306 | 2258.281 | 0.504779 | 0.504779 | 0.504450 | +0.018954 |
| 7 | Treatment-off | 0.509243 | 2127.924 | 0.504450 | 0.504450 | 0.504450 | +0.018891 |
| 7 | Treatment-shuffled | 0.509107 | 2114.808 | 0.504285 | 0.504285 | 0.504285 | +0.018755 |
| 23 | Full TGIIF | 0.923058 | 1075.943 | 0.924028 | 0.923698 | 0.922215 | +0.039380 |
| 23 | CF-off | 0.923882 | 927.408 | 0.924852 | 0.924522 | 0.923039 | +0.040204 |
| 23 | Treatment-off | 0.725452 | 2427.292 | 0.738299 | 0.724127 | 0.720007 | -0.158226 |
| 23 | Treatment-shuffled | 0.923882 | 927.408 | 0.924852 | 0.924522 | 0.923039 | +0.040204 |
| 41 | Full TGIIF | 0.726847 | 5021.209 | 0.731378 | 0.728576 | 0.724291 | +0.015669 |
| 41 | CF-off | 0.729632 | 4932.294 | 0.734179 | 0.731378 | 0.727093 | +0.018454 |
| 41 | Treatment-off | 0.529035 | 6402.574 | 0.540376 | 0.530323 | 0.523237 | -0.182143 |
| 41 | Treatment-shuffled | 0.714158 | 5506.796 | 0.717535 | 0.714898 | 0.712591 | +0.002980 |

## Three-seed descriptive summary

| Method | MRR mean +/- SD | H@10 mean +/- SD | H@3 mean +/- SD | H@1 mean +/- SD |
|---|---:|---:|---:|---:|
| Full TGIIF | 0.719091 +/- 0.207954 | 0.719293 +/- 0.211038 | 0.718249 +/- 0.210803 | 0.716326 +/- 0.209985 |
| CF-off | 0.720940 +/- 0.207425 | 0.721270 +/- 0.210334 | 0.720226 +/- 0.210094 | 0.718194 +/- 0.209436 |
| Treatment-off | 0.587910 +/- 0.119525 | 0.594375 +/- 0.125930 | 0.586300 +/- 0.120061 | 0.582565 +/- 0.119399 |
| Treatment-shuffled | 0.715716 +/- 0.207392 | 0.715557 +/- 0.210290 | 0.714568 +/- 0.210119 | 0.713305 +/- 0.209378 |

## Paired MRR evidence

| Comparison | Per-seed differences (7, 23, 41) | Mean +/- SD | 95% t CI | Exact one-sided sign p |
|---|---|---:|---:|---:|
| Full TGIIF - Official baseline | +0.017016, +0.039380, +0.015669 | +0.024022 +/- 0.013318 | [-0.009061, +0.057105] | 0.125 |
| CF-off - Full TGIIF | +0.001938, +0.000824, +0.002785 | +0.001849 +/- 0.000984 | [-0.000594, +0.004292] | 0.125 |
| Treatment-off - Full TGIIF | +0.001875, -0.197606, -0.197812 | -0.131181 +/- 0.115230 | [-0.417428, +0.155066] | 0.875 |
| Treatment-shuffled - Full TGIIF | +0.001739, +0.000824, -0.012689 | -0.003375 +/- 0.008079 | [-0.023444, +0.016694] | 0.500 |

With only three seeds, the t intervals are very wide and the smallest
possible one-sided exact sign-test p-value is 0.125. The table therefore
supports effect direction and magnitude, not a conventional significance claim.

## Model equations tied to implementation

For training triples `i=(h_i,r_i,t_i)`, train-only popularity is

```text
p_i = log(1 + deg_train(h_i)) + log(1 + deg_train(t_i)).
T_i = 1[p_i > median({p_j : r_j = r_i})].
```

The relation-preserving matched counterfactual is

```text
c(i) = argmin_j ||standardize(x_i) - standardize(x_j)||_2
       subject to r_j = r_i and T_j != T_i.
```

The predictor forms shared intersection representation `u_i`, invariant
representation `z_i^I`, popularity-aware bias representation `z_i^B`, and gate

```text
z_i^I = E_I(u_i),
z_i^B = E_B([u_i, p_i]),
g_i = sigmoid(G([u_i, p_i]) / tau_g),
d_i^I = alpha_I * g_i * D_I(z_i^I),
d_i^B = alpha_B * (1 - g_i) * D_B(z_i^B).
```

Under the paper's invariant-margin fusion, the bias branch remains auxiliary
and only `d_i^I` changes ranking:

```text
uDelta_qi = r_s * tanh(P_q(d_i^I)),
Delta_qi = clip(m * uDelta_qi, -f * R_q, f * R_q),
s_qi = s0_qi + Delta_qi,
R_q = max_j s0_qj - min_j s0_qj.
```

`P_q` denotes the configured query-level residual preparation. In the frozen
runs, centering and tie-preservation are disabled. The training objective is

```text
L = L_rank(s0) + lambda_rr L_res-rank + lambda_rp L_preserve
    + lambda_cf L_CF + lambda_t L_treat + lambda_o L_orth,
L_CF = SmoothL1(z_i^I, z_c(i)^I),
L_treat = BCEWithLogits(q_i, T_i),
L_orth = mean(cos(z_i^I, z_i^B)^2).
```

`L_rank` is intentionally computed from the official baseline score (`isolate_base_rank=True`),
so the baseline remains trainable while residual learning is separately supervised.

## Evidence interpretation

- Full exceeds its own `m=0` baseline for all three seeds. The paired MRR
  gains are reported above and are the strongest current 1000-epoch result.
- CF-off is slightly higher than Full for all three seeds. CF should therefore
  be described as an invariance constraint, not as a demonstrated source of
  validation-MRR improvement.
- Treatment-off causes a large drop for seeds 23 and 41 but not seed 7; the
  dependence is seed-sensitive rather than universally necessary.
- Treatment-shuffled degrades Full only at seed 41. It does not provide a
  consistent three-seed superiority claim for correctly aligned labels.
- Final 100-epoch-window branch and loss statistics are in
  `wn18rr_1000ep_mechanism_diagnostics.csv`.

## Provenance caveats

- Seed-23 ablation checkpoints were trained with recorded fraction 0.005; seeds
  7/41 used 0.25. All reported ranking results were re-evaluated with the same
  inference-time `f=0.0001, m=0.75` protocol. This should remain disclosed.
- Seed-41 CF-off reused its run tag, so its raw epoch JSONL has 2000 lines; the
  retained history/checkpoint/evaluation correspond to a complete 1000-epoch run.
- Seed-41 Treatment-off was partially relaunched after completion, appending 114
  raw epoch rows. The retained 1000-row history/checkpoint/evaluation predate that
  retry and are used here. Do not use the raw JSONL tail for this run.
- Dynamic verification used the WSL `intersection_feature_cu128` environment:
  all 52 unit tests passed. On the seed-7 Full checkpoint, disabling the bias
  branch exactly reproduced Full (`MRR=0.507368`), while disabling the invariant
  branch exactly reproduced its own `m=0` baseline (`MRR=0.490352`). This is the
  expected identity for invariant-margin fusion.
- Counterfactual-cache invariants and coverage are recorded in
  `wn18rr_cf_cache_audit.json`.

# Paper artifact manifest

Package scope: WN18RR TGIIF only. The package keeps the current treatment-aware
predictor call chain, data loader, hashing, counterfactual matcher, trainer,
evaluator, tests, WN18RR data, and train-only counterfactual cache.

## Retained evidence

- `v5c_counterfactual_*`: original 100-epoch V5c counterfactual reference,
  validation m-sweep, tie-audit JSON/log, and history JSON.
- `v5c_repro_fixed_ep100_20260902`: 100-epoch reproducibility checkpoint and
  logs.
- `v5c_invariant_margin_ep100_local_s23`: 100-epoch invariant-margin screen,
  fraction/m validation logs, diagnostics, history, and config.
- `v5c_invariant_margin_ep1000_local_s23`: 1000-epoch Full checkpoint and
  fraction/m validation sweep logs.
- `v5c_wn18rr_full_ep1000_local_s23`: historical 1000-epoch Full reference.
- `v5c_cf_off_ep100_s23_f005`, `v5c_cf_off_ep1000_s23_f005`.
- `v5c_treatment_off_ep100_s23_f005`, `v5c_treatment_off_ep1000_s23_f005`.
- `v5c_treatment_shuffled_ep100_s23_f005`,
  `v5c_treatment_shuffled_ep1000_s23_f005`.

- `tgiif_local_alignment_ep1000_20260906`: Full matched TGIIF rerun. Its
  `eval_valid_m0.log` is the Official baseline control,
  `eval_valid_m0.75.log` is the f=0.25 evaluation, and
  `eval_valid_f0.0001_m0.75.log` is the selected Full matched TGIIF result.
- `tgiif_random_cf_ep1000_s23`: 1000-epoch Random-CF control, including its
  random cache metadata, config, history, training log, and m=0/m=0.75
  validation evaluations. An additional `eval_valid_f0.0001_m0.75.log` uses
  the same selected cap as the Full matched TGIIF result.

The exact checkpoint, config, and evaluator log remain together under
`checkpoint/` and `logs/`.

## Explicit exclusions

Excluded are old V4/V6/V7/V8/V9 branches, tie-break and guard-only lines,
gate-soft legacy runs, generic FB15k-237 runs, 3000-epoch historical runs,
compatibility/module/package smoke probes, Python bytecode caches, and Codex
backup/audit metadata.

## Evidence caveat

The canonical 1000-epoch Full run is configured with
`fusion_mode=invariant-margin`, `invariant_margin_fraction=0.25`,
`residual_loss_schedule=auxiliary`, and factual treatment supervision. Its
evaluation directory also contains explicit cap sweeps (including 0.005 and
smaller values); those are inference-time analyses of the same checkpoint,
not separate training runs. The older additive/fusion clean-package reruns
are excluded from the paper evidence set because they do not represent this
canonical architecture.

Random-CF uses a train-only cache that preserves relation and flips treatment,
but samples the opposite-treatment triple uniformly instead of using nearest
intersection-feature matching. Its 95.01% valid coverage is matched to the
canonical cache; the cache records `random_cf_seed=23` and the source cache.

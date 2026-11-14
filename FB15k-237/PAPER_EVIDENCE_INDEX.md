# FB15k-237 paper-evidence index

This directory contains the complete FB15k-237 TGIIF code, data, evaluator,
tests, paper-selected checkpoints, and the logs/JSON needed to audit the main
and ablation results. Historical 1000/3000/3500/4500-epoch and smoke runs are
outside the submission package.

## Authoritative evidence

- `paper_evidence\fb15k237_ablation_alignment_20261007.md`: aligned ablation
  protocol, measurements, superseded values, and evidence boundaries.
- `paper_evidence\aligned_eval\eval_test_cf_off_f0.0175_m0.75.log`
- `paper_evidence\aligned_eval\eval_test_treatment_off_f0.0175_m0.75.log`
- `logs\fraction_m_sweep_full_ep4000_s23_20260919\VALID_TEST_COMPARISON.md`:
  legal-grid selection and the distinction between validation selection and
  diagnostic test scoring.

## Paper-to-artifact map

| Manuscript item | Checkpoint | Main evidence |
|---|---|---|
| Main table: TGIIF Full | `checkpoint\ctsnet_tgiif_fb15k237_full_ep4000_s23_FB15k-237_h1_0.005_Adam_ep4000.ckpt` | `logs\tgiif_fb15k237_full_ep4000_s23\` config/history; `logs\fraction_m_sweep_full_ep4000_s23_20260919\official_eval_test_f0.0175_m1.25.log` (the runtime clamps `m=1.25` to `1.0`) |
| Ablation: Full | Same Full checkpoint | `logs\fraction_m_sweep_full_ep4000_s23_20260919\test_grid.csv` / `test_grid.json` |
| Ablation: CF-off | `checkpoint\ctsnet_tgiif_fb15k237_cf_off_ep4000_s23_FB15k-237_h1_0.005_Adam_ep4000.ckpt` | `paper_evidence\aligned_eval\eval_test_cf_off_f0.0175_m0.75.log`; run directory under `logs\tgiif_fb15k237_cf_off_ep4000_s23\` |
| Ablation: Treatment-off | `checkpoint\ctsnet_tgiif_fb15k237_treatment_off_ep4000_s23_20261111_FB15k-237_h1_0.005_Adam_ep4000.ckpt` | `paper_evidence\aligned_eval\eval_test_treatment_off_f0.0175_m0.75.log`; run directory under `logs\tgiif_fb15k237_treatment_off_ep4000_s23_20261111\` |
| Ablation: shuffled labels | `checkpoint\ctsnet_tgiif_fb15k237_treatment_shuffled_ep4000_s23_FB15k-237_h1_0.005_Adam_ep4000.ckpt` | `logs\treatment_shuffled\eval_treatment_shuffled_ep4000_test_m0.75.log`; training records under `logs\tgiif_fb15k237_treatment_shuffled_ep4000_s23\` |
| Same-checkpoint `m=0` control | Full checkpoint | `logs\fraction_m_sweep_full_ep4000_s23_20260919\test_grid.csv` / `test_grid.json` |

## Retained checkpoints and SHA256

| Experiment checkpoint | SHA256 |
|---|---|
| `ctsnet_tgiif_fb15k237_full_ep4000_s23_FB15k-237_h1_0.005_Adam_ep4000.ckpt` | `90EF1994F9BCDC25672DACD87FBFB9B2E08E01EFBB08CFB3A48675DF18851DA4` |
| `ctsnet_tgiif_fb15k237_cf_off_ep4000_s23_FB15k-237_h1_0.005_Adam_ep4000.ckpt` | `E7317FE25EA38DADF37AF94F2A6FE48D5DDD7663D1C77FE1AA4F5F759987D575` |
| `ctsnet_tgiif_fb15k237_treatment_off_ep4000_s23_20261111_FB15k-237_h1_0.005_Adam_ep4000.ckpt` | `F2D3DAA43FCE0C8C2A3B2E3476C08235030F151C308DD62EA4E3AF0030BFA81E` |
| `ctsnet_tgiif_fb15k237_treatment_shuffled_ep4000_s23_FB15k-237_h1_0.005_Adam_ep4000.ckpt` | `3F40037AFC2D2AE626563DC3D9632F78A581527F41592AC10AA1039E5E09AE5C` |

## Reported protocol

- Main test result: split `test`, `fraction=0.0175`, `m=1.0`, 4000 epochs.
- Unified ablation block: split `test`, `fraction=0.0175`, `m=0.75`, 4000
  epochs. At this precision, the Full row is `0.467/0.455/0.458/0.500`.
- The `m=0` control is the same Full checkpoint with correction multiplier zero.
- The earlier 3000-epoch CF-off/Treatment-off values (`0.446` and `0.414`) are
  superseded and must not be used with the 4000-epoch Full row.

## Evidence boundaries

- These are single-seed (23), single-checkpoint measurements.
- The shuffled-training fraction is assumed to be `0.0175`; the source log did
  not record it. This assumption is disclosed in the manuscript evidence note.
- Exploratory rows with `m>1` are invalid because the production setter clamps
  the multiplier to `[0,1]`. They are excluded.
- Test-grid results are diagnostic for valid/test agreement; the formal
  hyperparameter choice should be described as validation-selected and reported
  once on test.

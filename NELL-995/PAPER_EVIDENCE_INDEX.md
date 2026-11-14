# NELL-995 paper-evidence index

This directory contains the complete NELL-995 TGIIF code, data, evaluator,
tests, source checkpoints, the final stagewise checkpoint, and the logs/JSON
needed to audit the manuscript. Smoke and unselected maturity-screen
checkpoints are outside the submission package.

## Manuscript result

The main table now uses the frozen-checkpoint filtered test evaluation:

| Setting | MRR | Hits@1 | Hits@3 | Hits@10 | MR |
|---|---:|---:|---:|---:|---:|
| `m=0` | 0.565736 | 0.558993 | 0.570641 | 0.570641 | 3773.388184 |
| `m=1.0` | 0.570008 | 0.558993 | 0.570641 | 0.584795 | 4251.164062 |

Both rows use the same final checkpoint, test split, seed, evaluator, and
residual cap. Only the correction multiplier changes.

## Paper-to-artifact map

| Manuscript item | Artifact |
|---|---|
| Main table: TGIIF Full and `m=0` | `checkpoint\ctsnet_tgiif_nell995_hybrid_base4000_gradscale001_treatmentoff_adapter500_s23.ckpt` |
| Test `m=0` evidence | `logs\paper_test_audit\eval_test_m0.log` |
| Test `m=1.0` evidence | `logs\paper_test_audit\eval_test_m1.log` |
| Test protocol and claim boundary | `logs\paper_test_audit\NELL995_TEST_EVALUATION.md` |
| Selected adapter configuration | `paper_evidence\selected_adapter_config.json` |
| Stagewise architecture and tensor inventory | `NELL995_ARCHITECTURE_DIFFERENCE_SPEC_20261114.txt` |
| Adapter-maturity validation, solely for architecture selection | `reference_evidence\STAGEWISE_ADAPTER_MATURITY_VALIDATION.md` |
| Mature 4000-epoch scorer source | `checkpoint\ctsnet_tgiif_nell995_full_ep4000_cloud_s23_NELL-995_h1_0.005_Adam_ep4000.ckpt`; `logs\tgiif_nell995_full_ep4000_cloud_s23\` |
| Separate 500-epoch Treatment-off adapter source | `checkpoint\ctsnet_tgiif_nell995_safe_gradscale001_cfdt005_treatmentoff_ep500_s23_NELL-995_h1_0.005_Adam_ep500.ckpt`; `logs\tgiif_nell995_safe_gradscale001_cfdt005_treatmentoff_ep500_s23\` |
| Evaluation runner/evaluator | `eval_tgiif_nell995.sh`; `examples\evaluate_tgiif_checkpoint_NELL995.py` |

## Retained checkpoints and SHA256

| Checkpoint | Role | SHA256 |
|---|---|---|
| `ctsnet_tgiif_nell995_hybrid_base4000_gradscale001_treatmentoff_adapter500_s23.ckpt` | Final composed paper checkpoint | `F9285508B8F21617A3EFED5BCE1EC5752B11A6B18518C460DF29395863C72A7F` |
| `ctsnet_tgiif_nell995_full_ep4000_cloud_s23_NELL-995_h1_0.005_Adam_ep4000.ckpt` | Mature scorer source | `8B61B9F38003D355E8D0A62128F981E2358204EFFA5529FC00508633E5020DA0` |
| `ctsnet_tgiif_nell995_safe_gradscale001_cfdt005_treatmentoff_ep500_s23_NELL-995_h1_0.005_Adam_ep500.ckpt` | Separate Treatment-off adapter source | `20FAEF7C6BE8ADF58DF8CE3F3858D26DA251A852F1600CD6995D91F36F6CCF6A` |

## Architecture and protocol

- Final model: 23 official scorer tensors from the mature 4000-epoch source plus
  44 non-official state items from the separate 500-epoch Treatment-off adapter
  source, for 67 items total.
- Composition happens after the two sources are trained. The adapter source's
  official path was not frozen during its 500 epochs; the official scorer is
  static only in the final composed checkpoint.
- Selected adapter settings: one hash hop; `hidden_dim=128`; `branch_dim=64`;
  ReLU; Adam; learning rate `0.005`; 100 batches per epoch; 25 entity negatives;
  ranking margin `0.5`; residual scale `0.1`; residual start epoch `100`;
  warm-up `150`; training correction fraction `0.001`; evaluation fraction
  `0.05`; `m=1.0`.
- Ranking path: ungated invariant-margin fusion with query centering,
  query-RMS normalization to `0.05`, exact-tie preservation, and floor at the
  baseline minimum. `lambda_t=0`; treatment, bias, and gate parameters remain in
  the network but do not determine the final invariant ranking.
- Evaluation: original OpenKE NELL-995 split, `test`, 3,992 test triples;
  `triple` features with one hash hop; official `Base.so` unconstrained filtered
  ranking in a separate process.

## Evidence boundaries

- The test result is a single-checkpoint, single-seed (23) measurement, not a
  multi-seed estimate or confidence interval.
- MRR gains `0.004272`; Hits@10 gains `0.014154`; Hits@1 and Hits@3 are equal;
  mean rank worsens by `477.775878`. The manuscript therefore must not claim
  improvement on every metric.
- The adapter-maturity study selected the architecture on validation. Its raw
  test result must not be used for parameter tuning.
- The phrase "4000-epoch scorer followed by a 500-epoch adapter stage" describes
  source order only. It does not mean that the scorer was frozen while the
  adapter source was trained.

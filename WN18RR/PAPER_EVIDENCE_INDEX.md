# WN18RR paper-evidence index

This directory contains the complete WN18RR TGIIF code, data, evaluator,
tests, paper-selected checkpoints, and the logs/JSON needed to audit the
experiments reported in the manuscript. Historical exploratory runs remain in
`F:\Codex\tgiif-project-excluded\WN18RR` and are not part of this submission
package.

## Authoritative evidence

- `paper_evidence\wn18rr_test_reevaluation_20261110\CORE_TEST_SCOPE_AND_RESULTS.md`:
  formal test results for the seven core Full/ablation reruns.
- `paper_evidence\wn18rr_matched_f005_full_20261111\MATCHED_F005_FULL_TEST_SUMMARY.md`:
  formal test result for the matched-fraction seed-23 Full checkpoint.
- `paper_evidence\WN18RR_1000EP_PAPER_EVIDENCE.md`: multi-seed results,
  provenance caveats, implementation equations, and validation-based
  mechanism/stability diagnostics. Its validation performance table is not
  the formal split used by the manuscript.
- `paper_evidence\wn18rr_1000ep_metrics.csv`: legacy validation per-run
  metrics.
- `paper_evidence\wn18rr_1000ep_aggregate.csv`: legacy validation descriptive
  aggregate.
- `paper_evidence\wn18rr_artifact_inventory_sha256.csv`: hashes for the legacy
  training/validation artifact set.
- `paper_evidence\wn18rr_test_reevaluation_20261110\` and
  `paper_evidence\wn18rr_matched_f005_full_20261111\`: complete `test`
  result bundles. Each run retains `metrics.json`,
  `resolved_eval_config.json`, `run_manifest.json`, `command.txt`, and
  `stdout.log`.
- `paper_evidence\mechanism_probe\MECHANISM_PROBE_REPORT.md`: score-pathway,
  representation, gradient, and saturation evidence for Figure 5 and the
  mechanism tables.

## Paper-to-artifact map

The common formal protocol is WN18RR **test**, filtered ranking, `f=0.0001`,
`m=0.75`, and `fusion_mode=invariant-margin`; the control is the same
checkpoint at `m=0`.

Training configs may still contain `eval_split=valid` because validation was
used for training-time monitoring. Formal split identity is determined by the
copied test-run `metrics.json` and `run_manifest.json`, both of which record
`split=test`.

| Manuscript item | Checkpoint | Configuration / history | Evaluation |
|---|---|---|---|
| Main table and matched-fraction Full, seed 23 | `checkpoint\ctsnet_v5c_full_f005_ep1000_s23_20261111_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `logs\v5c_full_f005_ep1000_s23_20261111\` (`*_config.json`, `*_history.json`) | `paper_evidence\wn18rr_matched_f005_full_20261111\core\exp37_full_f005_s23\test\m0_f0p0001\` and `m0p75_f0p0001\` |
| Ablation: CF-off, seed 23 | `checkpoint\ctsnet_v5c_cf_off_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `logs\v5c_cf_off_ep1000_s23_f005\` | `paper_evidence\wn18rr_test_reevaluation_20261110\core\exp04_cf_off_s23\test\m0_f0p0001\` and `m0p75_f0p0001\` |
| Ablation: Treatment-off, seed 23 | `checkpoint\ctsnet_v5c_treatment_off_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `logs\v5c_treatment_off_ep1000_s23_f005\` | `paper_evidence\wn18rr_test_reevaluation_20261110\core\exp05_treatment_off_s23\test\m0_f0p0001\` and `m0p75_f0p0001\` |
| Ablation: shuffled labels, seed 23 | `checkpoint\ctsnet_v5c_treatment_shuffled_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `logs\v5c_treatment_shuffled_ep1000_s23_f005\` | `paper_evidence\wn18rr_test_reevaluation_20261110\core\exp06_treatment_shuffled_s23\test\m0_f0p0001\` and `m0p75_f0p0001\` |
| Multi-seed Full: seed 7 | `checkpoint\ctsnet_tgiif_full_ep1000_s7_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `logs\tgiif_full_ep1000_s7\` | `paper_evidence\wn18rr_test_reevaluation_20261110\core\exp02_full_s7\test\m0_f0p0001\` and `m0p75_f0p0001\` |
| Multi-seed Full: seed 23 | Same matched-fraction checkpoint as the main result above | Same matched-fraction log directory as the main result above | Same `exp37` evaluation pair above |
| Multi-seed Full: seed 41 | `checkpoint\ctsnet_tgiif_full_ep1000_s41_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `logs\tgiif_full_ep1000_s41\` | `paper_evidence\wn18rr_test_reevaluation_20261110\core\exp03_full_s41\test\m0_f0p0001\` and `m0p75_f0p0001\` |
| Matched Treatment-off: seed 41 | `checkpoint\ctsnet_tgiif_treatment_off_ep1000_s41_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `logs\tgiif_treatment_off_ep1000_s41\` | `paper_evidence\wn18rr_test_reevaluation_20261110\core\exp07_treatment_off_s41\test\m0_f0p0001\` and `m0p75_f0p0001\` |
| Mechanism/checkpoint comparisons | Seed-23 Full, CF-off, and Treatment-off checkpoints above; additional seed-7/41 runs in `paper_evidence\mechanism_probe\` | `paper_evidence\mechanism_probe\RUN_COMMANDS.md` | `paper_evidence\mechanism_probe\score_pathways\`, `representations\`, `gradients\`, `cap_saturation\` |

## Retained checkpoints and SHA256

| Experiment checkpoint | SHA256 |
|---|---|
| `ctsnet_tgiif_local_alignment_ep1000_20260906_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `6B4CF7A5773BB39AAF8FC4556EE5B6EA28117619A419A5A99BCA4859FDCCF33D` |
| `ctsnet_v5c_full_f005_ep1000_s23_20261111_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `D1BE119350168F9B9670AAC631ECFF450222871E0D582CEA5C2E438E6FDAD59F` |
| `ctsnet_tgiif_full_ep1000_s7_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `C953DA0B4EFA298E9384CEBCC2D7F3C02F91CA027C924D057A71E58CBDDC312B` |
| `ctsnet_tgiif_full_ep1000_s41_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `17505A3DD5816A24783A9F1E940ABA601400E2E8010292A8AD43196B8E5C688E` |
| `ctsnet_v5c_cf_off_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `A9A880EFA0CABBE59EB79E2C18F27E122AB45A2B819E97B4162BF83ACD7B33D2` |
| `ctsnet_v5c_treatment_off_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `D627774AB76827D1C4A9BABC74C8A7311BF630DE26EE6808242C5B3C225D9A19` |
| `ctsnet_v5c_treatment_shuffled_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `0819B458BA896052120D3BD26F088F27E5C2E84A151183EC0D408706A836D02A` |
| `ctsnet_tgiif_cf_off_ep1000_s7_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `17819F397923F886CB66E1522412406F9F267E270C4B76A16195ECED0D0359E9` |
| `ctsnet_tgiif_treatment_off_ep1000_s7_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `D7FF10E8C584573F2455F9D6717CD7A9E12F24004AB2A82592CFD898D1C2DFE1` |
| `ctsnet_tgiif_treatment_shuffled_ep1000_s7_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `933BF706BB274A62AF5EA73EE8E6779A1C30B5915A5BB37B750566E151414D14` |
| `ctsnet_tgiif_cf_off_ep1000_s41_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `9A1BC5EDD5A43890F6717FB01384CEAAC50E265B4659FCCC3C0AA7E40E573571` |
| `ctsnet_tgiif_treatment_off_ep1000_s41_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `BD33325108C1B144F501F30F7E59A660675B6AB3C0051587979522A6C1F52D2D` |
| `ctsnet_tgiif_treatment_shuffled_ep1000_s41_WN18RR_h1_0.005_Adam_ep1000.ckpt` | `7B8B8E0CEAFE350221D1597007772D16026B4775FFA458F2D663BC6628851899` |

## Evidence boundaries

- The manuscript's WN18RR main table, seed-23 ablations, and multi-seed
  Full ranking values use the test split. The old validation performance
  table is retained only as mechanism/stability provenance.
- The matched seed-23 Full and all three seed-23 ablation checkpoints were
  trained with fraction `0.005`. Seeds 7 and 41 recorded training fraction
  `0.25`. All formal ranking results were evaluated with the same
  inference-time test protocol `f=0.0001, m=0.75`.
- Full seed 23 improves test MRR over its own `m=0` control by `0.043493`.
  CF-off and shuffled-label seed 23 are each `0.000373` above matched Full,
  so the package does not support a claim that Full outperforms either
  control.
- Treatment-off lowers test MRR by `0.211772` for seed 23 and by `0.213703`
  for seed 41. These are single-checkpoint comparisons and must not be
  generalized beyond the retained runs.
- The seed-41 CF-off epoch JSONL contains an appended retry, and the seed-41
  Treatment-off JSONL contains 114 appended rows. The retained checkpoint,
  1000-row history, and evaluation are the intended completed run; see the
  paper-evidence report before processing raw JSONL tails.
- Three seeds support direction and descriptive variability, not a conventional
  significance claim. The smallest possible one-sided exact sign-test p-value is
  0.125.

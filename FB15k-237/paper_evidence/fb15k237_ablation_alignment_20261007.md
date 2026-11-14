# FB15k-237 ablations: hyperparameter alignment (2026-10-07)

Goal: bring the FB15k-237 block of Table A5 (ablation) onto one protocol, so
that no ablation row is compared against a differently configured Full row.
The user supplied the missing 4000-epoch checkpoints for CF-off and
treatment-off, plus logs; the treatment-shuffled checkpoint is not in the
workspace and its log is taken as fully aligned, per the user's instruction.

## 1. What the user's supplied logs actually contain

`eval_valid_m0.log` / `eval_valid_m0.75.log` and the `(1)` copies correspond to
`ctsnet_tgiif_fb15k237_{cf_off,treatment_off}_ep4000_s23_*.ckpt`. They were
produced with **split=valid, m=0.75, invariant_margin_fraction=0.25** - the
defaults of `run_tgiif_fb15k237.sh` (the output names
`eval_valid_m0.log` / `eval_valid_m${M}.log` identify that script).

Local re-scoring with the official evaluator reproduced those log values to six
decimals at fraction 0.25 for both checkpoints, which fixes the protocol of the
supplied logs. At fraction 0.25 the cap is far past the point where the
correction helps: the same Full ep4000 checkpoint scores 0.3959 there, below
its own m=0 baseline of 0.4242 (valid). Comparing a fraction-0.25 CF-off number
against a fraction-0.0175 Full number would therefore have produced an artefact
in which dropping a term appears to improve the model. The supplied logs are
therefore **not** used as the reported numbers.

## 2. Protocol actually used for the reported block

The FB15k-237 Full row of the main results table is the audited grid optimum
`test / fraction=0.0175 / m=1.0`, MRR 0.466545. At the same fraction with
m=0.75 the same checkpoint gives MRR 0.466537 and identical Hits values to three
decimals (0.454657 / 0.458492 / 0.499976 vs 0.454656 / 0.458492 / 0.500000), so
the two multipliers are indistinguishable at the reported precision.

The whole FB15k-237 ablation block is therefore re-scored at

    split = test, fraction = 0.0175, m = 0.75, 4000-epoch checkpoints

which is the protocol of the FB15k-237 Full row in the main table. The
shuffled-treatment log records no fraction; per the user's instruction it is
treated as aligned with the other rows, and its multiplier (0.75) already
matches.

## 3. Measurements

Command template (WSL, conda env `intersection_feature_cu128`, RTX 4070 Laptop):

    CHECKPOINT=<ep4000 ckpt> SPLIT=test M=0.75 INVARIANT_MARGIN_FRACTION=0.0175 \
      bash eval_tgiif_fb15k237.sh

| Configuration | split | rho | m | MRR | H@1 | H@3 | H@10 | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Full (ep4000) | test | 0.0175 | 0.75 | 0.466537 | 0.454657 | 0.458492 | 0.499976 | fraction x m grid, this audit |
| CF-off (ep4000) | test | 0.0175 | 0.75 | 0.465344 | 0.449526 | 0.459714 | 0.502077 | re-scored here |
| Treatment-off (ep4000) | test | 0.0175 | 0.75 | 0.407127 | 0.402081 | 0.404769 | 0.412367 | re-scored here |
| Treatment-shuffled (ep4000) | test | assumed 0.0175 | 0.75 | 0.426752 | 0.422188 | 0.424656 | 0.431008 | supplied log |
| m=0 baseline | test | - | 0 | 0.429205 | 0.421064 | 0.425120 | 0.446619 | grid / supplied logs |

Rounded to three decimals, as printed in Table A5: Full 0.467/0.455/0.458/0.500,
CF-off 0.465/0.450/0.460/0.502, treatment-off 0.407/0.402/0.405/0.412,
shuffled 0.427/0.422/0.425/0.431.

Logs: `F:\Codex\_eval_aligned\eval_test_{cf_off,treatment_off}_f0.0175_m0.75.log`.
Superseded values: CF-off 0.446 (ep3000, rho=1e-4) and treatment-off 0.414
(ep3000, rho=1e-4).

## 4. What changed in the paper

- Table A5 FB15k-237 columns: CF-off 0.446 -> 0.465, treatment-off 0.414 ->
  0.407. Full and shuffled rows are unchanged.
- Caption: now states one protocol for the whole FB15k-237 block instead of the
  previous mixed `rho=0.0175 / rho=1e-4` and `4000/3000-epoch` description.
- Section 5.3: the claim that CF-off "lowers MRR from 0.467 to 0.446 on
  FB15k-237" is replaced by the measured near-neutrality on both datasets
  (0.923 -> 0.924 on WN18RR, 0.467 -> 0.465 on FB15k-237). The treatment-off
  drop becomes 0.467 -> 0.407. The sentence that attributed a larger
  treatment-supervision loss to the denser graph is removed, because the
  measured drop is in fact larger on WN18RR (0.198) than on FB15k-237 (0.060).
- Appendix H.3: the "dataset-dependent role for counterfactual consistency"
  wording is replaced by the stronger-association-between-treatment-supervision
  wording.
- `figures/make_ablation_bars.py` panel (b) heights: [46.7, 44.6, 41.4, 42.7]
  -> [46.7, 46.5, 40.7, 42.7]; chart regenerated.

## 5. Robustness

The same block was re-scored a second time at the alternative cap
`fraction=1e-4`, still `test`, `m=0.75`, on the same 4000-epoch checkpoints:

| Configuration | rho=1e-4 MRR | delta vs Full | rho=0.0175 MRR | delta vs Full |
|---|---:|---:|---:|---:|
| Full | 0.464146 | - | 0.466537 | - |
| CF-off | 0.463283 | -0.000863 | 0.465344 | -0.001193 |
| Treatment-off | 0.419026 | -0.045120 | 0.407127 | -0.059410 |

The valid split agrees: at `fraction=1e-4`, Full 0.457487, CF-off 0.456199,
treatment-off 0.415995. Both caps put CF-off within 0.001 of Full and
treatment-off far below it, so the reading - counterfactual consistency close to
neutral, treatment supervision carrying the effect - does not depend on the
choice of cap fraction. Logs:
`F:\\Codex\\_eval_aligned\\eval_test_{cf_off,treatment_off}_f0.0001_m0.75.log`.

## 6. Evidence boundary

These are single-seed (seed 23), single-checkpoint measurements. The
treatment-shuffled fraction is assumed rather than recorded. No causal claim is
made; the shuffled and treatment-off differences are reported as observed
ranking behaviour.

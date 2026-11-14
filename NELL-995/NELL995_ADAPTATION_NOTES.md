# NELL-995 adaptation: measured root causes and low-cost strategy

This note records the NELL-995-specific diagnosis and the runtime/training
controls added to the clean TGIIF package. It does not change the historical
default profile or the dataset split.

## Root-cause evidence

The 4000-epoch checkpoint
`checkpoint/ctsnet_tgiif_nell995_full_ep4000_cloud_s23_NELL-995_h1_0.005_Adam_ep4000.ckpt`
was evaluated with independent processes and filtered validation ranking.

| observation | measured evidence | consequence |
|---|---:|---|
| Sigmoid saturation | 50-query diagnostic: mean unique-score ratio `0.000298`, 99% of queries tied, median target tie group `17,213`, maximum `74,582` | Probability-space ranking has almost no numeric resolution on NELL-995 |
| Tiny residual is not harmless | `m=0`: MRR `0.584196`; `m=0.0001`: MRR `0.368386`; mean score change was only `9.98e-6` | A tiny perturbation reorders enormous baseline equivalence classes |
| Residual branch saturates | final training log: invariant raw magnitude `16.22`, residual saturation `0.996`, residual unit magnitude `0.09985` | Increasing/decreasing `m` cannot recover the lost ordering |
| Gate is unstable | 4000ep checkpoint gate mean `1.0`; 50ep safe gated screen gate mean about `0.01` | A learned gate can either suppress or monopolize invariant ranking |
| Treatment task is auxiliary, not ranking evidence | 4000ep treatment accuracy `0.962/0.922`, CF coverage `0.950`, gate still collapsed | Better treatment prediction does not imply better entity ranking |
| Objective scale mismatch | `MarginLoss` is applied to sigmoid scores in `[0,1]` with `margin=5` | The official branch is pushed toward `p=1,n=0` and vanishing sigmoid gradients |

The optional `--score-space logit` diagnostic confirms that simply replacing the
historical probability output is not a free fix: on the same checkpoint its
`m=0` validation was only MRR `0.010440`. Therefore the paper-facing default
remains probability-space filtered ranking; logit mode is only a saturation
diagnostic.

## Low-cost structural response

The runner now supports `NELL_SAFE_MODE=1`. It keeps all data, split, hash
hops, and checkpoint interfaces fixed while selecting:

* `margin=0.5` for new NELL training screens;
* query-RMS residual calibration with target RMS `0.05`;
* `invariant_margin_fraction=0.001`;
* near-tie mining (`25` candidates), no generic hard-negative mining;
* `residual_tie_band=0.05` and stronger baseline-preserve loss;
* centering, `preserve-base-ties`, and baseline-min floor.

In addition, `--invariant-margin-ungated` makes the invariant branch the sole
ranking branch in `invariant-margin` mode. The treatment gate remains available
for diagnostics and treatment supervision, but cannot turn the invariant rank
signal off. The flag is runtime-only and old checkpoints remain loadable.

## 50-epoch screen

The two profiles were trained with seed 23 and evaluated on validation only.
They are screens, not final NELL-995 results.

| profile | m=0 MRR / H@1,H@3,H@10 | m=0.75 MRR / H@1,H@3,H@10 | gate / residual diagnostics |
|---|---|---|---|
| safe, learned gate | `0.018330 / 0.002762, 0.006446, 0.064457` | `0.016766 / 0.001842, 0.003683, 0.062615` | gate collapsed to about `0.01` |
| safe + ungated invariant | `0.018330 / 0.002762, 0.006446, 0.064457` | `0.018278 / 0.002762, 0.006446, 0.064457` | gate fixed at diagnostic `0.5`; invariant raw `0.480`, saturation `0` |

The ungated profile reduces the m=0.75 deficit from `0.001564` to `0.000052`
in this short screen. The low absolute MRR is expected at 50 epochs after
changing the margin; it is not evidence for a final long-run gain.

## Recommended staged protocol

1. Run the existing 4000ep checkpoint through `--floor-residual-at-baseline-min`
   and record its own `m=0` comparison (already measured: MRR `0.585534` at
   `m=0.75`, versus `0.584196` at `m=0`).
2. Use the safe + ungated profile for one 100ep screen. Keep it only if its
   non-zero-m result is no worse than its own m=0 result and residual saturation
   stays below `0.2`.
3. Continue that single candidate to 500/1000 epochs. Do not start a 4000ep
   run until the 1000ep validation checkpoint passes the same baseline-preserve
   gate.
4. For WN18RR/FB15k-237, first measure score saturation. Use the historical
   profile when unique-score ratios are healthy; activate the NELL safe profile
   only when saturation/tie diagnostics cross the recorded thresholds.

## Files changed

* `openke/module/model/CTSNetIndependentResidualPredictor.py` — runtime score
  space switch for controlled saturation diagnostics.
* `openke/module/model/CTSNetTreatmentAwareResidualPredictor.py` — optional
  ungated invariant-margin fusion and logit diagnostic output.
* `examples/train_tgiif_from_scratch_NELL995.py` — training flag.
* `examples/evaluate_tgiif_checkpoint_NELL995.py` — evaluation flags.
* `run_tgiif_nell995.sh` — `NELL_SAFE_MODE` and `INVARIANT_MARGIN_UNGATED`
  parameter interfaces.
### Coherent logit-space diagnostic profile

`NELL_COHERENT_LOGIT_MODE=1` uses logit space for both the isolated official
rank loss and residual fusion/ranking. The filtered-ranking evaluator itself
is unchanged. Because logit gaps are much wider than probability gaps, this
diagnostic uses residual tie band `5.0` and bounded raw deltas. It is not the
default safe profile because the medium-run result below failed its gain gate.

### 500-epoch coherent-logit diagnosis and tie-safe revision

The seed-23 run `tgiif_nell995_safe_logit_band5_ep500_s23_retry` completed
without numerical saturation, but `m=0` and `m=0.75` produced exactly the same
filtered validation result: MRR `0.044061`, MR `7130.804688`, Hits@1
`0.001842`, Hits@3 `0.064457`, and Hits@10 `0.088398`.  In epochs 400--499,
the mean prepared residual was only `8.60e-6`, invariant gradient norm was
`2.35e-6`, and CF consistency loss was `2.29e-9`, while treatment accuracy
remained active (`0.934/0.868`) and CF coverage remained `0.950`.

This isolates the failure to the ranking path rather than the treatment/cache
path.  `preserve-base-ties` assigns one correction to every exact-score group,
so the residual rank loss cannot distinguish the positive from tied negatives.
The baseline-minimum floor additionally freezes the group that controls top-1.
With no surviving ranking gradient, the delta magnitude term favors the
all-zero invariant solution.

Two follow-up 100ep screens tested bounded `break-base-ties`, first ungated and
then with a regularized treatment gate. The ungated run still matched m=0
exactly; the gated run changed rankings but decreased MRR from `0.040443` to
`0.040014`. A post-hoc 100ep adapter initialized from the 4000ep official
branch also decreased MRR from `0.547895` to `0.371126`. Therefore these
settings remain optional diagnostic overrides, and `NELL_SAFE_MODE=1` is
restored to the previously measured positive 1000ep probability-space profile
(`0.102113 -> 0.136088`) with preserved ties, baseline-minimum floor, and
ungated invariant fusion. Full evidence is recorded in
`logs/ORGANIZED/NELL995_SAFE_CORRECTION_REPORT_20260915.md`.

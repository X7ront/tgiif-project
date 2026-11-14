# WN18RR matched-fraction Full test evaluation

Evaluation date: 2026-11-11

The new Full checkpoint and the seed-23 ablation checkpoints all use
`training invariant_margin_fraction=0.005`. Formal filtered-ranking evaluation
uses the previously frozen `evaluation fraction=0.0001` and `m=0.75`.

| Run | m | MRR | MR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|---:|
| Full f005 seed 23 | 0 | 0.881871 | 1346.184814 | 0.880823 | 0.882897 | 0.883216 |
| Full f005 seed 23 | 0.75 | 0.925364 | 1030.718994 | 0.924378 | 0.926292 | 0.926611 |
| CF-off f005 seed 23 | 0.75 | 0.925737 | 885.588989 | 0.924697 | 0.926771 | 0.927090 |
| Treatment-off f005 seed 23 | 0.75 | 0.713592 | 2577.535889 | 0.707881 | 0.712508 | 0.727824 |
| Treatment-shuffled f005 seed 23 | 0.75 | 0.925737 | 885.588989 | 0.924697 | 0.926771 | 0.927090 |

Full f005 improves MRR by `0.043493` over its own `m=0` control. Against the
matched Full result, CF-off and Treatment-shuffled are each `0.000373` higher
in MRR, while Treatment-off is `0.211772` lower. Therefore this seed does not
show a positive Full-over-CF-off or Full-over-shuffled effect, but it does show
a large degradation when treatment supervision is disabled.

All evaluations used independent evaluator processes. No validation sweep,
principle probe, per-query rank dump, or additional evaluation fraction was
run.

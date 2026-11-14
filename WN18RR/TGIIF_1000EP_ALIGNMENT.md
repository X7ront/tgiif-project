# TGIIF 1000-epoch alignment record

Run tag: `tgiif_local_alignment_ep1000_20260906`

The run was executed locally in WSL with the `intersection_feature_cu128`
environment on WN18RR, seed 23, and the canonical TGIIF configuration:

- `fusion_mode=invariant-margin`
- `invariant_margin_fraction=0.25`
- `residual_loss_schedule=auxiliary`
- `treatment_supervision=factual`
- `lambda_cf=0.1`, `lambda_treat=0.1`, `lambda_orth=0.01`
- `hash_hops=1`, `feature_mode=triple`
- `residual_start_epoch=100`, `residual_warmup_epochs=100`

Validation results:

| residual multiplier | MRR | MR | Hits@10 | Hits@3 | Hits@1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.883678 | 1316.778076 | 0.884641 | 0.884311 | 0.882828 |
| 0.75 | 0.834890 | 2583.934570 | 0.835695 | 0.835531 | 0.834047 |

For the paper's selected Full matched TGIIF setting, the same checkpoint was
also evaluated with `invariant_margin_fraction=0.0001` and `m=0.75`:

| fraction | MRR | MR | Hits@10 | Hits@3 | Hits@1 |
|---:|---:|---:|---:|---:|---:|
| 0.0001 | 0.923058 | 1075.943115 | 0.924028 | 0.923698 | 0.922215 |

The newly generated history JSON has the same SHA256 as the retained
`v5c_invariant_margin_ep1000_local_s23` history. The two checkpoints have the
same 62 state-dict tensors with `max_abs_diff=0`, confirming deterministic
architecture and training-path alignment rather than metric-only agreement.

Artifacts:

- `checkpoint/ctsnet_tgiif_local_alignment_ep1000_20260906_WN18RR_h1_0.005_Adam_ep1000.ckpt`
- `logs/tgiif_local_alignment_ep1000_20260906/`

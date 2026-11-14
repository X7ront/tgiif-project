# CTS-Net v5: Treatment-Aware Residual at hash_hops=1

## Failures established by full validation diagnostics

The v3 residual changed zero true-entity ranks across all 6,068 WN18RR
validation query sides. Three mechanisms caused the ineffective path:

1. Same-sketch opposite-popularity CF pairs made scalar invariant consistency
   nearly free (`cf_loss` approached `1e-14`).
2. The Bias encoder saw only the same 4D sketch as the Invariant encoder. It
   reached about 81% factual treatment accuracy but only about 30% on paired
   counterfactuals, because identical inputs cannot encode opposite treatment.
3. The residual was applied before sigmoid. Saturated official scores absorbed
   most logit corrections, leaving mean score changes near `1e-7`.

## v5 controlled changes

- Keep `hash_hops=1` and the official baseline branch unchanged and trainable.
- Derive endpoint popularity only from `train2id.txt`; no validation/test graph
  statistics are used.
- Feed normalized endpoint popularity only to the Bias branch and gate.
- Keep the Invariant branch restricted to the official 4D Triple Sketch.
- Apply the bounded residual after the official sigmoid score. Multiplier zero
  is therefore exactly the official score, while nonzero corrections survive
  official-score saturation.
- Apply CF consistency to `z_inv`, not to a single scalar score.
- Supervise treatment on both factual and CF Bias representations because the
  explicit popularity input makes their opposite labels identifiable.
- Train residual ranking on the five hardest of 25 sampled negatives, selected
  by the detached official score. Preserve margins on all sampled negatives
  that the official branch already orders correctly.

## Acceptance sequence

Run three 100-epoch validation-only ablations:

1. post-score residual + hard negatives;
2. add factual/CF treatment supervision;
3. add representation-level CF and orthogonality.

Each checkpoint is evaluated at residual multipliers 0, 0.25, 0.5, 0.75, and
1. Test remains untouched until one configuration exceeds its own multiplier-0
validation score without materially degrading Hits@1/3/10.

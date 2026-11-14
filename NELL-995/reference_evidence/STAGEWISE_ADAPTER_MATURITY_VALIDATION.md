# NELL-995 stagewise adapter-maturity validation (seed 23)

Date: 2026-09-16

All results are filtered validation results. No test split was evaluated and no
new training was performed for this study.

## Fixed official baseline

- Dataset: NELL-995
- Split: validation
- Seed: 23
- Feature mode / hash hops: triple / 1
- Official source:
  `checkpoint/ctsnet_tgiif_nell995_full_ep4000_cloud_s23_NELL-995_h1_0.005_Adam_ep4000.ckpt`
- Official source SHA256:
  `8B61B9F38003D355E8D0A62128F981E2358204EFFA5529FC00508633E5020DA0`
- Canonical `m=0` result: MRR `0.584196`, Hits@1 `0.578269`,
  Hits@3 `0.587477`, Hits@10 `0.588398`.

## Adapter-maturity screen

Every hybrid retained all adapter-side tensors from its source checkpoint and
replaced the 23 official scorer tensors with the fixed 4000-epoch official
source. The 50/100/1000 safe-ungated runs used the same configuration except
for the epoch budget and proportionally scaled residual schedule. The
500/1000 gradscale/CF runs form a second strictly matched configuration family.

| adapter source | best m at fraction 0.001 | MRR | delta vs fixed baseline | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|---:|
| safe-ungated 50ep | 0.25 | 0.584286 | +0.000090 | 0.578269 | 0.587477 | 0.588398 |
| safe-ungated 100ep | 0.25--1.0 | 0.584156 | -0.000040 | 0.578269 | 0.587477 | 0.588398 |
| gradscale/CF 500ep | 1.0 | **0.586224** | **+0.002028** | 0.578269 | 0.587477 | 0.588398 |
| gradscale/CF 1000ep | 1.0 | 0.585318 | +0.001122 | 0.578269 | 0.587477 | 0.588398 |
| prior safe-ungated 1000ep | 1.0 | 0.585448 | +0.001252 | 0.578269 | 0.587477 | 0.588398 |

The result is non-monotonic in adapter training time: an under-trained adapter
is neutral, the matched 500-epoch adapter is strongest, and continuing the
matched adapter to 1000 epochs reduces its compatibility with the mature
official scorer.

## Coarse fraction robustness for the 500-epoch adapter

Checkpoint:
`checkpoint/ctsnet_tgiif_nell995_hybrid_base4000_gradscale001_adapter500_s23.ckpt`

Checkpoint SHA256:
`41043AC886D1D1C576D41E319F8C65A4D736B669F60034001CF87F7233CFB0DA`

Adapter source SHA256:
`26922F9C0D30F59193DEE54790618117DB1A62B1CAAF4038BDB14B830A0EC4E2`

The multiplier was fixed at `m=1.0` for the coarse fraction check.

| fraction | MRR | delta MRR | Hits@1 | Hits@3 | Hits@10 |
|---:|---:|---:|---:|---:|---:|
| 0.0005 | 0.585809 | +0.001613 | 0.578269 | 0.587477 | 0.588398 |
| 0.0010 | 0.586224 | +0.002028 | 0.578269 | 0.587477 | 0.588398 |
| 0.0020 | 0.586111 | +0.001915 | 0.578269 | 0.587477 | 0.588398 |
| 0.0050 | 0.586379 | +0.002183 | 0.578269 | 0.587477 | 0.588398 |
| 0.0100 | 0.587019 | +0.002823 | 0.578269 | 0.587477 | 0.588398 |
| 0.0170 | 0.587555 | +0.003359 | 0.578269 | 0.587477 | 0.605893 |
| 0.0500 | **0.587819** | **+0.003623** | 0.578269 | 0.587477 | 0.605893 |

This was intentionally stopped after the preselected coarse points to avoid a
dense search on the 543-triple validation set.

## Multiplier robustness at fraction 0.05

| m | MRR | delta MRR | Hits@1 | Hits@3 | Hits@10 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.584196 | 0.000000 | 0.578269 | 0.587477 | 0.588398 |
| 0.25 | 0.587676 | +0.003480 | 0.578269 | 0.587477 | 0.605893 |
| 0.50 | 0.587696 | +0.003500 | 0.578269 | 0.587477 | 0.605893 |
| 0.75 | 0.587818 | +0.003622 | 0.578269 | 0.587477 | 0.605893 |
| 1.00 | **0.587819** | **+0.003623** | 0.578269 | 0.587477 | 0.605893 |

At the selected coarse fraction, all four non-zero multipliers improve MRR and
Hits@10 without reducing Hits@1 or Hits@3. The Hits@10-Hits@1 span is
`0.027624`.

## Training-diagnostic interpretation

The matched 500-to-1000 epoch comparison does not indicate branch collapse.
At the final epoch, the raw invariant magnitude increased from about `1985` to
`10362`, sampled-margin improvement increased from `62.1%` to `68.5%`, factual
treatment accuracy increased from about `91.8%` to `92.7%`, CF coverage stayed
near `95%`, and residual saturation stayed at zero. Nevertheless, the
1000-epoch adapter produced a smaller mature-baseline validation gain.

This supports an adapter-direction drift or training-candidate overfitting
hypothesis rather than a vanishing-gradient hypothesis. It does not establish
causality and does not yet establish multi-seed significance.

## Matched CF-off ablation

A new 500-epoch CF-off adapter was trained with the same seed, baseline path,
residual schedule, treatment loss, orthogonality loss, negative mining, and
optimizer settings as the Full 500-epoch adapter. The only intended mechanism
change was `lambda_cf: 0.1 -> 0.0`. It was then combined with the same fixed
4000-epoch official scorer and evaluated at fraction `0.05`.

CF-off adapter source SHA256:
`7308BB09E88FFEFC3DB08E18E63D0A81A649E0A3532C1D386581B6F01B43B40A`

CF-off hybrid SHA256:
`9E57C9B82889D700873535C1FB48AD1C237CB91EA607F94373D22C2289CF9D43`

| method | m | MRR | delta MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 0.25 | 0.587676 | +0.003480 | 0.578269 | 0.587477 | 0.605893 |
| Full | 0.50 | 0.587696 | +0.003500 | 0.578269 | 0.587477 | 0.605893 |
| Full | 0.75 | 0.587818 | +0.003622 | 0.578269 | 0.587477 | 0.605893 |
| Full | 1.00 | **0.587819** | **+0.003623** | 0.578269 | 0.587477 | **0.605893** |
| CF-off | 0.25 | **0.584681** | **+0.000485** | 0.578269 | 0.587477 | 0.588398 |
| CF-off | 0.50 | 0.584677 | +0.000481 | 0.578269 | 0.587477 | 0.588398 |
| CF-off | 0.75 | 0.584674 | +0.000478 | 0.578269 | 0.587477 | 0.588398 |
| CF-off | 1.00 | 0.584675 | +0.000479 | 0.578269 | 0.587477 | 0.588398 |

Removing CF eliminates all observed Hits@10 gain and about 87% of the best
MRR gain. CF-off still gives a small positive MRR change, so stagewise
composition and residual ranking may have a weak independent effect, but they
do not explain the Full result.

Last-50-epoch diagnostics also separate the mechanisms:

| diagnostic | Full | CF-off |
|---|---:|---:|
| raw invariant magnitude | 1352.556 | 3.049 |
| invariant gradient norm | 0.0000363 | 0.0000406 |
| sampled-margin improved ratio | 0.6086 | 0.3841 |
| prepared residual magnitude | 0.008953 | 0.004352 |
| factual treatment accuracy | 0.9209 | 0.9265 |
| counterfactual treatment accuracy | 0.8419 | 0.8542 |

CF-off has comparable invariant gradient norm and slightly higher treatment
accuracy, yet its invariant magnitude, training-margin transfer, and
validation gain are much smaller. The evidence therefore supports the CF
objective as a direction-forming signal for the invariant adapter, not merely
as a source of additional gradient magnitude. This is still a single-seed
mechanism result rather than a general causal conclusion.

## Evidence boundary and next gate

The result is stronger than baseline preservation on NELL-995 seed 23: it is a
positive MRR and Hits@10 result across several non-zero multipliers. It remains
a validation-selected, single-seed result. Before test evaluation or a general
claim, reproduce the fixed protocol on additional seeds and compare Full,
treatment-off, and shuffled-treatment adapters with the same mature official
scorer. The matched seed-23 CF-off comparison is now complete. No test score
should be used to tune the fraction or multiplier.

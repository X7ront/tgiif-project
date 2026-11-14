# Release verification manifest

Release date: 2026-09-19

## Architecture

- Dataset: NELL-995, OpenKE format, original train/valid/test split.
- Feature mode / hash hops: `triple / 1`.
- Mature scorer: 4000 epochs, seed 23.
- Adapter: 500 epochs, CF distance temperature 0.05.
- NELL safe path: query-RMS target 0.05, centered residual, preserve weight 2,
  invariant-margin ungated, training fraction 0.001.
- Frozen evaluation: fraction 0.05, multiplier 1.0.
- Default cloud variant: Treatment-off.  Full remains available through
  `ABLATION=full`.

## Verification

- Shell syntax: passed for all three runner/evaluator scripts.
- Python syntax: passed.
- Unit tests: `58/58 passed`.
- Hybrid composition: 23 scorer tensors and 44 adapter tensors verified;
  recomposed Treatment-off state matched the reference state tensor-for-tensor.
- Package-local filtered validation:
  - `m=0`: MRR 0.584196, MR 3629.885742, Hits@1 0.578269,
    Hits@3 0.587477, Hits@10 0.588398.
  - `m=1.0`, fraction 0.05, ungated: MRR 0.589245, MR 3994.688721,
    Hits@1 0.578269, Hits@3 0.587477, Hits@10 0.605893.

## Key SHA256 values

- Mature scorer: `8B61B9F38003D355E8D0A62128F981E2358204EFFA5529FC00508633E5020DA0`
- Treatment-off adapter: `20FAEF7C6BE8ADF58DF8CE3F3858D26DA251A852F1600CD6995D91F36F6CCF6A`
- Treatment-off hybrid: `F9285508B8F21617A3EFED5BCE1EC5752B11A6B18518C460DF29395863C72A7F`
- Train-only CF cache: `005641E97F3543AD13F68F404BCE8B0B991FC3EFA89AAC126B68BE7AC3CDD7E3`
- Treatment-aware predictor source: `13E8247753A74DBC812FC74C845797F86689298BE20917E0B2413E5BAC684D6A`
- CTS training strategy source: `CE7F51E52AD5A71DBAC17238EB9E5834C79D663D04A5BB59ABFF31B7D14D1684`

Validation scores are reference reproduction checks, not new multi-seed claims.

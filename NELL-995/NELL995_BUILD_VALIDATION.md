# NELL-995 project build validation

Validation date: 2026-09-08

## Dataset and interface audit

- Source revision: `2a1e389dab08f1239bf946c3d83be7600dc15937`
- OpenKE totals: 75,492 entities, 200 relations, 149,678 train triples
- Held-out totals: 543 validation triples, 3,992 test triples
- Split overlaps: train/valid 0, train/test 0, valid/test 0
- Feature mode: `triple`
- Hash hops: `1`
- Derived feature dimension: 4
- TGIIF `openke/` and `tests/`: byte-identical to the WN18RR clean
  template, excluding generated `__pycache__` files

## Train-only counterfactual cache

- Cache: `counterfactual_cache/NELL-995_popularity_triple_h1.pt`
- Training fingerprint:
  `5182b031f0a5ef03c7ec01a1f56f50c924708fc0f90fb1366cff9d8108142a9f`
- Treatment balance: low 80,583; high 69,095
- Valid matches: 142,205 / 149,678 (95.01%)
- Valid matches preserving relation: 142,205 / 142,205
- Valid matches changing treatment: 142,205 / 142,205
- Valid self-matches: 0
- Relation fallback list: empty

The cache builder reads only `train2id.txt`; neither validation nor test data
is passed to popularity estimation or matching.

## Software checks

- Python syntax compilation: passed
- Native OpenKE library load: passed in Ubuntu-22.04 WSL
- Unit tests: 52 / 52 passed
- Shell syntax: `run_tgiif_nell995.sh` and `eval_tgiif_nell995.sh` passed

## End-to-end CUDA smoke run

Run tag: `tgiif_nell995_smoke_ep2_s23`

The 2-epoch run exercised train-cache validation, official baseline training,
both residual branches, treatment supervision, checkpoint serialization,
checkpoint reload, and two independent filtered validation processes.

| multiplier | MRR | MR | Hits@1 | Hits@3 | Hits@10 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.006604 | 14729.043945 | 0.001842 | 0.001842 | 0.006446 |
| 0.75 | 0.026895 | 14806.227539 | 0.011050 | 0.011050 | 0.065378 |

These two-epoch values are pipeline-only smoke results and must not be used as
paper performance evidence.

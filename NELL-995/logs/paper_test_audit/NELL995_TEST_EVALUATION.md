# NELL-995 Test Evaluation Audit

Evaluation date: 2026-11-14  
Scope: frozen-checkpoint filtered test evaluation only; no retraining was performed.

## Checkpoint and model

- Checkpoint: `F:\Codex\CTS-Net-TGIIF-NELL995-cloud-latest-20260919\checkpoint\ctsnet_tgiif_nell995_hybrid_base4000_gradscale001_treatmentoff_adapter500_s23.ckpt`
- SHA256: `F9285508B8F21617A3EFED5BCE1EC5752B11A6B18518C460DF29395863C72A7F`
- File size: 747,088 bytes
- Tensor composition: 23 mature-scorer tensors + 44 adapter tensors = 67 tensors
- Parameter count: 179,256
- Seed: 23
- Training stage: 4000-epoch mature scorer followed by a 500-epoch frozen-scorer adapter stage
- Adapter variant: Treatment-off, ungated invariant-margin fusion
- Training residual cap: 0.001

## Evaluation protocol

- Dataset: NELL-995, original OpenKE split
- Split: test
- Test triples: 3,992
- Feature mode: triple
- Hash hops: 1
- Hidden/branch dimensions: 128 / 64
- Fusion: invariant-margin, fraction 0.05, ungated
- Evaluation residual cap: 0.05
- Ranking: official `Base.so` unconstrained filtered ranking
- Runtime: WSL Ubuntu 22.04, Python 3.10.20, PyTorch 2.1.0+cu118, CUDA on RTX 4070 Laptop GPU

The `m=0` and `m=1.0` runs used the same checkpoint, split, seed, model
configuration, evaluator, and residual cap. Only `--residual-multiplier`
changed. Both logs explicitly report `Filtered ranking (test)` and read
`benchmarks/NELL-995/test2id.txt`.

## Filtered test results

| Setting | MRR | MR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|
| `m=0` | 0.565736 | 3773.388184 | 0.558993 | 0.570641 | 0.570641 |
| `m=1.0` | 0.570008 | 4251.164062 | 0.558993 | 0.570641 | 0.584795 |
| Gain (`m=1.0 - m=0`) | +0.004272 | +477.775878 | 0.000000 | 0.000000 | +0.014154 |

The matched residual setting increases MRR by 0.004272, or 0.755% relative
to `m=0`, and increases Hits@10 by 0.014154. Hits@1 and Hits@3 are unchanged.
Mean rank worsens by 477.775878. Because lower MR is better, the MR change is
not a uniform improvement: the gain is concentrated in MRR and Hits@10.

Raw-ranking values are retained for audit:

| Setting | MRR | MR |
|---|---:|---:|
| `m=0` | 0.565732 | 3776.027832 |
| `m=1.0` | 0.569893 | 4253.756348 |

## Evidence files

- `F:\Codex\CTS-Net-TGIIF-NELL995-cloud-latest-20260919\logs\paper_test_audit\eval_test_m0.log`
  - SHA256: `FB9201B2770D232BC523FDC8055DCB2C5F2B8E2B8BC1AA0F4F49003A3111C00D`
- `F:\Codex\CTS-Net-TGIIF-NELL995-cloud-latest-20260919\logs\paper_test_audit\eval_test_m1.log`
  - SHA256: `8EE5B4F64F05748EB974A5EDC53B843CBF1F6DD4462FA187D6041D5B1722215E`
- Evaluation runner: `F:\Codex\CTS-Net-TGIIF-NELL995-cloud-latest-20260919\eval_tgiif_nell995.sh`
- Evaluator: `F:\Codex\CTS-Net-TGIIF-NELL995-cloud-latest-20260919\examples\evaluate_tgiif_checkpoint_NELL995.py`

## Reproducibility

From the project root in WSL, the two settings can be reproduced with:

```bash
export PYTHON_BIN=/home/xtront/miniconda3/envs/ctsnet/bin/python
export CHECKPOINT=./checkpoint/ctsnet_tgiif_nell995_hybrid_base4000_gradscale001_treatmentoff_adapter500_s23.ckpt
export SPLIT=test

M=0 OUTPUT=./logs/paper_test_audit/eval_test_m0.log ./eval_tgiif_nell995.sh
M=1.0 OUTPUT=./logs/paper_test_audit/eval_test_m1.log ./eval_tgiif_nell995.sh
```

## Claim boundary

These are single-checkpoint, single-seed test measurements, not a multi-seed
estimate or a confidence interval. They support a matched
Treatment-off, ungated stagewise-residual comparison on NELL-995 test. They do
not establish that a learned treatment gate is universally beneficial, and
only the MRR/Hits@10 gain is observed without a simultaneous MR improvement.

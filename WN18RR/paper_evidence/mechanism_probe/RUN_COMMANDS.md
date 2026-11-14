# Mechanism probe commands

All commands were run from `/mnt/f/Codex/CTS-Net-TGIIF-WN18RR-clean-20260906` after activating `intersection_feature_cu128`.

## Score pathway probe

```bash
python examples/probe_tgiif_score_pathways_WN18RR.py CHECKPOINT \
  --data-path ./benchmarks/WN18RR/ --split valid \
  --feature-mode triple --hash-hops 1 \
  --fusion-mode invariant-margin --invariant-margin-fraction 0.0001 \
  --residual-multiplier 0.75 --seed SEED --device cuda \
  --threads 8 --bootstrap 2000 --output-dir OUTPUT_DIR
```

Evaluated checkpoints:

- `checkpoint/ctsnet_tgiif_local_alignment_ep1000_20260906_WN18RR_h1_0.005_Adam_ep1000.ckpt` (`full_s23`)
- `checkpoint/ctsnet_v5c_cf_off_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` (`cf_off_s23`)
- `checkpoint/ctsnet_v5c_treatment_off_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` (`treatment_off_s23`)
- `checkpoint/ctsnet_tgiif_full_ep1000_s41_WN18RR_h1_0.005_Adam_ep1000.ckpt` (`full_s41`)
- `checkpoint/ctsnet_tgiif_treatment_off_ep1000_s41_WN18RR_h1_0.005_Adam_ep1000.ckpt` (`treatment_off_s41`)

## Representation probe

```bash
python examples/probe_tgiif_representations_WN18RR.py CHECKPOINT \
  --cache ./counterfactual_cache/WN18RR_popularity_h1.pt \
  --pair-sample 10000 --probe-sample 20000 --folds 5 \
  --bootstrap 2000 --seed 23 --device cuda --output OUTPUT_JSON
```

## Loss-gradient probe

```bash
python examples/probe_tgiif_loss_gradients_WN18RR.py \
  checkpoint/ctsnet_tgiif_local_alignment_ep1000_20260906_WN18RR_h1_0.005_Adam_ep1000.ckpt \
  --config logs/tgiif_local_alignment_ep1000_20260906/ctsnet_tgiif_local_alignment_ep1000_20260906_config.json \
  --cache ./counterfactual_cache/WN18RR_popularity_h1.pt \
  --batches 32 --seed 23 --device cuda \
  --output-dir paper_evidence/mechanism_probe/gradients/full_s23
```

## Cap saturation probe

```bash
python examples/probe_tgiif_cap_saturation_WN18RR.py CHECKPOINT \
  --data-path ./benchmarks/WN18RR/ --split valid \
  --feature-mode triple --hash-hops 1 \
  --invariant-margin-fraction 0.0001 --residual-multiplier 0.75 \
  --seed SEED --device cuda --threads 8 --output OUTPUT_JSON
```

## Regression test

```bash
python -m unittest discover -s tests -p 'test_*.py'
```


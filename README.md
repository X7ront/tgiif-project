# TGIIF Reviewer Reproduction Package

This package contains the runnable projects and retained evidence for the
WN18RR, FB15k-237, and NELL-995 experiments. Use the dataset evidence indexes
to locate the files associated with the manuscript, then use the included
checkpoints and scripts to re-evaluate or retrain the runs.

## Package layout

```text
tgiif-project/
├── WN18RR/
├── FB15k-237/
├── NELL-995/
└── README.md
```

Each dataset directory has the same general structure:

```text
<dataset>/
├── openke/                    # OpenKE sources, including the C++ Base.so sources
├── benchmarks/<dataset>/      # Data splits used by the evaluator
├── counterfactual_cache/      # Training-only counterfactual cache files
├── examples/                  # Training, evaluation, cache, and probe scripts
├── checkpoint/                # Retained checkpoints
├── logs/                      # Training configurations and histories
├── tests/                     # Unit tests
├── paper_evidence/            # Metrics, manifests, summaries, and diagnostics
├── PAPER_EVIDENCE_INDEX.md    # Evidence navigation for the dataset
├── requirements.txt
└── setup.py
```

## Finding evidence for the manuscript

Start with the dataset evidence index:

- [WN18RR evidence index](WN18RR/PAPER_EVIDENCE_INDEX.md)
- [FB15k-237 evidence index](FB15k-237/PAPER_EVIDENCE_INDEX.md)
- [NELL-995 evidence index](NELL-995/PAPER_EVIDENCE_INDEX.md)

Each index maps a manuscript item to its checkpoint, configuration, and
evaluation files. For exact values and execution conditions, open the
`metrics.json` and `run_manifest.json` files in the referenced result
directory. The manifest records the checkpoint, evaluator, data, cache,
command, and SHA256 values used for that run.

For WN18RR, the retained test reruns are stored in:

```text
WN18RR/paper_evidence/wn18rr_test_reevaluation_20261110/
WN18RR/paper_evidence/wn18rr_matched_f005_full_20261111/
```

Every run directory contains:

```text
metrics.json
resolved_eval_config.json
run_manifest.json
command.txt
stdout.log
```

The `command.txt` file is the authoritative record of how that evaluation was
executed.

## Environment

The scripts use Bash and should be run under WSL or Linux. The retained runs
used Python 3.10, PyTorch 2.1.0+cu118, and CUDA.

Install a dataset project from its own directory:

```bash
cd /path/to/tgiif-project/WN18RR
python -m pip install -r requirements.txt
python -m pip install -e .
(cd openke && bash make.sh)
```

For another dataset, replace `WN18RR` with `FB15k-237` or `NELL-995`.

`openke/make.sh` must be run inside the `openke` directory. Rebuild
`openke/release/Base.so` after changing the Linux, compiler, Python, or ABI
environment.

Run the tests from each dataset directory:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Re-evaluate retained checkpoints

### WN18RR

The WN18RR test evidence uses filtered ranking with
`fraction=0.0001`, `m=0.75`, and `fusion_mode=invariant-margin`. Each corrected
evaluation is paired with the same checkpoint at `m=0`.

The following checkpoint groups are retained:

| Run | Checkpoint |
|---|---|
| Full, seed 23 | `checkpoint/ctsnet_v5c_full_f005_ep1000_s23_20261111_WN18RR_h1_0.005_Adam_ep1000.ckpt` |
| Full, seed 7 | `checkpoint/ctsnet_tgiif_full_ep1000_s7_WN18RR_h1_0.005_Adam_ep1000.ckpt` |
| Full, seed 41 | `checkpoint/ctsnet_tgiif_full_ep1000_s41_WN18RR_h1_0.005_Adam_ep1000.ckpt` |
| CF-off, seed 23 | `checkpoint/ctsnet_v5c_cf_off_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` |
| Treatment-off, seed 23 | `checkpoint/ctsnet_v5c_treatment_off_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` |
| Treatment-shuffled, seed 23 | `checkpoint/ctsnet_v5c_treatment_shuffled_ep1000_s23_f005_WN18RR_h1_0.005_Adam_ep1000.ckpt` |
| Treatment-off, seed 41 | `checkpoint/ctsnet_tgiif_treatment_off_ep1000_s41_WN18RR_h1_0.005_Adam_ep1000.ckpt` |

Re-evaluate the matched-fraction Full seed-23 checkpoint:

```bash
cd /path/to/tgiif-project/WN18RR

CHECKPOINT=./checkpoint/ctsnet_v5c_full_f005_ep1000_s23_20261111_WN18RR_h1_0.005_Adam_ep1000.ckpt \
SPLIT=test \
M=0.75 \
INVARIANT_MARGIN_FRACTION=0.0001 \
OUTPUT=./logs/reproduction/wn18rr_full_s23_test_m0.75.log \
bash ./eval_v5c_wn18rr_cloud.sh

CHECKPOINT=./checkpoint/ctsnet_v5c_full_f005_ep1000_s23_20261111_WN18RR_h1_0.005_Adam_ep1000.ckpt \
SPLIT=test \
M=0 \
INVARIANT_MARGIN_FRACTION=0.0001 \
OUTPUT=./logs/reproduction/wn18rr_full_s23_test_m0.log \
bash ./eval_v5c_wn18rr_cloud.sh
```

To re-evaluate another checkpoint, replace `CHECKPOINT` and `OUTPUT`.
Training-time configurations may contain `eval_split=valid`; the rerun split
is the `SPLIT=test` setting above and is recorded in each
`run_manifest.json`.

Retrain the matched-fraction Full seed-23 checkpoint:

```bash
cd /path/to/tgiif-project/WN18RR

ABLATION=full \
EPOCHS=1000 \
SEED=23 \
INVARIANT_MARGIN_FRACTION=0.005 \
RUN_TAG=tgiif_repro_full_ep1000_s23 \
RUN_EVAL=0 \
PYTHON_BIN=python \
bash ./run_v5c_wn18rr_cloud.sh
```

Retrain the seed-23 ablations:

```bash
ABLATION=cf_off \
EPOCHS=1000 \
SEED=23 \
INVARIANT_MARGIN_FRACTION=0.005 \
RUN_TAG=tgiif_repro_cf_off_ep1000_s23 \
RUN_EVAL=0 \
bash ./run_v5c_wn18rr_cloud.sh

ABLATION=treatment_off \
EPOCHS=1000 \
SEED=23 \
INVARIANT_MARGIN_FRACTION=0.005 \
RUN_TAG=tgiif_repro_treat_off_ep1000_s23 \
RUN_EVAL=0 \
bash ./run_v5c_wn18rr_cloud.sh

ABLATION=treatment_shuffled \
SHUFFLE_TREATMENT_LABELS=1 \
EPOCHS=1000 \
SEED=23 \
INVARIANT_MARGIN_FRACTION=0.005 \
RUN_TAG=tgiif_repro_treat_shuf_ep1000_s23 \
RUN_EVAL=0 \
bash ./run_v5c_wn18rr_cloud.sh
```

### FB15k-237

The retained FB15k-237 test evaluations use filtered ranking with
`fraction=0.0175`.

Re-evaluate the Full checkpoint:

```bash
cd /path/to/tgiif-project/FB15k-237

CHECKPOINT=./checkpoint/ctsnet_tgiif_fb15k237_full_ep4000_s23_FB15k-237_h1_0.005_Adam_ep4000.ckpt \
SPLIT=test \
M=1.0 \
INVARIANT_MARGIN_FRACTION=0.0175 \
OUTPUT=./logs/reproduction/fb15k237_full_s23_test_m1.0.log \
bash ./eval_tgiif_fb15k237.sh
```

Re-evaluate the CF-off checkpoint at the ablation setting:

```bash
CHECKPOINT=./checkpoint/ctsnet_tgiif_fb15k237_cf_off_ep4000_s23_FB15k-237_h1_0.005_Adam_ep4000.ckpt \
SPLIT=test \
M=0.75 \
INVARIANT_MARGIN_FRACTION=0.0175 \
OUTPUT=./logs/reproduction/fb15k237_cf_off_s23_test_m0.75.log \
bash ./eval_tgiif_fb15k237.sh
```

Use the checkpoint and parameter files referenced in
[FB15k-237/PAPER_EVIDENCE_INDEX.md](FB15k-237/PAPER_EVIDENCE_INDEX.md) for
the remaining retained runs.

Retrain the Full checkpoint:

```bash
cd /path/to/tgiif-project/FB15k-237

ABLATION=full \
EPOCHS=4000 \
SEED=23 \
RUN_TAG=tgiif_repro_fb15k237_full_ep4000_s23 \
RUN_EVAL=0 \
PYTHON_BIN=python \
bash ./run_tgiif_fb15k237.sh
```

### NELL-995

The NELL-995 final checkpoint is a stagewise model built from a 4000-epoch
official scorer and a separately trained 500-epoch Treatment-off adapter.
The retained test evaluations use the original OpenKE NELL-995 split with
`fraction=0.05`, `m=0` and `m=1.0`, and ungated invariant margin.

Re-evaluate the final checkpoint:

```bash
cd /path/to/tgiif-project/NELL-995

CHECKPOINT=./checkpoint/ctsnet_tgiif_nell995_hybrid_base4000_gradscale001_treatmentoff_adapter500_s23.ckpt \
SPLIT=test \
M=0 \
INVARIANT_MARGIN_FRACTION=0.05 \
INVARIANT_MARGIN_UNGATED=1 \
OUTPUT=./logs/reproduction/nell995_test_m0.log \
bash ./eval_tgiif_nell995.sh

CHECKPOINT=./checkpoint/ctsnet_tgiif_nell995_hybrid_base4000_gradscale001_treatmentoff_adapter500_s23.ckpt \
SPLIT=test \
M=1.0 \
INVARIANT_MARGIN_FRACTION=0.05 \
INVARIANT_MARGIN_UNGATED=1 \
OUTPUT=./logs/reproduction/nell995_test_m1.log \
bash ./eval_tgiif_nell995.sh
```

Regenerate the stagewise checkpoint:

```bash
cd /path/to/tgiif-project/NELL-995

BASE_CHECKPOINT=./checkpoint/ctsnet_tgiif_nell995_full_ep4000_cloud_s23_NELL-995_h1_0.005_Adam_ep4000.ckpt \
ABLATION=treatment_off \
ADAPTER_EPOCHS=500 \
SEED=23 \
RUN_TAG=tgiif_repro_nell995_treatmentoff_adapter500_s23 \
HYBRID_CHECKPOINT=./checkpoint/ctsnet_tgiif_repro_nell995_treatmentoff_adapter500_s23_hybrid.ckpt \
RUN_TEST=0 \
PYTHON_BIN=python \
bash ./run_latest_nell995_cloud.sh
```

The exact base checkpoint, adapter checkpoint, configuration files, and test
logs are listed in [NELL-995/PAPER_EVIDENCE_INDEX.md](NELL-995/PAPER_EVIDENCE_INDEX.md).

## Artifact integrity

Check the following files before comparing a rerun with a retained result:

1. `run_manifest.json` for the checkpoint, evaluator, data, cache, command,
   and input SHA256 values.
2. `resolved_eval_config.json` for the effective split and evaluation
   parameters.
3. `metrics.json` for the reported metrics.
4. `stdout.log` for the evaluator output.

The dataset evidence indexes also record the retained checkpoint hashes and
point to the complete hash inventories where available.

## License

The original TGIIF project code and documentation are released under the MIT
License. See [LICENSE](LICENSE) for the full terms. Third-party datasets,
dependencies, and other externally sourced materials remain subject to their
respective licenses and terms of use.

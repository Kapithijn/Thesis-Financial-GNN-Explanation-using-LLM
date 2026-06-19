# Explanation of GNN Predictions with LLMs

Minimal thesis codebase for running the GNN explanation and LLM evaluation pipeline.

This repository intentionally contains the core pipeline source files plus one minimal Snellius job and example config. Datasets, plots, notebooks, cached models, and generated outputs are not included.

## Files

```text
Final_version/
├── main.py                 # Pipeline entry point and CLI
├── Data_File.py            # Dataset loading and preprocessing
├── GNN_Definition.py       # GCN, GAT, GIN, and GraphSAGE definitions
├── Train.py                # GNN training and evaluation helpers
├── Extracion.py            # Prediction, explanation, embedding, and subgraph extraction
├── Parallel_Extraction.py  # Worker helper for parallel extraction
├── LLM_Module.py           # Prompt construction, LLM inference, and response parsing
├── Evalueation.py          # Metrics and result saving
└── configs/
    └── snellius_smoke.json # Small example config for cluster runs
run_snellius.job            # Minimal Slurm entry point
```

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU runs, install the `torch` build that matches your CUDA environment before installing the rest of the requirements.

## Datasets

Datasets are not committed. Create the needed folders locally under `Final_version/data/`.

Common expected paths:

```text
Final_version/data/elliptic_dataset/
Final_version/data/elliptic_temporal_dataset/
Final_version/data/dgraphfin_dataset/raw/DGraphFin.zip
Final_version/data/tfinance_dataset/raw/
Final_version/data/ibm_aml_dataset/raw/HI-Small_Trans.csv.zip
```

The `.gitignore` prevents these dataset files from being committed.

## Quick Check

From the repository root:

```bash
python -m py_compile \
  Final_version/main.py \
  Final_version/Data_File.py \
  Final_version/GNN_Definition.py \
  Final_version/Train.py \
  Final_version/Extracion.py \
  Final_version/Parallel_Extraction.py \
  Final_version/LLM_Module.py \
  Final_version/Evalueation.py
```

## Dry Run

```bash
cd Final_version
python main.py --dry-run --device cpu --output-dir outputs/dry_run
```

## Example Run

This example runs data loading, training, and extraction while skipping LLM inference and final evaluation:

```bash
cd Final_version
python main.py \
  --datasets elliptic \
  --models GAT \
  --experiments embedding_classification \
  --num-target-nodes 1 \
  --device cpu \
  --output-dir outputs/local_smoke \
  --skip-llm \
  --skip-eval
```

Remove `--skip-llm --skip-eval` and set `--llms` for a full LLM run.

## Snellius Run

Submit from the repository root:

```bash
sbatch run_snellius.job
```

Useful overrides:

```bash
CONDA_ENV=ethfraud_clean RUN_LLM=false sbatch run_snellius.job
CONFIG_PATH=Final_version/configs/snellius_smoke.json sbatch run_snellius.job
```

Edit `Final_version/configs/snellius_smoke.json` for larger experiments, more target nodes, other datasets, other GNNs, or other LLMs.

## Notes

If `main.py` fails with `ModuleNotFoundError: No module named 'torch_geometric'`, install the dependencies in `requirements.txt` in the active environment.

On some macOS environments, mixed scientific packages can trigger an OpenMP `libomp.dylib already initialized` error. A clean Conda environment is preferred. For a temporary local smoke check, prefix the command with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python Final_version/main.py --dry-run --device cpu
```

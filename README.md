# Explanation of GNN Predictions with LLMs

This repository contains the runnable thesis code for experiments that combine graph neural networks, graph explanations, and large language models on financial fraud/anomaly graph datasets.

Datasets and generated outputs are intentionally not included. Place datasets in the folders described below before running full experiments.

## Repository Layout

```text
.
├── Final_version/
│   ├── main.py                         # Main pipeline entry point
│   ├── Data_File.py                    # Dataset loaders and preprocessing
│   ├── GNN_Definition.py               # GNN model definitions
│   ├── Train.py                        # Training helpers
│   ├── Extracion.py                    # GNNExplainer/subgraph extraction
│   ├── Parallel_Extraction.py          # Multiprocessing extraction helper
│   ├── LLM_Module.py                   # Prompting, generation, parsing
│   ├── Evalueation.py                  # Evaluation and metrics
│   ├── configs/                        # Example JSON configs
│   ├── data/                           # Dataset placeholders only
│   └── tests/                          # Unit tests
├── *.job                               # Slurm job scripts for cluster runs
├── requirements.txt
└── README.md
```

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU runs, install the `torch` build that matches your CUDA environment before installing the rest of the requirements. On Snellius or another cluster, using a Conda environment is usually easier.

## Datasets

No datasets are committed to git. The expected locations are:

```text
Final_version/data/elliptic_dataset/
Final_version/data/elliptic_temporal_dataset/
Final_version/data/dgraphfin_dataset/raw/DGraphFin.zip
Final_version/data/tfinance_dataset/raw/
Final_version/data/ibm_aml_dataset/raw/HI-Small_Trans.csv.zip
```

The Elliptic PyG datasets can be downloaded automatically by PyTorch Geometric when network access is available. DGraphFin, T-Finance, and IBM AML must usually be downloaded manually because they require external dataset access.

For IBM AML HI-Small, this helper can place the zip in the expected folder:

```bash
cd Final_version
./download_ibm_aml_hi_small.sh
```

## Quick Checks

From the repository root:

```bash
python -m py_compile Final_version/main.py Final_version/LLM_Module.py Final_version/Extracion.py Final_version/Evalueation.py
```

Run unit tests from inside the code folder:

```bash
cd Final_version
python -m unittest discover -s tests
```

## Dry Run

A dry run checks configuration and planned stages without loading data:

```bash
cd Final_version
python main.py --dry-run --config configs/aml_hi_small_mac_smoke.json --device cpu --output-dir outputs/dry_run
```

## Example Local Run

This runs the pipeline without LLM inference or evaluation, useful for checking data loading, training, and extraction first:

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

Remove `--skip-llm --skip-eval` and set `--llms` when you want full LLM inference.

## Slurm Runs

The `.job` files are written for a project layout where this repository root contains `Final_version/main.py`.

Example:

```bash
PROJECT_DIR=/scratch-shared/$USER/Explanation-of-GNN-using-LLM sbatch reconstruction_smoke.job
```

Adjust `PROJECT_DIR`, `CONDA_ENV`, dataset paths, and resource settings to match your cluster account.

## Troubleshooting

If `main.py` fails with `ModuleNotFoundError: No module named 'torch_geometric'`, install the dependencies in `requirements.txt` in the active environment.

On some macOS Python environments, mixed scientific packages can trigger an OpenMP `libomp.dylib already initialized` error. A clean Conda environment is preferred. For a temporary local smoke check, prefix the command with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python Final_version/main.py --dry-run --config Final_version/configs/aml_hi_small_mac_smoke.json
```

## Clean Git Upload

Generated outputs, cached models, datasets, notebooks, plots, and local environment files are ignored by `.gitignore`.

To initialize a clean repository:

```bash
git init
git add .
git commit -m "Initial clean thesis codebase"
```

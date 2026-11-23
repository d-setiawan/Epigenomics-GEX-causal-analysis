# Environments

Choose the environment for the analysis you want to reproduce. The November 22 handoff tested the exact-model CPU environment from a fresh installation. It used the existing Linux environment for empirical preprocessing. [Transfer guide](../TRANSFER.md) · [Verification record](../reproduce/VERIFICATION.md)

## Restore from a Git checkout

**Exact-model counts on CPU.** From the repository root, use Python 3.11. The verified export used 3.11.14. The CPU requirements include the numerical libraries, test runner and HDF5 support for the RNA barcode restorer. The independent PyTorch gradient audit used the research environment. [CPU requirements](requirements-replay-cpu.txt)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r envs/requirements-replay-cpu.txt
python -m pytest -q pipelines/simulation/integration_recoverability/tests/test_count_transport*.py
```

**Empirical Linux x86-64 environment.** The explicit Conda specification records package builds. Apply the pip overlay after creating the environment to restore the working numerical stack, including NumPy 2.2.6. The overlay records versions outside some upstream declared requirements. A fresh installation of the full Linux lock and GPU training remain unverified by the handoff. [Conda build list](scrna_r-linux-64.explicit.txt) · [Pip overlay](requirements-scrna_r-overlay.txt) · [Recorded CUDA runtime](../reproduce/records/cuda-13.0.96.json)

```bash
conda create -n scrna_replay --file envs/scrna_r-linux-64.explicit.txt
conda activate scrna_replay
python -m pip install --no-deps -r envs/requirements-scrna_r-overlay.txt
```

## Which environment runs which stage

| Workload | Environment and evidence |
| --- | --- |
| Count-transport generation, fitting and evaluation | Python 3.11 CPU environment above. The original full fit used one BLAS thread per worker. [Replay stages](../TRANSFER.md#exact-model-simulation-on-cpu) |
| CUT&Tag preprocessing, scGLUE, empirical per-gene runs | Primary `scrna_r` environment. Use the Linux specification above and consult the [empirical pipeline](../pipelines/empirical/README.md). |
| Original integration-recoverability suite | `scrna_r` for its simulator, adapters and RCIT evaluation. The [October dependency record](requirements-integration-recoverability-2026-10-04.txt) lists study-specific versions. [Execution ledger](../pipelines/simulation/integration_recoverability/IMPLEMENTATION_STATUS.md) |
| scConfluence on cached simulations and scMTR-seq S1 | Isolated overlay on `scrna_r` for the native fits, then the primary interpreter for scoring. [Overlay pins](requirements-scconfluence-pilot-2026-11-01.txt) · [Study guide](../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md) |
| Tetrad and alternative causal learners | `causaldisco` with Python 3.11 and OpenJDK 21. [Conda specification](environment-causaldisco.yml) · [Recorded causal packages and py-tetrad commit](requirements-causaldisco-causal.txt) |
| Earlier CLI and sensitivity analyses | [Base package inventory](requirements-base-full.txt) and [secondary cvbench specification](environment-cvbench.yml) |

## Earlier environment records

The July [scrna_r YAML](environment-scrna_r.yml) and [pip freeze](requirements-scrna_r.txt) describe the earlier empirical stack. The freeze contains machine-local package URLs. Use the November build list and overlay for the transferred Linux environment. The separate [scConfluence pilot overlay](requirements-scconfluence-pilot-2026-11-01.txt) uses NumPy 1.26.4 and AnnData 0.10.9 and inherits PyTorch from its parent environment. Its runtime directory is generated locally.

For Tetrad methods, activate the causal environment and point JPype to its JDK. The YAML captures the requested environment and the package inventory records the original py-tetrad revision.

```bash
conda env create -n causaldisco -f envs/environment-causaldisco.yml
conda activate causaldisco
export JAVA_HOME="$CONDA_PREFIX"
```

For data restoration, simulated world replay and per-study requirements, follow the commands from the repository root. [Transfer guide](../TRANSFER.md) · [Integration-recoverability guide](../pipelines/simulation/integration_recoverability/README.md)

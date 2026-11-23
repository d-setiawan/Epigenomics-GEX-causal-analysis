# Environments

Pinned environment specs for reproducibility. Captured 2026-07-08.

## Restore from a Git checkout

For the exact-model CPU simulation, install
[requirements-replay-cpu.txt](requirements-replay-cpu.txt) in a Python 3.11
virtual environment. See the [transfer guide](../TRANSFER.md) for generation
and fitting commands.

For the original Linux x86-64 empirical environment, the Conda package-build
lock and the pip overlay are captured separately. This preserves the pip
NumPy overlay on the Conda stack. Use a compatible NVIDIA driver for GPU fits.

```bash
conda create -n scrna_replay --file envs/scrna_r-linux-64.explicit.txt
conda activate scrna_replay
python -m pip install --no-deps -r envs/requirements-scrna_r-overlay.txt
```

These files record the working environment, including its pip overlays.
Some of those versions are outside the upstream requirements declared by
SnapATAC2 and igraph consumers. The CPU replay uses a separate set of pins.
The remaining specifications below document earlier environments.

## Which environment runs which stage

| Stage | Env | Why |
| --- | --- | --- |
| scGLUE integration (`pipelines/empirical/integration/`) | **`scrna_r`** | scglue 0.2.3, scanpy, snapatac2, torch |
| Causal discovery graph runs (`causal_cli.py graph/sweep/dagma-sweep`) | **`scrna_r`** | causal-learn 0.1.4.5, dagma (pinned git commit) |
| CUT&Tag preprocessing (`pipelines/empirical/data/scripts/`) | **`scrna_r`** | snapatac2 2.8.0 (fixed-bin / peak matrices) |
| CLI orchestration, elastic-net screen, external support, evaluate/plot | **base** | only needs pandas / scikit-learn / numpy / matplotlib / networkx / statsmodels |
| (secondary causal env, newer causal-learn 0.1.4.7) | `cvbench` | optional cross-check |
| **Phase D causal-discovery method zoo** (GFCI/RFCI/FCI-Max, RCD/CAM-UV/DirectLiNGAM, GIN, NOTEARS, Causal-Dropout CI) | **`causaldisco`** | one centralized env: py-tetrad + JDK 21, lingam, causal-learn 0.1.4.7, gcastle |

**`causaldisco` needs `JAVA_HOME` set to run the Tetrad (Java) methods:**
```bash
conda activate causaldisco
export JAVA_HOME="$CONDA_PREFIX"
python your_script.py
```
Verified working: GFCI/FCI/RCD/CAM-UV all run; JPype drives the in-env JDK 21 (system Java is 11, too
old for current Tetrad). SEACells was deliberately NOT installed here — it force-upgrades numpy and
would break the stack; metacells use native scanpy in `scrna_r` instead.

Pass the interpreter explicitly on the CLI, e.g.
`python3 pipelines/empirical/analysis/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python graph ...`.
The base env is `/home/dgsetiawan/anaconda3/bin/python`.

## Files

- **`environment-scrna_r.yml`** — the **recreation artifact** for the primary env
  (`conda env export --no-builds`). Recreate with:
  ```bash
  conda env create -n scrna_r -f envs/environment-scrna_r.yml
  ```
- `requirements-scrna_r.txt` — `pip freeze` of `scrna_r`. A **version record**, not directly
  pip-installable: conda-provided packages appear as `name @ file:///…` local paths. Use it to read
  exact versions (e.g. `causal-learn==0.1.4.5`, `scglue==0.2.3`, `snapatac2==2.8.0`, dagma pinned to a
  git commit); use the YAML to actually rebuild.
- `environment-cvbench.yml` — secondary causal env (causal-learn 0.1.4.7, scanpy 1.11.5).
- `requirements-base-full.txt` — full `pip freeze` of the base env (CLI/support/eval tooling).

Key pinned versions (scrna_r): python 3.11, `scglue==0.2.3`, `causal-learn==0.1.4.5`,
`dagma @ git+…@0886168`, `scanpy` 1.10.x, `anndata==0.9.2`, `snapatac2==2.8.0`, `torch` 2.x,
numpy 1.26.4.

## Installed for Phase D (2026-07-09): `causaldisco`

- **`environment-causaldisco.yml`** — recreation artifact (`conda env export --from-history`). Recreate:
  ```bash
  mamba env create -n causaldisco -f envs/environment-causaldisco.yml
  ```
- `requirements-causaldisco-causal.txt` — exact causal-package versions (causal-learn 0.1.4.7,
  lingam 1.12.2, gcastle 1.0.4, JPype1 1.7.1, py-tetrad pinned to a git commit).
- Covers all three method directions: **Java suite** (py-tetrad → GFCI/RFCI/FCI-Max/BOSS-FCI),
  **non-sufficiency functional** (lingam → RCD/CAM-UV/ParceLiNGAM), **latent-structure** (causal-learn
  → GIN), and the **Causal Dropout Model** (test-wise-deletion CI test `zerodel_fisherz`,
  from github.com/MarkDana/scRNA-Causal-Dropout — a drop-in CI test, not yet wired into `causal_cli`).

## Still not installed

- Phase C aggregation extras: **mcRigor** (metacell homogeneity QC), an OT library (POT/`ot`) for
  metacell-OT matching. (SEACells intentionally avoided — see the JAVA_HOME note above re numpy.)
- **RCoT** (fast KCI surrogate): causal-learn's `fastkci`/`rcit` cover this; a dedicated R RCoT is optional.

When new envs are installed, capture them the same way and add a row to the table above.

## Integration-recoverability benchmark (2026-10-04)

The new simulator, both integration adapters, and RCIT evaluator run in
`scrna_r`. The older environment exports above remain historical records;
they do not describe every current package. A focused current version record is
[requirements-integration-recoverability-2026-10-04.txt](requirements-integration-recoverability-2026-10-04.txt).
It is not a complete Conda/CUDA recreation lock.

This implementation added `scikit-misc==0.5.2` for native Seurat-v3 feature
selection and `tensorboard==2.20.0` with its missing reader dependencies for
validating native scGLUE training histories. Installation used pinned wheels
without dependency upgrades; existing numerical/training packages were not
changed. `tensorboardX` alone writes logs but does not provide the required
TensorBoard event reader. The first scGLUE fit completed training before that
missing-reader check failed; its native snapshots and original failure record
are preserved for audited export without retraining.

Each benchmark artifact records the actual applicable versions and source
hashes. See the [execution record](../pipelines/simulation/integration_recoverability/IMPLEMENTATION_STATUS.md)
for scientific scope, results, and remaining work.

## scConfluence pilot overlay (2026-11-01)

The [scConfluence/scMTR-seq study](../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md)
uses the workspace-local `envs/runtime/scconfluence_pilot_20261101` virtualenv
with `scrna_r` system site packages. Its
[minimal requirements](requirements-scconfluence-pilot-2026-11-01.txt) install
local NumPy, AnnData, MuData, POT and GeomLoss versions while retaining the
shared PyTorch/Lightning installation. The shared environment was not changed.
The pinned native source is vendored with its license beside the adapter.

All six scientific fits used this overlay and the verified process-local CUDA
runtime recorded in the study's execution plan. Causal and state scoring use
the original `scrna_r` interpreter; the overlay is not a general replacement
for that environment. Package/runtime receipts and the Python-version
compatibility deviation are documented in the study and adapter READMEs.

## Regenerating these files

```bash
conda env export -n scrna_r  --no-builds > envs/environment-scrna_r.yml
conda env export -n cvbench   --no-builds > envs/environment-cvbench.yml
/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python -m pip freeze > envs/requirements-scrna_r.txt
/home/dgsetiawan/anaconda3/bin/python -m pip freeze > envs/requirements-base-full.txt
```

# Environments

Pinned environment specs for reproducibility. Captured 2026-07-08.

## Which environment runs which stage

| Stage | Env | Why |
| --- | --- | --- |
| scGLUE integration (`Integration/`) | **`scrna_r`** | scglue 0.2.3, scanpy, snapatac2, torch |
| Causal discovery graph runs (`causal_cli.py graph/sweep/dagma-sweep`) | **`scrna_r`** | causal-learn 0.1.4.5, dagma (pinned git commit) |
| CUT&Tag preprocessing (`Data/scripts/`) | **`scrna_r`** | snapatac2 2.8.0 (fixed-bin / peak matrices) |
| CLI orchestration, elastic-net screen, external support, evaluate/plot | **base** | only needs pandas / scikit-learn / numpy / matplotlib / networkx / statsmodels |
| (secondary causal env, newer causal-learn 0.1.4.7) | `cvbench` | optional cross-check |
| **Phase D causal-discovery method zoo** (GFCI/RFCI/FCI-Max, RCD/CAM-UV/DirectLiNGAM, GIN, NOTEARS, Causal-Dropout CI) | **`causaldisco`** | one centralized env: py-tetrad + JDK 21, lingam, causal-learn 0.1.4.7, gcastle |

**`causaldisco` needs `JAVA_HOME` set to run the Tetrad (Java) methods:**
```bash
export JAVA_HOME=/home/dgsetiawan/anaconda3/envs/causaldisco   # JDK 21 lives inside the env
/home/dgsetiawan/anaconda3/envs/causaldisco/bin/python your_script.py
```
Verified working: GFCI/FCI/RCD/CAM-UV all run; JPype drives the in-env JDK 21 (system Java is 11, too
old for current Tetrad). SEACells was deliberately NOT installed here — it force-upgrades numpy and
would break the stack; metacells use native scanpy in `scrna_r` instead.

Pass the interpreter explicitly on the CLI, e.g.
`python3 CausalDiscovery/scripts/causal_cli.py --python-bin /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python graph ...`.
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

## Regenerating these files

```bash
conda env export -n scrna_r  --no-builds > envs/environment-scrna_r.yml
conda env export -n cvbench   --no-builds > envs/environment-cvbench.yml
/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python -m pip freeze > envs/requirements-scrna_r.txt
/home/dgsetiawan/anaconda3/bin/python -m pip freeze > envs/requirements-base-full.txt
```

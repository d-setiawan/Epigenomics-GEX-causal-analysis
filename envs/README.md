# Environments

Pinned environment specs for reproducibility. Captured 2026-07-08.

## Which environment runs which stage

| Stage | Env | Why |
| --- | --- | --- |
| scGLUE integration (`integration/`) | **`scrna_r`** | scglue 0.2.3, scanpy, snapatac2, torch |
| Causal discovery graph runs (`causal_cli.py graph/sweep/dagma-sweep`) | **`scrna_r`** | causal-learn 0.1.4.5, dagma (pinned git commit) |
| CUT&Tag preprocessing (`sc_scripts/`) | **`scrna_r`** | snapatac2 2.8.0 (fixed-bin / peak matrices) |
| CLI orchestration, elastic-net screen, external support, evaluate/plot | **base** | only needs pandas / scikit-learn / numpy / matplotlib / networkx / statsmodels |
| (secondary causal env, newer causal-learn 0.1.4.7) | `cvbench` | optional cross-check |

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

## Not yet installed — planned for the method upgrades (ROADMAP Phase C/D)

These are decided in `ROADMAP.md` but not installed anywhere yet. Install into a **new dedicated env**
(recommended, to avoid perturbing `scrna_r`) rather than the pinned envs above:

- Phase D causal-discovery: **py-tetrad** (needs JPype1 + a JDK for GFCI/RFCI/fGES + bootstrap),
  **lingam** (RCD/CAM-UV/DirectLiNGAM), **RCIT/RCoT** (fast KCI surrogate; R pkg or a Python port),
  **gcastle** (NOTEARS/GOLEM sensitivity).
- Phase C aggregation: **SEACells**, **mcRigor**, an OT library (e.g. POT / `ot`).
- Phase B validation: **pyliftover** (or the UCSC `liftOver` binary) for hg19↔hg38 (see `scripts/liftover.py`).

When any of these are installed and used, capture them the same way (`conda env export --no-builds`
into `envs/environment-<name>.yml`) and add a row to the table above.

## Regenerating these files

```bash
conda env export -n scrna_r  --no-builds > envs/environment-scrna_r.yml
conda env export -n cvbench   --no-builds > envs/environment-cvbench.yml
/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python -m pip freeze > envs/requirements-scrna_r.txt
/home/dgsetiawan/anaconda3/bin/python -m pip freeze > envs/requirements-base-full.txt
```

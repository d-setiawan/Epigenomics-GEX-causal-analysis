# PT-SEM reproducibility package

This repository contains the code, pinned environments, processed inputs,
experiment records, and paper outputs for *Causal DAG Identification for Count
Data via Poisson-Thinning Structural Equation Models*.

## Reproduce the paper outputs

Python 3.11 is required. The simulation and NBA analyses use separate pinned
NumPy versions, so create both environments:

```powershell
python -m venv .venv-simulation
.venv-simulation\Scripts\python.exe -m pip install -r requirements-simulation.txt

python -m venv .venv-nba
.venv-nba\Scripts\python.exe -m pip install -r requirements-nba.txt
```

Then rebuild and verify all six figures and the NBA table:

```powershell
.venv-simulation\Scripts\python.exe reproduce.py --nba-python .venv-nba\Scripts\python.exe
```

The command prints seven `PASS` lines and writes:

- `outputs/figures/figure1_all_poisson.pdf` through
  `outputs/figures/figure6_nba_season_estimates.pdf`
- `outputs/tables/nba_structural_recovery.csv`

The normal rebuild renders the figures and recomputes the table from committed
scientific result records; it does not copy pre-existing PDFs. PDF content
streams are checked against `manifests/PAPER_OBJECTS.csv`, while the table is
checked byte for byte.

The source, configurations, seeds, processed data, fits, and baseline records
needed to inspect or rerun the analyses are included. A one-command full refit
is not claimed for this release, and `reproduce.py` intentionally has no
`--refit` option. See [docs/REPRODUCING.md](docs/REPRODUCING.md) for the closure
and individual analysis programs.

## Data and external baselines

The ten processed NBA season files are committed under
`data/nba/processed_team_quarter/`. The upstream play-by-play source, hashes,
and citation are documented in [docs/DATA.md](docs/DATA.md).

PB-SCM and PB-SCM-PGF are fetched at pinned commits by
`scripts/fetch_external_baselines.py`. Their upstream repositories do not
provide redistribution licenses at those commits, so their source is not
included here.

The reported Proposed DP-BIC NBA result is 1.000 for skeleton and directed
precision, recall, and F1; the complete prespecified DAG is recovered in every
season.

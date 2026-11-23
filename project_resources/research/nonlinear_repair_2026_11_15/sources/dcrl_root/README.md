# Discrete Causal Representation Learning

## Installation

After unzipping the archive, open a terminal in the extracted folder and run

```bash
conda env create -f environment.yml
conda activate dcrl
python -m pip install --no-deps -e .
```

Run these commands from the repository root after cloning or extracting the
archive. The editable install makes `dcrl` available in Python.

For the exact Python package versions used in validation, use a Python 3.12
virtual environment instead:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproducible.txt
python -m pip install --no-deps -e .
```

The pip-only setup additionally needs the Graphviz `dot` executable on PATH
for seesaw graph PNGs. The Conda environment above supplies Graphviz.

## Run

To reproduce the Section 5 simulation table with the supplied settings:

```bash
python experiments/run_section5.py
```

This runs 90 settings (5 graphs, 2 Q matrices, 3 distributions and 3 sample
sizes), with 500 independent datasets per setting. Results are saved under
`results/section5`. Use the pinned environment above for reproduction.

### Section 5 settings

`configs/section5.json` contains the default experiment plan. The original
configuration is retained for 27 of the 30 graph/Q/distribution groups.
Three groups use the parameters tested in the subsequent independent-seed
validation:

| Graph | Q | Distribution | Outer iterations |
| --- | --- | --- | --- |
| Model-7 | 1 | Poisson | 400 |
| Model-8 | 2 | Lognormal | 100 |
| Model-13 | 1 | Lognormal | 100 |

For these three groups, `max_iter=min_outer_iter`, `inner_max_iter=200`,
`gain_numerator=1`, `gain_offset=10`, `C=3`, `kappa=0.01`,
`tau_multiplier=2`, and the graph structure prior is `0.01`. They use the
same analytic-objective/vectorized-Gibbs implementation as the validation.
The other 27 groups retain the original estimator, stopping rule, parameters
and seeds. These are fixed group-specific settings, not automatic tuning.

All groups use column-sum orientation, latent-sample alignment for evaluation,
and one latent-resampling graph fit with sample size N and ESS=1. Data are
generated independently at N=3000, 5000 and 7000; smaller datasets and fitted
estimates are not reused at larger N. The signal setting remains
`restore_minimum` with multiplier 1.25 throughout.

For parallel repetitions, a single setting, or an interrupted run:

```bash
python experiments/run_section5.py --jobs 4
python experiments/run_section5.py --task-id 39
python experiments/run_section5.py --resume
```

Choose an empty output directory for a new run (`--output PATH`). Resume
requires the same plan, source and numerical environment, and retains recorded
failures. On a cluster, run this through the scheduler. A Slurm array indexed
0–89 can pass `$SLURM_ARRAY_TASK_ID` to `--task-id`; allocate one CPU per
`--jobs` worker. Run the combined summary after all settings finish:

```bash
python experiments/run_section5.py --summary-only
```

`results/section5/overview.txt` shows overall SHD (`matched_full_shd`) by sample
size and counts the decreasing/nonincreasing groups. `section5_metrics.csv`
also includes Q, composite graph, latent CPDAG, skeleton, v-structure and
compelled directed-edge exact recovery, latent SHD, Monte Carlo SEs and
valid-dataset counts. Each setting retains its full `summary.json` and
replicate outputs. Missing or failed replicates prevent a complete trend
verdict. A decreasing sample mean is not a test of statistical significance.

See [the Section 5 reproduction notes](docs/section5-reproduction.md) for seed
formulas, the validation results and implementation details. The original
entry points and general diagnostic plans below remain available.

## Revision options and reproducible diagnostics

The general CLI/API default profile is `legacy`; the Section 5 plan selects
`revision`. The `revision` profile changes only the
default Bernoulli initialization clipping value from `1e-5` to `0.001`.
The Varimax cutoff constant `2.5`, penalty, coefficient bounds, gain and
stopping rules stay the same. This choice was tested in strong-signal settings.

### Adjust initialization for your data

Set either constant through the CLI, Python API or JSON plan. Explicit values
override profile defaults.

| Parameter | Meaning | Default |
| --- | --- | --- |
| `init_cutoff_c` | Positive c in the rotated-loading cutoff `1/(c*sqrt(J))` | `2.5` |
| `epsilon` | For Bernoulli, clips the low-rank probability estimate to `[epsilon, 1-epsilon]` before logit | `1e-5` in `legacy`; `0.001` in `revision` |

Larger c lowers the cutoff and retains more rotated loadings at initialization;
the final Q estimate need not vary monotonically with c. Bernoulli requires
`0 < epsilon < 0.5`.
For Poisson, epsilon is a lower bound on the reconstructed log(X+1) scale
(default `1e-50`); the Lognormal initializer does not use epsilon.

For example, these flags explicitly choose c=2.5 and epsilon=0.001:

```bash
python experiments/run_parallel.py --start 1 --end 11 --n 3000 --k 10 --dag_type Tree --distribution Bernoulli --init_cutoff_c 2.5 --epsilon 0.001 --output_path results/custom_initialization.csv
```

To fit your own observed N-by-J matrix `X` using K latent variables:

```python
from dcrl.config import ExperimentConfig
from dcrl.fitting import fit_observed

cfg=ExperimentConfig(N=X.shape[0], J=X.shape[1], K=K,
                     distribution="Bernoulli",
                     init_cutoff_c=2.5, epsilon=0.001)
fit=fit_observed(X, cfg)  # final column-sum orientation is mandatory
```

The original spectral initializer requires J>=3K and min(N,J)>=K. In a study
JSON, put these values in `model` to fix them, or use, for example,
`"grid": {"init_cutoff_c": [1.5, 2.5, 3.5], "epsilon": [0.0001, 0.001, 0.01]}`
for a Cartesian sensitivity study. These are example grids, not automatic
tuning rules. Study and CLI output manifests record the settings used.

### Graph discovery and diagnostic studies

Graph discovery has two options:

* `resampling`: multinomial latent resampling, as in the existing pipeline.
* `plugin`: BDeu evaluated at fractional sufficient counts `m * p_hat(z)`.
  It uses unrounded counts directly and is deterministic for fixed inputs
  and software.

To save initial/final estimates, objectives, optimizer diagnostics, exact
graph truth and Monte Carlo summaries:

```bash
python experiments/run_study.py --config configs/reviewer2_smoke.json --output results/reviewer2_smoke --jobs 2
python experiments/run_study.py --config configs/revision_model13_bernoulli.json --output results/model13_revision --jobs 4
```

The first command is a small three-distribution integration check. The second
compares both graph methods at N=3000/7000 with restored Q2 coefficients
multiplied by 1.25/1.5, with 20 repetitions per setting.
`configs/reviewer2_diagnostics.json` adds random/spectral starts, oracle
stages, multiple graph draws and ESSs, and takes substantially longer.
Set `repetitions` and freeze the plan before a larger study.

The supplied plans use `evaluation_alignment="latent_sample"` to match latent
coordinates from the final Gibbs sample. Q-support-based reference errors
are saved separately. Matching is used only for evaluation.

These plans cover general simulated measurement models. Final fitted outputs
are now column-sum oriented in the main simulation, TIMSS and seesaw entry
points; raw internal optimization traces are unchanged. Exactly zero column
sums remain unresolved and are explicitly reported when available. `run_timss.py` uses one graph-sampling draw; the newer TIMSS
200-candidate graph-constrained BIC selection and bootstrap analyses are not
included. This release covers only the experiments documented here.

The original simulation entry point also accepts explicit options:

```bash
python experiments/run_parallel.py --start 1 --end 11 --n 3000 --k 13 --dag_type Model-13 --distribution Bernoulli --profile revision --graph_method plugin --ess 1 --signal_scheme restore_minimum --signal_multiplier 1.25 --output_path results/model13_plugin.csv
```

`run_parallel.py` retains its historical three-size SHD output and sampled
reference graph. Use `run_study.py` for the revision's full diagnostics and
exact generator CPDAG. See [the revision-code guide](docs/revision-code.md)
for APIs, all tuning options, orientation, oracle definitions, and limitations.

To apply either graph method to a saved fit, including TIMSS or seesaw:

```bash
python experiments/learn_graph.py --p_path data/TIMSS/timss_refit.json --k 7 --effective_n 124 --method plugin --ess 1 --output results/timss_plugin
```

This graph-only command does not refit the measurement model or change the
default TIMSS analysis. `--effective_n` is the count scale m, whereas ESS is
the BDeu prior's equivalent sample size; they are different parameters.

## Original entry points

For the main simulation pipeline,

```bash
python experiments/run_parallel.py --start 1 --end 11 --n 4000 --k 10 --dag_type Tree --distribution Lognormal
```

For the seesaw experiment,

```bash
python experiments/run_seesaw.py --N 10000 --K 4 --seed 123 --bank_size 2000 --jitter 0.001 --noisy_z3 --p_up_if11 0.8 --p_up_else 0.2 --p_ball4_if_up 0.99 --p_ball4_if_down 0
```

For the seesaw experiment using the pre-generated dataset used in the paper, which reproduces the exact seesaw results reported in the paper,

```bash
python experiments/run_seesaw_from_dataset.py --dataset_path data/seesaw/seesaw_dataset.npz
```

For TIMSS, fit the original student responses and then use multinomial latent
resampling. The fitting seed is 48 (selected by likelihood in the prior
initialization comparison); the separate graph-sampling seed is 1. The default
ratio is `f(N)/N = 0.2`, selected by the lowest BIC in the complete graph-scale
comparison using multinomial sampling.

```bash
python experiments/run_timss.py --fn_ratio 0.2 --seed 48 --graph_seed 1
```

To reproduce the graph from the saved **fresh** latent distribution used in
the revised comparison:

```bash
python experiments/run_timss_from_saved_p.py --fn_ratio 0.2 --seed 1
```

Both commands use actual multinomial samples, with
`f(N) = max(1, floor(fn_ratio * N))`. With `N=620` and ratio `0.2`, they draw
124 latent observations and return a seven-edge CPDAG for the supplied fit
and seed. Ratio `0.5` gives ten edges and a higher BIC. See the
[complete latent-law and graph-scale comparison](docs/timss-comparison.md)
for all examined settings, saved fitted parameters, and score verification.

`--no_plot` skips PNG generation. Plotting uses Matplotlib. The JSON
summary and numeric graph are always written. `--N` in the saved-fit command
means original students, not the number of latent resamples. The new default
checkpoint is `data/TIMSS/timss_refit.json`, which includes its fit provenance
and input hashes. The historical `TIMSS_estimate_p.npz` is retained for explicit
legacy comparisons; it is not the revised default.

The fast TIMSS refit solves the same bounded fixed-Q lognormal item model by
weighted least squares and profiled variance updates, using observed cells.
Its reported likelihood includes the Gaussian normalizing constant on clipped
log times. It is a saturated latent-law fit; attaching a graph afterward does
not turn that likelihood into a graph-constrained likelihood or reduce its
parameter count. Graph-constrained BIC comparisons require separate refitting.

The original ball-image and `run_parallel.py` defaults are unchanged. Sparse
initialization changes storage, not their statistical settings. The Section 5
reproduction settings are listed above. The sampling
condition needed for ratios below one is explained in
[the consistency note](docs/resampling-consistency.md).

## Output

The Section 5 entry point writes per-setting results and the combined tables
to `results/section5/`.

The legacy simulation entry point writes to `results/`, for example `results/results_4000_Lognormal_Tree_2.txt` with columns `iter,shd_val,shd2_val,shd3_val`.

The seesaw script writes outputs to `results/seesaw/`, including `p_hat.npy`, `B_hat.npy`, `latent_graph_estimated.png`, `latent_graph_estimated.npy`, `summary.json`, and heatmaps under `results/seesaw/heatmaps/`.

The TIMSS script writes outputs to `results/timss/`, including `p_hat.npy`, `B_hat.npy`, `gamma_hat.npy`, `latent_samples_from_phat.npy`, `latent_counts.npy`, `latent_graph_estimated.npy`,
`latent_graph_estimated.png`, and `summary.json`. Summaries record the actual
resample size, seeds, input/sample hashes, and numerical package versions.

## File map

* `src/dcrl/utils.py`

  * generic helpers such as `binary`, `TLP`, `thres`, `sigmoid`
* `src/dcrl/config.py`

  * experiment configuration and default hyperparameter construction
* `src/dcrl/latent_models.py`

  * latent DAG / latent-state generators
  * `Markov`, `Tree`, `DiverseTree`, `Model-7`, `Model-8`, `Model-13`, `Model-16`
* `src/dcrl/data_generator.py`

  * `GenerateData` class
  * creates `Q`, `B`, `gamma`, latent `A`, and observed `X`
* `src/dcrl/initialization.py`

  * spectral / varimax initialization of `p`, `B`, `gamma`, `A`
* `src/dcrl/evaluation.py`

  * graph stitching and SHD code
  * resampling from estimated `p_hat` and evaluating recovered graphs
* `src/dcrl/estimator.py`

  * `DAGEstimator`
  * `PSAEM` and the main `estimate` method
* `experiments/run_section5.py`

  * Section 5 reproduction entry point; defaults in `configs/section5.json`
* `src/dcrl/section5.py`

  * task plans and sample-size comparison tables
* `src/dcrl/section5_backend/`

  * validated fitting implementation for the three overridden groups
* `src/dcrl/runner.py`

  * `ParallelDAGEstimator`
  * parallel streaming experiment runner with file locking
* `src/dcrl/seesaw/dataset.py`

  * seesaw image generation, masks, pooled observations, and latent variables
* `src/dcrl/seesaw/init.py`

  * custom initialization for the seesaw Bernoulli setting
* `src/dcrl/seesaw/psaem.py`

  * Bernoulli PSAEM for the seesaw experiment
* `src/dcrl/seesaw/plotting.py`

  * heatmap visualization for the estimated coefficient matrix `\\hat B`
* `src/dcrl/timss/em.py`

  * lognormal EM algorithm with missing data for the TIMSS experiment
* `experiments/truth_graph.py`

  * builds truth graphs for each named DAG family
* `experiments/run_parallel.py`

  * the executable CLI script for the main simulation pipeline
* `experiments/run_seesaw.py`

  * the executable CLI script for the seesaw Bernoulli experiment
* `experiments/run_seesaw_from_dataset.py`

  * the executable CLI script that uses the pre-generated seesaw dataset used in the paper and reproduces the exact seesaw results reported in the paper
* `experiments/run_timss.py`

  * the executable CLI script for the TIMSS experiment
* `experiments/run_timss_from_saved_p.py`

  * the saved-fit TIMSS CLI, using multinomial resampling from the fresh comparison fit
* `data/TIMSS/Q.csv`

  * Q-matrix for the TIMSS experiment
* `data/TIMSS/time.csv`

  * observed TIMSS data matrix used by `run_timss.py`
* `data/TIMSS/TIMSS_estimate_p.npz`

  * historical latent distribution, retained for explicit legacy comparisons

## Sparse probabilities in the main Gibbs–SAEM pipeline

`DAGEstimator.PSAEM()` now returns `p_hat` as a `SparseLatentLaw`. Initialization,
Gibbs updates, finalization and graph-evaluation resampling no longer allocate
arrays over all `2**K` states. The measurement estimates and Gibbs count updates
are unchanged. Integer state keys can exceed 64 bits.

```python
from dcrl import SparseLatentLaw

# After fitting an estimator:
p_hat = estimator.p_hat
p_hat.save_npz("p_hat_sparse.npz")
p_hat = SparseLatentLaw.load_npz("p_hat_sparse.npz")
latent_samples = p_hat.sample(1000)
probability = p_hat.probability(5)  # encoded binary configuration
```

The law stores encountered-state masses and, when needed, one aggregate uniform
mixture weight. Positive `L_final` preserves the previous public implementation's
rule of adding `L_final / U` to every state, where `U` is the number of unvisited
states. A zero-count law retains the previous uniform fallback. Sampling the
uniform component draws independent fair bits, without listing configurations.

For `S` stored states and `M` requested samples, finalization takes `O(S)` work
and storage, and resampling takes `O(S + M*K)` work and storage under machine-word
key accounting. Larger keys require `O(K*S)` bits and additional polynomial bit
operations. This removes mandatory state-space enumeration; it does not guarantee
polynomial convergence or polynomial worst-case unrestricted GES search.

Legacy consumers can explicitly request `DAGEstimator(..., dense_p_output=True)`
or call `p_hat.to_dense()`; these optional conversions cost `O(2**K)`. The separate
seesaw and TIMSS estimators retain their existing interfaces. The main simulation
runner and its GES evaluation accept the sparse result directly.


The ball-image initializer also counts encountered states without enumerating
all configurations when called by its PSAEM routine. Its final dense output
and the existing ball-image sampling settings are preserved.

## Verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests -q
```

The regression tests compare the initialization and main Gibbs–SAEM estimates
with public commit `190f198f9134655cca7c517905cdf61a994da6a8`, test sparse state
keys beyond 64 bits, and verify multinomial resampling and repeated TIMSS runs.
Use a normal clone with that commit in its history (CI fetches full history).

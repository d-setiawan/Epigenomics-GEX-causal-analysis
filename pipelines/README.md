# Pipelines

The project has two workflow families. The empirical workflow follows existing
biological data from preprocessing through validation. The simulation workflow
contains controlled aggregation studies, cached-world multimodal benchmarks,
and the integration-recoverability study.

- [Empirical pipeline](empirical/README.md): data → integration → analysis → evaluation.
- [Simulation pipelines](simulation/README.md): simulation tracks and their own
  inputs, scripts, runs, scoring, and results.

Each pipeline owns its generated outputs. Shared research notes, cross-project
reports, and archival material live in [`project_resources/`](../project_resources/README.md).

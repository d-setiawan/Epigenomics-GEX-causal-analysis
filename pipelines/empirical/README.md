# Empirical pipeline

This is the real-data workflow, kept in processing order:

1. [Data](data/README.md) — source inputs and preprocessing.
2. [Integration](integration/README.md) — method implementations, input
   manifests, isolated integration runs, and comparisons.
3. [Analysis](analysis/README.md) — pseudo-pairing, datasets, and causal graph
   discovery.
4. [Evaluation](evaluation/README.md) — empirical graph checks and external
   biological evidence.

The data stage is the source of truth for shared empirical inputs. Each stage
owns its derived files; integration run outputs are separated by run ID under
`integration/runs/`.

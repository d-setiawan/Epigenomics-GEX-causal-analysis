# Simulation pipelines

Simulation work has its own inputs, discovery code, scoring, run outputs, and
results. This keeps simulated-world evidence separate from validation against
the empirical biological data.

- [Aggregation identifiability](aggregation_identifiability/README.md) —
  controlled studies of aggregation and cross-mark identifiability.
- [Unpaired multimodal benchmark](unpaired_multimodal/README.md) — cached
  formal worlds, CA-BOSS discovery, truth-aware scoring, and formal runs.
- [Integration recoverability](integration_recoverability/README.md) — the
  integration benchmark before aggregation and the completed
  [exact-model count-source search](integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md).
  See the original benchmark's
  [implementation record](integration_recoverability/IMPLEMENTATION_STATUS.md)
  for executed stages and pending results.

Within a simulation track, discovery artifacts and truth-aware scoring are
kept distinct. Historical run files were moved intact and retain their original
run identifiers.

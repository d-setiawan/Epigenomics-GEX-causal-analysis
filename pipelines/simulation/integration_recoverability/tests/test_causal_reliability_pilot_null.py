"""Ensure exploratory null frequencies do not hide failed or within-assay graphs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import causal_reliability_pilot_null as null


def test_failed_graphs_stay_outside_null_denominators_and_within_edges_are_not_cross():
    candidates = [{"name": "a", "assay": "RNA"}, {"name": "b", "assay": "RNA"}, {"name": "c", "assay": "H3K27ac"}]
    edge = lambda a, b: {"node1": a, "node2": b, "endpoint1": "CIRCLE", "endpoint2": "CIRCLE"}
    results = [{"seed": 10000, "discovery": {"status": "ok", "edges": [edge("a", "c"), edge("a", "b")]}},
               {"seed": 10001, "discovery": {"status": "failed", "edges": None}},
               {"seed": 10002, "discovery": {"status": "ok", "edges": []}}]
    summary = null.summarize(results, candidates)
    assert summary["n_attempted"] == 3 and summary["n_successful"] == 2 and summary["n_failed"] == 1
    assert summary["any_cross_edge_fraction"] == .5
    assert summary["cross_edge_count_distribution"] == {0: 1, 1: 1}
    assert summary["edge_recurrence"] == [{"node1": "a", "node2": "c", "count": 1, "fraction": .5}]
    assert summary["per_seed"][1]["cross_edges"] is None


def test_all_failed_null_is_undefined_and_every_declared_seed_is_preserved():
    outcomes = [{"seed": seed, "discovery": {"status": "failed", "edges": None}} for seed in null.SEEDS]
    summary = null.summarize(outcomes, [])
    assert len(null.SEEDS) == 50 and summary["failed_seeds"] == list(range(10000, 10050))
    assert summary["any_cross_edge_fraction"] is None and summary["cross_edge_count_mean"] is None
    assert summary["edge_recurrence"] == [] and summary["n_failed"] == 50

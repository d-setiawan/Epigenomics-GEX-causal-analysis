#!/usr/bin/env python3
"""Select uniquely supported collider motifs from sparse within-assay evidence.

Pairwise positive-dependence tests answer whether an adjacency is plausible;
they do not, by themselves, separate a direct two-parent collider from an
isolated residual association or from the third side of a noisy triangle.
This discovery-only layer treats the high-confidence Bernoulli graph as a set
of *candidate* adjacencies and promotes only complete, uniquely supported
unshielded two-stars to ``within_assay_direct_candidate``.

For each connected candidate component, every two-star is scored by its weaker
arm's mean positive residual log odds.  The best wedge is accepted only when
both arms clear a practical floor and it beats the next-best competing wedge
by a frozen margin.  Base sparse-model edges are eligible even when the rescue
model narrowly misses its threshold, but all arms are scored by the same
empirical-null evidence.  Isolated edges, excess spokes, and statistically
symmetric triangles are returned as unresolved/demoted rather than forced into
the direct layer.  Disconnected components are handled independently, so the
rule does not assume one motif per assay.

No feature role, genomic annotation, truth, simulator parameter, paired cell,
or external prior is accepted.  The output is a conservative collider
candidate graph, not a claim that every biological direct effect is a collider.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from structure_aware_collider_rescue import (
    _assert_no_privileged_columns,
    _require_columns,
    aggregate_vote_evidence,
    canonical_edge,
)


@dataclass(frozen=True)
class ColliderMotifConfig:
    """Frozen thresholds for the v2 structure-aware direct-candidate layer."""

    min_rescue_vote_fraction: float = 5.0 / 6.0
    max_rescue_mean_empirical_p_positive: float = 0.020
    min_rescue_mean_pooled_log_odds: float = 1.00
    min_wedge_arm_pooled_log_odds: float = 0.85
    min_unique_wedge_margin: float = 0.15

    def validate(self) -> None:
        for name, value in (
            ("min_rescue_vote_fraction", self.min_rescue_vote_fraction),
            (
                "max_rescue_mean_empirical_p_positive",
                self.max_rescue_mean_empirical_p_positive,
            ),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name, value in (
            ("min_rescue_mean_pooled_log_odds", self.min_rescue_mean_pooled_log_odds),
            ("min_wedge_arm_pooled_log_odds", self.min_wedge_arm_pooled_log_odds),
            ("min_unique_wedge_margin", self.min_unique_wedge_margin),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class ColliderMotifResult:
    direct_candidates: pd.DataFrame
    demoted_or_unresolved: pd.DataFrame
    component_diagnostics: pd.DataFrame
    wedge_diagnostics: pd.DataFrame
    aggregated_evidence: pd.DataFrame
    metadata: dict[str, Any]


def _connected_components(edges: set[tuple[str, str]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    unseen = set(adjacency)
    components: list[set[str]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _direct_edge_row(
    assay: str,
    edge: tuple[str, str],
    *,
    center: str,
    component_id: str,
    wedge_score: float,
    runner_up_score: float,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    left, right = edge
    center_is_left = center == left
    return {
        "source_assay": assay,
        "node1": left,
        "endpoint1": "ARROW" if center_is_left else "TAIL",
        "node2": right,
        "endpoint2": "TAIL" if center_is_left else "ARROW",
        "edge_type": "within_assay_direct_candidate",
        "orientation_source": "unique_positive_residual_wedge",
        "collider_center": center,
        "component_id": component_id,
        "wedge_score": wedge_score,
        "runner_up_wedge_score": runner_up_score,
        "wedge_margin": (
            wedge_score - runner_up_score if np.isfinite(runner_up_score) else np.inf
        ),
        "selected_vote_fraction": float(evidence["selected_vote_fraction"]),
        "mean_empirical_p_positive": float(evidence["mean_empirical_p_positive"]),
        "mean_pooled_log_odds": float(evidence["mean_pooled_log_odds"]),
        "interpretation": "edge in a uniquely supported within-assay collider candidate",
    }


def select_unique_collider_motifs(
    base_edges: pd.DataFrame,
    vote_evidence: pd.DataFrame,
    config: ColliderMotifConfig | None = None,
) -> ColliderMotifResult:
    """Promote unique two-stars and demote unsupported pairwise associations."""

    cfg = config or ColliderMotifConfig()
    cfg.validate()
    _assert_no_privileged_columns(base_edges, "base_edges")
    _require_columns(base_edges, {"source_assay", "node1", "node2"}, "base_edges")
    base = base_edges.copy()
    if "selected" in base:
        base = base.loc[base["selected"].astype(bool)].copy()
    aggregated = aggregate_vote_evidence(vote_evidence)

    direct_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    wedge_rows: list[dict[str, Any]] = []

    for assay, assay_evidence in aggregated.groupby("source_assay", sort=True):
        evidence_map = {
            canonical_edge(str(row["node1"]), str(row["node2"])): row
            for row in assay_evidence.to_dict(orient="records")
        }
        base_pairs = {
            canonical_edge(str(row["node1"]), str(row["node2"]))
            for row in base.loc[
                base["source_assay"].astype(str) == str(assay)
            ].to_dict(orient="records")
        }
        rescue_pairs = {
            pair
            for pair, record in evidence_map.items()
            if float(record["selected_vote_fraction"])
            >= cfg.min_rescue_vote_fraction
            and float(record["mean_empirical_p_positive"])
            <= cfg.max_rescue_mean_empirical_p_positive
            and float(record["mean_pooled_log_odds"])
            >= cfg.min_rescue_mean_pooled_log_odds
        }
        candidate_pairs = base_pairs | rescue_pairs
        for component_index, component in enumerate(
            _connected_components(candidate_pairs)
        ):
            component_id = f"{assay}__component{component_index}"
            component_edges = {
                edge
                for edge in candidate_pairs
                if edge[0] in component and edge[1] in component
            }
            adjacency: dict[str, set[str]] = {node: set() for node in component}
            for left, right in component_edges:
                adjacency[left].add(right)
                adjacency[right].add(left)
            wedges: list[dict[str, Any]] = []
            for center in sorted(component):
                for outer1, outer2 in combinations(sorted(adjacency[center]), 2):
                    edge1 = canonical_edge(center, outer1)
                    edge2 = canonical_edge(center, outer2)
                    evidence1 = evidence_map.get(edge1)
                    evidence2 = evidence_map.get(edge2)
                    if evidence1 is None or evidence2 is None:
                        continue
                    arm1 = float(evidence1["mean_pooled_log_odds"])
                    arm2 = float(evidence2["mean_pooled_log_odds"])
                    wedges.append(
                        {
                            "source_assay": str(assay),
                            "component_id": component_id,
                            "center": center,
                            "outer1": outer1,
                            "outer2": outer2,
                            "edge1": edge1,
                            "edge2": edge2,
                            "arm1_log_odds": arm1,
                            "arm2_log_odds": arm2,
                            "wedge_score": min(arm1, arm2),
                            "outer_pair_is_candidate": (
                                canonical_edge(outer1, outer2) in candidate_pairs
                            ),
                        }
                    )
            wedges.sort(
                key=lambda row: (
                    -float(row["wedge_score"]),
                    str(row["center"]),
                    str(row["outer1"]),
                    str(row["outer2"]),
                )
            )
            best = wedges[0] if wedges else None
            runner_up_score = (
                float(wedges[1]["wedge_score"]) if len(wedges) > 1 else -np.inf
            )
            accepted = bool(
                best is not None
                and float(best["wedge_score"])
                >= cfg.min_wedge_arm_pooled_log_odds
                and (
                    len(wedges) == 1
                    or float(best["wedge_score"]) - runner_up_score
                    >= cfg.min_unique_wedge_margin
                )
            )
            selected_component_edges: set[tuple[str, str]] = set()
            if accepted and best is not None:
                selected_component_edges = {best["edge1"], best["edge2"]}
                for edge in sorted(selected_component_edges):
                    direct_rows.append(
                        _direct_edge_row(
                            str(assay),
                            edge,
                            center=str(best["center"]),
                            component_id=component_id,
                            wedge_score=float(best["wedge_score"]),
                            runner_up_score=runner_up_score,
                            evidence=evidence_map[edge],
                        )
                    )
            status = (
                "accepted_unique_wedge"
                if accepted
                else "unresolved_no_complete_wedge"
                if not wedges
                else "unresolved_weak_wedge"
                if best is not None
                and float(best["wedge_score"])
                < cfg.min_wedge_arm_pooled_log_odds
                else "unresolved_competing_wedges"
            )
            component_rows.append(
                {
                    "source_assay": str(assay),
                    "component_id": component_id,
                    "nodes": ";".join(sorted(component)),
                    "n_nodes": len(component),
                    "n_candidate_edges": len(component_edges),
                    "n_base_edges": len(component_edges & base_pairs),
                    "n_rescue_edges": len(component_edges & rescue_pairs),
                    "n_wedges": len(wedges),
                    "best_center": str(best["center"]) if best is not None else "",
                    "best_wedge_score": (
                        float(best["wedge_score"]) if best is not None else np.nan
                    ),
                    "runner_up_wedge_score": (
                        runner_up_score if np.isfinite(runner_up_score) else np.nan
                    ),
                    "accepted": accepted,
                    "status": status,
                }
            )
            for rank, wedge in enumerate(wedges, start=1):
                wedge_rows.append(
                    {
                        **{key: value for key, value in wedge.items() if key not in {"edge1", "edge2"}},
                        "edge1_node1": wedge["edge1"][0],
                        "edge1_node2": wedge["edge1"][1],
                        "edge2_node1": wedge["edge2"][0],
                        "edge2_node2": wedge["edge2"][1],
                        "rank": rank,
                        "accepted": bool(accepted and rank == 1),
                    }
                )
            for edge in sorted(component_edges - selected_component_edges):
                evidence = evidence_map[edge]
                unresolved_rows.append(
                    {
                        "source_assay": str(assay),
                        "node1": edge[0],
                        "endpoint1": "CIRCLE",
                        "node2": edge[1],
                        "endpoint2": "CIRCLE",
                        "edge_type": "unresolved",
                        "component_id": component_id,
                        "was_base_selected": edge in base_pairs,
                        "was_rescue_selected": edge in rescue_pairs,
                        "mean_empirical_p_positive": float(
                            evidence["mean_empirical_p_positive"]
                        ),
                        "mean_pooled_log_odds": float(
                            evidence["mean_pooled_log_odds"]
                        ),
                        "reason": (
                            "component_not_identifiable"
                            if not accepted
                            else "not_in_unique_best_wedge"
                        ),
                    }
                )

    direct = pd.DataFrame(direct_rows)
    unresolved = pd.DataFrame(unresolved_rows)
    components = pd.DataFrame(component_rows)
    wedge_diagnostics = pd.DataFrame(wedge_rows)
    metadata = {
        "method": "unique_positive_residual_collider_motif",
        "schema_version": "collider_motif_selector.v1",
        "config": asdict(cfg),
        "n_base_edges": int(len(base)),
        "n_aggregated_pairs": int(len(aggregated)),
        "n_direct_candidate_edges": int(len(direct)),
        "n_unresolved_or_demoted_edges": int(len(unresolved)),
        "n_accepted_components": int(
            components["accepted"].astype(bool).sum() if not components.empty else 0
        ),
        "n_unresolved_components": int(
            (~components["accepted"].astype(bool)).sum() if not components.empty else 0
        ),
        "identifiability_boundary": (
            "symmetric triangles and incomplete components remain unresolved"
        ),
    }
    return ColliderMotifResult(
        direct_candidates=direct,
        demoted_or_unresolved=unresolved,
        component_diagnostics=components,
        wedge_diagnostics=wedge_diagnostics,
        aggregated_evidence=aggregated,
        metadata=metadata,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-edges", type=Path, required=True)
    parser.add_argument("--vote-evidence", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    base = pd.read_csv(args.base_edges, sep="\t")
    votes = pd.concat(
        [pd.read_csv(path, sep="\t") for path in args.vote_evidence],
        ignore_index=True,
    )
    result = select_unique_collider_motifs(base, votes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in (
        ("direct_candidates.tsv", result.direct_candidates),
        ("demoted_or_unresolved.tsv", result.demoted_or_unresolved),
        ("component_diagnostics.tsv", result.component_diagnostics),
        ("wedge_diagnostics.tsv", result.wedge_diagnostics),
        ("aggregated_vote_evidence.tsv", result.aggregated_evidence),
    ):
        frame.to_csv(args.out_dir / filename, sep="\t", index=False)
    (args.out_dir / "motif_manifest.json").write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    _main()

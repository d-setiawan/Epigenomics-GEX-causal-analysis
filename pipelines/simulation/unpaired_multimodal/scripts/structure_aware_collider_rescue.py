#!/usr/bin/env python3
"""Conservative anchored-collider rescue for sparse within-assay edges.

The sparse Bernoulli layer can miss one arm of an otherwise well-supported
within-assay collider when one parent is detected only a few times.  A loose
union with a higher-powered all-cell model recovers that arm, but also admits
isolated residual associations.  This module applies a graph-structural filter
to such rescue evidence without using feature roles, truth, coordinates,
motifs, or simulator parameters.

The frozen rule is deliberately narrow.  Before a candidate is considered,
the base graph must contain exactly one edge incident to a proposed center and
no edge incident to the proposed new outer node.  The rescue edge must be
stable, positive, and practically non-trivial across aggregation votes.  The
remaining outer pair must be unsupported by the same empirical-null model.
At most one arm is added per center.  Thus the method can complete an anchored
unshielded two-star, but it never guesses a center in an unanchored triangle.

This is an adjacency/orientation *candidate*, not proof of a causal direction.
Its collider interpretation depends on the empirical outer-pair separation
remaining credible under the observed background-state/depth adjustment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_TOKENS = (
    "truth",
    "role",
    "oracle",
    "latent",
    "generative",
    "counterfactual",
    "parent",
    "child",
    "motif",
)


@dataclass(frozen=True)
class AnchoredColliderRescueConfig:
    """Frozen TRAIN-stage thresholds for the anchored-wedge filter."""

    min_candidate_vote_fraction: float = 5.0 / 6.0
    max_candidate_mean_empirical_p_positive: float = 0.020
    min_candidate_mean_pooled_log_odds: float = 1.00
    min_outer_mean_empirical_p_positive: float = 0.20
    max_outer_vote_fraction: float = 0.50
    base_center_degree: int = 1
    base_outer_degree: int = 0

    def validate(self) -> None:
        for name, value in (
            ("min_candidate_vote_fraction", self.min_candidate_vote_fraction),
            ("max_candidate_mean_empirical_p_positive", self.max_candidate_mean_empirical_p_positive),
            ("min_outer_mean_empirical_p_positive", self.min_outer_mean_empirical_p_positive),
            ("max_outer_vote_fraction", self.max_outer_vote_fraction),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not np.isfinite(self.min_candidate_mean_pooled_log_odds):
            raise ValueError("min_candidate_mean_pooled_log_odds must be finite")
        if self.base_center_degree != 1:
            raise ValueError("the preregistered anchored rule requires base_center_degree=1")
        if self.base_outer_degree != 0:
            raise ValueError("the preregistered anchored rule requires base_outer_degree=0")


@dataclass
class AnchoredColliderRescueResult:
    rescued_edges: pd.DataFrame
    candidate_diagnostics: pd.DataFrame
    aggregated_evidence: pd.DataFrame
    metadata: dict[str, Any]


def canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _assert_no_privileged_columns(frame: pd.DataFrame, source: str) -> None:
    forbidden = [
        column
        for column in frame.columns.astype(str)
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"{source} contains privileged columns: {sorted(forbidden)}")


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def aggregate_vote_evidence(vote_evidence: pd.DataFrame) -> pd.DataFrame:
    """Aggregate discovery-only positive-edge evidence across nested votes."""

    _assert_no_privileged_columns(vote_evidence, "vote_evidence")
    _require_columns(
        vote_evidence,
        {
            "source_assay",
            "node1",
            "node2",
            "selected",
            "empirical_p_positive",
            "pooled_log_odds",
        },
        "vote_evidence",
    )
    if vote_evidence.empty:
        return pd.DataFrame(
            columns=[
                "source_assay",
                "node1",
                "node2",
                "selected_vote_count",
                "vote_count",
                "selected_vote_fraction",
                "mean_empirical_p_positive",
                "mean_pooled_log_odds",
            ]
        )
    work = vote_evidence.copy()
    pairs = work.apply(
        lambda row: canonical_edge(str(row["node1"]), str(row["node2"])), axis=1
    )
    work["node1"] = [pair[0] for pair in pairs]
    work["node2"] = [pair[1] for pair in pairs]
    work["selected"] = work["selected"].astype(bool).astype(int)
    for column in ("empirical_p_positive", "pooled_log_odds"):
        work[column] = pd.to_numeric(work[column], errors="raise")
    grouped = (
        work.groupby(["source_assay", "node1", "node2"], as_index=False)
        .agg(
            selected_vote_count=("selected", "sum"),
            vote_count=("selected", "size"),
            mean_empirical_p_positive=("empirical_p_positive", "mean"),
            mean_pooled_log_odds=("pooled_log_odds", "mean"),
        )
        .sort_values(["source_assay", "node1", "node2"])
        .reset_index(drop=True)
    )
    grouped["selected_vote_fraction"] = (
        grouped["selected_vote_count"] / grouped["vote_count"].clip(lower=1)
    )
    return grouped


def select_anchored_collider_rescues(
    base_edges: pd.DataFrame,
    vote_evidence: pd.DataFrame,
    config: AnchoredColliderRescueConfig | None = None,
) -> AnchoredColliderRescueResult:
    """Select at most one empirically separated rescue arm per anchored center."""

    cfg = config or AnchoredColliderRescueConfig()
    cfg.validate()
    _assert_no_privileged_columns(base_edges, "base_edges")
    _require_columns(base_edges, {"source_assay", "node1", "node2"}, "base_edges")
    base = base_edges.copy()
    if "selected" in base:
        base = base.loc[base["selected"].astype(bool)].copy()
    aggregated = aggregate_vote_evidence(vote_evidence)
    diagnostics: list[dict[str, Any]] = []

    for assay, assay_evidence in aggregated.groupby("source_assay", sort=True):
        assay_base = base.loc[base["source_assay"].astype(str) == str(assay)]
        base_pairs = {
            canonical_edge(str(row["node1"]), str(row["node2"]))
            for row in assay_base.to_dict(orient="records")
        }
        neighbors: dict[str, set[str]] = {}
        for left, right in base_pairs:
            neighbors.setdefault(left, set()).add(right)
            neighbors.setdefault(right, set()).add(left)
        evidence_map = {
            canonical_edge(str(row["node1"]), str(row["node2"])): row
            for row in assay_evidence.to_dict(orient="records")
        }
        for record in assay_evidence.to_dict(orient="records"):
            candidate_pair = canonical_edge(str(record["node1"]), str(record["node2"]))
            if candidate_pair in base_pairs:
                continue
            for center, new_outer in (candidate_pair, candidate_pair[::-1]):
                center_neighbors = sorted(neighbors.get(center, set()))
                new_outer_degree = len(neighbors.get(new_outer, set()))
                structural_anchor = (
                    len(center_neighbors) == cfg.base_center_degree
                    and new_outer_degree == cfg.base_outer_degree
                )
                anchor = center_neighbors[0] if structural_anchor else ""
                outer_record = (
                    evidence_map.get(canonical_edge(anchor, new_outer))
                    if structural_anchor
                    else None
                )
                candidate_strength = (
                    float(record["selected_vote_fraction"])
                    >= cfg.min_candidate_vote_fraction
                    and float(record["mean_empirical_p_positive"])
                    <= cfg.max_candidate_mean_empirical_p_positive
                    and float(record["mean_pooled_log_odds"])
                    >= cfg.min_candidate_mean_pooled_log_odds
                )
                outer_separated = bool(
                    outer_record is not None
                    and float(outer_record["mean_empirical_p_positive"])
                    >= cfg.min_outer_mean_empirical_p_positive
                    and float(outer_record["selected_vote_fraction"])
                    <= cfg.max_outer_vote_fraction
                )
                diagnostics.append(
                    {
                        "source_assay": str(assay),
                        "center": center,
                        "anchor_outer": anchor,
                        "new_outer": new_outer,
                        "candidate_node1": candidate_pair[0],
                        "candidate_node2": candidate_pair[1],
                        "base_center_degree": len(center_neighbors),
                        "base_new_outer_degree": new_outer_degree,
                        "candidate_vote_fraction": float(record["selected_vote_fraction"]),
                        "candidate_mean_empirical_p_positive": float(
                            record["mean_empirical_p_positive"]
                        ),
                        "candidate_mean_pooled_log_odds": float(
                            record["mean_pooled_log_odds"]
                        ),
                        "outer_mean_empirical_p_positive": (
                            float(outer_record["mean_empirical_p_positive"])
                            if outer_record is not None
                            else np.nan
                        ),
                        "outer_vote_fraction": (
                            float(outer_record["selected_vote_fraction"])
                            if outer_record is not None
                            else np.nan
                        ),
                        "structural_anchor": structural_anchor,
                        "candidate_strength": candidate_strength,
                        "outer_separated": outer_separated,
                        "eligible": bool(
                            structural_anchor and candidate_strength and outer_separated
                        ),
                    }
                )

    diagnostic_frame = pd.DataFrame(diagnostics)
    selected_rows: list[dict[str, Any]] = []
    if not diagnostic_frame.empty:
        eligible = diagnostic_frame.loc[diagnostic_frame["eligible"].astype(bool)].copy()
        # One completion per pre-existing degree-one center.  Ranking is fixed
        # and deterministic: stronger empirical evidence, then effect, then ID.
        eligible = eligible.sort_values(
            [
                "source_assay",
                "center",
                "candidate_mean_empirical_p_positive",
                "candidate_mean_pooled_log_odds",
                "new_outer",
            ],
            ascending=[True, True, True, False, True],
        ).drop_duplicates(["source_assay", "center"], keep="first")
        for row in eligible.to_dict(orient="records"):
            left, right = canonical_edge(str(row["center"]), str(row["new_outer"]))
            center_is_left = left == str(row["center"])
            selected_rows.append(
                {
                    "source_assay": row["source_assay"],
                    "node1": left,
                    "endpoint1": "ARROW" if center_is_left else "TAIL",
                    "node2": right,
                    "endpoint2": "TAIL" if center_is_left else "ARROW",
                    "edge_type": "within_assay_direct_candidate",
                    "orientation_source": "anchored_empirical_null_collider_rescue",
                    "collider_center": row["center"],
                    "anchor_outer": row["anchor_outer"],
                    "candidate_vote_fraction": row["candidate_vote_fraction"],
                    "candidate_mean_empirical_p_positive": row[
                        "candidate_mean_empirical_p_positive"
                    ],
                    "candidate_mean_pooled_log_odds": row[
                        "candidate_mean_pooled_log_odds"
                    ],
                    "outer_mean_empirical_p_positive": row[
                        "outer_mean_empirical_p_positive"
                    ],
                    "interpretation": (
                        "anchored positive residual arm with empirically separated outer pair"
                    ),
                }
            )
    rescued = pd.DataFrame(selected_rows)
    metadata = {
        "method": "anchored_empirical_null_collider_rescue",
        "schema_version": "anchored_collider_rescue.v1",
        "config": asdict(cfg),
        "n_base_edges": int(len(base)),
        "n_vote_rows": int(len(vote_evidence)),
        "n_aggregated_pairs": int(len(aggregated)),
        "n_rescued_edges": int(len(rescued)),
        "identifiability_boundary": (
            "unanchored triangles are unresolved; no center is guessed without an outer separation"
        ),
    }
    return AnchoredColliderRescueResult(
        rescued_edges=rescued,
        candidate_diagnostics=diagnostic_frame,
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
    result = select_anchored_collider_rescues(base, votes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.rescued_edges.to_csv(args.out_dir / "rescued_edges.tsv", sep="\t", index=False)
    result.candidate_diagnostics.to_csv(
        args.out_dir / "candidate_diagnostics.tsv", sep="\t", index=False
    )
    result.aggregated_evidence.to_csv(
        args.out_dir / "aggregated_vote_evidence.tsv", sep="\t", index=False
    )
    (args.out_dir / "rescue_manifest.json").write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    _main()

#!/usr/bin/env python3
"""Restore the frozen collider-before-state-group stage for component axes.

This is a narrow, discovery-only harness for the component-alignment
prototype.  It refits the registered six beta-binomial votes against an
already frozen component embedding, selects unique rescue-supported collider
motifs, excludes their observed centers from the loading-group projection,
and only then estimates assay-spanning latent axes.

The harness deliberately supplies no preliminary pairwise direct edges to the
motif selector.  Consequently every selected motif is supported by the
registered beta-binomial rescue evidence itself; no output from a differently
aligned ordinary graph is mixed into this isolation check.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark_ca_boss_v4 import (
    FROZEN_BETA_BINOMIAL_CONFIG,
    FROZEN_COLLIDER_MOTIF_CONFIG,
    fit_collider_motif_vote_evidence,
    write_collider_motif_vote_evidence,
)
from latent_axis_clustering_v4 import (
    LatentAxisClusteringConfig,
    estimate_latent_axis_clustering,
)
from loading_group_projection_v4 import (
    LoadingProjectionConfig,
    estimate_loading_group_projection,
    load_evidence_view,
)
from structure_aware_collider_motif import select_unique_collider_motifs


_FORBIDDEN_PATH_PARTS = {
    "scorer",
    "truth",
    "ground_truth",
    "oracle",
    "counterfactual",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _validate_discovery_path(path: Path, *, directory: bool) -> Path:
    lexical = path.absolute()
    if lexical.is_symlink():
        raise ValueError(f"Discovery paths may not be symlinks: {lexical}")
    resolved = lexical.resolve(strict=True)
    if directory and not resolved.is_dir():
        raise ValueError(f"Expected discovery directory: {resolved}")
    if not directory and not resolved.is_file():
        raise ValueError(f"Expected discovery file: {resolved}")
    forbidden = sorted(
        part.casefold()
        for part in resolved.parts
        if part.casefold() in _FORBIDDEN_PATH_PARTS
    )
    if forbidden:
        raise ValueError(f"Refusing scorer/truth path: {forbidden}")
    return resolved


def _write_tables(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for filename, frame in tables.items():
        frame.to_csv(output / filename, sep="\t", index=False)


def run_restoration_check(
    world_discovery: Path,
    component_discovery: Path,
    output: Path,
) -> dict[str, Any]:
    """Run collider selection, exclusion, projection, and axis estimation."""

    world = _validate_discovery_path(world_discovery, directory=True)
    component = _validate_discovery_path(component_discovery, directory=True)
    out = output.absolute()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite output path: {out}")
    if any(part.casefold() in _FORBIDDEN_PATH_PARTS for part in out.parts):
        raise ValueError("Output must remain in the discovery information domain")

    embedding = _validate_discovery_path(
        component / "alignment_evidence" / "analysis_embeddings.tsv",
        directory=False,
    )
    eligibility_path = _validate_discovery_path(
        component / "axis_input_projection" / "alignment_eligibility.json",
        directory=False,
    )
    eligibility = json.loads(eligibility_path.read_text())
    eligible_modalities = tuple(sorted(map(str, eligibility["eligible_modalities"])))

    chrom_tables = sorted(world.glob("observed_focal__chrom_*.tsv"))
    if len(chrom_tables) != 6:
        raise ValueError("Expected exactly six observed chromatin focal tables")
    views_root = _validate_discovery_path(
        component / "loading_views", directory=True
    )
    view_dirs = sorted(path for path in views_root.iterdir() if path.is_dir())
    if len(view_dirs) != 6:
        raise ValueError("Expected exactly six frozen loading evidence views")

    out.mkdir(parents=True)
    combined, vote_frames, vote_metadata = fit_collider_motif_vote_evidence(
        chrom_tables, embedding
    )
    beta_hashes = write_collider_motif_vote_evidence(
        out, vote_frames, vote_metadata
    )
    rescue_only_base = pd.DataFrame(columns=["source_assay", "node1", "node2"])
    motif = select_unique_collider_motifs(
        rescue_only_base,
        combined,
        FROZEN_COLLIDER_MOTIF_CONFIG,
    )
    collider_centers = tuple(
        sorted(set(motif.direct_candidates.get("collider_center", pd.Series(dtype=str)).astype(str)))
    )

    motif_out = out / "collider_motif"
    _write_tables(
        motif_out,
        {
            "direct_candidates.tsv": motif.direct_candidates,
            "demoted_or_unresolved.tsv": motif.demoted_or_unresolved,
            "component_diagnostics.tsv": motif.component_diagnostics,
            "wedge_diagnostics.tsv": motif.wedge_diagnostics,
            "aggregated_vote_evidence.tsv": motif.aggregated_evidence,
        },
    )
    (motif_out / "motif_provenance.json").write_text(
        json.dumps(
            _json_ready(
                {
                    **motif.metadata,
                    "beta_binomial_config": asdict(FROZEN_BETA_BINOMIAL_CONFIG),
                    "collider_motif_config": asdict(FROZEN_COLLIDER_MOTIF_CONFIG),
                    "base_edge_policy": "empty_rescue_only_isolation_check",
                    "selected_collider_centers": collider_centers,
                    "applied_before_loading_group_projection": True,
                    "beta_vote_evidence_sha256": beta_hashes,
                    "truth_or_scorer_used": False,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    views = [load_evidence_view(path) for path in view_dirs]
    projection_config = LoadingProjectionConfig(
        cross_projection_empirical_q=0.40
    )
    projection = estimate_loading_group_projection(
        views,
        projection_config,
        excluded_features=collider_centers,
    )
    projection_out = out / "loading_group_projection"
    _write_tables(
        projection_out,
        {
            "feature_consensus.tsv": projection.feature_consensus,
            "within_group_candidates.tsv": projection.within_group_candidates,
            "loading_groups.tsv": projection.groups,
            "loading_group_membership.tsv": projection.group_membership,
            "group_match_evidence.tsv": projection.group_match_evidence,
            "typed_group_hyperedges.tsv": projection.typed_hyperedges,
            "sparse_pair_projection.tsv": projection.sparse_pair_projection,
        },
    )
    (projection_out / "projection_provenance.json").write_text(
        json.dumps(_json_ready(projection.provenance), indent=2, sort_keys=True)
        + "\n"
    )

    axis_groups = projection.groups.loc[
        projection.groups["modality"].astype(str).isin(eligible_modalities)
    ].copy()
    eligible_group_ids = set(axis_groups["loading_group"].astype(str))
    axis_membership = projection.group_membership.loc[
        projection.group_membership["loading_group"].astype(str).isin(
            eligible_group_ids
        )
    ].copy()
    axis = estimate_latent_axis_clustering(
        axis_groups,
        axis_membership,
        views,
        LatentAxisClusteringConfig(),
    )
    axis.provenance["alignment_eligibility"] = {
        **eligibility,
        "selection_source": "frozen_component_alignment_gate",
    }
    axis.provenance["collider_exclusion"] = {
        "selected_collider_centers": list(collider_centers),
        "applied_before_axis_clustering": True,
    }
    axis_out = out / "latent_axis_projection"
    _write_tables(
        axis_out,
        {
            "group_pair_view_evidence.tsv": axis.group_pair_view_evidence,
            "group_pair_evidence.tsv": axis.group_pair_evidence,
            "axis_candidates.tsv": axis.axis_candidates,
            "group_axis_view_evidence.tsv": axis.group_axis_view_evidence,
            "group_axis_evidence.tsv": axis.group_axis_evidence,
            "typed_axis_hyperedges.tsv": axis.typed_axis_hyperedges,
            "representative_evidence.tsv": axis.representative_evidence,
            "sparse_pair_projection.tsv": axis.sparse_pair_projection,
        },
    )
    (axis_out / "axis_provenance.json").write_text(
        json.dumps(_json_ready(axis.provenance), indent=2, sort_keys=True) + "\n"
    )

    input_hashes = {
        str(embedding): _sha256(embedding),
        str(eligibility_path): _sha256(eligibility_path),
        **{str(path): _sha256(path) for path in chrom_tables},
    }
    summary = {
        "schema_version": "component_collider_order_check_v4.discovery.v1",
        "world_discovery": str(world),
        "component_discovery": str(component),
        "input_sha256": input_hashes,
        "loading_view_ids": [view.view_id for view in views],
        "loading_projection_config": asdict(projection_config),
        "latent_axis_config": asdict(LatentAxisClusteringConfig()),
        "selected_collider_centers": list(collider_centers),
        "n_selected_collider_centers": len(collider_centers),
        "n_selected_local_groups": int(
            projection.groups["selection_status"].astype(str).eq("selected").sum()
        ),
        "n_selected_axes": int(
            axis.axis_candidates["selection_status"].astype(str).eq("selected").sum()
        ),
        "eligible_modalities": list(eligible_modalities),
        "stage_order": [
            "six_beta_binomial_votes",
            "unique_collider_motif",
            "selected_collider_center_exclusion",
            "loading_group_projection",
            "latent_axis_clustering",
        ],
        "thresholds_changed_relative_to_component_axis_only_check": False,
        "truth_or_scorer_used": False,
    }
    (out / "restoration_manifest.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-discovery", type=Path, required=True)
    parser.add_argument("--component-discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = run_restoration_check(
        args.world_discovery,
        args.component_discovery,
        args.out,
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

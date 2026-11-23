#!/usr/bin/env python3
"""Build re-thresholdable TRAIN evidence for refined structure-anchored groups.

The script is intentionally split by a global information barrier.  Phase A
recomputes the observed-only collider refinement, applies it to the frozen
motif, refits no cell model, and constructs all parent-to-nonmotif loading
hypotheses from the already frozen assay-local loading views.  Selected
collider centers and parent--parent pairs are structurally forbidden.  Every
requested world is content-addressed before Phase B reads TRAIN roles.

Phase B labels the frozen hypotheses for development-time threshold analysis.
It does not choose a fresh-validation rule and never reads DEV or LOCKED data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from loading_group_projection_v4 import (
    LoadingProjectionConfig,
    estimate_loading_group_projection,
    make_evidence_view,
)
from observed_collider_structure import (
    apply_observed_collider_refinements,
    select_observed_collider_refinements,
)
from structure_aware_collider_motif import ColliderMotifResult


TRAIN_SEEDS = tuple(range(33_001, 33_025))
SOURCE_NAMES = (
    "loading_group_projection_v4.py",
    "observed_collider_structure.py",
    "structure_aware_collider_motif.py",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: _sha256(root / name) for name in SOURCE_NAMES}


def _payload_hashes(root: Path, exclude: Iterable[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and str(path.relative_to(root)) not in excluded
    }


def _read_tsv(path: Path, columns: Sequence[str] = ()) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(columns))


def _assert_train_path(path: Path, label: str) -> None:
    folded = str(path.resolve(strict=False)).casefold()
    if "dev" in folded or "locked" in folded:
        raise ValueError(f"Refusing {label} DEV/LOCKED path: {path}")


def _old_candidate(world: Path) -> Path:
    candidates = sorted(
        path
        for path in world.glob("v4_*")
        if path.is_dir() and (path / "collider_motif").is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected one old frozen candidate in {world}")
    return candidates[0]


def _motif(candidate: Path) -> ColliderMotifResult:
    root = candidate / "collider_motif"
    return ColliderMotifResult(
        direct_candidates=_read_tsv(root / "direct_candidates.tsv"),
        demoted_or_unresolved=_read_tsv(
            root / "demoted_or_unresolved.tsv",
            ("source_assay", "component_id", "node1", "node2"),
        ),
        component_diagnostics=_read_tsv(root / "component_diagnostics.tsv"),
        wedge_diagnostics=_read_tsv(root / "wedge_diagnostics.tsv"),
        aggregated_evidence=_read_tsv(root / "aggregated_vote_evidence.tsv"),
        metadata=json.loads((root / "motif_provenance.json").read_text()),
    )


def _refined_motif(artifact_world: Path, candidate: Path) -> tuple[Any, Any]:
    motif = _motif(candidate)
    view_dirs = sorted(
        path
        for path in artifact_world.glob("cells*_seed*")
        if (path / "raw_pair_evidence.tsv").is_file()
    )
    if len(view_dirs) != 6:
        raise ValueError(f"Expected six frozen views in {artifact_world}")
    base_views = {
        view.name: _read_tsv(candidate / f"nested_{index}_edges.tsv")
        for index, view in enumerate(view_dirs)
    }
    residual_views = {
        view.name: _read_tsv(view / "raw_pair_evidence.tsv")
        for view in view_dirs
    }
    feature_views = {
        view.name: _read_tsv(view / "cell_loading_feature_models.tsv")
        for view in view_dirs
    }
    votes = pd.concat(
        [
            _read_tsv(path)
            for path in sorted(
                (artifact_world / "collider_motif_evidence").glob(
                    "beta_binomial_vote*.tsv"
                )
            )
        ],
        ignore_index=True,
    )
    refinement = select_observed_collider_refinements(
        motif.component_diagnostics,
        motif.wedge_diagnostics,
        motif.demoted_or_unresolved,
        votes,
        base_views,
        residual_views,
        feature_views,
    )
    return refinement, apply_observed_collider_refinements(motif, refinement)


def _projection_config(first_candidate_world: Path) -> LoadingProjectionConfig:
    payload = json.loads(
        (first_candidate_world / "candidate_manifest.json").read_text()
    )
    raw = payload["projection_config"]
    allowed = {field.name for field in fields(LoadingProjectionConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    values["allow_cross_assay_group_matching"] = False
    return LoadingProjectionConfig(**values)


def _loading_views(first_candidate_world: Path) -> tuple[list[Any], pd.DataFrame, pd.DataFrame]:
    views: list[Any] = []
    pair_frames: list[pd.DataFrame] = []
    uncertainty_rows: list[dict[str, Any]] = []
    for view_dir in sorted((first_candidate_world / "views").iterdir()):
        if not view_dir.is_dir():
            continue
        models = _read_tsv(view_dir / "feature_models.tsv")
        loadings = _read_tsv(view_dir / "loadings.tsv")
        pairs = _read_tsv(view_dir / "pair_evidence.tsv")
        view = make_evidence_view(
            view_id=view_dir.name,
            feature_models=models,
            loadings=loadings,
            pair_evidence=pairs,
        )
        views.append(view)
        tagged = pairs.copy()
        tagged["view_id"] = view_dir.name
        pair_frames.append(tagged)
        model_lookup = models.set_index("feature")
        for pair in tagged.loc[
            tagged["same_assay"].astype(bool)
            & tagged["modality1"].astype(str).ne("rna")
        ].itertuples(index=False):
            left = model_lookup.loc[str(pair.node1)]
            right = model_lookup.loc[str(pair.node2)]
            uncertainty_rows.append(
                {
                    "view_id": view_dir.name,
                    "pair_key": str(pair.pair_key),
                    "maximum_relative_loading_uncertainty": max(
                        float(left["relative_loading_uncertainty"]),
                        float(right["relative_loading_uncertainty"]),
                    ),
                }
            )
    if len(views) != 6:
        raise ValueError(f"Expected six refitted assay-local views in {first_candidate_world}")
    return (
        views,
        pd.concat(pair_frames, ignore_index=True),
        pd.DataFrame(uncertainty_rows),
    )


def _canonical(left: str, right: str) -> str:
    return "|||".join(sorted((str(left), str(right))))


def _motif_specs(direct: pd.DataFrame) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for (assay, center), frame in direct.groupby(
        ["source_assay", "collider_center"], sort=True
    ):
        if len(frame) != 2:
            raise ValueError(f"Refined motif {assay}/{center} does not have two arms")
        parents: set[str] = set()
        for row in frame.itertuples(index=False):
            nodes = {str(row.node1), str(row.node2)}
            if str(center) not in nodes:
                raise ValueError("Refined direct edge omits its explicit center")
            parents.update(nodes.difference({str(center)}))
        if len(parents) != 2:
            raise ValueError(f"Refined motif {assay}/{center} lacks two parents")
        if str(assay) in specs:
            raise ValueError(f"Refinement selected multiple motifs in assay {assay}")
        specs[str(assay)] = {
            "center": str(center),
            "parents": tuple(sorted(parents)),
        }
    return specs


def _structural_candidates(
    direct: pd.DataFrame,
    within: pd.DataFrame,
    pair_views: pd.DataFrame,
    uncertainty_views: pd.DataFrame,
    state_membership: pd.DataFrame,
) -> pd.DataFrame:
    specs = _motif_specs(direct)
    required_membership = {
        "feature",
        "selection_status",
        "feature_global_maxT_confirmation_gate",
        "assay_global_maxT_admission_gate",
        "conditional_wald_view_support",
    }
    missing_membership = sorted(required_membership - set(state_membership.columns))
    if missing_membership:
        raise ValueError(
            f"Formal state membership lacks columns {missing_membership}"
        )
    if state_membership["feature"].astype(str).duplicated().any():
        raise ValueError("Formal state membership contains duplicate features")
    membership_lookup = state_membership.set_index(
        state_membership["feature"].astype(str)
    )
    pair_lookup = {
        key: frame.sort_values("view_id").reset_index(drop=True)
        for key, frame in pair_views.groupby("pair_key", sort=True)
    }
    uncertainty_lookup = {
        key: frame.sort_values("view_id").reset_index(drop=True)
        for key, frame in uncertainty_views.groupby("pair_key", sort=True)
    }
    within_lookup = within.set_index("pair_key")
    rows: list[dict[str, Any]] = []
    for modality, spec in sorted(specs.items()):
        center = spec["center"]
        parents = set(spec["parents"])
        assay = within.loc[within["modality"].astype(str).eq(modality)]
        for candidate in assay.to_dict(orient="records"):
            nodes = {str(candidate["node1"]), str(candidate["node2"])}
            if center in nodes or nodes == parents:
                continue
            parent_members = nodes & parents
            outside = nodes.difference(parents | {center})
            if len(parent_members) != 1 or len(outside) != 1:
                continue
            parent = next(iter(parent_members))
            sibling = next(iter(outside))
            other_parent = next(iter(parents.difference({parent})))
            pair_key = str(candidate["pair_key"])
            opposite_key = _canonical(other_parent, sibling)
            if opposite_key not in within_lookup.index:
                raise ValueError(f"Missing opposite-parent candidate {opposite_key}")
            views = pair_lookup[pair_key]
            opposite = pair_lookup[opposite_key]
            if list(views["view_id"]) != list(opposite["view_id"]):
                raise ValueError("Parent-choice views are misaligned")
            similarity = views["absolute_loading_cosine"].to_numpy(dtype=float)
            opposite_similarity = opposite[
                "absolute_loading_cosine"
            ].to_numpy(dtype=float)
            advantage = similarity - opposite_similarity
            p_values = views["loading_empirical_null_p"].to_numpy(dtype=float)
            uncertainty = uncertainty_lookup[pair_key][
                "maximum_relative_loading_uncertainty"
            ].to_numpy(dtype=float)
            if sibling not in membership_lookup.index or parent not in membership_lookup.index:
                raise ValueError("Structural candidate is absent from formal state membership")
            sibling_membership = membership_lookup.loc[sibling]
            parent_membership = membership_lookup.loc[parent]
            record = {
                **candidate,
                "collider_center": center,
                "parent_candidate": parent,
                "other_parent_candidate": other_parent,
                "nonmotif_sibling_candidate": sibling,
                "opposite_parent_pair_key": opposite_key,
                "parent_win_view_support": float(np.mean(advantage > 0.0)),
                "parent_nonloss_view_support": float(np.mean(advantage >= 0.0)),
                "median_parent_similarity_advantage": float(np.median(advantage)),
                "lower_parent_similarity_advantage": float(
                    np.quantile(advantage, 0.10)
                ),
                "relative_median_parent_margin": float(
                    np.median(advantage) / max(np.median(similarity), 1e-12)
                ),
                "similarity_view_sd": float(np.std(similarity, ddof=1)),
                "similarity_view_range": float(np.ptp(similarity)),
                "empirical_null_median_p": float(np.median(p_values)),
                "empirical_null_support_p_le_0_125": float(
                    np.mean(p_values <= 0.125)
                ),
                "empirical_null_support_p_le_0_25": float(
                    np.mean(p_values <= 0.25)
                ),
                "empirical_null_support_p_le_0_50": float(
                    np.mean(p_values <= 0.50)
                ),
                "median_maximum_relative_loading_uncertainty": float(
                    np.median(uncertainty)
                ),
                "upper_maximum_relative_loading_uncertainty": float(
                    np.quantile(uncertainty, 0.90)
                ),
                "sibling_formal_state_selected": str(
                    sibling_membership["selection_status"]
                )
                == "selected",
                "sibling_feature_global_maxT_confirmation_gate": bool(
                    sibling_membership["feature_global_maxT_confirmation_gate"]
                ),
                "sibling_assay_global_maxT_admission_gate": bool(
                    sibling_membership["assay_global_maxT_admission_gate"]
                ),
                "sibling_conditional_wald_view_support": float(
                    sibling_membership["conditional_wald_view_support"]
                ),
                "parent_formal_state_selected_diagnostic_only": str(
                    parent_membership["selection_status"]
                )
                == "selected",
                "parent_membership_used_as_gate": False,
                "center_excluded": True,
                "parent_parent_pair_forbidden": True,
            }
            rows.append(record)
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("No structure-anchored candidates were generated")
    output["structural_rank_by_coherence"] = (
        output.groupby("modality")["median_coherence_group_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    output["structural_rank_by_cosine"] = (
        output.groupby("modality")["median_absolute_loading_cosine"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return output.sort_values(
        ["modality", "structural_rank_by_coherence", "pair_key"]
    ).reset_index(drop=True)


def _phase_a_world(
    seed: int,
    artifacts_root: str,
    first_diagnostic_root: str,
    candidates_root: str,
) -> dict[str, Any]:
    artifact_world = Path(artifacts_root) / f"world_seed{seed}"
    first_world = (
        Path(first_diagnostic_root)
        / "phase_a_candidates"
        / f"world_seed{seed}"
    )
    candidate = _old_candidate(artifact_world)
    refinement, applied = _refined_motif(artifact_world, candidate)
    views, pair_views, uncertainty_views = _loading_views(first_world)
    membership_path = (
        artifact_world
        / "shared_state_subspace_v4"
        / "subspace_feature_membership.tsv"
    )
    state_membership = _read_tsv(membership_path)
    centers = tuple(sorted(set(applied.direct_candidates["collider_center"].astype(str))))
    projection = estimate_loading_group_projection(
        views,
        _projection_config(first_world),
        excluded_features=centers,
    )
    if not projection.group_match_evidence.empty:
        raise RuntimeError("Cross-assay loading matching was not suppressed")
    structural = _structural_candidates(
        applied.direct_candidates,
        projection.within_group_candidates,
        pair_views,
        uncertainty_views,
        state_membership,
    )
    output_root = Path(candidates_root)
    temp_parent = output_root / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"world_seed{seed}__", dir=temp_parent))
    final = output_root / f"world_seed{seed}"
    try:
        frames = {
            "refined_direct_candidates.tsv": applied.direct_candidates,
            "refined_unresolved.tsv": applied.demoted_or_unresolved,
            "refinement_decisions.tsv": refinement.decisions,
            "refinement_wedge_view_evidence.tsv": refinement.wedge_view_evidence,
            "refinement_wedge_evidence.tsv": refinement.wedge_evidence,
            "refinement_completion_view_evidence.tsv": refinement.completion_view_evidence,
            "refinement_completion_evidence.tsv": refinement.completion_evidence,
            "within_group_candidates.tsv": projection.within_group_candidates,
            "empirical_null_pair_views.tsv": pair_views.loc[
                pair_views["same_assay"].astype(bool)
                & pair_views["modality1"].astype(str).ne("rna")
            ],
            "structural_loading_candidates.tsv": structural,
        }
        for filename, frame in frames.items():
            frame.to_csv(temporary / filename, sep="\t", index=False)
        _write_json(temporary / "refinement_metadata.json", refinement.metadata)
        _write_json(temporary / "applied_motif_metadata.json", applied.metadata)
        manifest = {
            "schema_version": "ca_boss_v4.structure_anchored_train.phase_a_world.v1",
            "seed": seed,
            "train_only": True,
            "roles_or_scorer_read": False,
            "observed_refinement_default_config": refinement.metadata["config"],
            "n_refinement_decisions": len(refinement.decisions),
            "n_refined_direct_edges": len(applied.direct_candidates),
            "n_refined_assays": applied.direct_candidates[
                "source_assay"
            ].astype(str).nunique(),
            "selected_collider_centers": list(centers),
            "center_exclusion_applied": True,
            "parent_parent_pairs_forbidden": True,
            "cross_assay_group_matching_suppressed": True,
            "n_structural_loading_candidates": len(structural),
            "formal_state_membership_sha256": _sha256(membership_path),
            "formal_state_membership_applied_to_nonmotif_sibling_only": True,
            "direct_parent_membership_used_as_gate": False,
            "source_sha256": _source_hashes(),
            "payload_sha256": _payload_hashes(temporary),
        }
        _write_json(temporary / "candidate_manifest.json", manifest)
        if final.exists():
            raise FileExistsError(f"Refusing to overwrite {final}")
        os.replace(temporary, final)
        return {
            "seed": seed,
            "n_candidates": len(structural),
            "manifest_sha256": _sha256(final / "candidate_manifest.json"),
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_candidate(path: Path, seed: int) -> dict[str, Any]:
    manifest = json.loads((path / "candidate_manifest.json").read_text())
    if manifest.get("seed") != seed or manifest.get("roles_or_scorer_read") is not False:
        raise ValueError(f"Invalid Phase-A manifest for seed {seed}")
    observed = _payload_hashes(path, {"candidate_manifest.json"})
    if observed != manifest.get("payload_sha256"):
        raise ValueError(f"Phase-A payload changed for seed {seed}")
    return manifest


def _phase_b_label(
    seeds: Sequence[int],
    worlds_root: Path,
    candidates_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    direct_rows: list[pd.DataFrame] = []
    for seed in seeds:
        candidate = candidates_root / f"world_seed{seed}"
        _verify_candidate(candidate, seed)
        roles = _read_tsv(
            worlds_root / f"world_seed{seed}" / "scorer" / "role_assignments.tsv"
        )
        role_lookup = roles.set_index("node")["role"].astype(str).to_dict()
        frame = _read_tsv(candidate / "structural_loading_candidates.tsv")
        frame.insert(0, "seed", seed)
        frame["parent_role"] = frame["parent_candidate"].map(role_lookup)
        frame["other_parent_role"] = frame["other_parent_candidate"].map(role_lookup)
        frame["sibling_role"] = frame["nonmotif_sibling_candidate"].map(role_lookup)
        frame["exact_ps_pair"] = frame.apply(
            lambda row: {str(row["parent_role"]), str(row["sibling_role"])}
            == {"P", "S"},
            axis=1,
        )
        frame["false_pair_type"] = frame.apply(
            lambda row: "" if row["exact_ps_pair"] else "-".join(
                sorted((str(row["parent_role"]), str(row["sibling_role"])))
            ),
            axis=1,
        )
        rows.append(frame)
        direct = _read_tsv(candidate / "refined_direct_candidates.tsv")
        direct.insert(0, "seed", seed)
        direct["node1_role"] = direct["node1"].map(role_lookup)
        direct["node2_role"] = direct["node2"].map(role_lookup)
        direct_rows.append(direct)
    score_root = output_root / "train_labels"
    score_root.mkdir(exist_ok=False)
    labeled = pd.concat(rows, ignore_index=True)
    labeled.to_csv(
        score_root / "labeled_structural_loading_candidates.tsv",
        sep="\t",
        index=False,
    )
    direct = pd.concat(direct_rows, ignore_index=True)
    direct.to_csv(
        score_root / "refined_direct_roles_audit.tsv", sep="\t", index=False
    )
    audit = {
        "schema_version": "ca_boss_v4.structure_anchored_train.labels.v1",
        "train_only": True,
        "n_worlds": len(seeds),
        "n_structural_candidates": len(labeled),
        "n_exact_ps_candidates": int(labeled["exact_ps_pair"].sum()),
        "refined_direct_exact_edge_fraction": float(
            np.mean(
                direct.apply(
                    lambda row: {
                        str(row["node1_role"]),
                        str(row["node2_role"]),
                    }
                    in ({"P", "C"}, {"Q", "C"}),
                    axis=1,
                )
            )
        ),
        "all_candidates_frozen_before_roles_open": True,
    }
    _write_json(score_root / "label_audit.json", audit)
    return audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifacts_root = args.artifacts_root.resolve(strict=True)
    first_root = args.first_diagnostic_root.resolve(strict=True)
    worlds_root = args.worlds_root.resolve(strict=True)
    output_root = args.out_root.absolute()
    for path, label in (
        (artifacts_root, "artifacts"),
        (first_root, "first diagnostic"),
        (worlds_root, "worlds"),
        (output_root, "output"),
    ):
        _assert_train_path(path, label)
    seeds = tuple(args.seeds)
    if any(seed not in TRAIN_SEEDS for seed in seeds):
        raise ValueError("Only TRAIN seeds 33001--33024 are allowed")
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be nonempty and unique")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    candidates_root = output_root / "phase_a_candidates"
    candidates_root.mkdir()
    hashes_before = _source_hashes()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _phase_a_world,
                seed,
                str(artifacts_root),
                str(first_root),
                str(candidates_root),
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Phase A complete: seed {result['seed']}", flush=True)
    manifests: dict[str, str] = {}
    for seed in seeds:
        candidate = candidates_root / f"world_seed{seed}"
        payload = _verify_candidate(candidate, seed)
        if payload["source_sha256"] != hashes_before:
            raise RuntimeError(f"Source drift during Phase A for seed {seed}")
        manifests[f"world_seed{seed}"] = _sha256(
            candidate / "candidate_manifest.json"
        )
    if _source_hashes() != hashes_before:
        raise RuntimeError("Source changed during Phase A; refusing roles")
    freeze = {
        "schema_version": "ca_boss_v4.structure_anchored_train.global_freeze.v1",
        "train_only": True,
        "seeds": list(seeds),
        "all_candidates_frozen_before_roles_open": True,
        "candidate_manifest_sha256": manifests,
        "source_sha256": hashes_before,
    }
    _write_json(output_root / "global_phase_a_freeze.json", freeze)
    audit = _phase_b_label(seeds, worlds_root, candidates_root, output_root)
    summary = {
        "schema_version": "ca_boss_v4.structure_anchored_train.run.v1",
        "train_only": True,
        "n_worlds": len(seeds),
        "phase_a_freeze_sha256": _sha256(
            output_root / "global_phase_a_freeze.json"
        ),
        "label_audit": audit,
    }
    _write_json(output_root / "run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--first-diagnostic-root", type=Path, required=True)
    parser.add_argument("--worlds-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))

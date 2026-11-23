#!/usr/bin/env python3
"""Truth-separated scorer for the unpaired MDCR KS baseline.

The discovery module freezes background-only component fits and cross-assay
assignments before this scorer is allowed to open simulator latent states.
This script verifies the frozen manifest and every output hash first.  Truth is
used only to diagnose whether the published all-assay rule, the CA-BOSS
maximum-consensus relaxation, its maximum matched-assay clique, and the two
background-feature folds correspond to one common biological state axis.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import linear_sum_assignment


SCHEMA = "unpaired_mdcr_ks_baseline_v4.scorer.v3"
EXPECTED_EVIDENCE_SCHEMA = "unpaired_mdcr_ks_baseline_v4.v3"
_FORBIDDEN_EVIDENCE_PARTS = {
    "counterfactual",
    "ground_truth",
    "oracle",
    "scorer",
    "truth",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_directory(path: Path, *, evidence: bool) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise FileNotFoundError(path)
    resolved = lexical.resolve(strict=True)
    if evidence and _FORBIDDEN_EVIDENCE_PARTS & {
        part.casefold() for part in resolved.parts
    }:
        raise ValueError(f"Evidence path crosses scorer/truth boundary: {resolved}")
    return resolved


def verify_frozen_evidence(evidence_dir: Path) -> dict[str, Any]:
    """Verify the discovery manifest and all bound output artifacts."""

    directory = _safe_directory(evidence_dir, evidence=True)
    manifest_path = directory / "frozen_evidence_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPECTED_EVIDENCE_SCHEMA:
        raise ValueError("Unexpected evidence schema")
    observed_payload = str(manifest.get("manifest_payload_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if observed_payload != _canonical_sha256(unsigned):
        raise ValueError("Frozen evidence manifest payload hash mismatch")
    if manifest.get("truth_or_scorer_used") is not False:
        raise ValueError("Discovery manifest does not certify truth separation")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Discovery manifest unexpectedly created data")
    for name, expected in manifest.get("output_file_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(expected):
            raise ValueError(f"Frozen output hash mismatch: {name}")
    return manifest


def align_components_to_latent_state(
    component_scores: NDArray[np.float64],
    latent_state: NDArray[np.float64],
) -> pd.DataFrame:
    """Find the maximum-total-absolute-correlation component/state matching."""

    scores = np.asarray(component_scores, dtype=float)
    state = np.asarray(latent_state, dtype=float)
    if scores.ndim != 2 or state.ndim != 2 or len(scores) != len(state):
        raise ValueError("Component scores and latent state must share cell rows")
    if scores.shape[1] != state.shape[1]:
        raise ValueError("Component and latent dimensions must match")
    if not np.isfinite(scores).all() or not np.isfinite(state).all():
        raise ValueError("Alignment inputs must be finite")
    n_components = scores.shape[1]
    correlations = np.corrcoef(scores.T, state.T)[
        :n_components, n_components:
    ]
    absolute = np.abs(correlations)
    component_indices, state_indices = linear_sum_assignment(-absolute)
    rows: list[dict[str, Any]] = []
    for component, state_dimension in sorted(
        zip(component_indices, state_indices, strict=True)
    ):
        alternatives = np.sort(absolute[int(component)])[::-1]
        best = float(absolute[int(component), int(state_dimension)])
        second = float(alternatives[1]) if len(alternatives) > 1 else 0.0
        rows.append(
            {
                "component": int(component),
                "state_dimension": int(state_dimension),
                "correlation": float(
                    correlations[int(component), int(state_dimension)]
                ),
                "absolute_correlation": best,
                "component_relative_margin": float(
                    (best - second) / max(best, 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows)


def _score_assignment(
    *,
    assignment_json: str,
    component_to_dimension: Mapping[tuple[str, int], int],
    modalities: Sequence[str] | None = None,
) -> dict[str, Any]:
    assignment = {
        str(modality): int(component)
        for modality, component in json.loads(assignment_json).items()
    }
    selected_modalities = (
        sorted(assignment)
        if modalities is None
        else sorted(str(value) for value in modalities if str(value))
    )
    if not selected_modalities:
        return {
            "scored_modalities": "",
            "n_scored_modalities": 0,
            "state_dimensions": "",
            "biological_axis_pure": False,
            "majority_state_dimension": None,
            "majority_assays": 0,
            "majority_assay_fraction": 0.0,
        }
    dimensions = [
        int(component_to_dimension[(modality, assignment[modality])])
        for modality in selected_modalities
    ]
    counts = pd.Series(dimensions).value_counts().sort_index()
    maximum = int(counts.max())
    majority_dimension = int(counts.loc[counts.eq(maximum)].index.min())
    return {
        "scored_modalities": ";".join(selected_modalities),
        "n_scored_modalities": len(selected_modalities),
        "state_dimensions": ";".join(str(value) for value in dimensions),
        "biological_axis_pure": len(set(dimensions)) == 1,
        "majority_state_dimension": majority_dimension,
        "majority_assays": maximum,
        "majority_assay_fraction": float(maximum / len(dimensions)),
    }


def _truth_alignment(
    evidence: Path, scorer: Path
) -> tuple[pd.DataFrame, dict[tuple[str, int], int]]:
    scores = pd.read_csv(evidence / "component_scores.tsv", sep="\t")
    component_columns = sorted(
        [column for column in scores if column.startswith("component_")],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    rows: list[pd.DataFrame] = []
    lookup: dict[tuple[str, int], int] = {}
    for modality, frame in scores.groupby("modality", sort=True):
        truth_path = scorer / f"cell_ground_truth__{modality}.npz"
        if truth_path.is_symlink() or not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        with np.load(truth_path, allow_pickle=False) as archive:
            required = {"cell_ids", "latent_state"}
            if missing := sorted(required.difference(archive.files)):
                raise ValueError(f"Truth archive is missing {missing}")
            truth_ids = np.asarray(archive["cell_ids"]).astype(str)
            latent_state = np.asarray(archive["latent_state"], dtype=float)
        observed_ids = frame["cell_id"].astype(str).to_numpy()
        if not np.array_equal(observed_ids, truth_ids):
            raise ValueError(f"Cell IDs do not align for {modality}")
        aligned = align_components_to_latent_state(
            frame.loc[:, component_columns].to_numpy(dtype=float), latent_state
        )
        aligned.insert(0, "modality", str(modality))
        rows.append(aligned)
        lookup.update(
            {
                (str(modality), int(row.component)): int(row.state_dimension)
                for row in aligned.itertuples(index=False)
            }
        )
    return pd.concat(rows, ignore_index=True), lookup


def score_frozen_evidence(
    evidence_dir: Path, scorer_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Verify Phase A, then score component assignments against latent truth."""

    evidence = _safe_directory(evidence_dir, evidence=True)
    manifest = verify_frozen_evidence(evidence)
    scorer = _safe_directory(scorer_dir, evidence=False)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    alignment, component_to_dimension = _truth_alignment(evidence, scorer)

    rows: list[dict[str, Any]] = []

    def append_assignments(
        frame: pd.DataFrame,
        *,
        assignment_type: str,
        id_column: str,
        feature_fold: int | None = None,
        clique: bool = False,
    ) -> None:
        for row in frame.itertuples(index=False):
            selected_modalities = None
            if clique:
                selected_modalities = str(
                    getattr(row, "maximum_match_clique_modalities")
                ).split(";")
            scored = _score_assignment(
                assignment_json=str(getattr(row, "component_assignment_json")),
                component_to_dimension=component_to_dimension,
                modalities=selected_modalities,
            )
            rows.append(
                {
                    "assignment_type": assignment_type,
                    "assignment_id": str(getattr(row, id_column)),
                    "feature_fold": feature_fold,
                    **scored,
                }
            )

    strict = pd.read_csv(evidence / "shared_component_tuples.tsv", sep="\t")
    if len(strict):
        append_assignments(
            strict,
            assignment_type="published_all_assay",
            id_column="shared_component_id",
        )
    consensus = pd.read_csv(
        evidence / "consensus_component_tuples.tsv", sep="\t"
    )
    append_assignments(
        consensus,
        assignment_type="maximum_consensus_all_assay",
        id_column="consensus_component_id",
    )
    append_assignments(
        consensus,
        assignment_type="maximum_consensus_clique",
        id_column="consensus_component_id",
        clique=True,
    )
    folds = pd.read_csv(
        evidence / "fold_consensus_component_tuples.tsv", sep="\t"
    )
    for fold, frame in folds.groupby("feature_fold", sort=True):
        append_assignments(
            frame,
            assignment_type="feature_fold_consensus_all_assay",
            id_column="consensus_component_id",
            feature_fold=int(fold),
        )
        append_assignments(
            frame,
            assignment_type="feature_fold_consensus_clique",
            id_column="consensus_component_id",
            feature_fold=int(fold),
            clique=True,
        )
    assignment_metrics = pd.DataFrame(rows)

    full = assignment_metrics.loc[
        assignment_metrics["assignment_type"].eq(
            "maximum_consensus_all_assay"
        )
    ].set_index("assignment_id")
    fold_all = assignment_metrics.loc[
        assignment_metrics["assignment_type"].eq(
            "feature_fold_consensus_all_assay"
        )
    ]
    stable_axes = 0
    for assignment_id, frame in fold_all.groupby("assignment_id", sort=True):
        if assignment_id not in full.index:
            continue
        full_row = full.loc[assignment_id]
        dimensions = set(frame["majority_state_dimension"].astype(int))
        if (
            bool(full_row["biological_axis_pure"])
            and frame["biological_axis_pure"].astype(bool).all()
            and dimensions == {int(full_row["majority_state_dimension"])}
        ):
            stable_axes += 1

    type_summary: dict[str, Any] = {}
    for assignment_type, frame in assignment_metrics.groupby(
        "assignment_type", sort=True
    ):
        type_summary[str(assignment_type)] = {
            "assignments": len(frame),
            "biologically_pure_assignments": int(
                frame["biological_axis_pure"].astype(bool).sum()
            ),
            "biological_purity_rate": float(
                frame["biological_axis_pure"].astype(bool).mean()
            ),
            "mean_majority_assay_fraction": float(
                frame["majority_assay_fraction"].mean()
            ),
        }
    summary = {
        "schema_version": SCHEMA,
        "evidence_schema_version": manifest["schema_version"],
        "evidence_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "truth_opened_only_after_evidence_hash_verification": True,
        "algorithm_used_truth_for_fit_or_selection": False,
        "new_worlds_or_datasets_created": False,
        "assignment_metrics": type_summary,
        "full_and_all_feature_folds_same_pure_axis_count": stable_axes,
        "full_and_all_feature_folds_same_pure_axis_rate": float(
            stable_axes / len(full) if len(full) else 0.0
        ),
        "exact_cross_assay_feature_pair_claimed": False,
        "cross_assay_direction_claimed": False,
    }
    out_dir.mkdir(parents=True)
    alignment.to_csv(
        out_dir / "component_truth_alignment.tsv", sep="\t", index=False
    )
    assignment_metrics.to_csv(
        out_dir / "assignment_truth_metrics.tsv", sep="\t", index=False
    )
    summary["output_sha256"] = {
        name: _sha256(out_dir / name)
        for name in (
            "component_truth_alignment.tsv",
            "assignment_truth_metrics.tsv",
        )
    }
    summary["summary_payload_sha256"] = _canonical_sha256(summary)
    (out_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = score_frozen_evidence(
        args.evidence_dir, args.scorer_dir, args.out_dir
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

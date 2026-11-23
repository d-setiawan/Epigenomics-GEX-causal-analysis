#!/usr/bin/env python3
"""Filter loading groups by a frozen observable alignment eligibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


_FORBIDDEN = {"scorer", "truth", "ground_truth", "oracle", "counterfactual"}


def _safe_file(path: Path | str, expected_name: str) -> Path:
    lexical = Path(path).absolute()
    if lexical.name != expected_name or lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"Input must be a direct {expected_name} file")
    resolved = lexical.resolve(strict=True)
    if any(part.casefold() in _FORBIDDEN for part in resolved.parts):
        raise ValueError("Refusing scorer/truth alignment input")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_axis_groups(
    projection_dir: Path | str,
    alignment_provenance: Path | str,
    out_dir: Path | str,
) -> dict[str, object]:
    projection = Path(projection_dir).absolute()
    if projection.is_symlink() or not projection.is_dir():
        raise ValueError("Projection input must be a direct directory")
    groups_path = _safe_file(projection / "loading_groups.tsv", "loading_groups.tsv")
    membership_path = _safe_file(
        projection / "loading_group_membership.tsv", "loading_group_membership.tsv"
    )
    provenance_path = _safe_file(alignment_provenance, "alignment_provenance.json")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("information_boundary", {}).get("truth_or_scorer_used") is not False:
        raise ValueError("Alignment provenance lacks a truth-blind information boundary")
    eligible = sorted(str(value) for value in provenance.get("accepted_assays", []))
    abstained = sorted(str(value) for value in provenance.get("abstained_assays", []))
    if not eligible or set(eligible) & set(abstained):
        raise ValueError("Alignment eligibility partition is invalid")
    groups = pd.read_csv(groups_path, sep="\t")
    membership = pd.read_csv(membership_path, sep="\t")
    required_groups = {"loading_group", "modality"}
    required_membership = {"loading_group", "modality"}
    if not required_groups <= set(groups) or not required_membership <= set(membership):
        raise ValueError("Projection tables lack loading-group modality columns")
    selected_groups = groups.loc[groups["modality"].astype(str).isin(eligible)].copy()
    group_ids = set(selected_groups["loading_group"].astype(str))
    selected_membership = membership.loc[
        membership["loading_group"].astype(str).isin(group_ids)
    ].copy()
    output = Path(out_dir).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output}")
    if output.resolve(strict=False) in {projection.resolve(), provenance_path.parent}:
        raise ValueError("Filtered output must differ from its inputs")
    output.mkdir(parents=True, exist_ok=True)
    selected_groups.to_csv(output / "loading_groups.tsv", sep="\t", index=False)
    selected_membership.to_csv(
        output / "loading_group_membership.tsv", sep="\t", index=False
    )
    manifest = {
        "schema_version": "ca_boss_v4.axis_alignment_eligibility_filter.v1",
        "eligible_modalities": eligible,
        "abstained_modalities": abstained,
        "n_input_groups": len(groups),
        "n_output_groups": len(selected_groups),
        "n_output_members": len(selected_membership),
        "input_sha256": {
            "loading_groups.tsv": _sha256(groups_path),
            "loading_group_membership.tsv": _sha256(membership_path),
            "alignment_provenance.json": _sha256(provenance_path),
        },
        "truth_or_scorer_used": False,
    }
    (output / "alignment_eligibility.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-dir", type=Path, required=True)
    parser.add_argument("--alignment-provenance", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = filter_axis_groups(
        args.projection_dir, args.alignment_provenance, args.out_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["filter_axis_groups"]

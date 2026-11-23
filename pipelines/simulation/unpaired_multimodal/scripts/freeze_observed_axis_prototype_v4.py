#!/usr/bin/env python3
"""Freeze truth-blind axis-prototype files before any scorer is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_ALLOWED_ROOT = re.compile(
    r"^sd(?:000|020)_(?:orthogonal_axis|k6_native_axis|ordered_variance_native_axis|global_coarse_mixture_native_axis)_discovery$"
)
_FORBIDDEN = {"scorer", "truth", "ground_truth", "oracle", "counterfactual"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_regular(path: Path, *, label: str) -> Path:
    lexical = path.absolute()
    if lexical.is_symlink() or not lexical.exists() or not lexical.is_file():
        raise ValueError(f"{label} must be a direct existing regular file")
    resolved = lexical.resolve(strict=True)
    if any(component.casefold() in _FORBIDDEN for component in resolved.parts):
        raise ValueError(f"Refusing scorer/truth {label}")
    return resolved


def _safe_root(path: Path) -> Path:
    lexical = path.absolute()
    if not _ALLOWED_ROOT.fullmatch(lexical.name):
        raise ValueError("Prototype root has an unapproved truth-blind family name")
    if lexical.is_symlink() or not lexical.exists() or not lexical.is_dir():
        raise ValueError("Prototype root must be a direct existing directory")
    resolved = lexical.resolve(strict=True)
    if any(component.casefold() in _FORBIDDEN for component in resolved.parts):
        raise ValueError("Refusing scorer/truth prototype root")
    return resolved


def freeze_block(
    roots: list[Path],
    sources: list[Path],
    output: Path,
    *,
    train_seeds: list[int] | None = None,
) -> dict[str, Any]:
    safe_roots = [_safe_root(path) for path in roots]
    if len({path.name for path in safe_roots}) != len(safe_roots):
        raise ValueError("Prototype roots must be unique")
    safe_sources = [_safe_regular(path, label="source") for path in sources]
    files: dict[str, dict[str, Any]] = {}
    for root in sorted(safe_roots):
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("Prototype evidence may not contain symlinks")
            if not path.is_file():
                continue
            resolved = _safe_regular(path, label="prototype evidence")
            relative = f"{root.name}/{resolved.relative_to(root)}"
            files[relative] = {
                "sha256": _sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
    frozen_seeds = sorted(train_seeds or [30018, 30019, 30021, 30024])
    if not frozen_seeds or len(set(frozen_seeds)) != len(frozen_seeds):
        raise ValueError("Frozen TRAIN seeds must be a non-empty unique list")
    expected_seeds = {f"seed{seed}" for seed in frozen_seeds}
    for root in safe_roots:
        observed = {path.name for path in root.iterdir() if path.is_dir()}
        if observed != expected_seeds:
            raise ValueError(f"{root.name} does not contain the frozen TRAIN seed block")
    is_k6 = all("_k6_native_axis_" in path.name for path in safe_roots)
    is_orthogonal = all("_orthogonal_axis_" in path.name for path in safe_roots)
    is_ordered_variance = all(
        "_ordered_variance_native_axis_" in path.name for path in safe_roots
    )
    is_global_mixture = all(
        "_global_coarse_mixture_native_axis_" in path.name for path in safe_roots
    )
    if sum((is_k6, is_orthogonal, is_ordered_variance, is_global_mixture)) != 1:
        raise ValueError("A freeze block may not mix prototype families")
    noise_lookup = {"sd000": 0.0, "sd020": 0.2}
    noise_levels = sorted(
        noise_lookup[path.name.split("_", maxsplit=1)[0]] for path in safe_roots
    )
    manifest = {
        "schema_version": (
            "ca_boss_v4.k6_native_axis_pretruth_freeze.v1"
            if is_k6
            else (
                "ca_boss_v4.ordered_variance_native_axis_pretruth_freeze.v1"
                if is_ordered_variance
                else (
                    "ca_boss_v4.global_coarse_mixture_native_axis_pretruth_freeze.v1"
                    if is_global_mixture
                    else "ca_boss_v4.orthogonal_axis_pretruth_freeze.v1"
                )
            )
        ),
        "frozen_before_scorer_access": True,
        "train_seeds": frozen_seeds,
        "noise_levels": noise_levels,
        "prototype_roots": [path.name for path in safe_roots],
        "source_sha256": {str(path): _sha256(path) for path in safe_sources},
        "files": files,
        "information_boundary": {
            "scorer_or_truth_read": False,
            "paired_cells_used": False,
            "external_data_read": False,
            "loading_views_are_independent_worlds": False,
        },
    }
    target = output.absolute()
    allowed_output = (
        "k6_axis_pretruth_freeze_manifest.json"
        if is_k6
        else (
            "ordered_variance_axis_pretruth_freeze_manifest.json"
            if is_ordered_variance
            else (
                "global_coarse_mixture_axis_pretruth_freeze_manifest.json"
                if is_global_mixture
                else "orthogonal_axis_pretruth_freeze.json"
            )
        )
    )
    if target.is_symlink() or target.name != allowed_output:
        raise ValueError(f"Freeze output must be named {allowed_output}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    digest = _sha256(target)
    target.with_suffix(".sha256").write_text(digest + "\n")
    return {**manifest, "manifest_sha256": digest}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--train-seed", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = freeze_block(
        args.root, args.source, args.output, train_seeds=args.train_seed
    )
    print(json.dumps({"manifest_sha256": result["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

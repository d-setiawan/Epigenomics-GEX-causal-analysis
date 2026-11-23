#!/usr/bin/env python3
"""Reorganize graph output directories into a method-first tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _common import infer_repo_root, sanitize_token


def infer_method(graph_dir: Path) -> str:
    summary_path = graph_dir / "run_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            method = summary.get("params", {}).get("method")
            if method:
                return sanitize_token(str(method))
        except json.JSONDecodeError:
            pass
    return sanitize_token(graph_dir.name.split("_", 1)[0])


def infer_dagma_family(graph_dir: Path) -> str | None:
    if infer_method(graph_dir) != "dagma":
        return None
    summary_path = graph_dir / "run_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            family = (
                summary.get("dagma", {}).get("family_config")
                or summary.get("params", {}).get("dagma_family_config")
            )
            if family:
                return sanitize_token(str(family))
        except json.JSONDecodeError:
            pass
    return None


def infer_target(outputs_root: Path, graph_dir: Path) -> Path | None:
    try:
        rel = graph_dir.resolve().relative_to(outputs_root.resolve())
    except ValueError:
        return None

    parts = rel.parts
    method = infer_method(graph_dir)
    methods_root = outputs_root / "methods" / method
    dagma_family = infer_dagma_family(graph_dir)

    if len(parts) >= 6 and parts[0] == "datasets":
        run_id = parts[1]
        dataset_name = parts[2]
        gene_name = parts[3]
        dataset_id = sanitize_token(f"{run_id}__{dataset_name}")
        if method == "dagma" and dagma_family:
            return methods_root / dataset_id / dagma_family / gene_name / graph_dir.name
        return methods_root / dataset_id / gene_name / graph_dir.name

    if len(parts) >= 4 and parts[0] in {"locus_matrices_matched", "locus_matrices"}:
        dataset_family = parts[0]
        run_id = parts[1]
        locus_name = parts[2]
        dataset_id = sanitize_token(f"{run_id}__{dataset_family}")
        if method == "dagma" and dagma_family:
            return methods_root / dataset_id / dagma_family / locus_name / graph_dir.name
        return methods_root / dataset_id / locus_name / graph_dir.name

    if len(parts) >= 5 and parts[0] == "methods":
        method_from_path = sanitize_token(parts[1])
        if method_from_path != method:
            return None
        dataset_id = parts[2]
        maybe_gene_or_family = parts[3]
        if method == "dagma" and dagma_family:
            if maybe_gene_or_family == dagma_family:
                return None
            return outputs_root / "methods" / method / dataset_id / dagma_family / maybe_gene_or_family / graph_dir.name
        return None

    return None


def candidate_graph_dirs(outputs_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in [
        outputs_root / "datasets",
        outputs_root / "locus_matrices_matched",
        outputs_root / "locus_matrices",
        outputs_root / "methods",
    ]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_dir():
                continue
            if not (path / "run_summary.json").exists():
                continue
            if not (path.name.startswith("pc_") or path.name.startswith("fci_") or path.name.startswith("dagma_")):
                continue
            candidates.append(path)
    return sorted(set(candidates))


def main() -> int:
    parser = argparse.ArgumentParser(description="Move graph output directories into pipelines/empirical/analysis/outputs/methods/<method>/<dataset>/<gene-or-locus>/<result>.")
    parser.add_argument("--repo-root", default=str(infer_repo_root()))
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    outputs_root = repo_root / "pipelines/empirical/analysis" / "outputs"
    moves = []
    for graph_dir in candidate_graph_dirs(outputs_root):
        target = infer_target(outputs_root, graph_dir)
        if target is None:
            continue
        if target.resolve() == graph_dir.resolve():
            continue
        moves.append((graph_dir, target))

    if not moves:
        print("No graph output directories needed reorganization.")
        return 0

    for src, dst in moves:
        print(f"{src} -> {dst}")
    if args.dry_run:
        print(f"Planned moves: {len(moves)}")
        return 0

    moved = 0
    skipped = 0
    for src, dst in moves:
        if dst.exists():
            print(f"Skipping existing destination: {dst}")
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1

    print(f"Moved: {moved}")
    print(f"Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

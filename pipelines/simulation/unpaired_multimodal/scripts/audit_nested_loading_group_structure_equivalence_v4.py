#!/usr/bin/env python3
"""Two-phase cached audit of the CA-BOSS v4.32 equivalence layer.

Phase A verifies and binds the valid v4.31 observable freeze, the fixed v4.32
set-valued transformation, and the already sampled cached panels.  It never
opens the v4.31 Phase-B predictions or simulator truth.  Phase B verifies that
closure, opens the existing labels, applies v4.32 to a truth-free projection
of each v4.31 decision table, and scores candidate-pair *coverage* with a
one-to-one bipartite matching.  Candidate alternatives are never expanded or
scored as four discovered edges.

The script has no world-generation or missing-data path and makes no fresh
locked claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_nested_loading_group_competing_parent_observability_v4 as v431_audit  # noqa: E402
import loading_group_competing_parent_observability_guard_v4 as v431_guard  # noqa: E402
import loading_group_structure_equivalence_v4 as equivalence  # noqa: E402


SCHEMA = "ca_boss_v4.nested_loading_group_structure_equivalence.v1"
WORLD_KEY = ["block", "seed"]
GROUP_KEY = ["block", "seed", "loading_group"]
PHASE_A_PANEL = "phase_a_random_panels.tsv"
PHASE_B_HYPOTHESES = "phase_b_structure_equivalence_hypotheses.tsv"
PHASE_B_WORLD = "phase_b_world_metrics.tsv"
PHASE_B_BLOCK = "phase_b_source_block_metrics.tsv"
PHASE_B_PANEL = "phase_b_random_panel_metrics.tsv"
PHASE_B_TABLES = (
    PHASE_B_HYPOTHESES,
    PHASE_B_WORLD,
    PHASE_B_BLOCK,
    PHASE_B_PANEL,
)
SOURCE_FILES = (
    "audit_nested_loading_group_structure_equivalence_v4.py",
    "loading_group_structure_equivalence_v4.py",
    "audit_nested_loading_group_competing_parent_observability_v4.py",
    "loading_group_competing_parent_observability_guard_v4.py",
)
DEFAULT_V431_DIR = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_nested_loading_parent_observability_v431_full_pool_final"
)
DEFAULT_V427_DIR = v431_audit.DEFAULT_V427_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(SCRIPT_DIR / name) for name in SOURCE_FILES}


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    observed = str(value.get(field, ""))
    unsigned = dict(value)
    unsigned.pop(field, None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError(f"{label} payload hash mismatch")
    return observed


def _safe_file(path: Path, *, phase_a: bool) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise FileNotFoundError(path)
    resolved = lexical.resolve(strict=True)
    if phase_a and {"scorer", "truth"} & {
        part.casefold() for part in resolved.parts
    }:
        raise ValueError(f"Phase A refuses scorer/truth path: {resolved}")
    return resolved


def _load_json(path: Path, *, phase_a: bool) -> tuple[Path, dict[str, Any]]:
    safe = _safe_file(path, phase_a=phase_a)
    value = json.loads(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {safe}")
    return safe, value


def run_phase_a(*, out_dir: Path, v431_dir: Path) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-A output: {out_dir}")
    upstream_root = v431_dir.expanduser().absolute().resolve(strict=True)
    upstream_manifest_path = _safe_file(
        upstream_root / "phase_a_manifest.json", phase_a=True
    )
    upstream_manifest, upstream_tables = v431_audit._load_phase_a(
        upstream_root
    )
    panels = upstream_tables["panels"].copy()
    required_panel = {"panel_id", "selection_seed", "block", "seed"}
    if missing := sorted(required_panel.difference(panels.columns)):
        raise ValueError(f"v4.31 cached panels are missing {missing}")
    if panels.duplicated(["panel_id", *WORLD_KEY]).any():
        raise ValueError("v4.31 cached panels contain duplicate worlds")

    out_dir.mkdir(parents=True)
    panel_path = out_dir / PHASE_A_PANEL
    panels.to_csv(panel_path, sep="\t", index=False)

    artifact = {
        "schema_version": equivalence.SCHEMA,
        "algorithm": equivalence.ALGORITHM,
        "fixed_rule": equivalence.FIXED_RULE,
        "artifact_role": "future_application_rule_binding",
        "application_requires_truth": False,
        "numeric_threshold_fitted": False,
        "state_route_output": equivalence.STATE_EQUIVALENCE_TYPE,
        "state_route_candidate_pair_count": 4,
        "axis_route_output": equivalence.AXIS_EXACT_TYPE,
        "axis_route_candidate_pair_count": 1,
        "candidate_pair_expansion_permitted": False,
        "direct_causal_relation_claimed": False,
        "exact_cross_assay_pair_claimed": False,
        "changes_v431_selection": False,
        "changes_state_v427_membership": False,
        "changes_direct_candidates": False,
        "new_simulation_worlds_or_datasets_created": False,
        "source_sha256": _source_hashes(),
    }
    artifact["artifact_payload_sha256"] = _canonical_sha256(artifact)
    artifact_path = out_dir / "phase_a_fixed_equivalence_rule.json"
    _write_json(artifact_path, artifact)

    manifest = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": equivalence.SCHEMA,
        "phase": "a",
        "cached_inputs_only": True,
        "all_observable_inputs_loaded_before_truth": True,
        "upstream_v431_phase_b_opened": False,
        "scorer_or_truth_opened": False,
        "new_simulation_worlds_or_datasets_created": False,
        "fresh_locked_claim": False,
        "retrospective_development_only": True,
        "upstream_v431_dir": str(upstream_root),
        "upstream_v431_phase_a_manifest": str(upstream_manifest_path),
        "upstream_v431_phase_a_manifest_sha256": _sha256(
            upstream_manifest_path
        ),
        "upstream_v431_phase_a_manifest_payload_sha256": upstream_manifest[
            "manifest_payload_sha256"
        ],
        "worlds": int(upstream_manifest["worlds"]),
        "source_blocks": int(upstream_manifest["source_blocks"]),
        "random_panels": int(upstream_manifest["random_panels"]),
        "panel_size": int(upstream_manifest["panel_size"]),
        "selection_seed": int(upstream_manifest["selection_seed"]),
        "fixed_equivalence_artifact_sha256": _sha256(artifact_path),
        "output_sha256": {PHASE_A_PANEL: _sha256(panel_path)},
        "source_sha256": _source_hashes(),
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _load_phase_a(
    out_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    _, manifest = _load_json(out_dir / "phase_a_manifest.json", phase_a=True)
    _validate_self_hash(
        manifest, field="manifest_payload_sha256", label="v4.32 Phase-A manifest"
    )
    required = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": equivalence.SCHEMA,
        "phase": "a",
        "all_observable_inputs_loaded_before_truth": True,
        "upstream_v431_phase_b_opened": False,
        "scorer_or_truth_opened": False,
        "new_simulation_worlds_or_datasets_created": False,
    }
    bad = {
        key: manifest.get(key)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if bad:
        raise ValueError(f"Unsupported or unsafe v4.32 Phase-A manifest: {bad}")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("v4.32 sources changed after Phase A")

    artifact_path = _safe_file(
        out_dir / "phase_a_fixed_equivalence_rule.json", phase_a=True
    )
    if manifest.get("fixed_equivalence_artifact_sha256") != _sha256(
        artifact_path
    ):
        raise ValueError("Fixed v4.32 equivalence artifact changed")
    _, artifact = _load_json(artifact_path, phase_a=True)
    _validate_self_hash(
        artifact,
        field="artifact_payload_sha256",
        label="fixed v4.32 equivalence rule",
    )
    if (
        artifact.get("schema_version") != equivalence.SCHEMA
        or artifact.get("fixed_rule") != equivalence.FIXED_RULE
        or artifact.get("numeric_threshold_fitted") is not False
        or artifact.get("candidate_pair_expansion_permitted") is not False
    ):
        raise ValueError("Fixed v4.32 equivalence rule changed")

    upstream_root = Path(str(manifest["upstream_v431_dir"]))
    upstream_manifest_path = _safe_file(
        Path(str(manifest["upstream_v431_phase_a_manifest"])), phase_a=True
    )
    if manifest.get("upstream_v431_phase_a_manifest_sha256") != _sha256(
        upstream_manifest_path
    ):
        raise ValueError("Bound v4.31 Phase-A manifest changed")
    upstream_manifest, _ = v431_audit._load_phase_a(upstream_root)
    if (
        manifest.get("upstream_v431_phase_a_manifest_payload_sha256")
        != upstream_manifest.get("manifest_payload_sha256")
    ):
        raise ValueError("Bound v4.31 Phase-A payload changed")

    panel_path = _safe_file(out_dir / PHASE_A_PANEL, phase_a=True)
    if manifest.get("output_sha256", {}).get(PHASE_A_PANEL) != _sha256(
        panel_path
    ):
        raise ValueError("v4.32 random panels changed")
    panels = pd.read_csv(panel_path, sep="\t", keep_default_na=False)
    return manifest, panels


def maximum_truth_matching(
    candidate_sets: Sequence[set[frozenset[str]]],
    truth_pairs: set[frozenset[str]],
) -> dict[int, frozenset[str]]:
    """Return a deterministic maximum matching from hypotheses to truth pairs."""

    truth_to_hypothesis: dict[frozenset[str], int] = {}

    def augment(hypothesis: int, seen: set[frozenset[str]]) -> bool:
        ordered = sorted(
            candidate_sets[hypothesis], key=lambda pair: tuple(sorted(pair))
        )
        for pair in ordered:
            if pair not in truth_pairs or pair in seen:
                continue
            seen.add(pair)
            incumbent = truth_to_hypothesis.get(pair)
            if incumbent is None or augment(incumbent, seen):
                truth_to_hypothesis[pair] = hypothesis
                return True
        return False

    for hypothesis in range(len(candidate_sets)):
        augment(hypothesis, set())
    return {
        hypothesis: pair for pair, hypothesis in truth_to_hypothesis.items()
    }


def _candidate_set(value: str) -> set[frozenset[str]]:
    return {
        frozenset(pair) for pair in equivalence.parse_candidate_pairs(value)
    }


def _pair_text(pair: frozenset[str] | None) -> str:
    return "|||".join(sorted(pair)) if pair else ""


def _score_hypotheses(
    frame: pd.DataFrame, *, total_true_pairs: int
) -> dict[str, Any]:
    selected = frame["hypothesis_selected"].astype(bool)
    matched = frame["matched_to_unique_truth_pair"].astype(bool)
    selected_count = int(selected.sum())
    matched_count = int((selected & matched).sum())
    return {
        "selected_hypotheses": selected_count,
        "matched_unique_truth_pairs": matched_count,
        "unmatched_hypotheses": int(selected_count - matched_count),
        "false_negative_truth_pairs": int(total_true_pairs - matched_count),
        "coverage_precision": (
            float(matched_count / selected_count) if selected_count else 0.0
        ),
        "coverage_recall": (
            float(matched_count / total_true_pairs) if total_true_pairs else 0.0
        ),
        "sentinel_contaminated_hypotheses": int(
            (selected & frame["contains_sentinel_candidate"].astype(bool)).sum()
        ),
        "exact_loading_groups": int(
            (selected & frame["exact_loading_group_selected"].astype(bool)).sum()
        ),
        "structure_equivalence_hyperedges": int(
            (selected & frame["structure_equivalence_selected"].astype(bool)).sum()
        ),
        "candidate_pairs_if_improperly_expanded": int(
            frame.loc[selected, "candidate_pair_count"].astype(int).sum()
        ),
    }


def _verify_upstream_phase_b(
    manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    root = Path(str(manifest["upstream_v431_dir"])).resolve(strict=True)
    summary_path, summary = _load_json(
        root / "phase_b_summary.json", phase_a=False
    )
    _validate_self_hash(
        summary, field="summary_payload_sha256", label="v4.31 Phase-B summary"
    )
    if (
        summary.get("algorithm_label") != "CA-BOSS v4.31"
        or summary.get("algorithm_schema_version") != v431_guard.SCHEMA
        or summary.get("all_gates_passed") is not True
        or summary.get("new_simulation_worlds_or_datasets_created") is not False
    ):
        raise ValueError("Upstream v4.31 Phase-B result is not the valid audit")
    if summary.get("phase_a_manifest_sha256") != manifest.get(
        "upstream_v431_phase_a_manifest_sha256"
    ):
        raise ValueError("v4.31 Phase A/B binding disagrees")
    predictions_path = _safe_file(
        root / "phase_b_outer_group_predictions.tsv", phase_a=False
    )
    expected = summary.get("output_sha256", {}).get(
        "phase_b_outer_group_predictions.tsv"
    )
    if expected != _sha256(predictions_path):
        raise ValueError("Upstream v4.31 group predictions changed")
    predictions = pd.read_csv(
        predictions_path, sep="\t", keep_default_na=False
    )
    return summary_path, summary, predictions


def run_phase_b(
    *,
    out_dir: Path,
    worlds_root: Path,
    v427_dir: Path,
) -> None:
    for name in (*PHASE_B_TABLES, "phase_b_summary.json"):
        if (out_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite Phase-B output: {name}")

    # Verify every Phase-A rule and observable binding before opening v4.31
    # predictions, v4.27 truth roles, or simulator truth below.
    manifest, panels = _load_phase_a(out_dir)
    upstream_summary_path, upstream_summary, predictions = (
        _verify_upstream_phase_b(manifest)
    )
    feature_predictions, v427_provenance = (
        v431_audit.v429_audit._verified_v427_predictions(v427_dir)
    )
    if predictions.duplicated(GROUP_KEY).any():
        raise ValueError("Upstream v4.31 predictions contain duplicate groups")
    if missing := sorted(
        {"truth_pair", "contains_sentinel", *GROUP_KEY}.difference(
            predictions.columns
        )
    ):
        raise ValueError(f"Upstream v4.31 predictions are missing {missing}")

    role_lookup = feature_predictions.set_index(
        ["block", "seed", "feature"]
    )["truth_role"].astype(str)
    output_frames: list[pd.DataFrame] = []
    world_rows: list[dict[str, Any]] = []
    truth_hashes: dict[str, str] = {}
    expected_truth_hashes = upstream_summary.get("truth_input_sha256", {})

    for world, world_predictions in predictions.groupby(
        WORLD_KEY, sort=True
    ):
        block, seed = str(world[0]), int(world[1])
        safe_inputs = world_predictions.loc[
            :, sorted(equivalence.REQUIRED_COLUMNS)
        ].copy()
        transformed = equivalence.build_loading_group_structure_equivalence(
            safe_inputs
        ).decisions
        transformed.insert(0, "seed", seed)
        transformed.insert(0, "block", block)

        truth_path = (
            worlds_root
            / block
            / f"world_seed{seed}"
            / "scorer"
            / "truth_typed_edges.tsv"
        )
        truth_pairs, truth_hash = v431_audit.v429_audit._truth_pairs(
            truth_path
        )
        key = f"{block}:{seed}"
        if expected_truth_hashes.get(key) != truth_hash:
            raise ValueError(f"Truth input changed since v4.31: {key}")
        truth_hashes[key] = truth_hash

        candidate_sets = [
            _candidate_set(value)
            for value in transformed["candidate_pairs_json"].astype(str)
        ]
        matching = maximum_truth_matching(candidate_sets, truth_pairs)
        transformed["candidate_truth_pair_count"] = [
            len(candidates.intersection(truth_pairs))
            for candidates in candidate_sets
        ]
        transformed["contains_any_true_pair"] = transformed[
            "candidate_truth_pair_count"
        ].gt(0)
        transformed["matched_to_unique_truth_pair"] = [
            index in matching for index in range(len(transformed))
        ]
        transformed["matched_truth_pair_key"] = [
            _pair_text(matching.get(index))
            for index in range(len(transformed))
        ]
        transformed["v431_nominal_truth_pair"] = world_predictions.sort_values(
            ["modality", "loading_group"]
        )["truth_pair"].astype(bool).to_numpy()

        sentinel_flags: list[bool] = []
        for candidates in candidate_sets:
            nodes = sorted(set().union(*(set(pair) for pair in candidates)))
            roles: list[str] = []
            for node in nodes:
                try:
                    roles.append(str(role_lookup.loc[(block, seed, node)]))
                except KeyError as error:
                    raise ValueError(
                        f"Candidate node lacks a bound truth-role row: {key}:{node}"
                    ) from error
            sentinel_flags.append(any(role in {"N1", "N2"} for role in roles))
        transformed["contains_sentinel_candidate"] = sentinel_flags
        transformed["truth_used_for_equivalence_application"] = False
        transformed["truth_used_for_phase_b_scoring"] = True
        output_frames.append(transformed)

        score = _score_hypotheses(
            transformed, total_true_pairs=len(truth_pairs)
        )
        world_rows.append(
            {
                "block": block,
                "seed": seed,
                "truth_pairs": len(truth_pairs),
                **score,
                "all_selected_hypotheses_match_unique_truth": bool(
                    score["unmatched_hypotheses"] == 0
                ),
                "zero_sentinel_contamination": bool(
                    score["sentinel_contaminated_hypotheses"] == 0
                ),
            }
        )

    hypotheses = pd.concat(output_frames, ignore_index=True)
    if hypotheses.duplicated(GROUP_KEY).any():
        raise RuntimeError("v4.32 hypotheses contain duplicate group keys")
    world_metrics = pd.DataFrame(world_rows).sort_values(WORLD_KEY).reset_index(
        drop=True
    )
    total_truth = int(world_metrics["truth_pairs"].sum())
    full_score = _score_hypotheses(hypotheses, total_true_pairs=total_truth)

    block_rows: list[dict[str, Any]] = []
    for block, frame in hypotheses.groupby("block", sort=True):
        block_total = int(
            world_metrics.loc[world_metrics["block"].eq(block), "truth_pairs"].sum()
        )
        block_rows.append(
            {"block": str(block), **_score_hypotheses(frame, total_true_pairs=block_total)}
        )
    block_metrics = pd.DataFrame(block_rows)

    panel_rows: list[dict[str, Any]] = []
    for panel, selected_worlds in panels.groupby("panel_id", sort=True):
        keys = set(
            zip(
                selected_worlds["block"].astype(str),
                selected_worlds["seed"].astype(int),
            )
        )
        use = hypotheses.apply(
            lambda row: (str(row["block"]), int(row["seed"])) in keys,
            axis=1,
        )
        world_use = world_metrics.apply(
            lambda row: (str(row["block"]), int(row["seed"])) in keys,
            axis=1,
        )
        panel_total = int(world_metrics.loc[world_use, "truth_pairs"].sum())
        score = _score_hypotheses(
            hypotheses.loc[use], total_true_pairs=panel_total
        )
        panel_rows.append(
            {
                "panel_id": str(panel),
                "selection_seed": int(selected_worlds["selection_seed"].iloc[0]),
                "worlds": len(keys),
                **score,
            }
        )
    panel_metrics = pd.DataFrame(panel_rows)

    hypotheses.to_csv(out_dir / PHASE_B_HYPOTHESES, sep="\t", index=False)
    world_metrics.to_csv(out_dir / PHASE_B_WORLD, sep="\t", index=False)
    block_metrics.to_csv(out_dir / PHASE_B_BLOCK, sep="\t", index=False)
    panel_metrics.to_csv(out_dir / PHASE_B_PANEL, sep="\t", index=False)

    state = hypotheses.loc[
        hypotheses["structure_equivalence_selected"].astype(bool)
    ]
    axis = hypotheses.loc[
        hypotheses["exact_loading_group_selected"].astype(bool)
    ]
    gates = {
        "all_selected_hypotheses_match_distinct_truth_pairs": (
            full_score["unmatched_hypotheses"] == 0
        ),
        "coverage_precision_is_one": full_score["coverage_precision"] == 1.0,
        "coverage_recall_at_least_0.83": full_score["coverage_recall"] >= 0.83,
        "zero_sentinel_contaminated_hypotheses": (
            full_score["sentinel_contaminated_hypotheses"] == 0
        ),
        "state_hypotheses_are_four_pair_unexpanded_equivalence_sets": bool(
            state["candidate_pair_count"].astype(int).eq(4).all()
            and ~state["exact_pair_identified"].astype(bool).any()
            and ~state["candidate_pairs_may_be_expanded_as_edges"].astype(bool).any()
        ),
        "axis_hypotheses_remain_one_pair_exact_groups": bool(
            axis["candidate_pair_count"].astype(int).eq(1).all()
            and axis["exact_pair_identified"].astype(bool).all()
        ),
        "no_direct_or_cross_assay_pair_claim": bool(
            ~hypotheses["direct_causal_relation_claimed"].astype(bool).any()
            and ~hypotheses["exact_cross_assay_pair_claimed"].astype(bool).any()
        ),
        "v431_selection_count_unchanged": int(
            predictions["retained"].astype(bool).sum()
        )
        == full_score["selected_hypotheses"],
        "no_new_worlds_or_datasets_created": True,
    }
    summary = {
        "schema_version": SCHEMA,
        "algorithm": equivalence.ALGORITHM,
        "algorithm_schema_version": equivalence.SCHEMA,
        "phase": "b",
        "cached_inputs_only": True,
        "new_simulation_worlds_or_datasets_created": False,
        "fresh_locked_claim": False,
        "retrospective_development_only": True,
        "worlds": int(world_metrics.shape[0]),
        "source_blocks": int(world_metrics["block"].nunique()),
        "truth_within_assay_shared_state_pairs": total_truth,
        "v431_nominal_point_baseline": upstream_summary[
            "v431_guarded_loading_groups"
        ],
        "v432_set_valued_hypotheses": full_score,
        "state_structure_equivalence": {
            "hyperedges": int(len(state)),
            "candidate_pairs_per_hyperedge": 4,
            "candidate_pairs_are_alternatives_not_edges": True,
            "exact_pair_identified": False,
            "matched_unique_truth_pairs": int(
                state["matched_to_unique_truth_pair"].astype(bool).sum()
            ),
        },
        "axis_exact_groups": {
            "groups": int(len(axis)),
            "candidate_pairs_per_group": 1,
            "matched_unique_truth_pairs": int(
                axis["matched_to_unique_truth_pair"].astype(bool).sum()
            ),
        },
        "random_cached_panels": {
            "panels": int(len(panel_metrics)),
            "world_evaluations": int(panel_metrics["worlds"].sum()),
            "selection_seeds": panel_metrics["selection_seed"].astype(int).tolist(),
            "mean_coverage_precision": float(
                panel_metrics["coverage_precision"].mean()
            ),
            "mean_coverage_recall": float(
                panel_metrics["coverage_recall"].mean()
            ),
        },
        "gates": gates,
        "all_gates_passed": bool(all(gates.values())),
        "identifiability_boundary": {
            "estimand": (
                "assay-local shared-state loading/structure equivalence class "
                "with one-to-one truth coverage scoring"
            ),
            "candidate_pairs_are_joint_alternatives": True,
            "candidate_pairs_may_be_expanded_as_edges": False,
            "cell_specific_cross_assay_direction": "not identifiable",
            "exact_cross_assay_feature_correspondence": "not claimed",
            "new_data_used": False,
        },
        "phase_a_manifest_sha256": _sha256(
            out_dir / "phase_a_manifest.json"
        ),
        "upstream_v431_phase_b_summary": str(upstream_summary_path),
        "upstream_v431_phase_b_summary_sha256": _sha256(
            upstream_summary_path
        ),
        "upstream_v431_phase_b_summary_payload_sha256": upstream_summary[
            "summary_payload_sha256"
        ],
        "upstream_v427": v427_provenance,
        "truth_input_sha256": truth_hashes,
        "source_sha256": _source_hashes(),
        "output_sha256": {
            name: _sha256(out_dir / name) for name in PHASE_B_TABLES
        },
    }
    summary["summary_payload_sha256"] = _canonical_sha256(summary)
    _write_json(out_dir / "phase_b_summary.json", summary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--v431-dir", type=Path, default=DEFAULT_V431_DIR)
    parser.add_argument(
        "--worlds-root", type=Path, default=SCRIPT_DIR.parent / "formal_worlds"
    )
    parser.add_argument("--v427-dir", type=Path, default=DEFAULT_V427_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.phase == "a":
        run_phase_a(out_dir=args.out_dir, v431_dir=args.v431_dir)
    else:
        run_phase_b(
            out_dir=args.out_dir,
            worlds_root=args.worlds_root,
            v427_dir=args.v427_dir,
        )


if __name__ == "__main__":
    main()

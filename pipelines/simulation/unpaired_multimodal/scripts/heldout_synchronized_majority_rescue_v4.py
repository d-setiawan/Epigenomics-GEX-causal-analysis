#!/usr/bin/env python3
"""Held-out synchronized majority rescue for CA-BOSS v4.46.

CA-BOSS v4.44 requires an isolated two-assay component to match in every
statistical context.  CA-BOSS v4.46 replaces that brittle special case with
two distinct observable requirements: strict-majority ordinary component
matching across the full fit and two disjoint background-feature folds, plus
reporter co-assignment in at least one *held-out-feature* global multi-assay
permutation frame from v4.45.  Only v4.44-unrepresented pairs are added.

The output is a reporter-level shared-state hypothesis.  It is never a direct
regulatory edge, never directed, and never expands feature alternatives.
This stage verifies frozen upstream hashes and has no scorer or truth input.
"""

from __future__ import annotations

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402


SCHEMA = "ca_boss_v4.heldout_synchronized_majority_rescue.v1"
ALGORITHM = "CA-BOSS v4.46 held-out synchronized strict-majority rescue"
EXPECTED_SYNCHRONIZATION_SCHEMA = (
    "ca_boss_v4.multi_assay_permutation_synchronization.v1"
)
FIXED_SELECTION_RULE = (
    "among_v444_unrepresented_assay_pairs_require_both_assays_to_have_"
    "component_resolved_within_assay_hypotheses_context_identified_"
    "reporters_and_all_context_reporter_bootstrap_support_then_require_"
    "ordinary_pairwise_component_matching_in_a_strict_majority_of_the_"
    "full_fit_and_two_disjoint_background_feature_folds_and_reporter_"
    "coassignment_in_at_least_one_of_the_two_heldout_feature_global_"
    "multi_assay_permutation_frames_then_emit_one_unexpanded_reporter_"
    "level_cross_assay_shared_state_hypothesis_with_no_direct_or_"
    "directional_claim"
)

CANDIDATE_COLUMNS = [
    "synchronized_majority_hyperedge_id",
    "modalities",
    "n_modalities",
    "candidate_assay_pairs_json",
    "candidate_assay_pair_count",
    "external_state_reporter_members",
    "candidate_cross_assay_feature_pairs_json",
    "candidate_cross_assay_feature_pair_count",
    "statistical_context_support_pattern",
    "statistical_context_support_count",
    "statistical_context_support_fraction",
    "global_reporter_coassignment_context_pattern",
    "global_reporter_coassignment_context_count",
    "heldout_global_reporter_coassignment_pattern",
    "heldout_global_reporter_coassignment_count",
    "both_assays_within_assay_pair_resolved_by_component",
    "both_reporters_context_components_identified",
    "both_reporters_all_context_bootstrap_supported",
    "v444_assay_pair_already_represented",
    "hypothesis_selected",
    "selection_status",
    "hypothesis_type",
    "within_assay_direct_structure_claimed",
    "exact_assay_participation_claimed",
    "assay_participation_is_set_valued",
    "exact_cross_assay_pair_claimed",
    "cross_assay_direction_claimed",
    "candidate_pairs_may_be_expanded_as_edges",
    "fixed_selection_rule",
]

AUDIT_COLUMNS = [
    "left_modality",
    "right_modality",
    "statistical_context_support_pattern",
    "statistical_context_support_count",
    "global_reporter_coassignment_context_pattern",
    "global_reporter_coassignment_context_count",
    "heldout_global_reporter_coassignment_pattern",
    "heldout_global_reporter_coassignment_count",
    "both_assays_within_assay_pair_resolved_by_component",
    "both_reporters_context_components_identified",
    "both_reporters_all_context_bootstrap_supported",
    "v444_assay_pair_already_represented",
    "hypothesis_selected",
    "selection_status",
    "candidate_id",
]


def verify_synchronization(
    synchronization_dir: Path,
    bridge: Path,
    bridge_provenance: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Verify the v4.45 pre-truth artifact and its v4.44 binding."""

    directory = synchronization._forbid_truth_path(synchronization_dir)
    manifest_path = directory / "synchronization_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPECTED_SYNCHRONIZATION_SCHEMA:
        raise ValueError("Unexpected synchronization schema")
    observed = str(manifest.get("manifest_payload_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if observed != synchronization._canonical_sha256(unsigned):
        raise ValueError("Synchronization manifest payload hash mismatch")
    boundary = manifest.get("information_boundary", {})
    if boundary.get("scorer_truth_used") is not False:
        raise ValueError("Synchronization did not certify truth separation")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("Synchronization did not satisfy the pre-truth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Synchronization created a new world or dataset")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if synchronization._sha256(path) != str(expected):
            raise ValueError(f"Synchronization output hash mismatch: {name}")
    if str(manifest.get("bridge_provenance_payload_sha256")) != str(
        bridge_provenance["provenance_payload_sha256"]
    ):
        raise ValueError("Synchronization and bridge payload hashes disagree")
    if str(manifest.get("bridge_provenance_file_sha256")) != (
        synchronization._sha256(bridge / "provenance.json")
    ):
        raise ValueError("Synchronization and bridge file hashes disagree")
    if (
        str(manifest.get("block")) != str(bridge_provenance.get("block"))
        or int(manifest.get("seed")) != int(bridge_provenance.get("seed"))
    ):
        raise ValueError("Synchronization and bridge identities disagree")
    return directory, manifest


def _context_pair_patterns(
    contexts: pd.DataFrame,
) -> tuple[list[str], dict[frozenset[str], str]]:
    """Return ordered context labels and reporter co-assignment patterns."""

    required = {"context_index", "context_label", "reporter_groups_json"}
    if missing := sorted(required.difference(contexts.columns)):
        raise ValueError(f"Synchronization contexts lack columns: {missing}")
    ordered = contexts.copy()
    ordered["context_index"] = pd.to_numeric(
        ordered["context_index"], errors="raise"
    ).astype(int)
    ordered = ordered.sort_values("context_index").reset_index(drop=True)
    if ordered["context_index"].tolist() != list(range(len(ordered))):
        raise ValueError("Synchronization context indices must be consecutive")
    labels = ordered["context_label"].astype(str).tolist()
    if len(labels) != 3 or labels[0] != "full_fit":
        raise ValueError("Expected full fit followed by two held-out folds")
    if any(not label.startswith("feature_fold_") for label in labels[1:]):
        raise ValueError("Held-out synchronization contexts are malformed")
    pairs_by_context: list[set[frozenset[str]]] = []
    all_pairs: set[frozenset[str]] = set()
    for row in ordered.itertuples(index=False):
        pairs: set[frozenset[str]] = set()
        for group in json.loads(str(row.reporter_groups_json)):
            modalities = sorted(str(value) for value in group)
            pairs.update(
                frozenset(pair) for pair in combinations(modalities, 2)
            )
        pairs_by_context.append(pairs)
        all_pairs.update(pairs)
    patterns = {
        pair: "".join(
            "1" if pair in context_pairs else "0"
            for context_pairs in pairs_by_context
        )
        for pair in all_pairs
    }
    return labels, patterns


def build_majority_rescues(
    assay: pd.DataFrame,
    hyperedges: pd.DataFrame,
    pair_decisions: pd.DataFrame,
    contexts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the fixed v4.46 rule to one world's frozen observables."""

    context_labels, coassignment_patterns = _context_pair_patterns(contexts)
    context_count = len(context_labels)
    strict_majority_count = context_count // 2 + 1
    if assay["modality"].astype(str).duplicated().any():
        raise ValueError("Assay assignment table contains duplicate modalities")
    assays = assay.set_index(assay["modality"].astype(str), drop=False)
    represented = synchronization._represented_assay_pairs(hyperedges)
    candidate_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for decision in pair_decisions.itertuples(index=False):
        left, right = sorted(
            (str(decision.left_modality), str(decision.right_modality))
        )
        if left not in assays.index or right not in assays.index:
            raise ValueError("Pair decision references an unknown assay")
        left_assay, right_assay = assays.loc[left], assays.loc[right]
        pair = frozenset((left, right))
        statistical_pattern = synchronization._context_support_pattern(
            decision.statistical_context_support_pattern, context_count
        )
        coassignment_pattern = coassignment_patterns.get(
            pair, "0" * context_count
        )
        heldout_pattern = coassignment_pattern[1:]
        statistical_count = statistical_pattern.count("1")
        coassignment_count = coassignment_pattern.count("1")
        heldout_count = heldout_pattern.count("1")
        both_resolved = all(
            synchronization._as_bool(value)
            for value in (
                left_assay["within_assay_pair_resolved_by_component"],
                right_assay["within_assay_pair_resolved_by_component"],
            )
        )
        both_context_identified = all(
            synchronization._as_bool(value)
            for value in (
                left_assay["reporter_context_components_identified"],
                right_assay["reporter_context_components_identified"],
            )
        )
        both_bootstrap = all(
            synchronization._as_bool(value)
            for value in (
                left_assay[
                    "reporter_all_context_bootstrap_component_assignments_supported"
                ],
                right_assay[
                    "reporter_all_context_bootstrap_component_assignments_supported"
                ],
            )
        )
        already_represented = pair in represented
        selected = bool(
            not already_represented
            and both_resolved
            and both_context_identified
            and both_bootstrap
            and statistical_count >= strict_majority_count
            and heldout_count >= 1
        )
        if selected:
            status = "selected_by_heldout_synchronized_strict_majority"
        elif already_represented:
            status = "diagnostic_duplicate_of_v444_output"
        elif not both_resolved:
            status = "unresolved_within_assay_direct_structure"
        elif not both_context_identified:
            status = "unresolved_reporter_context_component"
        elif not both_bootstrap:
            status = "unresolved_reporter_bootstrap_support"
        elif statistical_count < strict_majority_count:
            status = "unresolved_without_strict_statistical_context_majority"
        else:
            status = "unresolved_without_heldout_global_coassignment"

        candidate_id = ""
        if selected:
            candidate_id = (
                "heldout_synchronized_majority_shared_state_"
                f"{len(candidate_rows) + 1:03d}"
            )
            left_reporters = [
                value
                for value in str(
                    left_assay["external_state_reporter_members"]
                ).split(";")
                if value
            ]
            right_reporters = [
                value
                for value in str(
                    right_assay["external_state_reporter_members"]
                ).split(";")
                if value
            ]
            if not left_reporters or not right_reporters:
                raise ValueError("Selected assay pair lacks reporter features")
            feature_pairs = sorted(
                {
                    tuple(sorted((left_feature, right_feature)))
                    for left_feature, right_feature in product(
                        left_reporters, right_reporters
                    )
                }
            )
            candidate_rows.append(
                {
                    "synchronized_majority_hyperedge_id": candidate_id,
                    "modalities": f"{left};{right}",
                    "n_modalities": 2,
                    "candidate_assay_pairs_json": json.dumps(
                        [[left, right]], separators=(",", ":")
                    ),
                    "candidate_assay_pair_count": 1,
                    "external_state_reporter_members": ";".join(
                        sorted(set(left_reporters + right_reporters))
                    ),
                    "candidate_cross_assay_feature_pairs_json": json.dumps(
                        feature_pairs, separators=(",", ":")
                    ),
                    "candidate_cross_assay_feature_pair_count": len(
                        feature_pairs
                    ),
                    "statistical_context_support_pattern": statistical_pattern,
                    "statistical_context_support_count": statistical_count,
                    "statistical_context_support_fraction": float(
                        statistical_count / context_count
                    ),
                    "global_reporter_coassignment_context_pattern": (
                        coassignment_pattern
                    ),
                    "global_reporter_coassignment_context_count": (
                        coassignment_count
                    ),
                    "heldout_global_reporter_coassignment_pattern": (
                        heldout_pattern
                    ),
                    "heldout_global_reporter_coassignment_count": heldout_count,
                    "both_assays_within_assay_pair_resolved_by_component": (
                        both_resolved
                    ),
                    "both_reporters_context_components_identified": (
                        both_context_identified
                    ),
                    "both_reporters_all_context_bootstrap_supported": (
                        both_bootstrap
                    ),
                    "v444_assay_pair_already_represented": already_represented,
                    "hypothesis_selected": True,
                    "selection_status": status,
                    "hypothesis_type": "cross_assay_shared_state_equivalence",
                    "within_assay_direct_structure_claimed": False,
                    "exact_assay_participation_claimed": True,
                    "assay_participation_is_set_valued": False,
                    "exact_cross_assay_pair_claimed": False,
                    "cross_assay_direction_claimed": False,
                    "candidate_pairs_may_be_expanded_as_edges": False,
                    "fixed_selection_rule": FIXED_SELECTION_RULE,
                }
            )
        audit_rows.append(
            {
                "left_modality": left,
                "right_modality": right,
                "statistical_context_support_pattern": statistical_pattern,
                "statistical_context_support_count": statistical_count,
                "global_reporter_coassignment_context_pattern": (
                    coassignment_pattern
                ),
                "global_reporter_coassignment_context_count": (
                    coassignment_count
                ),
                "heldout_global_reporter_coassignment_pattern": heldout_pattern,
                "heldout_global_reporter_coassignment_count": heldout_count,
                "both_assays_within_assay_pair_resolved_by_component": (
                    both_resolved
                ),
                "both_reporters_context_components_identified": (
                    both_context_identified
                ),
                "both_reporters_all_context_bootstrap_supported": (
                    both_bootstrap
                ),
                "v444_assay_pair_already_represented": already_represented,
                "hypothesis_selected": selected,
                "selection_status": status,
                "candidate_id": candidate_id,
            }
        )
    return (
        pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS),
        pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS),
    )


def freeze_rescue(
    *, bridge_dir: Path, synchronization_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Freeze one truth-blind v4.46 rescue artifact."""

    bridge, bridge_provenance = synchronization.verify_bridge(bridge_dir)
    synchronized, synchronization_manifest = verify_synchronization(
        synchronization_dir, bridge, bridge_provenance
    )
    assay = pd.read_csv(
        bridge / "assay_axis_assignments.tsv", sep="\t", keep_default_na=False
    )
    hyperedges = pd.read_csv(
        bridge / "stable_cross_assay_hyperedges.tsv",
        sep="\t",
        keep_default_na=False,
    )
    pair_decisions = pd.read_csv(
        bridge / "context_majority_pair_decisions.tsv",
        sep="\t",
        keep_default_na=False,
    )
    contexts = pd.read_csv(
        synchronized / "multi_assay_synchronization_contexts.tsv",
        sep="\t",
        keep_default_na=False,
    )
    candidates, audit = build_majority_rescues(
        assay, hyperedges, pair_decisions, contexts
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "heldout_synchronized_majority_candidates.tsv": candidates,
        "heldout_synchronized_majority_pair_audit.tsv": audit,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    result = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "block": bridge_provenance["block"],
        "seed": int(bridge_provenance["seed"]),
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "information_boundary": {
            "frozen_v444_bridge_used": True,
            "frozen_v445_multi_assay_synchronization_used": True,
            "full_fit_plus_two_disjoint_background_feature_folds_used": True,
            "heldout_feature_global_permutation_frame_required": True,
            "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False,
            "scorer_truth_used": False,
            "external_features_or_priors_used": False,
        },
        "formal_pretruth_barrier_satisfied": bool(
            bridge_provenance["formal_pretruth_barrier_satisfied"]
            and synchronization_manifest["formal_pretruth_barrier_satisfied"]
        ),
        "retrospective_rule_development_on_cached_worlds": True,
        "new_worlds_or_datasets_created": False,
        "selected_hypothesis_count": len(candidates),
        "within_assay_direct_structure_claimed": False,
        "exact_cross_assay_pair_claimed": False,
        "cross_assay_direction_claimed": False,
        "candidate_pairs_may_be_expanded_as_edges": False,
        "bridge_provenance_payload_sha256": bridge_provenance[
            "provenance_payload_sha256"
        ],
        "bridge_provenance_file_sha256": synchronization._sha256(
            bridge / "provenance.json"
        ),
        "synchronization_manifest_payload_sha256": synchronization_manifest[
            "manifest_payload_sha256"
        ],
        "synchronization_manifest_file_sha256": synchronization._sha256(
            synchronized / "synchronization_manifest.json"
        ),
        "source_sha256": {
            Path(__file__).name: synchronization._sha256(
                Path(__file__).resolve()
            )
        },
        "input_sha256": {
            "assay_axis_assignments.tsv": synchronization._sha256(
                bridge / "assay_axis_assignments.tsv"
            ),
            "stable_cross_assay_hyperedges.tsv": synchronization._sha256(
                bridge / "stable_cross_assay_hyperedges.tsv"
            ),
            "context_majority_pair_decisions.tsv": synchronization._sha256(
                bridge / "context_majority_pair_decisions.tsv"
            ),
            "multi_assay_synchronization_contexts.tsv": (
                synchronization._sha256(
                    synchronized / "multi_assay_synchronization_contexts.tsv"
                )
            ),
        },
        "output_sha256": {
            name: synchronization._sha256(out_dir / name) for name in outputs
        },
        "published_method_inspiration": {
            "permutation_synchronization": (
                "Pachauri, Kondor, and Singh, NeurIPS 2013"
            ),
            "multiway_matching": "Tang and Jebara, AISTATS 2017",
            "unpaired_multi_domain_causal_representation": (
                "Sturma, Squires, Drton, and Uhler, NeurIPS 2023"
            ),
        },
    }
    result["manifest_payload_sha256"] = synchronization._canonical_sha256(
        result
    )
    (out_dir / "rescue_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--synchronization-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = freeze_rescue(
        bridge_dir=args.bridge_dir,
        synchronization_dir=args.synchronization_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

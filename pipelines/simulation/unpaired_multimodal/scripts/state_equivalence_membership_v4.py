#!/usr/bin/env python3
"""Annotate uncertainty-aware CA-BOSS v4 state-equivalence membership.

The strict shared-state feature layer intentionally retains one stable
representative when several within-assay features carry redundant state
evidence.  That is appropriate for a precision-first feature list, but it
throws away useful structure: a redundancy-demoted feature can remain a strong
member of the same assay-local state-equivalence class even when it is not a
unique representative.

This module adds a separate, strictly typed tier.  It never changes strict
membership, assay participation, a direct edge, or the shared-state hyperedge.
An equivalence-only member must have passed the pre-redundancy selector, have
been demoted specifically by redundancy handling, and pass the already-frozen
world-maxT strong-core threshold.  The result is a within-assay equivalence
annotation, not an exact cross-assay pair or a direct causal claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


SCHEMA = "state_equivalence_membership.v4.24"
STRICT_TIER = "strict_state_member"
EQUIVALENCE_TIER = "within_assay_state_equivalence_member"
UNRESOLVED_TIER = "unresolved"
GROUP_EDGE_TYPE = "within_assay_shared_state_equivalence_class"


@dataclass(frozen=True)
class StateEquivalenceConfig:
    """Frozen selection-only settings for the equivalence tier."""

    maximum_feature_global_maxT_p: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.maximum_feature_global_maxT_p <= 1.0:
            raise ValueError("maximum_feature_global_maxT_p must lie in (0, 1]")


def _boolean(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        values = sorted(set(normalized.loc[invalid]))
        raise ValueError(f"{name} contains non-boolean values {values}")
    return normalized.isin({"true", "1"})


def _strict_selection(membership: pd.DataFrame) -> pd.Series:
    if "audit_selected" in membership:
        return _boolean(membership["audit_selected"], name="audit_selected")
    if "selection_status" not in membership:
        raise ValueError("Membership table lacks strict selection status")
    return membership["selection_status"].fillna("").astype(str).eq("selected")


def annotate_state_equivalence_membership(
    membership: pd.DataFrame,
    *,
    config: StateEquivalenceConfig | None = None,
    strict_membership: pd.Series | Sequence[bool] | None = None,
) -> pd.DataFrame:
    """Return strict, equivalence-only, and unresolved feature tiers.

    ``strict_membership`` lets a layered producer supply its already frozen
    strict feature decision.  The default preserves the standalone/audit
    behavior, which reads ``audit_selected`` when present and otherwise uses
    ``selection_status``.  Supplying the decision prevents the equivalence
    tier from redefining a stricter downstream feature estimand.
    """

    cfg = config or StateEquivalenceConfig()
    required = {
        "feature",
        "modality",
        "pre_redundancy_selection_status",
        "redundancy_demoted",
        "feature_global_maxT_p",
    }
    missing = sorted(required - set(membership))
    if missing:
        raise ValueError(f"Membership table is missing columns {missing}")
    output = membership.copy()
    if strict_membership is None:
        strict = _strict_selection(output)
    else:
        if isinstance(strict_membership, pd.Series):
            if not strict_membership.index.equals(output.index):
                raise ValueError(
                    "strict_membership index must exactly match membership"
                )
            strict_values = strict_membership
        else:
            if len(strict_membership) != len(output):
                raise ValueError(
                    "strict_membership length must match membership"
                )
            strict_values = pd.Series(strict_membership, index=output.index)
        strict = _boolean(strict_values, name="strict_membership")
    pre_redundancy = (
        output["pre_redundancy_selection_status"]
        .fillna("")
        .astype(str)
        .eq("selected")
    )
    redundancy_demoted = _boolean(
        output["redundancy_demoted"], name="redundancy_demoted"
    )
    max_t_p = pd.to_numeric(output["feature_global_maxT_p"], errors="raise")
    if (~max_t_p.between(0.0, 1.0, inclusive="both")).any():
        raise ValueError("feature_global_maxT_p must lie in [0, 1]")
    equivalence_only = (
        ~strict
        & pre_redundancy
        & redundancy_demoted
        & max_t_p.le(cfg.maximum_feature_global_maxT_p)
    )
    supported = strict | equivalence_only
    tier = pd.Series(UNRESOLVED_TIER, index=output.index, dtype=object)
    tier.loc[equivalence_only] = EQUIVALENCE_TIER
    tier.loc[strict] = STRICT_TIER
    output["strict_state_member"] = strict
    output["state_equivalence_only_member"] = equivalence_only
    output["state_supported_member"] = supported
    output["state_membership_tier"] = tier
    output["state_equivalence_maxT_threshold"] = (
        cfg.maximum_feature_global_maxT_p
    )
    output["state_equivalence_basis"] = ""
    output.loc[
        equivalence_only, "state_equivalence_basis"
    ] = "pre_redundancy_selected;redundancy_demoted;world_maxT_strong_core"
    output["exact_cross_assay_pair_claimed"] = False
    output["direct_causal_claimed"] = False
    output["strict_membership_changed"] = False
    if not output.loc[equivalence_only, "redundancy_demoted"].astype(bool).all():
        raise ValueError("Equivalence tier contains a non-demoted feature")
    if (strict & equivalence_only).any():
        raise ValueError("Strict and equivalence-only tiers overlap")
    return output


def build_state_equivalence_groups(
    annotated_membership: pd.DataFrame,
) -> pd.DataFrame:
    """Build within-assay sets without expanding them into pairwise edges."""

    required = {
        "feature",
        "modality",
        "strict_state_member",
        "state_equivalence_only_member",
        "state_supported_member",
    }
    missing = sorted(required - set(annotated_membership))
    if missing:
        raise ValueError(f"Annotated membership is missing columns {missing}")
    rows: list[dict[str, Any]] = []
    for modality, group in annotated_membership.groupby("modality", sort=True):
        strict = sorted(
            group.loc[
                _boolean(group["strict_state_member"], name="strict_state_member"),
                "feature",
            ].astype(str)
        )
        equivalence = sorted(
            group.loc[
                _boolean(
                    group["state_equivalence_only_member"],
                    name="state_equivalence_only_member",
                ),
                "feature",
            ].astype(str)
        )
        supported = sorted(set(strict) | set(equivalence))
        rows.append(
            {
                "state_equivalence_group_id": f"state_equivalence__{modality}",
                "modality": str(modality),
                "edge_type": GROUP_EDGE_TYPE,
                "strict_members": ";".join(strict),
                "equivalence_only_members": ";".join(equivalence),
                "supported_members": ";".join(supported),
                "n_strict_members": len(strict),
                "n_equivalence_only_members": len(equivalence),
                "n_supported_members": len(supported),
                "selection_status": "selected" if supported else "unresolved",
                "pairwise_expansion_permitted": False,
                "exact_cross_assay_pair_claimed": False,
                "direct_causal_claimed": False,
                "interpretation": (
                    "within-assay features with strict or strong redundant "
                    "shared-state evidence"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_state_equivalence_outputs(
    membership: pd.DataFrame,
    *,
    out_dir: Path,
    config: StateEquivalenceConfig | None = None,
) -> dict[str, Path]:
    cfg = config or StateEquivalenceConfig()
    if out_dir.exists():
        raise FileExistsError(f"Output already exists: {out_dir}")
    annotated = annotate_state_equivalence_membership(membership, config=cfg)
    groups = build_state_equivalence_groups(annotated)
    out_dir.mkdir(parents=True)
    paths = {
        "feature_membership": out_dir / "state_equivalence_feature_membership.tsv",
        "groups": out_dir / "typed_state_equivalence_groups.tsv",
        "provenance": out_dir / "state_equivalence_provenance.json",
    }
    annotated.to_csv(paths["feature_membership"], sep="\t", index=False)
    groups.to_csv(paths["groups"], sep="\t", index=False)
    provenance = {
        "schema_version": SCHEMA,
        "config": {
            "maximum_feature_global_maxT_p": cfg.maximum_feature_global_maxT_p
        },
        "n_features": len(annotated),
        "n_strict_members": int(annotated["strict_state_member"].sum()),
        "n_equivalence_only_members": int(
            annotated["state_equivalence_only_member"].sum()
        ),
        "n_supported_members": int(annotated["state_supported_member"].sum()),
        "n_groups": len(groups),
        "strict_membership_changed": False,
        "assay_participation_changed": False,
        "pairwise_cross_assay_edges_created": 0,
        "direct_edges_created_or_changed": 0,
    }
    paths["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    membership = pd.read_csv(args.membership, sep="\t", keep_default_na=False)
    write_state_equivalence_outputs(membership, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

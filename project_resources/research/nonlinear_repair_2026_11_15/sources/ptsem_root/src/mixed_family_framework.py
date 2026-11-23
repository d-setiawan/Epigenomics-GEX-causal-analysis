#!/usr/bin/env python3
"""Standalone activation and invariants for the paper's mixed-family design."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np


if not hasattr(np, "asfarray"):
    def _numpy_asfarray_compat(values, dtype=float):
        requested = np.dtype(dtype)
        if requested.kind not in "fc":
            requested = np.dtype(float)
        return np.asarray(values, dtype=requested)

    np.asfarray = _numpy_asfarray_compat  # type: ignore[attr-defined]


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CORE_D = PACKAGE_ROOT / "src" / "simulation_core.py"
SIX_FAMILIES = ("Poisson", "NB", "ZIP", "Geom", "Binomial", "Bernoulli")

# These variables are set before importing the core and are inherited by Windows
# spawn workers.  The assignment is iid uniform, exactly as stated in the paper.
os.environ["PTSEM_FAMILY_ASSIGNMENT"] = "iid_uniform"
sys.path.insert(0, str(CORE_D.parent))

import simulation_core as d  # noqa: E402


if Path(d.__file__).resolve() != CORE_D.resolve():
    raise RuntimeError(f"Imported unexpected core: {d.__file__}")
if getattr(d, "NUMERICAL_CORE_VERSION", None) != "ptsem_submission_corrected_v3":
    raise RuntimeError("The required simulation core version is not active")
if tuple(d.FAMLIB) != SIX_FAMILIES or d.FAMILY_ASSIGNMENT_MODE != "iid_uniform":
    raise RuntimeError("The formal iid-uniform six-family design is not active")


def source_sha256() -> str:
    digest = hashlib.sha256()
    with CORE_D.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def framework_provenance() -> dict[str, object]:
    return {
        "core_d_path": str(CORE_D),
        "core_d_sha256": source_sha256(),
        "numerical_core_version": d.NUMERICAL_CORE_VERSION,
        "true_family_assignment": d.FAMILY_ASSIGNMENT_MODE,
        "family_library": list(d.FAMLIB),
        "simulation_function": "d.simulate_ptsem",
        "experiment_function": "d.run_experiment",
        "historical_results_consumed": False,
    }


def run_experiment(**kwargs):
    raw, summary = d.run_experiment(**kwargs)
    if not raw.empty:
        allowed = set(SIX_FAMILIES)
        for value in raw["true_families"].astype(str):
            if not set(value.split(",")).issubset(allowed):
                raise RuntimeError(f"Unknown generated family in {value}")
    return raw, summary

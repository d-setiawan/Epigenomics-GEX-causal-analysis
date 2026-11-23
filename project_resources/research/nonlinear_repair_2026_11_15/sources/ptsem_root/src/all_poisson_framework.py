#!/usr/bin/env python3
"""All-Poisson specialization of the paper's simulation core.

The sole DGP specialization is ``make_exogenous_specs``: every true node
family is Poisson and each rate is drawn independently from Uniform(2, 10).
The top-level wrappers keep the specialization active in spawned workers.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


# The formal run used NumPy 1.26.4.  NumPy 2 removed ``np.asfarray``, while
# KDEpy 1.1.4 (used by the pinned PB-SCM author code) still calls it.  This is
# the exact documented replacement for that removed conversion helper and does
# not alter the simulation core or pinned author source.
if not hasattr(np, "asfarray"):
    def _numpy_asfarray_compat(values, dtype=float):
        requested = np.dtype(dtype)
        if requested.kind not in "fc":
            requested = np.dtype(float)
        return np.asarray(values, dtype=requested)

    np.asfarray = _numpy_asfarray_compat  # type: ignore[attr-defined]


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PACKAGE_ROOT / "src"
CORE_D = CORE_DIR / "simulation_core.py"
SIX_FAMILIES = ("Poisson", "NB", "ZIP", "Geom", "Binomial", "Bernoulli")

# These must be set before importing the core and are inherited by spawned workers.
os.environ["PTSEM_FAMILY_ASSIGNMENT"] = "iid_uniform"
sys.path.insert(0, str(CORE_DIR))

import simulation_core as d  # noqa: E402


if Path(d.__file__).resolve() != CORE_D.resolve():
    raise RuntimeError(
        f"Imported unexpected simulation core: {Path(d.__file__).resolve()} != {CORE_D.resolve()}"
    )
if getattr(d, "NUMERICAL_CORE_VERSION", None) != "ptsem_submission_corrected_v3":
    raise RuntimeError("The required simulation core version is not active")
if tuple(d.FAMLIB) != SIX_FAMILIES:
    raise RuntimeError(f"Formal six-family candidate library is not active: {d.FAMLIB}")

_ORIGINAL_MAKE_EXOGENOUS_SPECS = d.make_exogenous_specs
_ORIGINAL_RUN_REPLICATE_TASK = d.run_replicate_task
_ORIGINAL_INITIALIZE_WORKER = d.initialize_worker


def make_all_poisson_specs(
    dimension: int, rng: np.random.Generator
) -> list[d.FamilyParam]:
    """Table 1 All-Poisson truth: lambda_i iid Uniform(2, 10)."""
    return [
        d.FamilyParam("Poisson", {"lam": float(rng.uniform(2.0, 10.0))})
        for _ in range(dimension)
    ]


def install_all_poisson_specialization() -> None:
    """Install the minimal specialization into the simulation core namespace."""
    d.make_exogenous_specs = make_all_poisson_specs
    d.run_replicate_task = run_all_poisson_replicate_task
    d.initialize_worker = initialize_all_poisson_worker


def initialize_all_poisson_worker(methods: Sequence[str]) -> None:
    """Spawn-safe initializer: specialize first, then run the formal initializer."""
    install_all_poisson_specialization()
    _ORIGINAL_INITIALIZE_WORKER(methods)


def run_all_poisson_replicate_task(task: d.ReplicateTask) -> d.ReplicateOutput:
    """Spawn-safe wrapper around the unchanged formal replicate implementation."""
    install_all_poisson_specialization()
    output = _ORIGINAL_RUN_REPLICATE_TASK(task)
    for row in output.rows:
        families = str(row["true_families"]).split(",")
        if families != ["Poisson"] * int(row["d"]):
            raise RuntimeError(f"All-Poisson invariant failed: {families}")
    return output


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
        "candidate_library": list(d.FAMLIB),
        "formal_data_function": "d.simulate_ptsem",
        "formal_experiment_function": "d.run_experiment",
        "specialized_function": "d.make_exogenous_specs",
        "true_family": "Poisson at every node",
        "poisson_lambda": "iid Uniform(2, 10)",
        "environment_compatibility": {
            "reason": "formal NumPy 1.26.4 versus current NumPy >=2",
            "shim": "removed np.asfarray mapped to its documented floating np.asarray replacement for KDEpy 1.1.4",
        },
    }


def run_experiment(**kwargs):
    """Call the unchanged formal runner and enforce the All-Poisson invariant."""
    install_all_poisson_specialization()
    raw, summary = d.run_experiment(**kwargs)
    if not raw.empty:
        expected = raw["d"].map(lambda value: ",".join(["Poisson"] * int(value)))
        if not raw["true_families"].astype(str).equals(expected):
            raise RuntimeError("Formal output contains a non-Poisson true family")
    return raw, summary


install_all_poisson_specialization()

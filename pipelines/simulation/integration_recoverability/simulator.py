"""Frozen, unpaired integration-recoverability simulation.

The simulator has two outputs with a hard conceptual access boundary: raw
observed assay matrices for integration, and private joint truth for evaluation.
The learner must not load the root truth directory or private_inputs.  This is
an experiment convention, not an OS-level security boundary.

The CLI creates immutable artifacts.  It never replaces a failed world, tunes
to graph recovery, forces detections, or invents paired training observations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import expit

try:
    from .calibration import (ASSAYS, DEFAULT_TOLERANCES, assess_profile_fit,
                              canonical_hash, file_sha256, fit_count_parameters,
                              panel_exposures, sample_counts)
except ImportError:
    from calibration import (ASSAYS, DEFAULT_TOLERANCES, assess_profile_fit,
                             canonical_hash, file_sha256, fit_count_parameters,
                             panel_exposures, sample_counts)

SCHEMA_VERSION = 1
SIMPLE_FEATURE_COUNTS = {"RNA": 25215, "H3K27ac": 30000, "H3K4me3": 30000}
EXTENDED_FEATURE_COUNTS = {"RNA": 2000, "H3K27ac": 15000, "H3K4me3": 15000}
EXTENDED_COHORT_SIZES = {"train": 4000, "val": 500, "discovery": 2000,
                         "probe": 1000, "assessment": 2000}
REGIME_MULTIPLIERS = {"realistic": {a: 1.0 for a in ASSAYS},
                      "informative": {"RNA": 5.0, "H3K27ac": 20.0, "H3K4me3": 20.0}}


def stream_seed(seed: int, *names: Any) -> int:
    """Stable named substreams, independent of call order and Python hash salt."""
    message = json.dumps([int(seed), *map(str, names)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little")


def rng_for(seed: int, *names: Any) -> np.random.Generator:
    return np.random.default_rng(stream_seed(seed, *names))


def _opaque(prefix: str, seed: int, *parts: Any) -> str:
    return prefix + hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()[:20]


def make_simple_spec(seed: int = 20261003) -> dict[str, Any]:
    return {"world_id": "simple_saturating_chain", "design": "simple", "seed": seed,
            "visible_seed": seed, "family": "saturating", "motif": "chain",
            "histone_order": ["H3K27ac", "H3K4me3"], "split": "development",
            "draw": 0, "state_dimension": 1}


def make_extended_specs(seed: int = 20261003) -> list[dict[str, Any]]:
    """Exactly 30 specifications: 9 development, 21 assessment, no resampling."""
    specs = []
    for family in ("linear", "constrained", "unrestricted"):
        for motif in ("chain", "collider"):
            for draw in range(4):
                name = f"{family}_{motif}_{draw}"
                world_seed = stream_seed(seed, name)
                specs.append({"world_id": name, "design": "extended", "seed": world_seed,
                              "visible_seed": world_seed, "family": family, "motif": motif,
                              "draw": draw, "state_dimension": 2,
                              "histone_order": (["H3K27ac", "H3K4me3"] if draw % 2 == 0
                                                else ["H3K4me3", "H3K27ac"]),
                              "split": "development" if draw == 0 else "assessment"})
    for draw in range(4):
        world_seed = stream_seed(seed, "shared_state", draw)
        specs.append({"world_id": f"shared_state_{draw}", "design": "extended",
                      "seed": world_seed, "visible_seed": world_seed, "family": "linear",
                      "motif": "shared_state", "draw": draw, "state_dimension": 2,
                      "histone_order": ["H3K27ac", "H3K4me3"],
                      "split": "development" if draw == 0 else "assessment"})
    twin_seed = stream_seed(seed, "ambiguity_pair")
    for rho in (0.0, 0.7):
        specs.append({"world_id": f"ambiguity_rho_{rho:.1f}", "design": "extended",
                      "seed": twin_seed, "visible_seed": twin_seed, "family": "linear",
                      "motif": "ambiguity", "rho": rho, "state_dimension": 2,
                      "draw": 0, "histone_order": ["H3K27ac", "H3K4me3"],
                      "split": "development", "visible_artifact_group": "ambiguity_pair"})
    return specs


def freeze_extended_bank(calibration: dict[str, Any], out_path: str | Path,
                          seed: int = 20261003) -> dict[str, Any]:
    """Freeze specifications and biological parameters, without cohort draws.

    Measurement parameters will be generated deterministically on independent
    calibration draws and saved before each world's first cohort.  Source hashes
    pin that procedure, so changes require an explicit protocol amendment, not
    silently different worlds bearing the same names.
    """
    path = Path(out_path)
    if path.exists():
        raise FileExistsError(f"Frozen bank manifest already exists: {path}")
    if canonical_hash({k: v for k, v in calibration.items() if k != "calibration_hash"}) != calibration["calibration_hash"]:
        raise ValueError("Calibration hash mismatch")
    specs = make_extended_specs(seed)
    manifest = {
        "schema_version": SCHEMA_VERSION, "bank_seed": seed,
        "n_specifications": len(specs), "n_development": 9, "n_assessment": 21,
        "specifications": specs,
        "biology_parameters": {s["world_id"]: freeze_biology(s) for s in specs},
        "calibration_hash": calibration["calibration_hash"],
        "source_sha256": {"simulator.py": file_sha256(Path(__file__)),
                           "calibration.py": file_sha256(Path(__file__).with_name("calibration.py"))},
        "generation_config": {"feature_counts": EXTENDED_FEATURE_COUNTS,
                               "cohort_sizes": EXTENDED_COHORT_SIZES,
                               "regimes": ["realistic", "informative"],
                               "calibration_n": 2048, "background_calibration_n": 256,
                               "feature_chunk_size": 256, "conditional_mean_repetitions": 128},
        "development_sentinel_panel": ["linear_chain_0", "constrained_chain_0", "unrestricted_chain_0", "shared_state_0"],
        "independence_note": "Ambiguity members share one visible input; regimes and integration seeds are not independent biological worlds.",
        "status": "specifications_and_biology_frozen; cohorts_not_generated",
    }
    manifest["bank_hash"] = canonical_hash(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
    return manifest


def sample_state(spec: dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    if spec["design"] == "simple":
        return ((rng.beta(2, 5, size=n) - 2 / 7) / np.sqrt(10 / 392))[:, None]
    # Population moments are fixed analytically.  No cohort-specific scaling.
    component = rng.random(n) < 0.65
    raw = rng.normal(np.where(component, -0.5, 1.5), np.where(component, 0.55, 0.8))
    mixture_mean = 0.65 * (-0.5) + 0.35 * 1.5
    mixture_var = 0.65 * (0.55 ** 2 + (-0.5 - mixture_mean) ** 2) + 0.35 * (0.8 ** 2 + (1.5 - mixture_mean) ** 2)
    return np.column_stack(((raw - mixture_mean) / np.sqrt(mixture_var), rng.normal(size=n)))


def _mechanism(family: str, n_parents: int, rng: np.random.Generator) -> dict[str, Any]:
    if family == "linear":
        return {"family": family, "weights": rng.uniform(0.7, 1.3, n_parents).tolist()}
    if family == "constrained":
        weights = rng.uniform(0.25, 1.0, (n_parents, 32))
        output = rng.uniform(0.5, 1.5, 32)
        return {"family": family, "weights": weights.tolist(),
                "bias": rng.uniform(-1.5, 1.5, 32).tolist(),
                "output": (4 * output / output.sum()).tolist(),
                "cooperation": float(rng.uniform(0.15, 0.35)) if n_parents > 1 else 0.0}
    if family == "unrestricted":
        return {"family": family, "weights": rng.normal(0, 0.8, (n_parents, 32)).tolist(),
                "bias": rng.normal(0, 0.5, 32).tolist(),
                "output": (rng.normal(size=32) / np.sqrt(32)).tolist(), "cooperation": 0.0}
    raise ValueError(f"Unknown mechanism family {family}")


def apply_mechanism(parents: np.ndarray, mechanism: dict[str, Any]) -> np.ndarray:
    x = np.asarray(parents, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    weights = np.asarray(mechanism["weights"])
    if mechanism["family"] == "linear":
        return x @ weights
    hidden = x @ weights + np.asarray(mechanism["bias"])
    if mechanism["family"] == "constrained":
        value = (expit(hidden) - 0.5) @ np.asarray(mechanism["output"])
        # A bounded positive interaction has nonnegative partial derivatives.
        if x.shape[1] > 1:
            value += mechanism["cooperation"] * np.prod(expit(x), axis=1)
        return value
    return np.tanh(hidden) @ np.asarray(mechanism["output"])


def freeze_biology(spec: dict[str, Any], calibration_n: int = 2048) -> dict[str, Any]:
    if spec["design"] == "simple":
        return {"noise_scales": {a: 0.5 for a in ASSAYS},
                "state_distribution": "standardized Beta(2,5)", "mechanisms": {},
                "residual_calibration": "fixed README equations, not recalibrated"}
    rng = rng_for(spec["seed"], "biology_parameters")
    params = {"state_weights": {a: [float(rng.uniform(0.8, 1.2)), float(rng.uniform(-0.35, 0.35))] for a in ASSAYS},
              "decoy_weights": {a: [float(rng.uniform(0.6, 1.0)), float(rng.uniform(0.5, 1.0))] for a in ASSAYS},
              "noise_scales": {a: 1.0 for a in ASSAYS}, "mechanisms": {},
              "state_distribution": {"weight": 0.65, "means": [-0.5, 1.5], "sds": [0.55, 0.8], "second_program": "independent N(0,1)"},
              "residual_calibration": {"target_r_squared": 0.5, "n_state_pairs": calibration_n,
                                       "scope": "aggregate E[Var(signal|S)] / (E[Var(signal|S)] + sigma^2), not a per-state guarantee",
                                       "estimator": "half mean squared difference of independent ancestral signals conditional on same S"}}
    if spec["motif"] not in {"chain", "collider"}:
        return params
    first, second = spec["histone_order"]
    children = [second, "RNA"] if spec["motif"] == "chain" else ["RNA"]
    for child in children:
        parents = [first] if child == second else ([second] if spec["motif"] == "chain" else [first, second])
        params["mechanisms"][child] = {"parents": parents, **_mechanism(spec["family"], len(parents), rng)}
        state = sample_state(spec, calibration_n, rng_for(spec["seed"], "effect_calibration", child, "state"))
        signals = []
        for rep in range(2):
            core = _sample_core(spec, params, state, rng_for(spec["seed"], "effect_calibration", child, rep))
            signals.append(apply_mechanism(np.column_stack([core[p] for p in parents]), params["mechanisms"][child]))
        conditional_variance = float(np.mean((signals[0] - signals[1]) ** 2) / 2)
        if not np.isfinite(conditional_variance) or conditional_variance < 1e-10:
            raise ValueError("Predeclared numerical degeneracy: zero/non-finite conditional mechanism variation")
        params["noise_scales"][child] = float(np.sqrt(conditional_variance))
        params["mechanisms"][child]["conditional_signal_variance"] = conditional_variance
    return params


def _sample_core(spec: dict[str, Any], biology: dict[str, Any], state: np.ndarray,
                 rng: np.random.Generator, assigned: np.ndarray | None = None) -> dict[str, np.ndarray]:
    n = len(state)
    if spec["design"] == "simple":
        a = state[:, 0] + 0.5 * rng.normal(size=n)
        b = np.tanh(a) + 0.5 * rng.normal(size=n)
        return {"H3K27ac": a, "H3K4me3": b, "RNA": np.tanh(b) + 0.5 * rng.normal(size=n)}
    if spec["motif"] == "ambiguity":
        # Coupling the missing views conditional on the one observed view keeps
        # every observed single-assay realization identical across the twins.
        rho = spec["rho"]
        u, v, w = rng.normal(size=(3, n))
        ac, rna = u.copy(), v.copy()
        if assigned is None:
            rna = rho * u + np.sqrt(1 - rho ** 2) * v
        else:
            rna_rows = assigned == "RNA"
            ac[rna_rows] = rho * v[rna_rows] + np.sqrt(1 - rho ** 2) * u[rna_rows]
            not_rna = ~rna_rows
            rna[not_rna] = rho * u[not_rna] + np.sqrt(1 - rho ** 2) * v[not_rna]
        return {"H3K27ac": state[:, 0] + ac, "RNA": 0.8 * state[:, 0] + rna,
                "H3K4me3": 0.7 * state[:, 0] + w}
    core = {a: state @ np.asarray(biology["state_weights"][a]) + biology["noise_scales"][a] * rng.normal(size=n) for a in ASSAYS}
    first, second = spec["histone_order"]
    order = [second, "RNA"] if spec["motif"] == "chain" else ["RNA"]
    if spec["motif"] == "shared_state":
        return core
    for child in order:
        mechanism = biology["mechanisms"].get(child)
        if mechanism is not None:
            core[child] = apply_mechanism(np.column_stack([core[p] for p in mechanism["parents"]]), mechanism) + biology["noise_scales"][child] * rng.normal(size=n)
    return core


def candidate_definitions(spec: dict[str, Any]) -> list[dict[str, str]]:
    roles = ["core", "sentinel"] if spec["design"] == "simple" else ["core", "decoy", "sentinel"]
    return [{"name": f"{a}_{role}", "assay": a, "role": role} for a in ASSAYS for role in roles]


def sample_activities(spec: dict[str, Any], biology: dict[str, Any], state: np.ndarray,
                      rng: np.random.Generator, assigned: np.ndarray | None = None) -> np.ndarray:
    core = _sample_core(spec, biology, state, rng, assigned)
    columns = []
    for candidate in candidate_definitions(spec):
        assay, role = candidate["assay"], candidate["role"]
        if role == "core":
            value = core[assay]
        elif role == "decoy":
            value = state @ np.asarray(biology["decoy_weights"][assay]) + 0.5 * rng.normal(size=len(state))
        else:
            value = 0.5 * rng.normal(size=len(state))
        columns.append(value)
    return np.column_stack(columns)


def biological_edges(spec: dict[str, Any], biology: dict[str, Any]) -> list[list[str]]:
    if spec["design"] == "simple":
        return [["S0", "H3K27ac_core"], ["H3K27ac_core", "H3K4me3_core"], ["H3K4me3_core", "RNA_core"]]
    edges = []
    if spec["motif"] == "ambiguity":
        edges.extend([["S0", a + "_core"] for a in ASSAYS])
        if spec["rho"] != 0:
            edges.append(["H3K27ac_core", "RNA_core"])
    else:
        for assay in ASSAYS:
            if assay not in biology["mechanisms"]:
                edges.extend([[f"S{k}", assay + "_core"] for k in range(2)])
            else:
                edges.extend([[parent + "_core", assay + "_core"] for parent in biology["mechanisms"][assay]["parents"]])
    for assay in ASSAYS:
        edges.extend([[f"S{k}", assay + "_decoy"] for k in range(2)])
    return edges


def _profile_subset(source: list[dict[str, Any]], n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    # Exact source universe when sizes agree; otherwise declared sampled subset.
    if n <= len(source):
        indices = rng.choice(len(source), n, replace=False)
    else:
        indices = rng.choice(len(source), n, replace=True)
    return [source[int(i)] for i in indices]


def _make_layout(spec: dict[str, Any], calibration: dict[str, Any],
                 feature_counts: dict[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = spec["visible_seed"]
    candidates = candidate_definitions(spec)
    layout = {}
    # Shared annotation channels are unrelated to the planted focal graph.
    channels = [_opaque("g_", seed, "gene", j) for j in range(feature_counts["RNA"])]
    channel_rng = rng_for(seed, "channel_background")
    shared_loadings = channel_rng.normal(0, 0.7, (len(channels), spec["state_dimension"]))
    shared_forms = channel_rng.integers(0, 2, len(channels))
    proportions = [0.8, 0.0, 0.2] if spec["design"] == "simple" else [0.6, 0.2, 0.2]
    shared_classes = channel_rng.choice(["shared", "private", "null"], len(channels), p=proportions)
    for assay in ASSAYS:
        n = feature_counts[assay]
        selected = [x for x in candidates if x["assay"] == assay]
        if n < len(selected) + 3:
            raise ValueError("Need at least three background features in every assay")
        rng = rng_for(seed, "layout", assay)
        feature_ids = [_opaque("f_", seed, assay, j) for j in range(n)]
        positions = rng.choice(n, len(selected), replace=False)
        for candidate, position in zip(selected, positions):
            candidate.update(feature_id=feature_ids[int(position)], feature_index=int(position))
        if assay == "RNA":
            channel_indices = np.arange(n)
        else:
            channel_indices = rng.integers(len(channels), size=n)
        assay_channels = np.asarray(channels)[channel_indices].tolist()
        # Scored candidates have no known cross-assay annotation correspondence.
        # They stay eligible in native matrices, but cannot leak planted links.
        for candidate, position in zip(selected, positions):
            assay_channels[int(position)] = _opaque("g_unlinked_", seed, assay, candidate["feature_id"])
        profiles = calibration["assays"][assay]
        panel_name = "encoder" if spec["design"] == "simple" or assay == "RNA" else "retained"
        targets = _profile_subset(profiles[panel_name]["features"], n, rng_for(seed, "source_profiles", assay))
        retained_source = set(profiles["retained_feature_ids"])
        reconstruction_mask = [assay == "RNA" or x["feature_id"] in retained_source for x in targets]
        # This is a *source-panel* mask.  Main integration additionally records
        # its training-selected mask; it must never force focal inclusion.
        for candidate in selected:
            targets[candidate["feature_index"]] = profiles["focal_profile"]
        classes = shared_classes[channel_indices].copy()
        weights = shared_loadings[channel_indices] + rng.normal(0, 0.05, (n, spec["state_dimension"]))
        layout[assay] = {"feature_ids": feature_ids, "gene_ids": assay_channels,
                         "channel_indices": channel_indices.tolist(),
                         "background_classes": classes.tolist(), "weights": weights.tolist(),
                         "forms": shared_forms[channel_indices].tolist(),
                         "private_weights": rng.normal(0, 0.7, n).tolist(),
                         "background_noise_scale": 0.5,
                         "source_panel": panel_name, "source_profiles": targets,
                         "source_reconstruction_mask": reconstruction_mask,
                         "scored_positions": positions.tolist()}
    return layout, candidates


def _background_activity(layout: dict[str, Any], state: np.ndarray, private: np.ndarray,
                          start: int, stop: int, rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray(layout["weights"])[start:stop]
    classes = np.asarray(layout["background_classes"])[start:stop]
    linear = state @ weights.T
    forms = np.asarray(layout["forms"])[start:stop]
    value = np.where(forms[None, :] == 1, np.tanh(linear), linear)
    value[:, classes == "private"] = private[:, None] * np.asarray(layout["private_weights"])[start:stop][None, classes == "private"]
    value[:, classes == "null"] = 0.0
    return value + layout["background_noise_scale"] * rng.normal(size=value.shape)


def _fit_background_parameters(activities: np.ndarray, exposure: np.ndarray,
                                targets: list[dict[str, Any]], distribution: str,
                                kappa: float = 0.8) -> list[dict[str, Any]]:
    means = np.asarray([t["mean"] for t in targets])
    variances = np.asarray([t["variance"] for t in targets])
    sensitivity = kappa * activities
    low = np.full(len(targets), -80.0)
    high = np.maximum(80.0, 2 * means / float(exposure.mean()) + np.max(np.abs(sensitivity), axis=0))
    for _ in range(30):
        alpha = (low + high) / 2
        observed_means = np.mean(exposure[:, None] * np.logaddexp(0, alpha[None, :] + sensitivity), axis=0)
        smaller = observed_means < means
        low = np.where(smaller, alpha, low)
        high = np.where(smaller, high, alpha)
    alpha = (low + high) / 2
    mu = exposure[:, None] * np.logaddexp(0, alpha[None, :] + sensitivity)
    residual = variances - (mu.mean(axis=0) + mu.var(axis=0))
    result = []
    for j in range(len(targets)):
        theta = float(max(np.mean(mu[:, j] ** 2) / residual[j], 1e-4)) if distribution == "nb" and residual[j] > 0 else None
        result.append({"alpha": float(alpha[j]), "kappa": kappa, "theta": theta,
                       "distribution": distribution, "structural_zero": bool(means[j] == 0)})
    return result


def prepare_world(spec: dict[str, Any], calibration: dict[str, Any],
                   feature_counts: dict[str, int] | None = None,
                   calibration_n: int = 2048, background_calibration_n: int = 256,
                   feature_chunk_size: int = 256) -> dict[str, Any]:
    """Freeze all mechanisms and count parameters using independent draws."""
    defaults = SIMPLE_FEATURE_COUNTS if spec["design"] == "simple" else EXTENDED_FEATURE_COUNTS
    if canonical_hash({k: v for k, v in calibration.items() if k != "calibration_hash"}) != calibration["calibration_hash"]:
        raise ValueError("Calibration artifact content does not match its frozen hash")
    feature_counts = dict(defaults if feature_counts is None else feature_counts)
    biology = freeze_biology(spec, calibration_n)
    layout, candidates = _make_layout(spec, calibration, feature_counts)
    state = sample_state(spec, calibration_n, rng_for(spec["visible_seed"], "measurement_calibration", "state"))
    # Ambiguity marginals do not depend on rho; use the same canonical draw to
    # make fitted observations bit-identical (finite-MC fitting must not leak rho).
    calibration_spec = dict(spec, rho=0.0) if spec["motif"] == "ambiguity" else spec
    activities = sample_activities(calibration_spec, biology, state, rng_for(spec["visible_seed"], "measurement_calibration", "activity"))
    parameters, audits = {}, {}
    for assay in ASSAYS:
        source = calibration["assays"][assay]
        exposure = rng_for(spec["visible_seed"], "measurement_calibration", assay, "exposure").choice(
            panel_exposures(source, layout[assay]["source_panel"]), calibration_n)
        distribution = "poisson" if spec["design"] == "simple" and assay != "RNA" else "nb"
        private = rng_for(spec["visible_seed"], "measurement_calibration", assay, "private").normal(size=calibration_n)
        assay_parameters = []
        n_cal = min(calibration_n, background_calibration_n)
        for start in range(0, feature_counts[assay], feature_chunk_size):
            stop = min(start + feature_chunk_size, feature_counts[assay])
            background = _background_activity(layout[assay], state[:n_cal], private[:n_cal], start, stop,
                                              rng_for(spec["visible_seed"], "measurement_calibration", assay, "background", start))
            assay_parameters.extend(_fit_background_parameters(background, exposure[:n_cal], layout[assay]["source_profiles"][start:stop], distribution))
        audits[assay] = {}
        for j, candidate in enumerate(candidates):
            if candidate["assay"] != assay:
                continue
            fitted = fit_count_parameters(activities[:, j], exposure, source["focal_profile"], distribution)
            assay_parameters[candidate["feature_index"]] = fitted
            audits[assay][candidate["name"]] = assess_profile_fit(source["focal_profile"], fitted["expected"], calibration["tolerances"])
        parameters[assay] = assay_parameters
    edges = biological_edges(spec, biology)
    return {"schema_version": SCHEMA_VERSION, "spec": copy.deepcopy(spec), "biology": biology,
            "feature_counts": feature_counts, "layout": layout, "candidates": candidates,
            "observation_parameters": parameters, "calibration_hash": calibration["calibration_hash"],
            "exposure_distributions": {a: panel_exposures(calibration["assays"][a], layout[a]["source_panel"]).tolist() for a in ASSAYS},
            "calibration_n": calibration_n, "background_calibration_n": min(calibration_n, background_calibration_n),
            "feature_chunk_size": feature_chunk_size, "calibration_audit": audits,
            "biological_dag_edges": edges,
            "core_dag_edges": [e for e in edges if all(x.endswith("_core") for x in e)],
            "passed_focal_calibration": all(v["passed"] for a in audits.values() for v in a.values()),
            "warnings": ["Source-panel moment matching is not a uniquely identified biological/technical decomposition.",
                         "Library and background realism require an independent generated-count audit before recovery claims."]}


def _cohort_population(prepared: dict[str, Any], cohort: str, size: int) -> dict[str, np.ndarray]:
    spec = prepared["spec"]
    private = cohort in {"probe", "assessment"}
    n = size if private else size * len(ASSAYS)
    seed = spec["visible_seed"]
    # Arbitrary IDs do not encode latent state, candidate role, or partner ID.
    cell_ids = np.asarray([_opaque("c_", seed, cohort, i) for i in range(n)])
    assigned = np.full(n, "paired", dtype="U10") if private else np.repeat(np.asarray(ASSAYS), size)
    if not private:
        assigned = assigned[rng_for(seed, cohort, "assignment").permutation(n)]
    state = sample_state(spec, n, rng_for(seed, cohort, "state"))
    activities = sample_activities(spec, prepared["biology"], state,
                                   rng_for(seed, cohort, "activity"), None if private else assigned)
    return {"cell_ids": cell_ids, "assigned_assay": assigned, "state": state, "activities": activities}


def _sample_full_count_panel(prepared: dict[str, Any], population: dict[str, np.ndarray],
                             exposures: np.ndarray, cohort: str, regime: str) -> np.ndarray:
    n = len(population["cell_ids"])
    counts = np.zeros((n, len(prepared["candidates"])), dtype=np.int32)
    seed = prepared["spec"]["visible_seed"]
    for j, candidate in enumerate(prepared["candidates"]):
        assay = candidate["assay"]
        assay_index = ASSAYS.index(assay)
        parameters = prepared["observation_parameters"][assay][candidate["feature_index"]]
        visible = population["assigned_assay"] == assay
        # Isolate visible and hidden RNG calls.  Changed hidden means in an
        # ambiguity twin must not consume random draws used for visible cells.
        for label, rows in (("visible", visible), ("hidden", ~visible)):
            counts[rows, j] = sample_counts(rng_for(seed, cohort, regime, "candidate_counts", j, label),
                                            population["activities"][rows, j], exposures[rows, assay_index],
                                            parameters, REGIME_MULTIPLIERS[regime][assay])
    return counts


def conditional_activity_means(prepared: dict[str, Any], state: np.ndarray,
                                seed: int, repetitions: int = 128,
                                cell_chunk_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Evaluator-only Monte Carlo E[V|S] and standard error, not eps=0."""
    means = np.zeros((len(state), len(prepared["candidates"])))
    errors = np.zeros_like(means)
    if repetitions < 2:
        raise ValueError("Conditional mean Monte Carlo needs at least 2 repetitions")
    for start in range(0, len(state), cell_chunk_size):
        local = state[start:start + cell_chunk_size]
        values = sample_activities(prepared["spec"], prepared["biology"], np.repeat(local, repetitions, axis=0),
                                   rng_for(seed, "conditional_means", start))
        values = values.reshape(len(local), repetitions, -1)
        means[start:start + len(local)] = values.mean(axis=1)
        errors[start:start + len(local)] = values.std(axis=1, ddof=1) / np.sqrt(repetitions)
    return means, errors


def _assay_matrix(prepared: dict[str, Any], population: dict[str, np.ndarray],
                   exposures: np.ndarray, counts: np.ndarray, assay: str,
                   cohort: str, regime: str) -> tuple[sparse.csr_matrix, np.ndarray]:
    n = len(population["cell_ids"])
    rows = np.arange(n) if cohort in {"probe", "assessment"} else np.flatnonzero(population["assigned_assay"] == assay)
    layout = prepared["layout"][assay]
    seed = prepared["spec"]["visible_seed"]
    private = rng_for(seed, cohort, "background_private", assay).normal(size=n)[rows]
    state = population["state"][rows]
    exposure = exposures[rows, ASSAYS.index(assay)]
    chunks = []
    candidate_by_position = {c["feature_index"]: j for j, c in enumerate(prepared["candidates"]) if c["assay"] == assay}
    feature_chunk = prepared["feature_chunk_size"]
    for start in range(0, len(layout["feature_ids"]), feature_chunk):
        stop = min(start + feature_chunk, len(layout["feature_ids"]))
        activities = _background_activity(layout, state, private, start, stop,
                                           rng_for(seed, cohort, "background_activity", assay, start))
        block = np.zeros(activities.shape, dtype=np.int32)
        for position in range(start, stop):
            if position in candidate_by_position:
                block[:, position - start] = counts[rows, candidate_by_position[position]]
            else:
                block[:, position - start] = sample_counts(rng_for(seed, cohort, regime, "background_counts", assay, position),
                                                           activities[:, position - start], exposure,
                                                           prepared["observation_parameters"][assay][position],
                                                           REGIME_MULTIPLIERS[regime][assay])
        chunks.append(sparse.csr_matrix(block))
    return sparse.hstack(chunks, format="csr"), rows


def _write_assay(path: Path, matrix: sparse.csr_matrix, rows: np.ndarray,
                  prepared: dict[str, Any], population: dict[str, np.ndarray], assay: str) -> None:
    import anndata as ad
    import pandas as pd

    layout = prepared["layout"][assay]
    obs = pd.DataFrame({"assay": assay, "total_counts": np.asarray(matrix.sum(axis=1)).ravel(),
                        "n_features": np.diff(matrix.indptr)}, index=population["cell_ids"][rows])
    # Synthetic coordinates provide only annotation channels, not the planted
    # DAG.  Scored candidates have unique unlinked gene IDs in every assay.
    start = np.asarray(layout["channel_indices"], dtype=np.int64) * 10000
    var = pd.DataFrame({"gene_id": layout["gene_ids"], "channel_id": layout["gene_ids"],
                        "chrom": "chrSynthetic", "chromStart": start, "chromEnd": start + (1000 if assay == "RNA" else 500),
                        "strand": "+"}, index=layout["feature_ids"])
    # Public matrices contain no source masks, roles, state, true exposure,
    # source-feature IDs, innovation, partner IDs, or mechanism parameters.
    data = ad.AnnData(X=matrix, obs=obs, var=var)
    data.uns["schema_version"] = SCHEMA_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(path, compression="gzip")


def _generated_summary(matrix: sparse.csr_matrix) -> dict[str, Any]:
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    detected = np.diff(matrix.indptr)
    return {"n_cells": matrix.shape[0], "n_features": matrix.shape[1], "nnz": matrix.nnz,
            "zero_rows": int((detected == 0).sum()),
            "library_quantiles": np.quantile(totals, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            "detected_features_quantiles": np.quantile(detected, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()}


def _assess_generated_panel(summary: dict[str, Any], prepared: dict[str, Any],
                             calibration: dict[str, Any], assay: str) -> dict[str, Any]:
    """Predeclared library/detection check on the generated feature universe.

    A subset has fewer expected total counts than the full empirical panel.
    Scale the empirical quantile reference by the summed coherent feature means
    (or detections), not merely by the feature-count ratio.  This is an explicit
    panel-size approximation, never an assertion of exact quantile equality.
    """
    layout = prepared["layout"][assay]
    source = calibration["assays"][assay][layout["source_panel"]]
    targets = layout["source_profiles"]
    checks = {}
    for label, moment, tolerance in (("library_quantiles", "mean", "library_quantile_relative"),
                                      ("detected_features_quantiles", "detection", "detected_features_quantile_relative")):
        source_sum = sum(x[moment] for x in source["features"])
        target_sum = sum(x[moment] for x in targets)
        ratio = target_sum / source_sum if source_sum > 0 else 1.0
        target = np.asarray(source[label]) * ratio
        observed = np.asarray(summary[label])
        allowed = calibration["tolerances"][tolerance] * np.maximum(target, 1.0)
        errors = np.abs(observed - target)
        checks[label] = {"target": target.tolist(), "observed": observed.tolist(),
                          "relative_panel_scale": ratio, "allowed_absolute_errors": allowed.tolist(),
                          "passed_each_quantile": (errors <= allowed).tolist(),
                          "passed": bool(np.all(errors <= allowed))}
    return {"passed": all(x["passed"] for x in checks.values()), "checks": checks,
            "reference": "Empirical source-panel quantiles scaled by sum of assigned coherent feature means/detections; approximate for subsets.",
            "source_panel": layout["source_panel"], "zero_rows": summary["zero_rows"]}


def write_world(spec: dict[str, Any], calibration: dict[str, Any], out_dir: str | Path,
                 feature_counts: dict[str, int] | None = None,
                 cohort_sizes: dict[str, int] | None = None,
                 regimes: tuple[str, ...] | None = None,
                 calibration_n: int = 2048, background_calibration_n: int = 256,
                 feature_chunk_size: int = 256, conditional_mean_repetitions: int = 128,
                 visible_reference: str | Path | None = None,
                 prepared: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate once; partial or completed directories are never overwritten.

    Cohort sizes are per-assay for public cohorts and complete-cell totals for
    private cohorts.  Simple defaults to train=2000, hence one 6000-cell world.
    A second ambiguity member MUST reference the first member's public files.
    """
    root = Path(out_dir)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite frozen or partial world: {root}")
    if spec["motif"] == "ambiguity" and spec["rho"] != 0 and visible_reference is None:
        raise ValueError("Nonzero-rho ambiguity member requires visible_reference to guarantee exact public inputs")
    if regimes is None:
        regimes = ("realistic",) if spec["design"] == "simple" else ("realistic", "informative")
    if set(regimes) - set(REGIME_MULTIPLIERS):
        raise ValueError("Unknown observation regime")
    if cohort_sizes is None:
        cohort_sizes = {"train": 2000} if spec["design"] == "simple" else dict(EXTENDED_COHORT_SIZES)
    if any(size < 1 for size in cohort_sizes.values()):
        raise ValueError("All cohort sizes must be positive")
    if prepared is None:
        prepared = prepare_world(spec, calibration, feature_counts, calibration_n,
                                  background_calibration_n, feature_chunk_size)
    if prepared["spec"] != spec:
        raise ValueError("Prepared world specification does not match requested specification")
    if prepared["calibration_hash"] != calibration["calibration_hash"]:
        raise ValueError("Prepared world uses a different frozen calibration")
    if feature_counts is not None and prepared["feature_counts"] != feature_counts:
        raise ValueError("Prepared world uses different feature counts")
    root.mkdir(parents=True)
    (root / "truth").mkdir()
    # Persist frozen mechanisms BEFORE sampling any benchmark population.
    with (root / "truth/frozen_parameters.json").open("x") as handle:
        json.dump(prepared, handle, sort_keys=True, allow_nan=False)
    manifest = {"schema_version": SCHEMA_VERSION, "world_id": spec["world_id"],
                "design": spec["design"], "split": spec["split"], "regimes": list(regimes),
                "cohort_sizes": cohort_sizes, "feature_counts": prepared["feature_counts"],
                "calibration_hash": calibration["calibration_hash"],
                "frozen_parameters_sha256": file_sha256(root / "truth/frozen_parameters.json"),
                "status": "generating", "files": {}, "count_summaries": {}, "panel_calibration_audits": {},
                "focal_calibration_passed": prepared["passed_focal_calibration"],
                "information_boundary": "Only observed/ is learner-visible; private_inputs/ and truth/ are evaluator-only"}
    reference = Path(visible_reference) if visible_reference is not None else None
    if reference is not None:
        ref_manifest = json.loads((reference / "manifest.json").read_text())
        if ref_manifest["status"] != "complete" or ref_manifest["cohort_sizes"] != cohort_sizes or ref_manifest["feature_counts"] != prepared["feature_counts"] or ref_manifest["regimes"] != list(regimes):
            raise ValueError("Ambiguity reference must be complete with identical cohorts/features/regimes")
        ref_parameters = json.loads((reference / "truth/frozen_parameters.json").read_text())
        if (spec["motif"] != "ambiguity" or ref_parameters["spec"]["motif"] != "ambiguity"
                or ref_parameters["spec"]["rho"] != 0.0
                or ref_parameters["spec"]["visible_seed"] != spec["visible_seed"]
                or ref_parameters["calibration_hash"] != prepared["calibration_hash"]
                or ref_parameters["layout"] != prepared["layout"]
                or ref_parameters["observation_parameters"] != prepared["observation_parameters"]):
            raise ValueError("Reference is not this exact ambiguity pair with identical measurement parameters")
    for cohort, size in cohort_sizes.items():
        population = _cohort_population(prepared, cohort, size)
        n = len(population["cell_ids"])
        exposures = np.column_stack([rng_for(spec["visible_seed"], cohort, "exposure", a).choice(prepared["exposure_distributions"][a], n) for a in ASSAYS])
        truth = {**population, "exposures": exposures,
                 "candidate_feature_ids": np.asarray([x["feature_id"] for x in prepared["candidates"]]),
                 "candidate_names": np.asarray([x["name"] for x in prepared["candidates"]]),
                 "candidate_assays": np.asarray([x["assay"] for x in prepared["candidates"]]),
                 "candidate_roles": np.asarray([x["role"] for x in prepared["candidates"]])}
        if conditional_mean_repetitions:
            means, errors = conditional_activity_means(prepared, population["state"],
                                                       stream_seed(spec["seed"], cohort), conditional_mean_repetitions)
            truth.update(conditional_mean_activities=means, conditional_mean_standard_errors=errors,
                         conditional_mean_repetitions=np.array(conditional_mean_repetitions))
        for regime in regimes:
            counts = _sample_full_count_panel(prepared, population, exposures, cohort, regime)
            truth["counts_" + regime] = counts
            for assay in ASSAYS:
                area = "private_inputs" if cohort in {"probe", "assessment"} else "observed"
                relative_path = Path(area) / regime / cohort / f"{assay}.h5ad"
                path = root / relative_path
                if reference is not None and area == "observed":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(reference / relative_path, path)
                    manifest["count_summaries"][str(relative_path)] = ref_manifest["count_summaries"][str(relative_path)]
                else:
                    matrix, rows = _assay_matrix(prepared, population, exposures, counts, assay, cohort, regime)
                    _write_assay(path, matrix, rows, prepared, population, assay)
                    manifest["count_summaries"][str(relative_path)] = _generated_summary(matrix)
                if regime == "realistic":
                    manifest["panel_calibration_audits"][str(relative_path)] = _assess_generated_panel(
                        manifest["count_summaries"][str(relative_path)], prepared, calibration, assay)
                manifest["files"][str(relative_path)] = file_sha256(path)
        truth_path = root / "truth" / f"{cohort}.npz"
        np.savez_compressed(truth_path, **truth)
        manifest["files"][str(truth_path.relative_to(root))] = file_sha256(truth_path)
    truth_manifest = {"schema_version": SCHEMA_VERSION, "spec": spec,
                      "family": spec["family"], "motif": spec["motif"],
                      "candidates": prepared["candidates"],
                      "biological_dag_edges": prepared["biological_dag_edges"],
                      "core_dag_edges": prepared["core_dag_edges"],
                      "state_nodes": [f"S{k}" for k in range(spec["state_dimension"])],
                      "measurement_graph": {"activity_to_count": [[x["name"], x["name"] + "_count"] for x in prepared["candidates"]],
                                             "exposure_to_count": [["L_" + x["assay"], x["name"] + "_count"] for x in prepared["candidates"]]},
                      "conditional_mean_method": "independent ancestral Monte Carlo conditional on same state, not zero innovations",
                      "conditional_mean_repetitions": conditional_mean_repetitions,
                      "source_reconstruction_masks": {a: prepared["layout"][a]["source_reconstruction_mask"] for a in ASSAYS},
                      "calibration_audit": prepared["calibration_audit"],
                      "ambiguity_reference": str(reference.resolve()) if reference else None}
    with (root / "truth/manifest.json").open("x") as handle:
        json.dump(truth_manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    manifest["files"]["truth/manifest.json"] = file_sha256(root / "truth/manifest.json")
    manifest["status"] = "complete"
    manifest["panel_calibration_passed"] = all(x["passed"] for x in manifest["panel_calibration_audits"].values())
    manifest["count_calibration_status"] = ("passed_declared_checks" if manifest["focal_calibration_passed"] and manifest["panel_calibration_passed"]
                                            else "count_model_check_failed; do not label this a validated-realism benchmark")
    manifest["manifest_hash"] = canonical_hash(manifest)
    with (root / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    return manifest


def validate_world(root: str | Path, verify_hashes: bool = True) -> dict[str, Any]:
    """Audit truth separation, exact measured readouts, disjointness and hashes."""
    import anndata as ad

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete world")
    expected_hash = manifest.pop("manifest_hash")
    if canonical_hash(manifest) != expected_hash:
        raise ValueError("Manifest hash mismatch")
    if verify_hashes:
        if file_sha256(root / "truth/frozen_parameters.json") != manifest["frozen_parameters_sha256"]:
            raise ValueError("Frozen parameter hash mismatch")
        for name, expected in manifest["files"].items():
            if file_sha256(root / name) != expected:
                raise ValueError(f"Artifact hash mismatch: {name}")
    seen_public, all_bio_ids = set(), set()
    checked = 0
    for cohort in manifest["cohort_sizes"]:
        with np.load(root / "truth" / f"{cohort}.npz", allow_pickle=False) as truth:
            cell_ids = truth["cell_ids"].astype(str)
            if len(set(cell_ids)) != len(cell_ids) or all_bio_ids.intersection(cell_ids):
                raise ValueError("Biological cell IDs repeat within/across cohorts")
            all_bio_ids.update(cell_ids)
            index = {x: i for i, x in enumerate(cell_ids)}
            area = "private_inputs" if cohort in {"probe", "assessment"} else "observed"
            for regime in manifest["regimes"]:
                for assay in ASSAYS:
                    data = ad.read_h5ad(root / area / regime / cohort / f"{assay}.h5ad")
                    if set(data.obs) != {"assay", "total_counts", "n_features"} or set(data.var) != {"gene_id", "channel_id", "chrom", "chromStart", "chromEnd", "strand"}:
                        raise ValueError("Unexpected metadata at inference boundary")
                    if data.obsm or data.layers or data.raw is not None or set(data.uns) != {"schema_version"}:
                        raise ValueError("Unexpected rich object at inference boundary")
                    matrix = sparse.csr_matrix(data.X)
                    if (matrix.data < 0).any() or not np.equal(matrix.data, np.floor(matrix.data)).all():
                        raise ValueError("Invalid raw counts")
                    rows = np.array([index[str(x)] for x in data.obs_names])
                    if area == "observed" and not np.all(truth["assigned_assay"][rows] == assay):
                        raise ValueError("Observed assay mismatch")
                    if regime == manifest["regimes"][0] and area == "observed":
                        if seen_public.intersection(data.obs_names):
                            raise ValueError("Cell identity overlaps public assay cohorts")
                        seen_public.update(data.obs_names)
                    columns = np.flatnonzero(truth["candidate_assays"] == assay)
                    feature_ids = truth["candidate_feature_ids"][columns].astype(str)
                    observed = data[:, feature_ids].X.toarray()
                    expected = truth["counts_" + regime][np.ix_(rows, columns)]
                    if not np.array_equal(observed, expected):
                        raise ValueError(f"Observed scored counts disagree with truth: {cohort}/{regime}/{assay}")
                    checked += 1
    return {"passed": True, "h5ad_checked": checked, "public_cells": len(seen_public),
            "biological_cells": len(all_bio_ids), "manifest_hash": expected_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--design", choices=["simple", "extended"], default="simple")
    parser.add_argument("--world-id", help="Required for one extended bank specification")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-bank", action="store_true")
    parser.add_argument("--visible-reference", type=Path)
    parser.add_argument("--conditional-mean-repetitions", type=int, default=128)
    args = parser.parse_args()
    if args.list_bank:
        print(json.dumps(make_extended_specs(args.seed), indent=2))
        return
    if args.validate_only:
        print(json.dumps(validate_world(args.out), indent=2))
        return
    if args.calibration is None:
        parser.error("--calibration is required to generate a world")
    calibration = json.loads(args.calibration.read_text())
    if args.design == "simple":
        spec = make_simple_spec(args.seed)
    else:
        selected = [x for x in make_extended_specs(args.seed) if x["world_id"] == args.world_id]
        if not selected:
            parser.error("--world-id must identify one frozen extended bank specification")
        spec = selected[0]
    manifest = write_world(spec, calibration, args.out, visible_reference=args.visible_reference,
                            conditional_mean_repetitions=args.conditional_mean_repetitions)
    print(json.dumps({"world_id": manifest["world_id"], "path": str(args.out),
                      "manifest_hash": manifest["manifest_hash"], "validation": validate_world(args.out)}, indent=2))


if __name__ == "__main__":
    main()

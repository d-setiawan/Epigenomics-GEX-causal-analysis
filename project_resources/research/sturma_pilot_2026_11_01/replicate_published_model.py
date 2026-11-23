"""Reduced, frozen replication of the unchanged Sturma et al. published control."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
import warnings

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from scipy.stats import beta, chi2, expon, gumbel_r, lognorm, skewnorm, weibull_min

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
sys.path.insert(0, str(UPSTREAM))
from experiments.rand import rand_model
from src.learner import LinearMDCRL

PROTOCOL = HERE / "published_replication_protocol.json"
RESULTS = HERE / "published_replication_results"
COMMIT = "578b5900590164b2ed1750fb6a7f98ae7153e220"
SOURCE_FILES = ["experiments/rand.py", "experiments/experiments.py", "experiments/utils.py",
                "src/learner.py", "src/scoring.py", "experiments/experiment_l=3/run_ndom=2.py",
                "experiments/experiment_l=3/run_ndom=3.py"]
REFERENCE = UPSTREAM / "experiments/experiment_l=3/results_gamma=0.2/stats.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    return {"python": sys.version, "executable": sys.executable,
            "packages": {p: importlib.metadata.version(p) for p in
                         ["numpy", "scipy", "scikit-learn", "causaldag", "networkx"]},
            "blas_threads": {k: os.environ[k] for k in
                             ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]}}


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError("Protocol already exists; it will not be overwritten.")
    head = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    if head != COMMIT or dirty:
        raise RuntimeError("Upstream pin or clean-tree check failed.")
    protocol = {
        "study": "Reduced published-model replication; not the original 1000-model experiment",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream_commit": COMMIT, "upstream_hashes": {f: sha(UPSTREAM / f) for f in SOURCE_FILES},
        "runner_sha256": sha(__file__), "test_sha256": sha(HERE / "test_published_replication.py"),
        "reference_stats_sha256": sha(REFERENCE), "environment": environment(),
        "settings": {"l": 3, "domains": [2, 3], "sample_sizes_per_domain": [1000, 10000, 25000],
                     "replicate_seeds": list(range(10)), "measure": "ks-test", "alpha": .05, "gamma": .2,
                     "dims": {"2": [15, 15], "3": [10, 10, 10]}, "private_latents_per_domain": 2,
                     "graph_density": .75, "mixing_density": .9, "mixing_distribution": "unif",
                     "two_pure_children": True, "fit_timeout_seconds": 120},
        "noise_distributions": ["beta(2,3)", "beta(2,5)", "chi2(df=4)", "gumbel_r", "lognorm(s=1)",
                                "weibull_min(c=2)", "expon(scale=.1)", "skewnorm(a=6)", "skewnorm(a=12)"],
        "rng": "Reset NumPy and Python random to replicate seed before generation, then to seed+100000 before native ICA. Seeds repeat across settings; comparisons across settings are not independent.",
        "observations": "Native rand_model produces independent cell draws per domain. No biological benchmark worlds, count layer or integration embeddings are added.",
        "metrics": {
            "dimension": "P(lhat=3), under/over/zero rates; all ten assigned replicates remain in denominators.",
            "mixing_native": "Unchanged native raw shared-column score divided by beta=min(lhat,3)*30, as upstream utils; lhat=0 is unavailable.",
            "mixing_paper_normalization": "Same native alignment numerator divided by sqrt(beta), separately labeled; alignment itself remains the native two-step heuristic.",
            "graph": "Unchanged native coefficient score divided by3, only when lhat=3 and graph finite; no adjacency threshold or exact-DAG claim.",
            "unconditional_coefficient_criterion": "Correct dimension and finite native graph_error/3<=0.1; failures and missing graphs count as false. Descriptive coefficient criterion, not exact causal graph recovery.",
            "convergence": "Capture all warnings; ConvergenceWarning marks ICA convergence concern. No-warning is not an identification certificate.",
            "graph_diagnostic": "Count nonzero diagonal and bidirectional coefficient pairs above1e-12 solely as numerical diagnostics, not causal adjacency selection."
        },
        "known_limitations": ["Current environment differs from upstream pinned historical dependencies.",
                              "Covariance numerical rank determines ICA component count; gamma is not the ICA rank threshold.",
                              "Native graph pure-child heuristic has reviewed duplicate-endpoint and candidate-exhaustion edge cases; unchanged and all exceptions retained.",
                              "Native signed/permutation scorers choose an absolute-entry permutation first; not exhaustive optimization of the final signed error.",
                              "Ten replicates per setting provide a small descriptive check, not a reliability guarantee.",
                              "Native graph estimate may be dense/cyclic; coefficient error is not exact DAG accuracy."]
    }
    write_json(PROTOCOL, protocol)
    print(json.dumps({"protocol": str(PROTOCOL), "sha256": sha(PROTOCOL), "fits": 60}), flush=True)


def check_protocol():
    p = json.loads(PROTOCOL.read_text())
    if sha(__file__) != p["runner_sha256"] or sha(HERE / "test_published_replication.py") != p["test_sha256"]:
        raise RuntimeError("Frozen runner/test changed.")
    for f, expected in p["upstream_hashes"].items():
        if sha(UPSTREAM / f) != expected:
            raise RuntimeError(f"Frozen upstream changed: {f}")
    if environment() != p["environment"] or sha(REFERENCE) != p["reference_stats_sha256"]:
        raise RuntimeError("Environment/reference differs from frozen protocol.")
    return p


def model_specs(m, n):
    return {"nr_doms": m, "joint_idx": [0, 1, 2],
            "domain_spec_idx": [[3 + 2*i, 4 + 2*i] for i in range(m)],
            "noise_rvs": [beta(2, 3), beta(2, 5), chi2(df=4), gumbel_r, lognorm(s=1),
                          weibull_min(c=2), expon(scale=.1), skewnorm(a=6), skewnorm(a=12)],
            "sample_sizes": [n] * m, "dims": [30 // m] * m,
            "graph_density": .75, "mixing_density": .9,
            "mixing_distribution": "unif", "two_pure_children": True}


def normalized_mixing(raw, lhat):
    if lhat == 0 or raw is None or not np.isfinite(raw):
        return None, None
    beta_size = min(lhat, 3) * 30
    return float(raw / beta_size), float(raw / np.sqrt(beta_size))


def timeout(_signum, _frame):
    raise TimeoutError("Frozen per-fit wall-time cap exceeded.")


def run_one(m, n, seed, protocol):
    out = RESULTS / f"m{m}_n{n}_seed{seed}.json"
    if out.exists():
        old = json.loads(out.read_text())
        if old["protocol_sha256"] != sha(PROTOCOL):
            raise RuntimeError("Existing result has a different protocol.")
        return old
    r = {"domains": m, "n_per_domain": n, "seed": seed, "ica_seed": seed+100000,
         "protocol_sha256": sha(PROTOCOL), "status": "started", "nr_joint": None,
         "mixing_error_raw": None, "mixing_error_native_normalized": None,
         "mixing_error_sqrt_beta_normalized": None, "graph_error_raw": None,
         "graph_error_normalized": None, "unconditional_coefficient_criterion": False}
    arrays = {}
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            random.seed(seed); np.random.seed(seed)
            data, graph, B_true = rand_model(model_specs(m, n))
            A_true = graph.to_amat().T
            arrays.update(B_true=B_true, A_true=A_true)
            r["data_sha256"] = [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in data]
            random.seed(seed+100000); np.random.seed(seed+100000)
            model = LinearMDCRL(measure="ks-test", alpha=.05, gamma=.2)
            signal.signal(signal.SIGALRM, timeout)
            signal.alarm(protocol["settings"]["fit_timeout_seconds"])
            fit_started = time.monotonic()
            try:
                model.fit(data)
            finally:
                signal.alarm(0)
                r["fit_seconds"] = time.monotonic()-fit_started
                r["nr_joint"] = int(model.nr_joint) if hasattr(model, "nr_joint") else None
                r["ica_components"] = [int(x.shape[1]) for x in getattr(model, "mixings", [])]
                if hasattr(model, "joint_mixing"): arrays["B_estimated"] = model.joint_mixing
                if hasattr(model, "A"): arrays["A_estimated"] = model.A
            lhat = r["nr_joint"]
            raw = float(model.score_shared_columns(B_true, 3))
            r["mixing_error_raw"] = raw if lhat > 0 and np.isfinite(raw) else None
            r["mixing_error_native_normalized"], r["mixing_error_sqrt_beta_normalized"] = normalized_mixing(raw, lhat)
            r["status"] = "no_shared_factors" if lhat == 0 else "underestimated_dimension" if lhat < 3 else "overestimated_dimension" if lhat > 3 else "correct_dimension"
            if lhat == 3:
                error, aligned = model.score_graph_param_matrix(A_true)
                arrays["A_aligned_native"] = aligned
                if np.isfinite(error) and np.all(np.isfinite(model.A)):
                    r["graph_error_raw"] = float(error)
                    r["graph_error_normalized"] = float(error/3)
                    r["unconditional_coefficient_criterion"] = bool(error/3 <= .1)
                    r["status"] = "correct_dimension_graph_available"
                    active = np.abs(model.A) > 1e-12
                    r["graph_numeric_diagnostics"] = {"diagonal_entries_above_1e_minus12": int(np.diag(active).sum()),
                        "bidirectional_pairs_above_1e_minus12": int(np.triu(active & active.T, 1).sum())}
                else:
                    r["status"] = "correct_dimension_graph_nonfinite"
        except Exception as exc:
            r["status"] = "exception"
            r["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        finally:
            signal.alarm(0)
    r["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    r["ica_convergence_warning"] = any(w["category"] == "ConvergenceWarning" for w in r["warnings"])
    r["elapsed_seconds"] = time.monotonic()-started
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    r["matrices_sha256"] = sha(out.with_suffix(".npz"))
    write_json(out, r)
    print(json.dumps({k:r[k] for k in ["domains", "n_per_domain", "seed", "status", "nr_joint", "graph_error_normalized", "elapsed_seconds"]}), flush=True)
    return r


def distribution(values):
    x = [v for v in values if v is not None and np.isfinite(v)]
    return {"n": len(x), "median": float(np.median(x)) if x else None,
            "q25": float(np.quantile(x,.25)) if x else None,
            "q75": float(np.quantile(x,.75)) if x else None}


def summarize(protocol):
    reference = json.loads(REFERENCE.read_text())
    groups = []
    for m in protocol["settings"]["domains"]:
        for n in protocol["settings"]["sample_sizes_per_domain"]:
            rows = []
            for seed in protocol["settings"]["replicate_seeds"]:
                f = RESULTS/f"m{m}_n{n}_seed{seed}.json"
                if f.exists():
                    row=json.loads(f.read_text())
                    if row["protocol_sha256"] != sha(PROTOCOL): raise RuntimeError("Wrong protocol result")
                    if sha(f.with_suffix(".npz")) != row["matrices_sha256"]: raise RuntimeError("Matrix seal mismatch")
                    rows.append(row)
            denom=len(protocol["settings"]["replicate_seeds"])
            group={"domains":m,"n_per_domain":n,"planned":denom,"completed":len(rows),
                   "status_counts":{s:sum(r["status"]==s for r in rows) for s in sorted({r["status"] for r in rows})},
                   "dimension_correct_rate":sum(r["nr_joint"]==3 for r in rows)/denom,
                   "graph_available_rate":sum(r["graph_error_normalized"] is not None for r in rows)/denom,
                   "unconditional_coefficient_criterion_rate":sum(r["unconditional_coefficient_criterion"] for r in rows)/denom,
                   "ica_convergence_warning_count":sum(r["ica_convergence_warning"] for r in rows),
                   "average_number_shared":float(np.mean([r["nr_joint"] for r in rows if r["nr_joint"] is not None])) if any(r["nr_joint"] is not None for r in rows) else None,
                   "too_many_shared_rate":sum(r["nr_joint"] is not None and r["nr_joint"]>3 for r in rows)/denom}
            for key in ["mixing_error_native_normalized","mixing_error_sqrt_beta_normalized","graph_error_normalized"]:
                group[key]=distribution([r[key] for r in rows])
                group[key+"_conditional_correct_dimension"]=distribution([r[key] for r in rows if r["nr_joint"]==3])
            ix=[1000,2500,5000,10000,25000].index(n)
            group["upstream_saved_stats"]={k:reference[str(m)][k]["mean"][ix] for k in ["number_shared","too_many_shared_rate","mixing_error","graph_error"]}
            groups.append(group)
    summary={"protocol_sha256":sha(PROTOCOL),"reference_stats_sha256":sha(REFERENCE),
             "scope":"Reduced60-fit check with current dependencies; original saved stats aggregate1000 models per setting. Their error keys named mean actually contain medians.",
             "groups":groups}
    write_json(HERE/"published_replication_summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument("command", choices=["freeze","run","summarize"]);args=parser.parse_args()
    if args.command=="freeze": freeze();return
    p=check_protocol()
    if args.command=="run":
        for m in p["settings"]["domains"]:
            for n in p["settings"]["sample_sizes_per_domain"]:
                for seed in p["settings"]["replicate_seeds"]: run_one(m,n,seed,p)
    summary=summarize(p)
    print(json.dumps({"summary":str(HERE/"published_replication_summary.json"),"completed":sum(g["completed"] for g in summary["groups"])}),flush=True)


if __name__=="__main__": main()

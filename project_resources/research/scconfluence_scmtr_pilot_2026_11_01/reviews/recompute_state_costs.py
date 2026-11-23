"""Independently reconstruct learned assignments and state costs, without fitting."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = REPO / "pipelines/simulation/integration_recoverability"
ASSAYS = ("RNA", "H3K27ac", "H3K4me3")


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def rows(source, wanted):
    lookup = {str(v): i for i, v in enumerate(source)}
    assert len(lookup) == len(source) and set(wanted) <= set(lookup)
    return np.asarray([lookup[v] for v in wanted])


def review(path):
    report = json.loads(path.read_text())
    target = HERE / f"{report['world_id']}_seed{report['seed']}_state_audit.json"
    binding = {"report": str(path), "report_sha256": sha(path), "reviewer_sha256": sha(__file__)}
    if target.exists():
        assert json.loads(target.read_text())["binding"] == binding
        return
    world, seed = report["world_id"], report["seed"]
    label = f"scconfluence_seed{seed}"
    model = BASE / "runs/integration_validation/20261101_scconfluence_v1" / world / f"seed{seed}"
    encoding = json.loads((model / "val/encoding_manifest.json").read_text())
    pools = report["pools"]
    z = {}
    for a in ASSAYS:
        file = model / "val" / f"{a}.npz"
        assert sha(file) == encoding["exports"][a]["sha256"]
        with np.load(file, allow_pickle=False) as data:
            z[a] = data["shared"][rows(data["cell_ids"], pools["val"][a])]
    chosen = {"RNA": np.arange(len(z["RNA"]))}
    for a in ASSAYS[1:]:
        source, destination = linear_sum_assignment(cdist(z["RNA"], z[a], metric="euclidean"))
        np.testing.assert_array_equal(source, chosen["RNA"])
        chosen[a] = destination
    for a in ASSAYS:
        np.testing.assert_array_equal(np.asarray(pools["val"][a])[chosen[a]],
                                      report["assignments"][label]["cell_ids"][a])
    states = {}
    for cohort in ("train", "val"):
        file = BASE / "runs/worlds" / world / "truth" / f"{cohort}.npz"
        assert sha(file) == report["truth_hashes"][cohort]
        with np.load(file, allow_pickle=False) as truth:
            states[cohort] = {a: truth["state"][rows(truth["cell_ids"], pools[cohort][a])] for a in ASSAYS}
    pooled = np.concatenate([states["train"][a] for a in ASSAYS])
    center, scale = pooled.mean(axis=0), pooled.std(axis=0)
    np.testing.assert_allclose(center, report["state_center"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(scale, report["state_scale"], rtol=0, atol=1e-12)
    aligned = {a: (states["val"][a][chosen[a]] - center) / scale for a in ASSAYS}
    costs = {}
    for a, b in itertools.combinations(ASSAYS, 2):
        distance = np.sum((aligned[a][:, None] - aligned[b][None, :]) ** 2, axis=2)
        i, j = linear_sum_assignment(distance)
        matched, oracle, random = np.diag(distance).mean(), distance[i, j].mean(), distance.mean()
        values = {"matched_squared_cost": float(matched), "oracle_squared_cost": float(oracle),
                  "random_expected_squared_cost": float(random), "normalized_gap": float((matched - oracle) / (random - oracle))}
        for key, value in values.items():
            np.testing.assert_allclose(value, report["results"][label]["pairwise_state_transport"][f"{a}--{b}"][key], rtol=0, atol=1e-10)
        costs[f"{a}--{b}"] = values
    worst = max(v["normalized_gap"] for v in costs.values())
    np.testing.assert_allclose(worst, report["results"][label]["worst_pair_gap"], rtol=0, atol=1e-10)
    with target.open("x") as handle:
        json.dump({"status": "passed", "binding": binding, "assignments_exact": True,
                   "train_state_scaling_reproduced": True, "costs": costs, "worst_pair_gap": worst,
                   "scope": "Independent NumPy/SciPy assignment, scaling and all state-cost formulas; no scorer helpers or model refits",
                   "not_checked": "Secondary decoder and geometry metrics are outside this numeric audit"}, handle, indent=2)
    print(target.name, "passed", worst)


if __name__ == "__main__":
    reports = BASE / "runs/integration_validation/20261101_scconfluence_state_v1/reports"
    for path in sorted(reports.glob("*.json")):
        review(path)

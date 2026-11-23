"""Compare sealed cap20 and cap6 readouts on unchanged data/targets."""
from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
OLD = STUDY.parent / "count_extensions_2026_11_01"
ASSAYS = ["RNA", "H3K27ac", "H3K4me3"]
VIEWS = ["full_panel", "original30_anchor"]
METRICS = ["model_full", "model_offdiagonal", "diagonal_training_baseline_full", "diagonal_training_baseline_offdiagonal"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sealed(study):
    for stage in ["runs_v1", "evaluation_v1"]:
        seal = json.loads((study / stage / "provenance.json").read_text())
        for name, digest in seal["artifacts"].items():
            assert sha(study / stage / name) == digest
    return {r["arm"]: r for r in json.loads((study / "evaluation_v1/summary.json").read_text())}


def tally(values):
    x = np.asarray(values)
    return {"unique_dataset_assay_units": len(x), "lower_error": int(np.sum(x < -1e-12)),
            "tie_at_1e_12": int(np.sum(np.abs(x) <= 1e-12)), "higher_error": int(np.sum(x > 1e-12)),
            "median_cap20_minus_cap6_rmse": float(np.median(x))}


def main():
    current, previous = load_sealed(STUDY), load_sealed(OLD)
    protocol = json.loads((STUDY / "protocol.json").read_text())
    stats = {"warning_records": 0, "source_bootstrap_fitting_failures": 0,
             "current_cap_binding_assay_fits": 0, "unchanged_nuisance_array_checks": 0}
    eligible, units, detailed, states = Counter(), {}, [], []
    seed_delta, control_delta = 0., 0.
    fit_records = {}
    for name, record in current.items():
        before = previous[name]
        new_fit = json.loads((STUDY / "runs_v1" / name / "result.json").read_text())
        old_fit = json.loads((OLD / "runs_v1" / name / "result.json").read_text())
        assert new_fit["variant"] == old_fit["variant"]
        assert new_fit["exposure_parameters"] == old_fit["exposure_parameters"]
        assert new_fit["input_hashes"] == old_fit["input_hashes"]
        stats["warning_records"] += len(new_fit["warnings"])
        fit_records[name] = new_fit
        for a, b in zip(new_fit["assays"], old_fit["assays"]):
            assert a["assay"] == b["assay"]
            assert a["rank"]["uncapped"] == b["rank"]["uncapped"]
            assert a["k"] == a["rank"]["uncapped"]
            stats["current_cap_binding_assay_fits"] += int(a["rank"]["cap_binding"])
            stats["source_bootstrap_fitting_failures"] += len(a["bootstrap_failures"])
            eligible[record["dataset"], record["control"]] += a["eligible_source_count"]
        with np.load(STUDY / "runs_v1" / name / "arrays.npz", allow_pickle=False) as now, \
                np.load(OLD / "runs_v1" / name / "arrays.npz", allow_pickle=False) as old:
            for assay in ASSAYS:
                for key in ["features", "fit_indices", "calibration_indices", "mean", "scale", "noise_diagonal", "poisson_noise_diagonal"]:
                    np.testing.assert_array_equal(now[f"{key}_{assay}"], old[f"{key}_{assay}"])
                    stats["unchanged_nuisance_array_checks"] += 1
        if record["control"] != "original":
            for assay in ASSAYS:
                for view in VIEWS:
                    for metric in METRICS:
                        delta = abs(record["within_assay"][assay]["views"][view][metric]["rmse"]
                                    - before["within_assay"][assay]["views"][view][metric]["rmse"])
                        control_delta = max(control_delta, delta)
                        assert delta <= 1e-12
        if "simulation_state" in record:
            for assay in ASSAYS:
                states.append({"dataset": record["dataset"], "variant": record["variant"]["name"],
                               "source_seed": record["source_seed"], "control": record["control"], "assay": assay,
                               "cap20_all_sources": record["simulation_state"][assay]["all_sources"],
                               "cap6_all_sources": before["simulation_state"][assay]["all_sources"]})
    for variant in protocol["variants"]:
        name = variant["name"]
        units[name] = {view: {metric: [] for metric in ["model_full", "model_offdiagonal"]} for view in VIEWS}
        for dataset in protocol["datasets"]:
            for assay in ASSAYS:
                item = {"variant": name, "dataset": dataset, "assay": assay, "views": {}}
                for view in VIEWS:
                    item["views"][view] = {}
                    for metric in METRICS:
                        pairs = {}
                        for label, collection in [("cap6", previous), ("cap20", current)]:
                            values = [collection[f"{dataset}__{name}__seed{seed}"]["within_assay"][assay]["views"][view][metric]["rmse"] for seed in [0, 1]]
                            seed_delta = max(seed_delta, abs(values[0] - values[1]))
                            assert abs(values[0] - values[1]) <= 1e-12
                            pairs[label] = float(np.mean(values))
                        item["views"][view][metric] = {**pairs, "cap20_minus_cap6": pairs["cap20"] - pairs["cap6"]}
                        if metric in units[name][view]:
                            units[name][view][metric].append(pairs["cap20"] - pairs["cap6"])
                fit = fit_records[f"{dataset}__{name}__seed0"]
                own = next(x for x in fit["assays"] if x["assay"] == assay)
                item.update(cap6_fitted_rank=6, cap20_fitted_rank=own["k"], uncapped_selection=own["rank"]["uncapped"])
                detailed.append(item)
    contrasts = {variant: {view: {metric: tally(values) for metric, values in metrics.items()}
                           for view, metrics in views.items()} for variant, views in units.items()}
    state_extrema = {}
    for dataset in ["linear_chain_0", "constrained_collider_0"]:
        state_extrema[dataset] = {}
        for cap in ["cap6", "cap20"]:
            eligible_states = []
            for entry in states:
                if entry["dataset"] != dataset or entry["control"] != "original":
                    continue
                for coordinate, value in enumerate(entry[f"{cap}_all_sources"]["max_absolute_correlation_per_state"]):
                    if value is not None:
                        eligible_states.append((value, entry["variant"], entry["source_seed"], entry["assay"], coordinate))
            largest = max(x[0] for x in eligible_states)
            state_extrema[dataset][cap] = {"descriptive_maximum": largest,
                "attaining_records": [{"variant": x[1], "source_seed": x[2], "assay": x[3], "state_coordinate": x[4]}
                                      for x in eligible_states if abs(x[0] - largest) <= 1e-14]}
    output = {"status": "complete", "script_sha256": sha(Path(__file__)),
              "cap20_fit_seal_sha256": sha(STUDY / "runs_v1/provenance.json"),
              "cap20_evaluation_seal_sha256": sha(STUDY / "evaluation_v1/provenance.json"),
              "cap6_fit_seal_sha256": sha(OLD / "runs_v1/provenance.json"),
              "cap6_evaluation_seal_sha256": sha(OLD / "evaluation_v1/provenance.json"),
              "independent_metric_replay_sha256": sha(STUDY / "reviews/extension_metric_audit.json"),
              "evaluated_arms": len(current), "checks": stats,
              "eligible_source_instances": [{"dataset": k[0], "control": k[1], "count": v} for k, v in sorted(eligible.items())],
              "max_seed_covariance_difference": seed_delta, "max_control_covariance_change": control_delta,
              "comparison_unit": "9 unique dataset-by-assay units per variant, averaging numerically invariant covariance seeds; not independent biological replicates",
              "within_assay_contrasts": contrasts, "unit_details": detailed,
              "all_source_state_maxima": states, "descriptive_state_extrema_on_matched_variant_subset": state_extrema,
              "state_caveat": "Maxima span components, variants and seeds; larger source sets change multiplicity. No source/state or graph identities established.",
              "moment_pair_matches": sum(r["moment_pair_match_count"] for r in current.values()),
              "confirmed_pair_matches": sum(r["confirmed_pair_match_count"] for r in current.values()),
              "model_or_seed_selected": False, "scientific_fit_or_setting_changes": False}
    path = STUDY / "reviews/cap6_comparison.json"
    with path.open("x") as f:
        json.dump(output, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": "complete", "path": str(path), "sha256": sha(path),
                      "max_seed_difference": seed_delta, "max_control_change": control_delta, "checks": stats}))


if __name__ == "__main__":
    main()

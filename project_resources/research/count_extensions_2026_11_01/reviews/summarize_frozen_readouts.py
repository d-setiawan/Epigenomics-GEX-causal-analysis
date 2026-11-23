"""Descriptive contrasts of sealed results; no fits or model selection.

Covariance comparisons collapse numerically invariant source seeds into one
dataset-by-assay unit. Source-state correlations retain both source seeds.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-12
ASSAYS = ["RNA", "H3K27ac", "H3K4me3"]
METRICS = ["model_full", "model_offdiagonal", "diagonal_training_baseline_full",
           "diagonal_training_baseline_offdiagonal"]
VIEWS = ["full_panel", "original30_anchor"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcomes(differences):
    values = np.asarray(differences, float)
    return {"unique_dataset_assay_units": len(values),
            "lower_error": int(np.sum(values < -TOLERANCE)),
            "tie_at_1e_12": int(np.sum(np.abs(values) <= TOLERANCE)),
            "higher_error": int(np.sum(values > TOLERANCE)),
            "median_variant_minus_comparator_rmse": float(np.median(values))}


def main():
    protocol = json.loads((STUDY / "protocol.json").read_text())
    rows = json.loads((STUDY / "evaluation_v1/summary.json").read_text())
    seal = json.loads((STUDY / "evaluation_v1/provenance.json").read_text())
    for name, digest in seal["artifacts"].items():
        assert sha(STUDY / "evaluation_v1" / name) == digest
    index = {(r["dataset"], r["variant"]["name"], r["source_seed"], r["control"]): r for r in rows}
    original, max_seed_delta = {}, 0.
    for dataset in protocol["datasets"]:
        for variant in protocol["variants"]:
            seed_rows = [index[dataset, variant["name"], seed, "original"] for seed in [0, 1]]
            for assay in ASSAYS:
                unit = {}
                for view in VIEWS:
                    unit[view] = {}
                    for metric in METRICS:
                        values = [r["within_assay"][assay]["views"][view][metric]["rmse"] for r in seed_rows]
                        assert all(value is not None for value in values)
                        delta = abs(values[0] - values[1])
                        max_seed_delta = max(max_seed_delta, delta)
                        assert delta <= TOLERANCE, "Covariance seeds are not invariant; cannot collapse"
                        unit[view][metric] = float(np.mean(values))
                original[dataset, variant["name"], assay] = unit
    contrasts = {}
    for variant in protocol["variants"]:
        name = variant["name"]
        baseline = f"poisson{variant['width']}_{variant['depth']}"
        entry = {"same_width_same_depth_poisson_comparator": baseline, "views": {}}
        for view in VIEWS:
            point = {}
            for metric, diagonal in [("model_full", "diagonal_training_baseline_full"),
                                     ("model_offdiagonal", "diagonal_training_baseline_offdiagonal")]:
                against_poisson, against_diagonal = [], []
                for dataset in protocol["datasets"]:
                    for assay in ASSAYS:
                        this = original[dataset, name, assay][view]
                        other = original[dataset, baseline, assay][view]
                        against_poisson.append(this[metric] - other[metric])
                        against_diagonal.append(this[metric] - this[diagonal])
                point[metric] = {"versus_poisson": outcomes(against_poisson),
                                 "versus_diagonal_training": outcomes(against_diagonal)}
            entry["views"][view] = point
        contrasts[name] = entry
    panel_contrasts = []
    for small in protocol["variants"]:
        if small["width"] != 30:
            continue
        compatible = [large for large in protocol["variants"] if large["width"] == 300
                      and all(small[key] == large[key] for key in ["depth", "theta", "retention", "orientation", "correction"])]
        for large in compatible:
            entry = {"small": small["name"], "large": large["name"], "view": "original30_anchor", "metrics": {}, "unit_details": []}
            differences = {key: [] for key in ["model_full", "model_offdiagonal"]}
            for dataset in protocol["datasets"]:
                for assay in ASSAYS:
                    a = original[dataset, small["name"], assay]["original30_anchor"]
                    b = original[dataset, large["name"], assay]["original30_anchor"]
                    unit = {"dataset": dataset, "assay": assay}
                    for key in differences:
                        differences[key].append(b[key] - a[key])
                        unit[key] = {"small": a[key], "large": b[key], "large_minus_small": b[key] - a[key]}
                    entry["unit_details"].append(unit)
            entry["metrics"] = {key: outcomes(values) for key, values in differences.items()}
            panel_contrasts.append(entry)
    fit_index = {}
    for entry in json.loads((STUDY / "runs_v1/summary.json").read_text()):
        result = json.loads((STUDY / "runs_v1" / entry["path"] / "result.json").read_text())
        fit_index[result["dataset"], result["variant"]["name"], result["source_seed"], result["control"]] = result
    real_table = []
    for variant in protocol["variants"]:
        if variant["depth"] != "unit":
            continue
        fitted = fit_index["scmtr_seq_s1", variant["name"], 0, "original"]
        by_assay = {r["assay"]: r for r in fitted["assays"]}
        for assay in ASSAYS:
            record = by_assay[assay]
            real_table.append({"variant": variant["name"], "assay": assay, "width": variant["width"],
                               "selected_rank": record["k"], "uncapped_rank": record["rank"]["uncapped"],
                               "cap_binding": record["rank"]["cap_binding"],
                               "views": original["scmtr_seq_s1", variant["name"], assay]})
    depth_parameters = {}
    for dataset in protocol["datasets"]:
        depth_parameters[dataset] = {}
        for mode in ["unit", "offset", "elastic"]:
            key = dataset, f"poisson300_{mode}", 0, "original"
            params = index[key]["exposure_parameters"]
            depth_parameters[dataset][mode] = params
            for result in rows:
                if result["dataset"] == dataset and result["variant"]["depth"] == mode:
                    assert result["exposure_parameters"] == params
    state_maxima = []
    for result in rows:
        if "simulation_state" not in result:
            continue
        for assay in ASSAYS:
            state = result["simulation_state"][assay]
            state_maxima.append({"dataset": result["dataset"], "variant": result["variant"]["name"],
                                 "source_seed": result["source_seed"], "control": result["control"],
                                 "assay": assay, "all_sources": state["all_sources"], "matched_levels": state["levels"]})
    statuses = Counter(value["status"] for r in rows for own in r["within_assay"].values() for value in own["views"].values())
    cross_values = [view for r in rows for pair in r.get("real_cross_assay", {}).values()
                    for level in pair["levels"].values() for view in level["views"].values()]
    caps = [a for r in fit_index.values() if r["control"] == "original" and r["variant"]["width"] == 300
            and r["variant"]["depth"] == "unit" for a in r["assays"]]
    receipt = {"status": "complete", "script_sha256": sha(Path(__file__)),
               "evaluation_seal_sha256": sha(STUDY / "evaluation_v1/provenance.json"),
               "independent_metric_audit_sha256": sha(STUDY / "reviews/extension_metric_audit.json"),
               "comparison_unit": "One dataset-by-assay unit per variant; covariance errors average two numerically invariant source seeds",
               "not_independent_biological_replicates": True, "max_covariance_seed_difference": max_seed_delta,
               "unit_count_per_contrast": 9, "comparison_tolerance": TOLERANCE,
               "width_comparison_rule": "Only original30 anchors and identical kernel, exposure, orientation, correction; never compare different full-panel targets",
               "fit_and_evaluation_arms": len(rows), "within_view_statuses": dict(statuses),
               "real_pair_level_view_count": len(cross_values),
               "real_primary_rmse_improvement_values": sorted({r["rmse_improvement_over_zero"] for r in cross_values}),
               "total_moment_pair_matches": sum(r["moment_pair_match_count"] for r in rows),
               "total_confirmed_pair_matches": sum(r["confirmed_pair_match_count"] for r in rows),
               "original_unit300_assay_fits": len(caps), "original_unit300_cap_binding_count": sum(a["rank"]["cap_binding"] for a in caps),
               "unit300_uncapped_rank_range": [min(a["rank"]["uncapped"] for a in caps), max(a["rank"]["uncapped"] for a in caps)],
               "cap_scope": "Every original unit300 assay hit cap6, making matching ineligible by protocol; this does not test unrestricted source dimensions",
               "variant_contrasts": contrasts, "nested_panel_anchor_contrasts": panel_contrasts,
               "real_unit_covariance_table": real_table, "frozen_exposure_parameters": depth_parameters,
               "all_source_state_maxima": state_maxima,
               "state_scope": "Both source seeds and controls retained; component maxima are descriptive with multiplicity, not posterior biology, source matching, or causal directions",
               "model_or_seed_selected": False, "fitting_or_threshold_changes": False}
    target = STUDY / "reviews/frozen_readout_contrasts.json"
    with target.open("x") as f:
        json.dump(receipt, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": "complete", "artifact": str(target), "sha256": sha(target),
                      "max_covariance_seed_difference": max_seed_delta,
                      "original_unit300_cap_binding_count": receipt["original_unit300_cap_binding_count"]}))


if __name__ == "__main__":
    main()

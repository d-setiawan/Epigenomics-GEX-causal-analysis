"""Summarize all declared arms without selecting a preferred seed or threshold."""

import json
import numpy as np

import causal_reliability_pilot as simulation
import causal_reliability_pilot_empirical as empirical


def ratio(value):
    return "undefined" if value is None else f"{value:.3f}"


def main():
    reports, sources = {}, {}
    for world in simulation.WORLDS:
        reports[world] = {}
        for arm in simulation.ARMS:
            path = simulation.ROOT / world / "arms" / f"{arm}.json"
            reports[world][arm] = simulation.read(path)
            sources[str(path)] = simulation.digest(path)
    real = {}
    for arm in empirical.ARMS:
        path = empirical.ROOT / "arms" / f"{arm}.json"
        real[arm] = empirical.read(path)
        sources[str(path)] = simulation.digest(path)
    out = {"source_reports": sources, "simulation": {}, "empirical": {}}
    lines = ["# Fixed integration reliability pilot", "",
             "All declared arms and seeds are reported. Two existing development worlds were reused; no new worlds or private assessment cohorts were opened.", "",
             "## Known synthetic graph targets", "",
             "Counts target the measurement PAG; activities target the biological activity projection. Discovery uses the frozen RCIT/FCI configuration on `log1p` original counts, with no assay-depth normalization or adjustment. Exact biological directions are an identifiability stress test.", ""]
    for scale in ("counts", "activities"):
        lines += [f"### {scale.capitalize()}", "", "| World | Arm | TP | FP | FN | Precision | Recall |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for world, arms in reports.items():
            for arm, report in arms.items():
                measured = report["results"]["full"]["scales"][scale]
                metrics = measured["metrics"]
                if measured["discovery"]["status"] != "ok":
                    lines.append(f"| {world} | {arm} | failed | failed | failed | undefined | undefined |")
                    continue
                cross = metrics["by_assay_pair"]["cross_assay"]
                out["simulation"].setdefault(world, {}).setdefault(arm, {})[scale] = {
                    "cross_assay": cross, "biological_pair_stress": metrics["exact_cross_assay_stress_test"]["pair"],
                    "biological_direction_stress": metrics["exact_cross_assay_stress_test"]["direction"]}
                lines.append(f"| {world} | {arm} | {cross['true_positive']} | {cross['false_positive']} | {cross['false_negative']} | {ratio(cross['precision'])} | {ratio(cross['recall'])} |")
        lines += [""]
    lines += ["### Exact biological pairs and directions: identifiability stress test", "",
              "Each world has two known direct biological cross-assay edges. The comparison below is a stress test, not an expected success criterion: measurement, latent confounding, observational equivalence and unpaired residuals can prevent direct-pair or direction recovery. Values are correct/extra edges against that biological target, separately for unordered pairs and exact directions.", "",
              "| World | Arm | Count pairs | Count directions | Activity pairs | Activity directions |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for world, arms in reports.items():
        for arm, report in arms.items():
            cells = []
            for scale in ("counts", "activities"):
                result = report["results"]["full"]["scales"][scale]
                if result["discovery"]["status"] != "ok":
                    cells.extend(("failed", "failed"))
                else:
                    for metric in ("pair", "direction"):
                        stress = result["metrics"]["exact_cross_assay_stress_test"][metric]
                        cells.append(f"{stress['true_positive']}/{stress['false_positive']}")
            lines.append(f"| {world} | {arm} | {' | '.join(cells)} |")
    lines += [""]
    lines += ["### Count subsampling", "",
              "Each triple lists TP/FP/FN for independent 80% assay subsamples at seeds 7000, 7001 and 7002, with rematching. These use fixed encoders and measure coarse conditional stability, not confidence intervals or false-discovery control.", "",
              "| World | Arm | Three cross-assay count readouts |", "| --- | --- | --- |"]
    for world, arms in reports.items():
        for arm, report in arms.items():
            entries = []
            for seed in simulation.SUBSAMPLE_SEEDS:
                result = report["results"][f"subsample_{seed}"]["scales"]["counts"]
                if result["discovery"]["status"] != "ok":
                    entries.append("failed")
                else:
                    score = result["metrics"]["by_assay_pair"]["cross_assay"]
                    entries.append("/".join(str(score[k]) for k in ("true_positive", "false_positive", "false_negative")))
            lines.append(f"| {world} | {arm} | {', '.join(entries)} |")
    lines += ["", "## Empirical scMTR holdout", "",
              "The 595-cell paired reference has no known causal graph or direction labels. All nine variables were excluded from integration with their overlapping/alias features. The primary paired reference has zero cross-assay adjacencies; coverage against that reference is undefined. Additional inferred adjacencies are disagreement with this empirical reference, not established false causal edges.", "",
              "| Arm | Cross edges | Paired-reference coverage | Log-count cross Pearson MAE | Cross Spearman MAE |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm, report in real.items():
        full = report["results"]["full"]
        agreement = full["paired_graph_agreement"]
        errors = {}
        for kind in ("pearson", "spearman"):
            valid = [r[f"{kind}_absolute_error"] for r in full["dependence"] if r["cross_assay"] and r[f"{kind}_absolute_error"] is not None]
            errors[kind] = float(np.mean(valid)) if valid else None
        cross = agreement.get("by_assay_pair", {}).get("cross_assay", {})
        out["empirical"][arm] = {"graph_agreement": agreement, "cross_dependence_error": errors,
                                  "retrieval": full["retrieval"], "day_transfer": report["day_transfer"]}
        lines.append(f"| {arm} | {cross.get('predicted', 'failed')} | {ratio(cross.get('reference_coverage'))} | {ratio(errors['pearson'])} | {ratio(errors['spearman'])} |")
    lines += ["", "### Exact heldout pair retrieval", "",
              "Fractions refer to full one-to-one assignments of the same 595 heldout cells, with opaque assay-specific IDs. Random expectation is 1/595 = 0.001681. Exact cell retrieval is a strict identity diagnostic, distinct from shared-state alignment.", "",
              "| Arm | RNA–H3K27ac | RNA–H3K4me3 | H3K27ac–H3K4me3 |", "| --- | ---: | ---: | ---: |"]
    for arm, report in real.items():
        pairs = report["results"]["full"]["retrieval"]["assignment"]
        lines.append("| " + arm + " | " + " | ".join(ratio(pairs[key]["exact_pair_fraction"]) for key in ("RNA--H3K27ac", "RNA--H3K4me3", "H3K27ac--H3K4me3")) + " |")
    lines += ["", "### Differentiation-day transfer", "",
              "Balanced accuracy uses a fixed source-assay train-only day decoder applied unchanged to every test assay. Day labels are evaluator-only and never select an encoder, panel or coupling. Four-class chance balanced accuracy is 0.25; these cells are not independent biological replicates.", "",
              "| Model seed | Training assay | RNA test | H3K27ac test | H3K4me3 test |", "| --- | --- | ---: | ---: | ---: |"]
    for arm in empirical.ARMS[2:]:
        day = real[arm]["day_transfer"]
        for source in empirical.ASSAYS:
            lines.append("| " + arm + " | " + source + " | " + " | ".join(ratio(day.get(f"{source}-->{target}", {}).get("balanced_accuracy")) for target in empirical.ASSAYS) + " |")
    null_root = simulation.ROOT.parent / "20261101_scmtr_product_null_audit_v1"
    null = simulation.read(null_root / "summary.json")
    diagnostic = simulation.read(null_root / "recurrent_edge_count_diagnostic.json")
    out["exploratory_product_null"] = {"summary": null, "descriptive_overlap_diagnostic": diagnostic}
    for filename in ("summary.json", "recurrent_edge_count_diagnostic.json"):
        path = null_root / filename
        sources[str(path)] = simulation.digest(path)
    lines += ["", "## Exploratory sparse-count null audit", "",
              f"A separately frozen check added after the primary random control produced at least one cross-assay edge in {null['any_cross_edge_count']}/{null['n_successful']} full-pool random couplings ({100 * null['any_cross_edge_fraction']:.1f}%). All {null['n_attempted']} outcomes were retained; {null['n_failed']} graphs failed. Mean cross-edge count was {null['cross_edge_count_mean']:.1f}, median 3, range 0–6. This conditional artificial-null frequency is not a biological FDR estimate.", "",
              "The most recurrent edge appeared in 23/50 couplings, exactly when its two variables had no co-detections. With 23 and 25 detections among 595 cells, zero overlap has 36.6% probability under independent fixed-margin permutation, yet the existing marginal test assigned p≈0.0004415 to this pattern. The binary-overlap calculation is descriptive and does not replace the CI test or alter a primary outcome.", "",
              "[Full null results](../20261101_scmtr_product_null_audit_v1/RESULTS.md) include every seed and independent source-count replay. This calibration failure limits biological causal interpretation of the sparse real-panel count graphs; it does not establish that the measurements lack biological relationships.", "",
              "## Interpretation and validation", "",
              "Weak paired-count controls limit what this pilot can establish about integration. Original assay depths are unadjusted in every count CI test. Histone test-panel detections are only 23–35/595. The random-coupling control can produce recurring cross-assay edges, so stability alone is not evidence of a correct causal graph.", "",
              "The simulations retain focal features in the original integration feature universes, which permits matching endogeneity. The real panel is held out from integration but is sparse and covers only three promoter loci in one pooled-day S1 sublibrary. Source DNA filtering across six targets was inherited from the released data. No empirical causal direction or false-positive-rate ground truth is available.", "",
              "Focused tests passed for both scorers; a second agent independently rebuilt and reran all eight empirical controls from prepared source counts with exact graph/endpoint agreement. A separate audit reconstructed train/test splits, training-only panel eligibility, and all overlapping-feature exclusions. The native loss correction independently passed all-pair gradient/order and empty-assay finite-backward checks.", "",
              "The machine-readable summary records hashes of every input arm report. Full query logs, graphs, assignments, endpoint stress metrics, and export receipts remain in the per-arm artifacts. Scientific sources and decision thresholds were frozen before these readouts.", ""]
    simulation.write(simulation.ROOT / "all_arm_summary.json", out)
    path = simulation.ROOT / "RESULTS.md"
    with path.open("x") as handle:
        handle.write("\n".join(lines))
    receipts = {}
    for world in simulation.WORLDS:
        simulation.context(world)
        for seed in (0, 1):
            key = f"{world}_scconfluence_seed{seed}"
            receipts[key] = simulation.pin(simulation.ROOT / f"{key}_execution.json")
    empirical.context()
    for seed in (0, 1):
        key = f"scmtr_scconfluence_seed{seed}"
        receipts[key] = simulation.pin(empirical.ROOT / f"{key}_execution.json")
    statuses = [s["discovery"]["status"] for arms in reports.values() for report in arms.values()
                for result in report["results"].values() for s in result["scales"].values()]
    statuses += [result["discovery"]["status"] for report in real.values() for result in report["results"].values()]
    completion = {"status": "complete" if all(s == "ok" for s in statuses) else "complete_with_failures",
                  "primary_graphs_attempted": len(statuses), "primary_graphs_successful": statuses.count("ok"),
                  "primary_graphs_failed": len(statuses) - statuses.count("ok"),
                  "exploratory_null_graphs": {k: null[k] for k in ("n_attempted", "n_successful", "n_failed")},
                  "report": simulation.pin(path), "summary": simulation.pin(simulation.ROOT / "all_arm_summary.json"),
                  "fit_scoring_receipts": receipts,
                  "protocols": {"simulation": simulation.pin(simulation.ROOT / "protocol.json"),
                                "empirical": simulation.pin(empirical.ROOT / "protocol.json"),
                                "exploratory_null": simulation.pin(null_root / "protocol.json")},
                  "summary_source": simulation.pin(__file__),
                  "interpretation": "Evaluation completed; successful execution does not establish reliable causal recovery"}
    simulation.write(simulation.ROOT / "completion.json", completion)
    print(json.dumps({"report": str(path), "summary": str(simulation.ROOT / "all_arm_summary.json"),
                      "completion": str(simulation.ROOT / "completion.json"), "status": completion["status"]}))


if __name__ == "__main__":
    main()

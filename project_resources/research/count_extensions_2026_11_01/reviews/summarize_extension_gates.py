"""Describe sealed training gate failures without changing any decision."""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np


def summarize(study):
    runs=study/"runs_v1"
    seal=runs/"provenance.json"
    assert seal.is_file()
    protocol=json.loads((study/"protocol.json").read_text())
    summary=json.loads((runs/"summary.json").read_text())
    overall=Counter(); statuses=Counter(); reasons=Counter(); warnings_count=Counter()
    gates=Counter(); by_dataset=defaultdict(Counter); by_control=defaultdict(Counter)
    by_assay=defaultdict(Counter); focused=defaultdict(Counter)
    closest={}; cap_sources=[]; bootstrap_reasons=Counter(); positive_control=[]
    for row in summary:
        record=json.loads((runs/row["path"]/"result.json").read_text())
        variant=record["variant"]; control=record["control"]
        overall["arms"]+=1;statuses["arm:"+record["status"]]+=1
        by_control[control]["arms"]+=1
        overall["raw_moment_equivalent_pairs"]+=len(record.get("moment_equivalent_pairs",[]))
        overall["moment_unique_pairs"]+=len(record.get("moment_pair_matches",[]))
        overall["confirmed_unique_pairs"]+=len(record.get("confirmed_pair_matches",[]))
        overall["confirmed_all_assay_candidates"]+=len(record.get("confirmed_all_assay_matches",[]))
        for w in record.get("warnings",[]):
            warnings_count[w["category"]+": "+w["message"]]+=1
        for assessment in record.get("cf_assessment",{}).values():
            overall["cf_assay:"+assessment["status"]]+=1
        for ar in record.get("assays",[]):
            stats=Counter(assay_fits=1,cap_binding_assay_fits=int(ar["rank"]["cap_binding"]),zero_rank_fits=int(ar["k"]==0))
            statuses["assay:"+ar["status"]]+=1
            if ar["status"]!="ok":
                reasons[str(ar.get("reason"))]+=1
                stats["abstained_assay_fits"]+=1
            if control!="original" and ar["k"]>0:
                positive_control.append(dict(path=row["path"],assay=ar["assay"],k=ar["k"],eligible=ar["eligible_source_count"]))
            for failure in ar.get("bootstrap_failures",[]):
                bootstrap_reasons[str(failure["reason"])]+=1
            stats["bootstrap_estimator_failures"]+=len(ar.get("bootstrap_failures",[]))
            if ar["status"]=="ok" and ar["k"]>0:
                k=ar["k"]
                tests={
                    "calibration_moments":np.asarray(ar["point_moment_valid"],bool),
                    "bootstrap_valid_count":np.asarray(ar["valid_bootstrap_counts"])>=protocol["bootstrap"]["min_valid"],
                    "median_loading_stability":np.asarray(ar["loading_cosine_median"],float)>=protocol["bootstrap"]["median_cosine"],
                    "lower_loading_stability":np.asarray(ar["loading_cosine_10th"],float)>=protocol["bootstrap"]["lower_cosine"],
                    "orientation_residual":np.full(k,ar["orientation_residual"]<=protocol["orientation"]["max_residual"]),
                    "pgf_effective_cells":np.full(k,bool(ar["effective_cells_gate"]))}
                other=np.logical_and.reduce(list(tests.values()))
                cap=bool(ar["rank"]["cap_binding"])
                stats.update(fitted_directions=k,pass_all_noncap_gates=int(other.sum()),cap_only_exclusions=int(other.sum())*cap,
                             eligible_directions=ar["eligible_source_count"],directions_with_another_failure=int((~other).sum()))
                variance=np.asarray(ar["calibration_variance"],float)
                stats["nonpositive_or_nonfinite_calibration_variance"]+=int((~np.isfinite(variance)|(variance<=0)).sum())
                for name,value in tests.items():
                    gates[name]+=int((~value).sum())
                    stats["fail:"+name]+=int((~value).sum())
                if cap:
                    gates["rank_cap"]+=k
                    for index in np.flatnonzero(other):
                        cap_sources.append(dict(path=row["path"],dataset=record["dataset"],assay=ar["assay"],source=int(index),
                                                uncapped_rank=ar["rank"]["uncapped"],fitted_rank=k))
                assert int((other & ~np.full(k,cap)).sum())==ar["eligible_source_count"]
            overall.update(stats);by_dataset[record["dataset"]].update(stats)
            by_control[control].update(stats);by_assay[ar["assay"]].update(stats)
            if variant["width"]==300 and variant["depth"]=="unit" and control=="original":
                focused[record["dataset"]].update(stats)
        for d in record.get("moment_diagnostics",[]):
            if not d["source_gates_pass"]:
                continue
            overall["eligible_signed_pair_comparisons"]+=1
            if d["point_scaled_distance"] is None or d["bootstrap_band_radius"] is None:
                continue
            score=d["point_scaled_distance"]+d["bootstrap_band_radius"]
            if not np.isfinite(score):
                continue
            overall["eligible_signed_pairs_with_finite_bands"]+=1
            item=dict(path=row["path"],dataset=record["dataset"],variant=variant["name"],control=control,source_seed=record["source_seed"],
                      assay_a=d["a"],assay_b=d["b"],source_a=d["ia"],source_b=d["ib"],sign=d["sign"],
                      point_scaled_distance=d["point_scaled_distance"],band_radius=d["bootstrap_band_radius"],
                      distance_plus_band=score,acceptance_boundary=1.)
            for key in ["overall","control:"+control,"dataset:"+record["dataset"]]:
                if key not in closest or score<closest[key]["distance_plus_band"]:
                    closest[key]=item
    return dict(status="descriptive_only",sealed_provenance_sha256=hashlib.sha256(seal.read_bytes()).hexdigest(),
                summary_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),overall=dict(overall),status_counts=dict(statuses),
                model_abstention_reasons=dict(reasons),bootstrap_estimator_failure_reasons=dict(bootstrap_reasons),warnings=dict(warnings_count),
                source_gate_failure_counts_nonexclusive=dict(gates),by_dataset={k:dict(v) for k,v in by_dataset.items()},
                by_control={k:dict(v) for k,v in by_control.items()},by_assay={k:dict(v) for k,v in by_assay.items()},
                original_unit300_by_dataset={k:dict(v) for k,v in focused.items()},cap_only_source_directions=cap_sources,
                positive_rank_control_assay_fits=positive_control,closest_eligible_signed_pairs=closest,
                interpretation="Counts are repeated fitted directions/assay fits across exploratory variants, not independent biological discoveries. Gate counts overlap. Cap-only means all other gates pass in the EXISTING truncated fit; it does not predict a higher-rank refit. Closest comparisons are descriptive and do not change thresholds. Unassessed CFs are not failures or confirmations.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    result=summarize(args.study.resolve())
    (args.study/"reviews/gate_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ["overall","status_counts","source_gate_failure_counts_nonexclusive","model_abstention_reasons","bootstrap_estimator_failure_reasons","warnings","closest_eligible_signed_pairs"]},indent=2))


if __name__=="__main__":
    main()

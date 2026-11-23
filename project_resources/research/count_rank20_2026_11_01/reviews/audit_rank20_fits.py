"""Bind the unchanged independent extension replay to the rank20 sibling."""
import importlib.util
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
HELPER = STUDY.parent/"count_extensions_2026_11_01/reviews/audit_extension_fits.py"
spec = importlib.util.spec_from_file_location("sealed_extension_fit_audit",HELPER)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def verify_rank_preservation():
    original=STUDY.parent/"count_extensions_2026_11_01"
    old_seal=original/"runs_v1/provenance.json"
    assert old_seal.is_file()
    old_provenance=json.loads(old_seal.read_text())
    new_protocol=json.loads((STUDY/"protocol.json").read_text())
    summary=json.loads((STUDY/"runs_v1/summary.json").read_text())
    assert new_protocol["rank"]["cap"]==20
    for name in ["apply_extensions.py","extension_estimators.py","evaluate_extensions.py"]:
        assert audit_module.sha(STUDY/name)==audit_module.sha(original/name)
    assay_records=0
    for arm in summary:
        relative=Path(arm["path"])
        before_path=original/"runs_v1"/relative/"result.json"
        after_path=STUDY/"runs_v1"/relative/"result.json"
        assert audit_module.sha(before_path)==old_provenance["artifacts"][str(relative/"result.json")]
        before,after=json.loads(before_path.read_text()),json.loads(after_path.read_text())
        for key in ["dataset","variant","source_seed","control","input_hashes","exposure_parameters"]:
            assert before[key]==after[key],(arm["path"],key)
        assert len(before["assays"])==len(after["assays"])==3
        for old,new in zip(before["assays"],after["assays"]):
            assert old["assay"]==new["assay"]
            assert old["bootstrap_seed"]==new["bootstrap_seed"]
            for key in ["status","uncapped","eigenvalues","pvalues","null_maxima"]:
                assert old["rank"][key]==new["rank"][key],(arm["path"],old["assay"],key)
            assert new["rank"]["k"]==min(new["rank"]["uncapped"],20)
            assert new["rank"]["cap_binding"]==(new["rank"]["uncapped"]>20)
            assay_records+=1
        old_arrays=original/"runs_v1"/relative/"arrays.npz"
        new_arrays=STUDY/"runs_v1"/relative/"arrays.npz"
        assert audit_module.sha(old_arrays)==old_provenance["artifacts"][str(relative/"arrays.npz")]
        with np.load(old_arrays,allow_pickle=False) as a,np.load(new_arrays,allow_pickle=False) as b:
            for assay in new_protocol["assays"]:
                for key in ["features","fit_indices","calibration_indices","mean","scale","working_scale","constant_mask","covariance","noise_diagonal","poisson_noise_diagonal"]:
                    name=key+"_"+assay
                    if a[name].dtype.kind in "fci":
                        equal=np.array_equal(a[name],b[name],equal_nan=True)
                    else:
                        equal=np.array_equal(a[name],b[name])
                    assert equal,(arm["path"],name)
    return dict(status="passed",arms=len(summary),assay_records=assay_records,
                original_provenance_sha256=audit_module.sha(old_seal),
                exact_equal_fields=["all49 null maxima","all eigenvalues","all rank p-values","uncapped rank",
                                    "training input hashes","features and split indices","exposure parameters","bootstrap seeds",
                                    "point mean/covariance/noise/observed scale/working scale/constant mask"],
                intentional_change="Requested source count uses cap20 instead of cap6; source orientation, stability and matching are allowed to change.")


def main():
    audit_module.STUDY = STUDY
    result = audit_module.audit(STUDY/"runs_v1")
    result.update(bound_study=str(STUDY),wrapper_sha256=audit_module.sha(Path(__file__)),
                  original_audit_source_sha256=audit_module.sha(HELPER),rank_and_input_preservation=verify_rank_preservation())
    (STUDY/"reviews/audit_rank20_fits.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ["status","arms","sealed_provenance_sha256"]}))


if __name__ == "__main__":
    main()

"""Independent replay of sealed extension fits; no refits or heldout reads."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


STUDY = Path(__file__).resolve().parents[1]
COMMON = STUDY.parent / "count_mdcrl_2026_11_01/reviews/audit_count_fits.py"
spec = importlib.util.spec_from_file_location("previous_independent_audit", COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
sha, check, same, close = common.sha, common.check, common.same, common.close


def unique(candidates):
    out = []
    for candidate in candidates:
        group = [d for d in candidates if (d["a"],d["b"]) == (candidate["a"],candidate["b"])]
        if sum(d["ia"] == candidate["ia"] for d in group) == 1 and sum(d["ib"] == candidate["ib"] for d in group) == 1:
            out.append(candidate)
    return out


def tensor_residual(tensor, rotation):
    transformed = np.einsum("ia,jb,kc,ijk->abc",rotation,rotation,rotation,tensor,optimize=True)
    off = transformed.copy()
    for i in range(len(rotation)):
        off[i,i,i] = 0
    norm = np.linalg.norm(tensor)
    return float(np.linalg.norm(off)/(norm if norm else 1.))


def assay_replay(ar, arrays, assay, variant, protocol):
    get = lambda key: arrays[key+"_"+assay]
    width,k = variant["width"],ar["k"]
    common.rank_check(ar["rank"],protocol)
    mean,scale,working = get("mean"),get("scale"),get("working_scale")
    covariance,noise,poisson = get("covariance"),get("noise_diagonal"),get("poisson_noise_diagonal")
    mixing,unmixing = get("mixing"),get("unmixing")
    features = get("features")
    check(features.shape == (width,) and len(set(features)) == width,assay+": retained feature identifiers")
    close(working,np.where(scale > 0,scale,1.),assay+": working preconditioner")
    constants = get("constant_mask")
    check(constants.dtype == bool and constants.shape == (width,),assay+": constant mask")
    check(np.all(scale[constants] == 0),assay+": constant observed SD")
    if "constant_feature_count" in ar:
        check(ar["constant_feature_count"] == int(constants.sum()),assay+": constant count")
    check(np.all(poisson >= 0),assay+": positive Poisson noise")
    close(np.diag(covariance)+noise,scale**2,assay+": observed variance",atol=1e-8)
    if variant["correction"]:
        invtheta = 0. if variant["theta"] is None else 1/variant["theta"]
        alpha = variant["retention"]/(1+invtheta)
        expected_noise = (1-alpha)*(np.diag(covariance)+noise+mean**2)+alpha*poisson
        close(noise,expected_noise,assay+": NB/ZI noise identity",atol=1e-8)
    else:
        close(noise,np.zeros(width),assay+": ordinary noise convention")
    if ar["rank"]["status"] == "ok":
        rank_covariance = covariance if variant["correction"] else covariance-np.diag(poisson)
        eigen = np.linalg.eigvalsh(rank_covariance/np.outer(working,working))[::-1]
        close(eigen,ar["rank"]["eigenvalues"],assay+": rank covariance eigenvalues",atol=1e-7)
    fit,cal = get("fit_indices"),get("calibration_indices")
    check(len(fit) == ar["fit_cells"] and len(cal) == ar["calibration_cells"],assay+": split sizes")
    check(not set(fit) & set(cal),assay+": split overlap")
    check(set(fit)|set(cal) == set(range(len(fit)+len(cal))),assay+": split coverage")
    if ar["status"] != "ok":
        check(ar["status"] == "abstained",assay+": source status")
        check(mixing.shape == (width,0) and unmixing.shape == (0,width),assay+": abstained directions")
        check(np.isnan(get("predicted_observed_covariance")).all(),assay+": abstained prediction")
        check("fingerprint_value_"+assay not in arrays,assay+": abstained fingerprints")
        return None
    close(unmixing @ mixing,np.eye(k),assay+": W B identity")
    close(unmixing @ covariance @ unmixing.T,np.eye(k),assay+": unit source variance")
    close(get("predicted_observed_covariance"),mixing @ mixing.T+np.diag(noise),assay+": covariance prediction")
    fp = {key:get("fingerprint_"+key) for key in ["value","bootstrap","eligible"]}
    check(fp["value"].shape == (k,2),assay+": moment shape")
    check(fp["bootstrap"].shape == (protocol["bootstrap"]["replicates"],k,2),assay+": bootstrap moment shape")
    if k == 0:
        check(ar["eligible_source_count"] == 0 and not fp["eligible"].any(),assay+": zero-source gates")
        return fp
    whitening,tensor = get("whitening"),get("white_third_tensor")
    rotation = whitening @ mixing
    close(rotation.T @ rotation,np.eye(k),assay+": orthogonal orientation")
    close(rotation.T @ whitening,unmixing,assay+": oriented demixing")
    raw_residual = tensor_residual(tensor,rotation)
    same(ar["third_offdiagonal_residual"],raw_residual,assay+": raw C3 residual")
    if variant["orientation"] == "pgf":
        matrices = get("generalized_covariance_matrices")
        centered = matrices-np.trace(matrices,axis1=1,axis2=2)[:,None,None]*np.eye(k)[None,:,:]/k
        rotated = np.einsum("ia,gij,jb->gab",rotation,centered,rotation,optimize=True)
        for i in range(k):
            rotated[:,i,i] = 0
        norm = np.linalg.norm(centered)
        residual = float(np.linalg.norm(rotated)/(norm if norm else 1.))
        effective_cells = get("pgf_effective_cells")
        same(ar["pgf_effective_cells"],effective_cells,assay+": saved PGF ESS")
        effective = bool(np.min(effective_cells) >= max(protocol["orientation"]["pgf_min_effective_cells"],protocol["orientation"]["pgf_min_effective_fraction"]*len(fit)))
        same(ar["orientation_fit_residual"],residual,assay+": PGF residual")
    else:
        residual,effective = raw_residual,True
        target = get("orientation_target_tensor")
        if variant["orientation"] == "c3_shrink":
            groups,sizes = get("shrink_group_tensors"),get("shrink_group_sizes")
            check(sizes.sum() == len(fit) and len(sizes) == min(5,len(fit)),assay+": shrink groups")
            close(np.einsum("g,gabc->abc",sizes,groups)/sizes.sum(),tensor,assay+": common-centered group tensor")
            variance = np.einsum("g,gabc->abc",sizes,(groups-tensor)**2)/((len(sizes)-1)*sizes.sum())
            close(get("shrink_variance_proxy"),variance,assay+": shrink variance proxy")
            denominator = tensor**2+variance
            expected_target = np.divide(tensor**3,denominator,out=np.zeros_like(tensor),where=denominator > 0)
            close(target,expected_target,assay+": shrink target")
        else:
            close(target,tensor,assay+": unshrunk target")
        same(ar["orientation_fit_residual"],tensor_residual(target,rotation),assay+": orientation target residual")
    same(ar["orientation_residual"],residual,assay+": eligibility residual")
    check(ar["effective_cells_gate"] == effective,assay+": point ESS gate")
    point_valid = np.isfinite(fp["value"]).all(axis=1) & (np.asarray(ar["calibration_variance"],float) > 0)
    same(ar["point_moment_valid"],point_valid,assay+": moment validity")
    valid_n = np.isfinite(fp["bootstrap"]).all(axis=2).sum(axis=0)
    cosine = get("fingerprint_cosine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        median,low = np.nanmedian(cosine,axis=0),np.nanquantile(cosine,protocol["bootstrap"]["lower_cosine_quantile"],axis=0)
    same(ar["valid_bootstrap_counts"],valid_n,assay+": bootstrap valid counts")
    same(ar["loading_cosine_median"],median,assay+": loading median")
    same(ar["loading_cosine_10th"],low,assay+": loading lower quantile")
    eligible = (point_valid & (valid_n >= protocol["bootstrap"]["min_valid"]) & (median >= protocol["bootstrap"]["median_cosine"])
                & (low >= protocol["bootstrap"]["lower_cosine"]) & (residual <= protocol["orientation"]["max_residual"])
                & effective & (not ar["rank"]["cap_binding"]))
    close(fp["eligible"],eligible,assay+": eligibility mask")
    same(ar["eligible_sources"],eligible,assay+": recorded eligibility")
    check(ar["eligible_source_count"] == int(eligible.sum()),assay+": eligible count")
    boot_effective = get("fingerprint_bootstrap_effective_gate")
    check(np.isnan(fp["bootstrap"][~boot_effective]).all(),assay+": failed ESS fingerprints masked")
    if variant["orientation"] == "pgf":
        minima = get("fingerprint_bootstrap_min_effective_cells")
        expected_effective = np.isfinite(minima) & (minima >= max(protocol["orientation"]["pgf_min_effective_cells"],protocol["orientation"]["pgf_min_effective_fraction"]*len(fit)))
        close(boot_effective,expected_effective,assay+": bootstrap numeric ESS gate")
    reference = mixing/working[:,None]
    for b,candidate in enumerate(get("fingerprint_bootstrap_mixing")):
        if not np.isfinite(candidate).all():
            continue
        close(get("fingerprint_bootstrap_unmixing")[b] @ candidate,np.eye(k),assay+": bootstrap W B")
        normalized = candidate/working[:,None]
        cos = reference.T @ normalized/np.outer(np.linalg.norm(reference,axis=0),np.linalg.norm(normalized,axis=0))
        close(np.diag(cos),cosine[b],assay+": aligned loading cosines")
        row,col = linear_sum_assignment(-np.abs(cos))
        close(np.array([np.abs(cos[row,col]).sum()]),np.array([np.trace(cos)]),assay+": optimal within-assay loading alignment")
    return fp


def cf_gate(values,se,max_log,numerical,proxy,variant,protocol):
    settings = protocol["matching"]
    gate = (np.isfinite(values).all(axis=-1) & numerical.all(axis=-1)
            & (se <= settings["cf_max_se"]).all(axis=-1)
            & (np.abs(values) <= settings["cf_max_abs"]).all(axis=-1)
            & (max_log <= settings["cf_max_logweight"]).all(axis=-1))
    if variant["correction"] and variant["theta"] is not None:
        gate &= (np.isfinite(proxy) & (proxy <= 1e-8)).all(axis=-1)
    return gate


def cf_replay(assay,fp,arrays,report,endpoints,variant,protocol):
    get = lambda key: arrays["cf_"+key+"_"+assay]
    k,ngrid = len(fp["value"]),len(protocol["matching"]["cf_grid"])
    assessed = np.zeros(k,bool)
    assessed[endpoints] = True
    same(report["endpoints"],endpoints,assay+": selected CF endpoints")
    check(report["status"] == ("assessed_moment_candidates" if endpoints else "not_assessed_no_moment_candidates"),assay+": CF assessment status")
    close(get("assessed"),assessed,assay+": CF assessed mask")
    value,bootstrap = get("value"),get("bootstrap")
    check(value.shape == (k,ngrid),assay+": CF shape")
    check(np.isnan(value[~assessed]).all() and np.isnan(bootstrap[:,~assessed]).all(),assay+": unassessed CF values")
    point_valid = np.zeros(k,bool)
    if endpoints:
        diagnostics = report["point_diagnostics"]
        values = value[endpoints]
        se = np.asarray(diagnostics["se"],float)
        numerical = np.asarray(diagnostics.get("numerical_valid",np.isfinite(values)),bool)
        max_log = np.asarray(diagnostics["max_logweight"],float)
        max_log = np.where(np.isnan(max_log),np.where(numerical,-np.inf,np.inf),max_log)
        proxy = np.asarray(diagnostics["mean_absolute_conditioning_proxy"],float)
        point_valid[endpoints] = cf_gate(values,se,max_log,numerical,proxy,variant,protocol)
        same(diagnostics["valid"],point_valid[endpoints],assay+": point CF diagnostic gate")
    else:
        check(report["point_diagnostics"] == {},assay+": unexpected unassessed CF diagnostics")
    close(get("point_valid"),point_valid,assay+": point CF mask")
    if endpoints:
        raw = get("bootstrap_raw_value")
        valid_bootstrap = cf_gate(raw,get("bootstrap_se"),get("bootstrap_max_logweight"),get("bootstrap_numerical_valid"),
                                 get("bootstrap_mean_absolute_conditioning_proxy"),variant,protocol)
        expected_bootstrap = np.full_like(raw,complex(np.nan,np.nan))
        expected_bootstrap[valid_bootstrap] = raw[valid_bootstrap]
        close(bootstrap,expected_bootstrap,assay+": bootstrap CF gate application")
    valid_n = np.isfinite(bootstrap).all(axis=2).sum(axis=0)
    same(report["valid_bootstrap_counts"],valid_n,assay+": recorded CF bootstrap counts")
    close(get("valid_bootstrap_counts"),valid_n,assay+": saved CF bootstrap counts")
    eligible = fp["eligible"] & point_valid & (valid_n >= protocol["bootstrap"]["min_valid"])
    same(report["eligible"],eligible,assay+": recorded CF eligibility")
    close(get("eligible"),eligible,assay+": CF eligibility mask")
    return dict(value=np.c_[fp["value"],value.real,value.imag],
                bootstrap=np.concatenate([fp["bootstrap"],bootstrap.real,bootstrap.imag],axis=2),eligible=eligible)


def audit(run_dir):
    seal_path,summary_path = run_dir/"provenance.json",run_dir/"summary.json"
    check(seal_path.is_file() and summary_path.is_file(),"Completed provenance and summary required before audit")
    seal,summary = json.loads(seal_path.read_text()),json.loads(summary_path.read_text())
    check(seal["evaluator_read"] is False,"Training seal reports evaluator access")
    for name,digest in seal["source_sha256"].items():
        check(sha(STUDY/name) == digest,"Training source hash: "+name)
    for name,digest in seal["support_sha256"].items():
        check(sha(STUDY.parent/name) == digest,"Supporting source hash: "+name)
    for name,digest in seal["artifacts"].items():
        check(sha(run_dir/name) == digest,"Sealed artifact hash: "+name)
    protocol = json.loads((STUDY/"protocol.json").read_text())
    data = STUDY/protocol["data"]
    check(sha(data/"manifest.json") == protocol["data_manifest_sha256"] == seal["data_manifest_sha256"],"Data manifest hash")
    check(len(summary) == protocol["expected_arm_records"],"Complete arm count")
    expected_names = {f"{dataset}__{variant['name']}__seed{seed}"+("__independent_columns" if control != "original" else "")
                      for dataset in protocol["datasets"] for variant in protocol["variants"]
                      for control,seeds in protocol["controls"].items() for seed in seeds}
    check({entry["path"] for entry in summary} == expected_names,"Complete prespecified variant/control/seed coverage")
    ngrid = len(protocol["matching"]["cf_grid"])
    margins = np.asarray(protocol["matching"]["moment_margins"])
    full_margins = np.r_[margins,np.repeat(protocol["matching"]["cf_margin"],2*ngrid)]
    results,exposure_parameters = [],{}
    for entry in summary:
        directory = run_dir/entry["path"]
        record = json.loads((directory/"result.json").read_text())
        check(record["status"] == entry["status"],"Summary status")
        variant = record["variant"]
        same(variant,next(v for v in protocol["variants"] if v["name"] == variant["name"]),"Frozen variant")
        if record["status"] == "failed":
            check(record.get("error") is not None,"Failed arm reason")
            if "partial_arrays_sha256" in record:
                check(sha(directory/"partial_arrays.npz") == record["partial_arrays_sha256"],"Partial array hash")
            results.append(dict(path=entry["path"],status="recorded_failure",error=record["error"]))
            continue
        check(sha(directory/"arrays.npz") == record["arrays_sha256"],"Model-array hash")
        required = {f"public/{record['dataset']}/train/{a}.npz" for a in protocol["assays"]}
        check(set(record["input_hashes"]) == required,"Only permitted training source paths")
        for name,digest in record["input_hashes"].items():
            check(sha(data/name) == digest,"Training input hash: "+name)
        with np.load(directory/"arrays.npz",allow_pickle=False) as archive:
            arrays = {name:archive[name] for name in archive.files}
        check(all(a.dtype != object for a in arrays.values()),"Unexpected object array")
        assay_records = {a["assay"]:a for a in record["assays"]}
        check(list(assay_records) == protocol["assays"],"Assay coverage/order")
        for assay,ar in assay_records.items():
            seed_parts = [protocol["bootstrap"]["seed"],record["dataset"],record["control"],assay,record["source_seed"]]
            expected_seed = int(hashlib.sha256(":".join(map(str,seed_parts)).encode()).hexdigest()[:8],16)
            check(ar["bootstrap_seed"] == expected_seed,"Local bootstrap seed")
            params = record["exposure_parameters"][assay]
            check(params["mode"] == variant["depth"],"Declared exposure treatment")
            check(np.isfinite(params["normalization"]) and params["normalization"] > 0,"Positive exposure normalizer")
            check(np.isfinite(params["beta"]) and 0 <= params["beta"] <= 2,"Exposure elasticity range")
            if variant["depth"] == "unit":
                check(params["beta"] == 0 and params["normalization"] == 1,"Unit exposure constants")
            elif variant["depth"] == "offset":
                check(params["beta"] == 1,"Offset exposure elasticity")
            exposure_key = record["dataset"],variant["depth"],assay
            if exposure_key in exposure_parameters:
                same(params,exposure_parameters[exposure_key],"Shared fixed exposure nuisance parameters")
            else:
                exposure_parameters[exposure_key] = params
        fps = {a:assay_replay(assay_records[a],arrays,a,variant,protocol) for a in protocol["assays"]}
        details = []
        for a,b in combinations(protocol["assays"],2):
            details.extend(common.pair_decisions(a,b,fps[a],fps[b],margins,ngrid)[1])
        candidates = [d for d in details if d["equivalent"]]
        matches = unique(candidates)
        full = {}
        for assay in protocol["assays"]:
            if fps[assay] is None:
                check(assay not in record["cf_assessment"],"Abstained CF assessment")
                continue
            endpoints = sorted({d["ia"] if d["a"] == assay else d["ib"] for d in candidates if assay in [d["a"],d["b"]]})
            full[assay] = cf_replay(assay,fps[assay],arrays,record["cf_assessment"][assay],endpoints,variant,protocol)
        all_confirmations = {}
        for a,b in combinations(protocol["assays"],2):
            if a not in full or b not in full:
                continue
            for d in common.pair_decisions(a,b,full[a],full[b],full_margins,ngrid)[1]:
                all_confirmations[a,b,d["ia"],d["ib"],d["sign"]] = d
        confirmation_details = [all_confirmations[d["a"],d["b"],d["ia"],d["ib"],d["sign"]] for d in candidates]
        confirmations = unique([d for d in confirmation_details if d["equivalent"]])
        for key,value in [("moment_diagnostics",details),("moment_equivalent_pairs",candidates),("moment_pair_matches",matches),
                          ("confirmation_diagnostics",confirmation_details),("confirmed_pair_matches",confirmations),
                          ("moment_all_assay_matches",common.clique_decisions(matches,protocol["assays"])),
                          ("confirmed_all_assay_matches",common.clique_decisions(confirmations,protocol["assays"]))]:
            same(record[key],value,entry["path"]+"/"+key)
        ambiguity = []
        for a,b in combinations(protocol["assays"],2):
            if fps[a] is None:
                continue
            for ia in range(len(fps[a]["value"])):
                selected = lambda d: (d["a"],d["b"],d["ia"]) == (a,b,ia)
                ambiguity.append(dict(a=a,b=b,ia=ia,
                    moment_equivalent=[dict(ib=d["ib"],sign=d["sign"]) for d in candidates if selected(d)],
                    cf_equivalent=[dict(ib=d["ib"],sign=d["sign"]) for d in confirmation_details if selected(d) and d["equivalent"]],
                    not_excluded_by_moment_band=[dict(ib=d["ib"],sign=d["sign"]) for d in details if selected(d) and d["not_excluded_by_band"]]))
        same(record["ambiguity_sets"],ambiguity,"Source ambiguity sets")
        for label,pairs in [("moment",matches),("confirmed",confirmations)]:
            for a,b in combinations(protocol["assays"],2):
                cross = np.zeros((variant["width"],variant["width"]))
                for d in pairs:
                    if (d["a"],d["b"]) == (a,b):
                        cross += d["sign"]*arrays["mixing_"+a][:,d["ia"],None]*arrays["mixing_"+b][None,:,d["ib"]]
                close(arrays[f"predicted_cross_covariance_{label}_{a}_{b}"],cross,label+" candidate covariance")
        same(entry["ranks"],[a["k"] for a in record["assays"]],"Summary ranks")
        same(entry["eligible_sources"],[a["eligible_source_count"] for a in record["assays"]],"Summary source eligibility")
        for key,expected in [("moment_equivalent_pairs",len(candidates)),("moment_pair_matches",len(matches)),("confirmed_pair_matches",len(confirmations)),
                             ("confirmed_all_assay_matches",len(record["confirmed_all_assay_matches"]))]:
            check(entry[key] == expected,"Summary "+key)
        check(entry["graph_status"] == record["graph_status"],"Summary graph status")
        if len(record["confirmed_all_assay_matches"]) < 2:
            check(record["graph_status"] == "fewer_than_two_confirmed_all_assay_candidates","Graph extraction gate")
            check("exploratory_native_A" not in arrays,"Unexpected graph coefficients")
        results.append(dict(path=entry["path"],status="passed",ranks=entry["ranks"],eligible_sources=entry["eligible_sources"],
                            moment_equivalent_pairs=len(candidates),moment_pair_matches=len(matches),confirmed_pair_matches=len(confirmations),
                            confirmed_all_assay_matches=len(record["confirmed_all_assay_matches"])))
    return dict(status="passed",arms=len(results),results=results,source_sha256=sha(Path(__file__)),common_audit_sha256=sha(COMMON),
                sealed_provenance_sha256=sha(seal_path),protocol_sha256=sha(STUDY/"protocol.json"),
                scope="Independent NumPy/SciPy replay of saved training-fit arrays and declared decisions; no runner/estimator imports, source fits, test or evaluator references.",
                limitations="Arithmetic/record validation does not establish biological kernel correctness, source-law equality, or causal identifiability.")


def self_test():
    """Logical saved-array fixtures, without raw observations or source fits."""
    common.self_test()
    protocol = json.loads((STUDY/"protocol.json").read_text())
    variant = next(v for v in protocol["variants"] if v["name"] == "shrink30_unit")
    p,k,repeats = 30,2,protocol["bootstrap"]["replicates"]
    mixing = np.eye(p)[:,:k]
    variance = np.r_[np.ones(k),np.zeros(p-k)]
    scale = np.sqrt(2*variance)
    tensor = np.zeros((k,k,k))
    tensor[0,0,0],tensor[1,1,1] = .8,1.4
    point = np.array([[.8,.5],[1.4,1.]])
    rank = dict(status="ok",k=k,uncapped=k,cap_binding=False,
                eigenvalues=(.5*variance).tolist(),null_maxima=[.25]*49,
                pvalues=[.02]*k+[1.]*(p-k))
    ar = dict(k=k,status="ok",rank=rank,constant_feature_count=p-k,fit_cells=75,calibration_cells=25,
              third_offdiagonal_residual=0.,orientation_fit_residual=0.,orientation_residual=0.,effective_cells_gate=True,
              calibration_variance=[1.,1.],point_moment_valid=[True,True],valid_bootstrap_counts=[repeats,repeats],
              loading_cosine_median=[1.,1.],loading_cosine_10th=[1.,1.],eligible_sources=[True,True],eligible_source_count=2)
    data = dict(features=np.array([f"fixture-{i}" for i in range(p)]),mean=3*variance,scale=scale,
                working_scale=np.where(scale > 0,scale,1.),constant_mask=variance == 0,
                covariance=np.diag(variance),noise_diagonal=variance,poisson_noise_diagonal=variance,
                mixing=mixing,unmixing=mixing.T,fit_indices=np.arange(75),calibration_indices=np.arange(75,100),
                predicted_observed_covariance=np.diag(2*variance),whitening=mixing.T,
                white_third_tensor=tensor,orientation_target_tensor=tensor,shrink_variance_proxy=np.zeros_like(tensor),
                shrink_group_sizes=np.repeat(15,5),shrink_group_tensors=np.repeat(tensor[None],5,axis=0),
                fingerprint_value=point,fingerprint_bootstrap=np.repeat(point[None],repeats,axis=0),fingerprint_eligible=np.ones(k,bool),
                fingerprint_cosine=np.ones((repeats,k)),fingerprint_bootstrap_effective_gate=np.ones(repeats,bool),
                fingerprint_bootstrap_mixing=np.repeat(mixing[None],repeats,axis=0),
                fingerprint_bootstrap_unmixing=np.repeat(mixing.T[None],repeats,axis=0))
    arrays = {key+"_RNA":value for key,value in data.items()}
    fp = assay_replay(ar,arrays,"RNA",variant,protocol)
    check(fp["eligible"].all(),"Logical source-gate fixture")
    values = np.full((k,3),complex(np.nan,np.nan))
    values[0] = [.99+.01j,.98+.02j,.9+.03j]
    raw = np.repeat(values[None],repeats,axis=0)
    raw[0,0,0] = 2.
    numerical = np.zeros(raw.shape,bool)
    numerical[:,0] = True
    se,logweight,proxy = np.full(raw.shape,np.nan),np.full(raw.shape,np.nan),np.full(raw.shape,np.nan)
    se[:,0],logweight[:,0],proxy[:,0] = .01,0.,0.
    filtered = raw.copy()
    filtered[0,0] = complex(np.nan,np.nan)
    cf_data = dict(assessed=np.array([True,False]),value=values,bootstrap=filtered,bootstrap_raw_value=raw,
                   bootstrap_se=se,bootstrap_max_logweight=logweight,bootstrap_numerical_valid=numerical,
                   bootstrap_mean_absolute_conditioning_proxy=proxy,point_valid=np.array([True,False]),
                   valid_bootstrap_counts=np.array([repeats-1,0]),eligible=np.array([True,False]))
    arrays.update({"cf_"+key+"_RNA":value for key,value in cf_data.items()})
    report = dict(status="assessed_moment_candidates",endpoints=[0],valid_bootstrap_counts=[repeats-1,0],eligible=[True,False],
                  point_diagnostics=dict(se=[[.01]*3],max_logweight=[[0.]*3],numerical_valid=[[True]*3],
                                         mean_absolute_conditioning_proxy=[[0.]*3],valid=[True]))
    full = cf_replay("RNA",fp,arrays,report,[0],variant,protocol)
    check(full["eligible"].tolist() == [True,False],"Logical CF eligibility fixture")
    check(np.isnan(full["bootstrap"][0,0,2:]).all(),"Rejected raw CF is excluded from matching")
    check(unique([dict(a="A",b="B",ia=0,ib=0,sign=1),dict(a="A",b="B",ia=0,ib=0,sign=-1)]) == [],"Ambiguous sign self-check")
    print("PASS: source/shrink/CF gate replay on logical saved-array fixtures; no run artifacts accessed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=STUDY/"runs_v1")
    parser.add_argument("--output",type=Path,default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = audit(args.runs.resolve())
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ["status","arms","sealed_provenance_sha256"]}))


if __name__ == "__main__":
    main()

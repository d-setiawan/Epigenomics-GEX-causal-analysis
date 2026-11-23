"""Independent literal row-resample comparison; no project data or model fits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY))
from evaluate_extensions import bootstrap_cross_covariance_errors


def main():
    n=7
    weights=np.array([[1,1,1,1,1,1,1],[7,0,0,0,0,0,0],
                      [0,1,0,2,0,3,1],[3,2,1,1,0,0,0]],dtype=float)
    checked=0
    largest=0.
    largest_nondegenerate=0.
    for p,q in [(3,5),(30,30),(300,300)]:
        row=np.arange(n)[:,None]
        left=((row+2)*(np.arange(p)[None,:]+3)%13).astype(float)
        right=((row+5)*(np.arange(q)[None,:]+1)%17).astype(float)
        ls=.7+(np.arange(p)+1)/(p+1)
        rs=1.1+(np.arange(q)+1)/(q+1)
        nonzero=np.sin(np.arange(p)[:,None]+np.arange(q)[None,:]+.3)/7
        for prediction in [np.zeros((p,q)),nonzero]:
            for offset in [0.,1e6]:
                got=bootstrap_cross_covariance_errors(left+offset,right-offset,prediction,ls,rs,weights)
                literal=[]
                zero=[]
                for multiplicity in weights:
                    rows=np.repeat(np.arange(n),multiplicity.astype(int))
                    l,r=(left+offset)[rows],(right-offset)[rows]
                    l,r=l-l.mean(0),r-r.mean(0)
                    observed=l.T@r/n/np.outer(ls,rs)
                    expected=prediction/np.outer(ls,rs)
                    literal.append(np.sqrt(np.mean((observed-expected)**2)))
                    zero.append(np.sqrt(np.mean(observed**2)))
                for actual,expected in [(got["model_rmse"],literal),(got["zero_rmse"],zero),
                                        (got["rmse_improvement_over_zero"],np.asarray(zero)-literal)]:
                    np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=2e-7)
                    largest=max(largest,float(np.max(np.abs(actual-expected))))
                    largest_nondegenerate=max(largest_nondegenerate,float(np.max(np.abs(
                        np.asarray(actual)[[0,2,3]]-np.asarray(expected)[[0,2,3]]))))
                cached=bootstrap_cross_covariance_errors(left+offset,right-offset,prediction,ls,rs,weights,
                                                         got["observed_squared_frobenius_cache"])
                np.testing.assert_array_equal(cached["model_rmse"],got["model_rmse"])
                checked+=len(weights)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    result={"status":"passed","literal_row_resamples_checked":checked,
            "width_pairs":[[3,5],[30,30],[300,300]],"nonzero_and_zero_predictions":True,
            "degenerate_single_cell_resample":True,"large_constant_shift_check":True,
            "cache_replay":True,"maximum_absolute_rmse_difference":largest,
            "maximum_nondegenerate_rmse_difference":largest_nondegenerate,
            "project_data_read":False,"model_fits":0,
            "source_sha256":{p.name:sha(p) for p in [Path(__file__),STUDY/"evaluate_extensions.py"]}}
    Path(__file__).with_name("gram_algebra_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

"""Fixed algebra fixtures verify relabeling, gradients, and information limits."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.special import logit, logsumexp
from scipy.stats import poisson

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import count_transport_label_optimizer as model
import count_transport_model as legacy


def fixture():
    return {'a':np.array([[0,1,0],[2,0,1],[1,2,0],[0,0,3]],float),
            'b':np.array([[1,0,2],[0,2,1],[1,1,0],[3,0,2]],float)}


@pytest.mark.parametrize('seed',[0,1,2,3])
def test_physical_initialization_matches_legacy_exactly(seed):
    data,theta,_,_=model._setup(fixture(),3,seed)
    _,old,_,_=legacy._setup(fixture(),3,'strict',seed)
    widths={a:x.shape[1] for a,x in data.items()}
    newdecoded=model._decode(theta,widths,3)
    olddecoded=legacy._decode(old,widths,3,'strict')
    np.testing.assert_allclose(newdecoded[0],olddecoded[0],atol=1e-15)
    np.testing.assert_allclose(newdecoded[3],olddecoded[3],atol=1e-15)
    np.testing.assert_allclose(np.exp(newdecoded[4]),olddecoded[4],atol=1e-15)
    for a in data:
        np.testing.assert_allclose(newdecoded[-1][a],olddecoded[-1][a],atol=1e-14)
    np.testing.assert_allclose(model._objective(theta,data,3)[0],legacy._objective(old,data,3,'strict')[0],atol=1e-14)


@pytest.mark.parametrize('probabilities',[[.22,.73],[.3,.5,.8]])
def test_exact_gradient_including_probabilities_above_half(probabilities):
    d=len(probabilities)
    data,theta,_,_=model._setup(fixture(),d,2)
    theta[:d]=logit(probabilities)
    theta[d:2*d]=np.linspace(-.4,.6,d)
    _,analytic=model._objective(theta,data,d)
    numeric=np.empty_like(theta)
    for j in range(len(theta)):
        plus,minus=theta.copy(),theta.copy()
        plus[j]+=1e-5;minus[j]-=1e-5
        numeric[j]=(model._objective(plus,data,d)[0]-model._objective(minus,data,d)[0])/2e-5
    np.testing.assert_allclose(analytic,numeric,rtol=3e-5,atol=2e-7)


def test_global_canonicalization_preserves_full_joint_law_and_gradient():
    data,theta,bounds,_=model._setup(fixture(),3,3)
    theta[:3]=logit([.2,.7,.6]);theta[3:6]=[.3,-.5,.2]
    widths={a:x.shape[1] for a,x in data.items()}
    canonical,flips=model.canonicalize(theta,widths,3)
    assert flips==[1,2]
    before,after=model._decode(theta,widths,3),model._decode(canonical,widths,3)
    np.testing.assert_allclose(after[0],[.2,.3,.4],atol=1e-15)
    lookup={tuple(row):i for i,row in enumerate(before[1])}
    corresponding=after[1].copy();corresponding[:,flips]=1-corresponding[:,flips]
    permutation=np.array([lookup[tuple(row)] for row in corresponding])
    np.testing.assert_allclose(after[4],before[4][permutation],atol=1e-14)
    joint_before=before[4].copy()[None,:]+np.zeros((4,1))
    joint_after=after[4].copy()[None,:]+np.zeros((4,1))
    for a in data:
        np.testing.assert_allclose(after[-1][a],before[-1][a][permutation],atol=1e-14)
        cb=poisson.logpmf(data[a][:,None,:],np.exp(before[-1][a])[None,:,:]).sum(2)
        ca=poisson.logpmf(data[a][:,None,:],np.exp(after[-1][a])[None,:,:]).sum(2)
        np.testing.assert_allclose(logsumexp(cb+before[4],axis=1),logsumexp(ca+after[4],axis=1),atol=1e-13)
        joint_before+=cb;joint_after+=ca
    np.testing.assert_allclose(logsumexp(joint_before,axis=1),logsumexp(joint_after,axis=1),atol=1e-13)
    lb,gb=model._objective(theta,data,3);la,ga=model._objective(canonical,data,3)
    np.testing.assert_allclose(la,lb,atol=1e-14)
    signs=np.ones(len(theta));signs[:3]=[1,-1,-1];signs[3:6]=[1,-1,-1]
    signs[6:]=np.tile([1,-1,-1],sum(widths.values()))
    np.testing.assert_allclose(ga,gb*signs,atol=1e-13)


def test_exactly_symmetric_probability_is_reported_without_clipping():
    _,theta,_,_=model._setup(fixture(),1,0);theta[0]=0
    with pytest.raises(ValueError,match='0.5'):
        model.canonicalize(theta,{'a':3,'b':3},1)


@pytest.mark.parametrize('probability',[.8,.99,.5])
def test_returned_model_has_explicit_new_schema_and_global_flip(monkeypatch,probability):
    def fake_driver(objective,initial,bounds,chunk_iterations,chunk_evaluations,gtol):
        x=initial.copy();x[0]=logit(probability)
        loss,gradient=objective(x)
        norm=float(np.max(np.abs(x-np.clip(x-gradient,np.array(bounds)[:,0],np.array(bounds)[:,1]))))
        result=OptimizeResult(x=x,fun=loss,jac=gradient,status=0,success=True,nit=7,nfev=11,
            message='fixed fixture',strongly_converged=False,projected_gradient_inf_norm=norm)
        continuation={'iterations':7,'objective_evaluations':11,'stop_reason':'floating_point_stagnation'}
        return result,result,continuation
    monkeypatch.setattr(model.convergence,'_drive',fake_driver)
    result=model.fit_label_model(fixture(),1,0)
    json.dumps(result,allow_nan=False)
    if probability==.5:
        assert result['model'] is None
        assert '0.5' in result['error']
        assert result['raw_optimizer']['probabilities']==[.5]
        assert result['raw_optimizer']['flipped_coordinates'] is None
        return
    assert result['error'] is None
    assert result['raw_optimizer']['flipped_coordinates']==[0]
    assert result['model']['parameter_schema']==model.PARAMETER_SCHEMA
    np.testing.assert_allclose(result['model']['probabilities'],[1-probability],atol=1e-15)
    np.testing.assert_allclose(result['raw_optimizer']['probabilities'],[probability],atol=1e-15)
    assert result['model']['canonical_outside_legacy_probability_range']==(probability==.99)
    assert not result['model']['converged']

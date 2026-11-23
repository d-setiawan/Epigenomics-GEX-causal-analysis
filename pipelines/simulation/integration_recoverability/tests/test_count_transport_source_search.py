"""Fixed algebra and optimizer-control fixtures, not new scientific worlds."""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.special import logit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import count_transport_source_search as search
import count_transport_model as legacy


def fixture():
    return {'a':np.array([[0,1,2],[2,0,1],[1,2,0],[0,0,3]],float),
            'b':np.array([[1,0,2],[0,2,1],[1,1,0],[3,0,2]],float)}


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('seed',[0,1])
def test_plain_start_is_physically_identical_to_legacy(dimension,seed):
    data=fixture();widths={a:x.shape[1] for a,x in data.items()}
    new=search.physical_legacy_start(data,dimension,seed)
    _,old,_,_=legacy._setup(data,dimension,'strict',seed)
    original=legacy._decode(old,widths,dimension,'strict')
    actual=search._decode(new,widths,dimension)
    np.testing.assert_allclose(actual[0],original[0],atol=1e-15)
    np.testing.assert_allclose(np.exp(actual[3]),original[4],atol=1e-15)
    for a in data:
        np.testing.assert_allclose(actual[-1][a],original[-1][a],atol=1e-14)
    np.testing.assert_allclose(search._objective(new,data,dimension)[0],legacy._objective(old,data,dimension,'strict')[0],atol=1e-14)


def test_exact_new_coordinate_gradient_and_no_old_probability_box():
    data=fixture();theta=search.physical_legacy_start(data,2,0)
    theta[:2]=logit([.001,.73]);theta[2:4]=[.3,-.4]
    loss,analytic=search._objective(theta,data,2)
    assert np.isfinite(loss)
    numeric=np.empty_like(theta)
    for j in range(len(theta)):
        plus,minus=theta.copy(),theta.copy();plus[j]+=1e-5;minus[j]-=1e-5
        numeric[j]=(search._objective(plus,data,2)[0]-search._objective(minus,data,2)[0])/2e-5
    np.testing.assert_allclose(analytic,numeric,atol=2e-7,rtol=3e-5)


def test_cube_orbit_quotient_covers_every_vertex_assignment():
    for d,expected in ((1,1),(2,3),(3,840)):
        representatives=search.cube_orbits(d)
        assert len(representatives)==expected
        bits=search.cube_bits(d).astype(int);powers=2**np.arange(d-1,-1,-1)
        transformations=[]
        for order in itertools.permutations(range(d)):
            for flips in itertools.product((0,1),repeat=d):
                transformations.append(((bits[:,order]^np.array(flips))@powers).astype(int))
        all_labels={tuple(label[transformation]) for label in representatives for transformation in transformations}
        assert len(all_labels)==search.math.factorial(2**d)


@pytest.mark.parametrize('d',[1,2,3])
def test_cube_geometry_recovers_known_exact_atoms(d):
    bits=search.cube_bits(d);p=np.array([.12,.27,.4])[:d]
    A=np.array([[.4,.1,-.2],[-.2,.35,.15],[.3,-.25,.4]])[:,:d]
    h=np.array([.2,-.1,.15])[:d]
    weights=np.prod(np.where(bits,p,1-p),axis=1);rates=np.exp((bits+h)@A.T)
    order=np.arange(2**d)[::-1]
    cube=search.cube_geometry(weights[order],rates[order],d)
    np.testing.assert_allclose(cube['probabilities'],p,atol=1e-13)
    np.testing.assert_allclose(cube['A'],A,atol=1e-13)
    np.testing.assert_allclose(cube['affine_intercept'],A@h,atol=1e-13)
    assert abs(cube['score'])<1e-13
    joint=search.joint_initializer({'a':cube,'b':cube},['a','b'],{'a':3,'b':3},d)
    np.testing.assert_allclose(joint[d:2*d],h,atol=1e-13)


def test_h_translation_and_global_sort_preserve_joint_atoms_and_standardized_model():
    data=fixture();widths={a:x.shape[1] for a,x in data.items()};d=2
    theta=search.physical_legacy_start(data,d,0);theta[:d]=logit([.7,.15]);theta[d:2*d]=[.2,-.4]
    canonical,labels=search.canonicalize(theta,widths,d)
    before,after=search._decode(theta,widths,d),search._decode(canonical,widths,d)
    assert labels=={'flipped_coordinates':[0],'axis_order':[1,0]}
    np.testing.assert_allclose(after[1],[-.4,-1.2],atol=1e-14)
    original_bits=np.empty_like(after[2]);order=labels['axis_order']
    flip=np.array([True,False])[order]
    original_bits[:,order]=np.where(flip,1-after[2],after[2])
    lookup={tuple(b):i for i,b in enumerate(before[2])}
    permutation=np.array([lookup[tuple(b)] for b in original_bits])
    np.testing.assert_allclose(after[3],before[3][permutation],atol=1e-14)
    for a in data:
        np.testing.assert_allclose(after[-1][a],before[-1][a][permutation],atol=1e-13)
    loss,gradient=search._objective(theta,data,d)
    result=OptimizeResult(fun=loss,jac=gradient,strongly_converged=False,status=0,message='fixture',success=True)
    trace={'stop_reason':'fixture','iterations':0,'objective_evaluations':1}
    model=search._pack(canonical,result,data,d,0,loss,1e-6,trace)
    json.dumps(model,allow_nan=False)
    for a in data:
        np.testing.assert_allclose(np.array(model['state_sources'])@np.array(model['loadings'][a]).T,model['state_log_rates'][a],atol=1e-13)


def test_em_single_component_reaches_exact_raw_count_mle():
    x=np.array([[0,1],[1,2],[3,1],[4,0]],float)
    result=search.poisson_mixture(x,1,3)
    assert result['converged'] and result['error'] is None
    np.testing.assert_allclose(result['rates'],x.mean(0,keepdims=True),atol=1e-13)
    assert result['gradient_inf_norm']<=1e-6
    assert result['total_iteration_limit'] is None


def test_zero_rate_em_failure_is_explicit_not_clipped():
    result=search.poisson_mixture(np.zeros((4,2)),2,3)
    assert not result['converged']
    assert 'zero' in result['error'].lower()


def test_unbounded_optimizer_can_pass_old_boxes_and_resume_budgets():
    def objective(x):
        delta=x-np.array([42.,-31.]);scale=np.array([1.,7.])
        return float(np.sum(scale*delta**2)/2),scale*delta
    result,run=search._optimize(objective,np.zeros(2),1,100,1e-7)
    assert result.strongly_converged
    assert run['resumed_chunks']>1 and run['parameter_bounds'] is None
    np.testing.assert_allclose(result.x,[42.,-31.],atol=1e-7)


def test_nonfinite_trial_is_rejected_for_backtracking(monkeypatch):
    def objective(x):
        if x[0]>100:
            raise FloatingPointError('overflow fixture')
        return float((x[0]-1)**2),np.array([2*(x[0]-1)])
    def minimizing(fun,x0,**kwargs):
        value,gradient=fun(np.array([1000.]))
        assert np.isinf(value) and np.isfinite(gradient).all()
        value,gradient=fun(np.array([1.]))
        kwargs['callback'](np.array([1.]))
        return OptimizeResult(x=np.array([1.]),fun=value,jac=gradient,nit=1,nfev=2,status=0,success=True,message='gradient')
    monkeypatch.setattr(search,'minimize',minimizing)
    result,run=search._optimize(objective,np.zeros(1),600,15000,1e-6)
    assert result.strongly_converged and run['rejected_nonfinite_trials']==1


def test_rejected_point_cannot_be_certified_with_placeholder_zero_gradient(monkeypatch):
    def objective(x):
        if x[0]>100:
            raise FloatingPointError('rejected point')
        return float((x[0]-1)**2),np.array([2*(x[0]-1)])
    def invalid_result(*args,**kwargs):
        return OptimizeResult(x=np.array([1000.]),fun=0.,jac=np.zeros(1),nit=1,nfev=2,
                              status=0,success=True,message='spurious success fixture')
    monkeypatch.setattr(search,'minimize',invalid_result)
    result,run=search._optimize(objective,np.zeros(1),600,15000,1e-6)
    assert not result.strongly_converged
    assert run['stop_reason']=='numerical_failure'
    np.testing.assert_array_equal(result.x,[0.])


def test_every_failed_mixture_attempt_is_retained(monkeypatch):
    def failed(*args,**kwargs):
        return {'converged':False,'error':'fixture failure'}
    monkeypatch.setattr(search,'poisson_mixture',failed)
    result=search.fit_source_search(fixture(),1,0)
    assert result['model'] is None and result['error']
    assert result['candidates']==[]
    assert all(len(records)==4 and all(r['error'] for r in records) for records in result['search_diagnostics']['mixtures'].values())

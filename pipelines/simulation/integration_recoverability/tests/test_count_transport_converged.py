"""Optimizer-control tests use fixed functions/arrays, not recovery datasets."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import count_transport_converged as converged
import count_transport_model as legacy


def quadratic(theta):
    scale = np.array([1., 13.])
    return float(np.sum(scale*theta**2)/2), scale*theta


def test_iteration_budget_is_resumed_until_gradient_converges():
    capped, final, run = converged._drive(quadratic, [3., -2.], [(-5.,5.)]*2, 1, 100, 1e-8)
    assert not capped.strongly_converged
    assert final.strongly_converged
    assert run['resumed_chunks'] > 2
    assert run['budget_limit_resumptions'] == run['resumed_chunks']
    assert run['iterations'] > 1
    assert run['total_iteration_limit'] is None and run['total_evaluation_limit'] is None
    np.testing.assert_allclose(final.x, 0, atol=1e-8)
    assert all(b['objective'] <= a['objective']+1e-14 for a,b in zip(run['chunks'],run['chunks'][1:]))


def test_evaluation_budget_is_resumed_until_gradient_converges():
    capped, final, run = converged._drive(quadratic, [3., -2.], [(-5.,5.)]*2, 100, 1, 1e-8)
    assert capped.status == 1
    assert final.strongly_converged
    assert run['resumed_chunks'] > 2
    assert run['objective_evaluations'] > 1


def test_box_stationarity_is_distinct_from_small_raw_gradient():
    x = np.array([-1.,1.,0.])
    grad = np.array([10.,-5.,.2])
    np.testing.assert_allclose(converged.projected_gradient(x,grad,[(-1.,1.)]*3),[0.,0.,.2])


def test_function_tolerance_alone_is_never_accepted(monkeypatch):
    calls = []
    def stalled(objective,initial,bounds,iterations,evaluations,gtol,ftol):
        calls.append(ftol)
        x = np.asarray(initial,float)
        value,grad = objective(x)
        return OptimizeResult(x=x,fun=value,jac=grad,nit=1,nfev=1,status=0,success=True,
            message='CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH',
            projected_gradient_inf_norm=float(np.max(np.abs(grad))),strongly_converged=False)
    monkeypatch.setattr(converged,'_chunk',stalled)
    cap,final,run = converged._drive(quadratic,[1.,1.],[(-5.,5.)]*2,2,10,1e-6)
    assert calls == [1e-9,0.]
    assert not cap.strongly_converged and not final.strongly_converged
    assert run['stop_reason']=='floating_point_stagnation'


def test_line_search_failure_does_not_loop(monkeypatch):
    def failed(*args):
        return OptimizeResult(x=np.array([1.,1.]),fun=7.,jac=np.array([1.,13.]),nit=0,nfev=4,
            status=2,success=False,message='ABNORMAL_TERMINATION_IN_LNSRCH',
            projected_gradient_inf_norm=6.,strongly_converged=False)
    monkeypatch.setattr(converged,'_chunk',failed)
    _,final,run=converged._drive(quadratic,[1.,1.],[(-5.,5.)]*2,2,10,1e-6)
    assert not final.strongly_converged and run['resumed_chunks']==0
    assert run['stop_reason']=='line_search_failure'


def test_solver_failure_is_not_accepted_despite_small_reported_gradient(monkeypatch):
    def failed(*args, **kwargs):
        return OptimizeResult(x=np.zeros(2),fun=0.,jac=np.zeros(2),nit=0,nfev=1,
            status=2,success=False,message='ABNORMAL_TERMINATION_IN_LNSRCH')
    monkeypatch.setattr(converged,'minimize',failed)
    result=converged._chunk(quadratic,[1.,1.],[(-5.,5.)]*2,2,10,1e-6,0.)
    assert not result.strongly_converged


def test_fixed_count_fit_matches_legacy_checkpoint_and_serializes():
    data={'a':np.array([[0,1],[2,0],[1,2],[0,3]]),
          'b':np.array([[1,0],[0,2],[1,1],[3,0],[0,1]])}
    old=legacy.fit_model(data,1,'strict',2,maxiter=20)
    pair=converged.fit_convergence_pair(data,1,2,chunk_iterations=20)
    json.dumps(pair,allow_nan=False)
    np.testing.assert_allclose(pair['capped']['parameter_vector'],old['parameter_vector'],atol=1e-14)
    np.testing.assert_allclose(pair['capped']['objective'],old['objective'],atol=1e-14)
    assert pair['uncapped']['objective'] <= pair['capped']['objective']+1e-12
    assert pair['uncapped']['converged'] or pair['continuation']['stop_reason'].endswith(('failure','stagnation'))
    for fit in (pair['capped'],pair['uncapped']):
        if fit['converged']:
            assert fit['projected_gradient_inf_norm']<=fit['convergence_tolerance']


@pytest.mark.parametrize('kwargs',[{'chunk_iterations':0},{'chunk_evaluations':0},{'gtol':0},{'gtol':np.nan}])
def test_invalid_numerical_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        converged.fit_convergence_pair({'a':np.array([[1],[2]])},1,0,**kwargs)

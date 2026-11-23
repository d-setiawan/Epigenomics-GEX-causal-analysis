import sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson, nbinom
sys.path.insert(0,sys.argv[1])
import model, evaluate
x=np.array([[0.,1.],[2.,4.],[5.,0.],[3.,2.]])
rates=np.array([[1.,3.],[4.,.6]])
w=np.array([.3,.7])
for kernel in ['poisson','nb2']:
    got=model.log_components(x,rates,kernel)
    expected=np.column_stack([sum((poisson.logpmf(x[:,j],r[j]) if kernel=='poisson' else nbinom.logpmf(x[:,j],2,2/(2+r[j]))) for j in range(2)) for r in rates])
    np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-12)
    for j in range(2):
        q=np.array([.1,.5,.8,.3]); target=q@x[:,j]/q.sum()
        fun=lambda v: -np.dot(q,poisson.logpmf(x[:,j],v) if kernel=='poisson' else nbinom.logpmf(x[:,j],2,2/(2+v)))
        opt=minimize_scalar(fun,bounds=(1e-6,20),method='bounded')
        np.testing.assert_allclose(opt.x,target,rtol=2e-5)
    mu,cov,z=evaluate.moments(w,rates,kernel)
    np.testing.assert_allclose(mu,w@rates)
    np.testing.assert_allclose(z,w@(np.exp(-rates) if kernel=='poisson' else (2/(2+rates))**2))
    var=w@(rates if kernel=='poisson' else rates+rates**2/2)+(w@(rates**2))-(w@rates)**2
    np.testing.assert_allclose(np.diag(cov),var)
print('PASS independent SciPy likelihood, weighted M-step, mixture mean/variance/zero checks; both kernels')

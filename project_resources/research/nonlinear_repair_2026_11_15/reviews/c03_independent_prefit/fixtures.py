import sys,numpy as np
from scipy.optimize import minimize
sys.path.insert(0,sys.argv[1]);from adapter import simplex_qp
from common import score_rows,score_summary
rng=np.random.default_rng(7854)
for _ in range(40):
 X=rng.normal(size=(4,4));W=X.T@X+.3*np.eye(4);b=rng.normal(size=4)*3;lb=.025
 sol=simplex_qp(b,W,lb);f=lambda x:(x-b)@W@(x-b);g=lambda x:2*W@(x-b)
 fit=minimize(f,np.ones(4)/4,jac=g,method='SLSQP',bounds=[(lb,None)]*4,constraints={'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(4)},options={'ftol':1e-11,'maxiter':500})
 assert fit.success;np.testing.assert_allclose(f(sol),f(fit.x),rtol=1e-7,atol=1e-7)
assert score_summary(score_rows(np.array([[1.,0],[0,0]]),np.ones(4)/4,np.zeros((4,2)))[0])=={'rows':2,'log_score':None,'log_score_status':'negative_infinity','zero_probability_rows':1}
print('PASS 40 deterministic SPD QPs vs SLSQP and exact zero support mixture')

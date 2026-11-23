import itertools,json,sys
import numpy as np
from scipy.stats import poisson
sys.path.insert(0,sys.argv[1]);from core import exact_logprob,moments,q_certificate
from evaluate import metrics
x=np.array([[1.,0.],[0.,0.]]); B=np.zeros((2,3)); p=np.array([1.,0,0,0]); lp=exact_logprob(x,B,p);assert np.isneginf(lp[0]) and lp[1]==0
mean,cov,zero=moments(B,p); record=metrics(x,mean,cov,zero,float(lp.mean()),[1.,1.],2)
try:json.dumps(record,allow_nan=False)
except ValueError:print('CONFIRMED zero-rate test -inf breaks strict JSON')
else:raise AssertionError('expected JSON serialization failure')
def brute(q):
 n=len(q)
 if not(np.any((q[:,0]==1)&(q[:,1]==0)) and np.any((q[:,1]==1)&(q[:,0]==0))):return False
 for a,b,c,d in itertools.permutations(range(n),4):
  if q[a,0] and q[b,1] and q[c,0] and q[d,1]:
   rest=[i for i in range(n) if i not in (a,b,c,d)]
   if np.all(q[rest].sum(axis=0)>0):return True
 return False
for rows in itertools.product(((0,0),(1,0),(0,1),(1,1)),repeat=6):
 q=np.asarray(rows);assert q_certificate(q)['pass']==brute(q)
print('PASS all 4096 six-row Q matrices against separate enumeration')
B=np.array([[.2,1.1,.4],[.3,.7,1.6],[1.8,.1,.5]]);p=np.array([.08,.22,.41,.29]);x=np.array([[0,0,0],[1,2,3],[2,0,4],[5,2,0]])
z=np.array([[0,0],[0,1],[1,0],[1,1]]);rate=B[:,0]+z@B[:,1:].T
ll=np.array([np.log(sum(p[k]*np.prod(poisson.pmf(row,rate[k])) for k in range(4))) for row in x]);np.testing.assert_allclose(exact_logprob(x,B,p),ll,atol=1e-12)
mu,cov,zero=moments(B,p);np.testing.assert_allclose(mu,p@rate);np.testing.assert_allclose(zero,p@np.exp(-rate));np.testing.assert_allclose(np.diag(cov),mu+p@((rate-mu)**2));print('PASS independent Poisson likelihood and moment fixture')

"""Deterministic fixtures; no sampled or heldout scientific data."""
import itertools,unittest,sys,json
from pathlib import Path
import numpy as np
from scipy.stats import poisson
from core import *

def brute(q):
    j=len(q)
    return bool(any(q[a,0] and q[b,1] and q[c,0] and q[d,1] and all(q[[i for i in range(j) if i not in (a,b,c,d)]].sum(0)>0) and np.any((q[:,0]==1)&(q[:,1]==0)) and np.any((q[:,1]==1)&(q[:,0]==0)) for a,b,c,d in itertools.permutations(range(j),4)))
class Checks(unittest.TestCase):
    def test_certificate_exhaustive(self):
        for rows in itertools.product(((0,0),(1,0),(0,1),(1,1)),repeat=5):
            q=np.array(rows);self.assertEqual(q_certificate(q)['pass'],brute(q))
        self.assertTrue(q_certificate(np.array([[1,1],[1,1],[1,0],[0,1],[1,1]]))['pass'])
        self.assertFalse(q_certificate(np.ones((6,2)))['pass'])
    def test_likelihood_zero_and_moments(self):
        b=np.array([[0,1,2],[2,1,0]],float);p=np.array([.2,.3,.1,.4]);x=np.array([[0,1],[2,3]])
        z=np.array([[0,0],[0,1],[1,0],[1,1]]);r=b[:,0]+z@b[:,1:].T
        truth=np.log(sum(p[k]*np.prod(poisson.pmf(x,r[k]),axis=1) for k in range(4)))
        np.testing.assert_allclose(exact_logprob(x,b,p),truth)
        mean,cov,_=moments(b,p);np.testing.assert_allclose(mean,p@r);np.testing.assert_allclose(np.diag(cov),mean+p@((r-mean)**2))
        with self.assertRaises(ValueError):exact_logprob(x,np.full((2,3),-1),p)
    def test_official_state_order_and_zero_rates(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'sources/dcrl_root/src'))
        from dcrl.fitting import observed_loglikelihood
        from dcrl.latent_law import SparseLatentLaw
        b=np.array([[0.,1.,3.],[0.,4.,.5],[2.,0.,1.]])
        law=SparseLatentLaw(2,{0:.11,1:.19,2:.27,3:.43})
        x=np.array([[0,0,0],[1,2,3],[0,4,1],[3,0,2]])
        official=observed_loglikelihood(x,b,law,None,'Poisson')
        self.assertAlmostEqual(official,float(exact_logprob(x,b,np.array([law.probability(i) for i in range(4)])).sum()),places=11)
        nozero=SparseLatentLaw(2,{1:.2,2:.3,3:.5})
        self.assertAlmostEqual(observed_loglikelihood(x,b,nozero,None,'Poisson'),float(exact_logprob(x,b,np.array([nozero.probability(i) for i in range(4)])).sum()),places=11)
    def test_zero_probability_serialization(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'sources/dcrl_root/src'))
        from dcrl.fitting import observed_loglikelihood
        from dcrl.latent_law import SparseLatentLaw
        from evaluate import metrics
        x=np.array([[1.,0.],[0.,0.]])
        b=np.zeros((2,3));p=np.array([1.,0.,0.,0.]); scores=exact_logprob(x,b,p)
        self.assertTrue(np.isneginf(scores[0]));self.assertEqual(scores[1],0.)
        self.assertTrue(np.isneginf(observed_loglikelihood(x,b,SparseLatentLaw(2,{0:1.}),None,'Poisson')))
        summary=score_summary(scores)
        self.assertEqual(summary,{'log_score':None,'log_score_status':'negative_infinity','zero_probability_rows':1,'rows':2})
        mean,cov,zero=moments(b,p); report=metrics(x,mean,cov,zero,scores,np.ones(2),2)
        self.assertEqual(report['zero_probability_rows'],1);self.assertEqual(report['rows'],2)
        json.dumps({'calibration_score':summary,'test_metrics':report},allow_nan=False)
        for bad in ([float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):score_summary(bad)
    def test_upstream_selection_semantics(self):
        rows=[{'status':'failed','penalized_objective':100}, {'status':'ok','penalized_objective':5,'id':1}, {'status':'ok','penalized_objective':5,'id':2}, {'status':'ok','penalized_objective':float('nan')}]
        self.assertEqual(select_start(rows)['id'],1)
        rows[2]['penalized_objective']=6
        self.assertEqual(select_start(rows)['id'],2)
        self.assertIsNone(select_start(rows[:1]))
    def test_split_null(self):
        ids=np.array(['a','b','c','d','e','f','g','h']);f,c=split(ids);x=np.arange(8*300).reshape(8,300);y=control(x,[f,c],seed(1000,'a','b'))
        for rows in (f,c):
            for j in range(300):np.testing.assert_array_equal(np.sort(x[rows,j]),np.sort(y[rows,j]))
        self.assertTrue(set(f).isdisjoint(c));self.assertEqual(len(f),6)
if __name__=='__main__':unittest.main()

"""Deterministic algebra fixtures only; no scientific worlds."""
import unittest
import numpy as np
from scipy.stats import poisson,nbinom
from model import *

class ModelChecks(unittest.TestCase):
    def test_likelihood(self):
        x=np.array([[0,1,3],[2,0,5]]); rates=np.array([[.1,2,3],[1,3,7.]])
        for i in range(2):
            np.testing.assert_allclose(log_components(x,rates,'poisson')[:,i],poisson.logpmf(x,rates[i]).sum(1))
            np.testing.assert_allclose(log_components(x,rates,'nb2')[:,i],nbinom.logpmf(x,2,2/(2+rates[i])).sum(1))
        np.testing.assert_allclose(nbinom.stats(2,2/(2+rates),moments='v'),rates+rates**2/2)
    def test_em_baseline_and_responsibilities(self):
        x=np.array([[0,1],[1,0],[2,3],[4,0],[0,2]],float)
        for kernel in ('poisson','nb2'):
            m=em(x,1,kernel,0); self.assertTrue(m['converged']); np.testing.assert_allclose(m['rates'][0],x.mean(0))
            m=em(x,2,kernel,1); self.assertTrue(m['converged']); self.assertTrue(np.all(np.diff(m['trace'])>=-1e-8)); lj=log_components(x,m['rates'],kernel)+np.log(m['weights']); np.testing.assert_allclose(np.exp(lj-logsumexp(lj,axis=1)[:,None]).sum(1),1)
            np.testing.assert_allclose(logprob(x,m['weights'],m['rates'],kernel),logprob(x,m['weights'][::-1],m['rates'][::-1],kernel))
        m=em(np.zeros((10,2)),1,'poisson',0); self.assertTrue(np.isfinite(logprob(np.ones((1,2)),m['weights'],m['rates'],'poisson')).all())
    def test_schema(self):
        ids=np.array([f'c{i}' for i in range(4)]); f=np.array([f'f{i}' for i in range(300)]); x=np.zeros((4,300))
        validate(x,ids,f,f.tolist())
        with self.assertRaises(ValueError):validate(x,ids,f,f[::-1].tolist())
        x[0,0]=-1
        with self.assertRaises(ValueError):validate(x,ids,f,f.tolist())
    def test_completion_fixture(self):
        w=np.array([.2,.3,.5]); rates=[np.array([[1,5],[3,2],[6,4.]]),np.array([[4,1],[1,7],[8,3.]]),np.array([[2,3],[3,4],[5,1.]])]
        pts=[{'weights':w,'rates':r} for r in rates]; st=[{'pass':True,'weight_intervals':[[v-.01,v+.01] for v in w]} for _ in pts]
        matches=compatible(pts,st); self.assertEqual(len(matches),1)
        wp,_,cov=completion(pts,matches[0]); np.testing.assert_allclose(wp,w)
        np.testing.assert_allclose(cov[0,1],sum(w[i]*np.outer(rates[0][i],rates[1][i]) for i in range(3))-np.outer(w@rates[0],w@rates[1]))
        st=[{'pass':True,'weight_intervals':[[0,1]]*3} for _ in pts]; self.assertEqual(len(compatible(pts,st)),36)
        x=np.arange(1200).reshape(4,300); a,b=split(np.array(['a','b','c','d'])); y=permute(x,[a,b],123); self.assertTrue(all(sorted(y[a,j])==sorted(x[a,j]) for j in range(300)))
    def test_bootstrap_near_ties_and_restart_repeatability(self):
        point={'weights':np.array([.50,.50]),'rates':np.array([[1.,1.],[9.,9.]]),'effective':np.array([50.,50.]),'converged':True}
        boots=[{'weights':np.array([.50,.50]),'rates':point['rates'].copy(),'effective':np.array([50.,50.]),'converged':True} for _ in range(32)]
        st=stability(point,boots,100)
        self.assertTrue(st['pass']); self.assertFalse(st['disjoint_weights'])
        self.assertEqual(len(compatible([point]*3,[st]*3)),4)
        x=np.array([[0,1],[1,0],[2,3],[4,0],[0,2]],float)
        a,b=em(x,2,'poisson',7),em(x,2,'poisson',7)
        self.assertEqual(a['converged'],b['converged']); np.testing.assert_allclose(a['rates'],b['rates']); self.assertEqual(a['trace'],b['trace'])
        for kernel in ('poisson','nb2'):
            c=-gammaln(x+1).sum(axis=1) if kernel=='poisson' else np.log1p(x).sum(axis=1)
            np.testing.assert_allclose(log_components(x,a['rates'],kernel,c),log_components(x,a['rates'],kernel))
if __name__=='__main__':unittest.main()

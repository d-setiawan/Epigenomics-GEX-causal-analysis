"""Small deterministic fixtures; no benchmark or heldout inputs."""
import itertools,json,sys,unittest
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from adapter import *
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'sources/zk_meitd_root'))
from solver.MEITD import MEITD
from common import active_status,score_rows,score_summary

class Checks(unittest.TestCase):
    def test_qp_kkt_against_slsqp(self):
        matrices=[np.diag([1.,2.,3.,4.]),np.array([[3.,.2,.3,0],[.2,2.,0,.1],[.3,0,1.,.1],[0,.1,.1,2.]])]
        for W in matrices:
            for b in (np.array([.2,.3,.1,.4]),np.array([-2.,2.,1.,-.5])):
                x=simplex_qp(b,W,.025)
                result=minimize(lambda z:(z-b)@W@(z-b),np.ones(4)/4,jac=lambda z:2*W@(z-b),bounds=[(.025,None)]*4,constraints={'type':'eq','fun':lambda z:z.sum()-1,'jac':lambda z:np.ones(4)},method='SLSQP',options={'ftol':1e-12,'maxiter':500})
                self.assertTrue(result.success);np.testing.assert_allclose(x,result.x,atol=2e-6)
                g=W@(x-b);free=x>.025+1e-7;nu=-g[free].mean();self.assertLess(np.max(abs(g[free]+nu)),1e-7);self.assertTrue(np.all(g[~free]+nu>=-1e-7))
    def test_compat_and_restoration(self):
        W=np.array([[3.,1.],[1.,2.]]);b=np.array([1.,4.]);orig=la.solve;before=np.random.get_state();had='int' in np.__dict__
        with scipy_compatibility():
            np.testing.assert_allclose(la.solve(W,b,sym_pos=True),np.linalg.solve(W,b));self.assertIs(np.int,int);np.random.seed(9)
        self.assertIs(la.solve,orig);self.assertEqual('int' in np.__dict__,had);np.testing.assert_array_equal(np.random.get_state()[1],before[1])
    def test_objective_direct_distinct_coordinates(self):
        V=np.arange(1,46,dtype=float).reshape(9,5)/10;M=(np.arange(1,37).reshape(9,4)%11+1)/4;w=np.array([.1,.2,.3,.4]);s=MEITD(d=4);s._setup(V,M,w,False,None)
        reported=s.eval_cost(); direct=constant=0.
        for k in range(1,5):
            for idx in itertools.combinations(range(9),k):
                observed=np.prod(V[list(idx)],axis=0).mean(); predicted=np.prod(s.M[list(idx)],axis=0)@w
                factor=np.math.factorial(k)*s.tau[k-1] if hasattr(np,'math') else __import__('math').factorial(k)*s.tau[k-1]
                direct+=factor*(predicted-observed)**2;constant+=factor*observed**2
        self.assertAlmostEqual(reported+constant,direct,places=7)
        self.assertAlmostEqual(s.tau[0],1.);self.assertAlmostEqual(s.tau[1],1/8)
    def test_boundary_and_stable_inverse(self):
        self.assertEqual(active_status(8),'theorem_dimension_abstention');self.assertEqual(active_status(9),'source_design_unsupported_n9');self.assertEqual(active_status(10),'eligible')
        V=np.array([[0.,1.,2.],[1.,2.,3.]]);M=np.array([[0.,2.],[1.,3.]]);s=MEITD(d=1);s._setup(V,M,np.array([.5,.5]),True,'data')
        rates,err=stable_inverse(s,V);np.testing.assert_allclose(rates,M);self.assertLess(err,1e-10)
        s.M[0,0]=s.Mlb[0,0]-.01
        with self.assertRaises(ValueError):stable_inverse(s,V)
    def test_author_one_iteration_deterministic_fixture(self):
        i=np.arange(10)[:,None];j=np.arange(31)[None,:]
        V=((i*13+j*7+i*j*3+j*j)%17).astype(float);M=V[:,[0,3,7,12]];s=MEITD(d=4)
        with scipy_compatibility():
            _,w,_=s.solve_mean_weight(V,M,np.full(4,.25),1,Mtol=1e-4,wtol=1e-4,ftol=None,translate_init='data',regularize_data=True,update_weights=True,warmup=0,qp_solver=simplex_qp,AA_depth=0,monit=0)
            final=float(s.eval_cost()); rates,_=stable_inverse(s,V)
        self.assertEqual(s.niter,1);self.assertTrue(np.isfinite(final));self.assertTrue(np.all(rates>=0));self.assertAlmostEqual(w.sum(),1)
    def test_poisson_zero(self):
        vals,_=score_rows(np.array([[1,0],[0,0]]),np.array([.25]*4),np.zeros((4,2)));r=score_summary(vals);self.assertEqual(r['zero_probability_rows'],1);self.assertEqual(r['log_score_status'],'negative_infinity');json.dumps(r,allow_nan=False)
if __name__=='__main__':unittest.main()

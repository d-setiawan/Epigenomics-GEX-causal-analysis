"""Exact expectation/algebra fixtures; no benchmark worlds or model-data fits."""
import unittest
import numpy as np
from scipy.stats import poisson
import count_sources as cs


class CountSourceChecks(unittest.TestCase):
    def test_scalar_factorial_moments_exact_poisson_expectation(self):
        # Exhaustive discrete expectation with negligible, bounded omitted tail.
        counts = np.array([(a,b) for a in range(30) for b in range(30)])
        rate = np.array([.7,1.2]); center=np.array([.2,.4]); w=np.array([[.7,-.3]])
        target=float(w[0]@(rate-center))
        for exposure in [.4,1.5]:
            weights=poisson.pmf(counts[:,0],exposure*rate[0])*poisson.pmf(counts[:,1],exposure*rate[1])
            self.assertLess(1-weights.sum(),1e-14)
            got=cs.projected_factorial_moments(counts,exposure,w,center)
            expectation=weights@got["moment_contributions"][:,0,:]
            np.testing.assert_allclose(expectation,[target**r for r in range(1,5)],atol=2e-12)

    def test_tensor_and_independent_scalar_formulas_agree(self):
        y=np.array([[0,1,0],[2,0,1],[1,3,0],[4,1,2],[0,2,3],[3,0,2],[2,1,4],[1,0,0]])
        t=np.array([.5,1.,2.,1.5,.75,1.25,2.5,1.])
        a=np.array([[.4,-.2,1.],[.1,.5,.2]])
        for corrected in [True,False]:
            mu,_,cov,third,_,_=cs._cumulants(y.astype(float),t,corrected)
            scalar=cs.projected_factorial_moments(y,t,a,mu,corrected)
            np.testing.assert_allclose(np.diag(a@cov@a.T),scalar["variance"],atol=2e-12)
            contracted=np.einsum("ai,aj,ak,ijk->a",a,a,a,third)
            np.testing.assert_allclose(contracted,scalar["moments"][:,2],atol=2e-11)

    def test_exact_tensor_recovers_signed_mixing(self):
        B=np.array([[1,.2,0],[0,1,.3],[.4,0,1],[.8,0,0],[0,1.1,0],[0,0,.9]])
        skew=np.array([.8,1.5,-.6]); cov=B@B.T
        third=np.einsum("ir,jr,kr,r->ijk",B,B,B,skew)
        scale=np.sqrt(np.diag(cov)+np.array([2,3,5,2,4,3]))
        for seed in [0,1]:
            result=cs._spectral_sources(cov,third,scale,3,seed)
            self.assertEqual(result["status"],"ok")
            aligned=cs.align_mixing(B,result["mixing"])
            np.testing.assert_allclose(aligned["aligned_mixing"],B,atol=2e-12)
            np.testing.assert_allclose(result["unmixing"]@result["mixing"],np.eye(3),atol=2e-12)
            self.assertLess(result["third_offdiagonal_residual"],1e-12)

    def test_no_ridge_and_zero_source_behavior(self):
        y=np.array([[0,0],[0,1],[1,0],[1,1]])
        bad=cs.fit_counts(y,np.ones(4),1)
        self.assertEqual(bad["status"],"abstained")
        self.assertEqual(bad["mixing"].shape,(2,0))
        zero=cs.fit_counts(np.zeros((4,3)),np.ones(4),0)
        self.assertEqual(zero["status"],"ok")
        self.assertEqual(zero["unmixing"].shape,(0,3))
        ordinary=cs.fit_counts(y,np.ones(4),0,correction=False)
        np.testing.assert_array_equal(ordinary["noise_diagonal"],np.zeros(2))
        np.testing.assert_allclose(ordinary["covariance"],np.cov(y,rowvar=False,ddof=0))
        with self.assertRaises(ValueError): cs.fit_counts(y,np.zeros(4),1)

    def test_condition_and_uninformative_tensor_abstention(self):
        bad=cs._spectral_sources(np.diag([1.,1e-10]),np.zeros((2,2,2)),np.ones(2),2,0)
        self.assertEqual(bad["status"],"abstained")
        null=cs._spectral_sources(np.eye(2),np.zeros((2,2,2)),np.ones(2),2,0)
        self.assertEqual(null["status"],"abstained")


if __name__=="__main__": unittest.main(verbosity=2)

"""Exact count-kernel and algebraic fixtures; no benchmark worlds generated."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import nbinom, poisson

import extension_estimators as ext


FIXTURE = np.array([[0,1,2], [2,0,1], [1,2,0], [0,3,1], [3,1,2],
                    [1,0,4], [4,2,1], [2,4,3], [3,0,0], [1,3,2]], float)


class ExtensionChecks(unittest.TestCase):
    def test_poisson_projection_matches_original_full_tensor(self):
        path = Path(__file__).resolve().parent.parent / "count_mdcrl_2026_11_01/count_sources.py"
        spec = importlib.util.spec_from_file_location("frozen_count_sources", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        t = np.array([.7, 1., 1.2, .9, 1.4, 1.1, .8, 1.3, .95, 1.05])
        w = np.array([[.2, -.4, .7], [.3, .5, -.1]])
        for correction in [True, False]:
            mean, scale, cov, third, _, noise = old._cumulants(FIXTURE, t, correction)
            stats = ext.count_covariance(FIXTURE, t, correction=correction)
            np.testing.assert_allclose(stats["covariance"], cov, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(stats["noise_diagonal"], noise, atol=1e-12)
            r, h2, h3 = ext._pseudo(FIXTURE, np.broadcast_to(t[:,None], FIXTURE.shape), np.inf, 1., 3, correction)
            projected = ext._third_projected(r, h2, h3, w, mean)
            full_projection = np.einsum("ai,bj,ck,ijk->abc", w, w, w, third)
            np.testing.assert_allclose(projected, full_projection, rtol=1e-12, atol=1e-12)
            new_moments = ext.projected_moments(FIXTURE, t, w, mean, correction=correction)
            old_moments = old.projected_factorial_moments(FIXTURE, t, w, mean, correction)
            for key in ["moment_contributions", "variance", "skewness", "excess_kurtosis"]:
                np.testing.assert_allclose(new_moments[key], old_moments[key], rtol=1e-11, atol=1e-11, equal_nan=True)

    def test_exact_nb_and_zero_inflated_projected_moments(self):
        support = np.arange(180)
        y = np.stack(np.meshgrid(support, support, indexing="ij"),axis=-1).reshape(-1,2)
        rate, t = np.array([.25,.5]), np.array([.7,1.1])
        w, center = np.array([[.7,-.3], [.5,.9]]), np.array([.1,.1])
        expected = np.array([((rate-center) @ w.T)**r for r in range(1,5)]).T
        for theta in [np.inf,10.,2.,.5]:
            for q in [1.,.75,.5]:
                probabilities = []
                for mu in t*rate/q:
                    p = poisson.pmf(support, mu) if np.isinf(theta) else nbinom.pmf(support, theta, theta/(theta+mu))
                    p *= q
                    p[0] += 1-q
                    probabilities.append(p)
                weights = np.outer(*probabilities).ravel()
                self.assertGreater(weights.sum(), 1-1e-13)
                moments = ext.projected_moments(y, np.broadcast_to(t,y.shape), w, center, theta=theta, retention=q)
                actual = np.einsum("n,nkr->kr", weights, moments["moment_contributions"])
                np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=2e-10)

    def test_noise_sensitivities_preserve_mean_and_add_variance_correction(self):
        base = ext.count_covariance(FIXTURE, 1.)
        for theta in [np.inf,10.,2.,.5]:
            for q in [1.,.75,.5]:
                stats = ext.count_covariance(FIXTURE, 1., theta=theta, retention=q)
                np.testing.assert_array_equal(stats["mean"], base["mean"])
                np.testing.assert_array_equal(stats["scale"], base["scale"])
                self.assertTrue(np.all(stats["noise_diagonal"] >= base["noise_diagonal"]-1e-14))
                np.testing.assert_allclose(stats["covariance"]+np.diag(stats["noise_diagonal"]), stats["observed_covariance"], atol=1e-12)

    def test_projected_nb_tensor_agrees_with_independent_bell_contractions(self):
        t = np.ones(FIXTURE.shape)*np.array([.7,1.1,1.5])
        w = np.array([[.3,-.2,.6], [.2,.7,-.4]])
        mean = (FIXTURE/t).mean(axis=0)
        for theta,q in [(np.inf,.5),(2.,1.),(.5,.75)]:
            r,h2,h3 = ext._pseudo(FIXTURE,t,theta,q,3,True)
            tensor = ext._third_projected(r,h2,h3,w,mean)
            for direction in [np.array([1.,0.]), np.array([0.,1.]), np.array([.4,-.7])]:
                scalar = ext.projected_moments(FIXTURE,t,(direction @ w)[None,:],mean,theta=theta,retention=q)
                contraction = np.einsum("a,b,c,abc",direction,direction,direction,tensor)
                self.assertAlmostEqual(contraction,scalar["moments"][0,2],places=10)

    def test_pgf_hessian_matches_finite_difference_log_pgf(self):
        y = FIXTURE[:,:2]
        t = np.linspace(.7,1.3,len(y))[:,None]*np.ones_like(y)
        w = np.array([[.6,-.2],[.3,.5]])
        matrices, info = ext._pgf_matrices(y,t,(y/t).mean(axis=0),w,seed=0)
        self.assertGreaterEqual(info["pgf_min_observed_base"],.5-1e-14)
        self.assertTrue(np.all(info["pgf_log_weight_ranges"] <= 0))
        h = 2e-4
        def log_pgf(s):
            return np.log(np.exp((y*np.log1p(s[None,:]/t)).sum(axis=1)).mean())
        for index,s in enumerate(info["pgf_processing_points"][:3]):
            numeric = np.zeros((2,2))
            for a in range(2):
                for b in range(2):
                    ea,eb = np.eye(2)[a]*h,np.eye(2)[b]*h
                    numeric[a,b]=(log_pgf(s+ea+eb)-log_pgf(s+ea-eb)-log_pgf(s-ea+eb)+log_pgf(s-ea-eb))/(4*h*h)
            np.testing.assert_allclose(matrices[index],w @ numeric @ w.T,rtol=2e-5,atol=3e-7)

    def test_tensor_shrinkage_and_generalized_orientation(self):
        t = np.ones_like(FIXTURE)
        w = np.array([[.3,-.2,.6],[.2,.7,-.4]])
        mean = FIXTURE.mean(axis=0)
        r,h2,h3 = ext._pseudo(FIXTURE,t,np.inf,1.,3,True)
        tensor = ext._third_projected(r,h2,h3,w,mean)
        target, metadata = ext._shrink_tensor(r,h2,h3,w,mean,tensor)
        np.testing.assert_allclose(np.einsum("g,gabc->abc",metadata["shrink_group_sizes"],metadata["shrink_group_tensors"])/len(r),tensor,atol=1e-12)
        self.assertTrue(np.all(np.abs(target) <= np.abs(tensor)+1e-14))
        self.assertGreater(np.ptp(target[tensor != 0]/tensor[tensor != 0]),1e-3)
        np.testing.assert_allclose(target,target.transpose(2,1,0),atol=1e-14)
        q = np.array([[.6,-.8],[.8,.6]])
        matrices = np.array([q @ np.diag(d) @ q.T for d in [[1,2],[3,1],[.3,.8]]])
        learned, info = ext._orient_matrices(matrices)
        row,col = linear_sum_assignment(-np.abs(q.T @ learned))
        np.testing.assert_allclose(np.abs(q.T @ learned)[row,col],1,atol=1e-12)
        self.assertLess(info["orientation_fit_residual"],1e-12)

    def test_300_features_zero_rank_and_invalid_combinations(self):
        y = np.tile(FIXTURE,(1,100))
        result = ext.fit_counts(y,1.,0)
        self.assertEqual(result["status"],"ok")
        self.assertEqual(result["mixing"].shape,(300,0))
        self.assertIsNone(result["white_third_tensor"])
        r,h2,h3 = ext._pseudo(y,np.ones_like(y),np.inf,1.,3,True)
        w = np.tile(np.array([[.3,-.2,.6],[.2,.7,-.4]]),(1,100))/100
        tensor = ext._third_projected(r,h2,h3,w,y.mean(axis=0))
        self.assertEqual(tensor.shape,(2,2,2))
        for kwargs in [dict(theta=2.,orientation="pgf"),dict(retention=.5,orientation="pgf"),dict(correction=False,orientation="pgf"),dict(retention=0.)]:
            with self.assertRaises(ValueError):
                ext.fit_counts(FIXTURE,1.,1,**kwargs)
        bernoulli = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.assertEqual(ext.fit_counts(bernoulli,1.,1)["status"],"abstained")

    def test_successful_fit_unit_demixing_and_original_residual_gate(self):
        y = np.vstack([np.zeros((9,3)), [10,20,30]])
        for theta in [np.inf,10.,2.,.5]:
            for retention in [1.,.75,.5]:
                for orientation in ["c3", "c3_shrink"]:
                    model = ext.fit_counts(y,1.,1,theta=theta,retention=retention,orientation=orientation)
                    self.assertEqual(model["status"],"ok",model["reason"])
                    np.testing.assert_allclose(model["unmixing"] @ model["mixing"],np.eye(1),atol=1e-12)
                    np.testing.assert_allclose(model["source_variance"],1,atol=1e-12)
                    self.assertEqual(model["orientation_residual"],model["third_offdiagonal_residual"])
        pgf = ext.fit_counts(FIXTURE*10,1.,2,orientation="pgf")
        self.assertEqual(pgf["status"],"ok",pgf["reason"])
        np.testing.assert_allclose(pgf["unmixing"] @ pgf["mixing"],np.eye(2),atol=1e-12)
        np.testing.assert_allclose(pgf["source_variance"],1,atol=1e-12)

    def test_retained_constant_columns_preserve_positive_modes_and_sources(self):
        base = FIXTURE*10
        expanded = np.c_[base,np.zeros(len(base)),np.ones(len(base))]
        for correction in [True,False]:
            for orientation in ["c3","c3_shrink"]:
                original = ext.fit_counts(base,3.,2,correction=correction,orientation=orientation)
                augmented = ext.fit_counts(expanded,3.,2,correction=correction,orientation=orientation)
                self.assertEqual(original["status"],"ok",original["reason"])
                self.assertEqual(augmented["status"],"ok",augmented["reason"])
                self.assertEqual(augmented["constant_feature_count"],2)
                np.testing.assert_array_equal(augmented["constant_mask"],[False,False,False,True,True])
                np.testing.assert_array_equal(augmented["scale"][-2:],[0.,0.])
                np.testing.assert_array_equal(augmented["working_scale"][-2:],[1.,1.])
                self.assertEqual(augmented["mean"][-1],1/3)
                np.testing.assert_allclose(augmented["eigenvalues"][:2],original["eigenvalues"][:2],atol=1e-12)
                np.testing.assert_allclose(augmented["covariance"][:3,:3],original["covariance"],atol=1e-12)
                np.testing.assert_array_equal(augmented["covariance"][:3,3:],np.zeros((3,2)))
                similarities = original["mixing"].T @ augmented["mixing"][:3]
                row,col = linear_sum_assignment(-np.abs(similarities))
                sign = np.sign(similarities[row,col])
                np.testing.assert_allclose(augmented["mixing"][:3,col]*sign,original["mixing"],atol=1e-10)
                np.testing.assert_allclose(augmented["mixing"][3:],0,atol=1e-12)
                np.testing.assert_allclose(augmented["unmixing"][:,3:],0,atol=1e-12)
                np.testing.assert_allclose(augmented["unmixing"] @ augmented["mixing"],np.eye(2),atol=1e-12)
                np.testing.assert_allclose(augmented["source_variance"],1,atol=1e-12)

    def test_generalized_cf_exact_kernel_expectations_and_signs(self):
        support = np.arange(90)
        y = np.stack(np.meshgrid(support,support,indexing="ij"),axis=-1).reshape(-1,2)
        t = np.broadcast_to([.7,1.1],y.shape)
        rate,mean,w = np.array([.1,.15]),np.array([.03,.02]),np.array([.2,-.1])
        for theta in [np.inf,10.,2.,.5]:
            for retention in [1.,.75,.5]:
                probabilities = []
                for mu in np.array([.7,1.1])*rate/retention:
                    p = poisson.pmf(support,mu) if np.isinf(theta) else nbinom.pmf(support,theta,theta/(theta+mu))
                    p *= retention
                    p[0] += 1-retention
                    probabilities.append(p)
                mass = np.outer(*probabilities).ravel()
                for frequency in [.125,.25,.5]:
                    logs,proxy,reason = ext._cf_log_weights(y,t,w,mean,frequency,theta,retention)
                    self.assertIsNotNone(logs,reason)
                    self.assertLessEqual(proxy,ext.CF_ABSOLUTE_CONDITIONING_TOLERANCE)
                    expected = np.exp(1j*frequency*(w @ (rate-mean)))
                    self.assertLess(abs(mass @ np.exp(logs)-expected),1e-12)
                a = ext.generalized_cf(FIXTURE,1.,np.array([[.2,-.1,.1]]),FIXTURE.mean(axis=0),theta,retention)
                b = ext.generalized_cf(FIXTURE,1.,np.array([[-.2,.1,-.1]]),FIXTURE.mean(axis=0),theta,retention)
                np.testing.assert_allclose(a["value"],np.conjugate(b["value"]),atol=1e-13)

    def test_cf_overflow_conditioning_and_exact_zero_product(self):
        bad = ext.generalized_cf(np.full((10,2),1000.),1.,np.full((1,2),100.),np.zeros(2))
        self.assertFalse(bad["valid"][0])
        self.assertFalse(bad["numerical_valid"].all())
        y,t = np.ones((2,1)),np.ones((2,1))
        with patch.object(ext,"hyp1f1",side_effect=[np.zeros((2,1),complex),np.ones((2,1))]):
            logs,proxy,reason = ext._cf_log_weights(y,t,np.ones(1),np.zeros(1),.125,2.,1.)
        self.assertIsNotNone(logs,reason)
        np.testing.assert_array_equal(np.exp(logs),np.zeros(2))
        self.assertLess(proxy,1e-8)
        with patch.object(ext,"hyp1f1",side_effect=[np.ones((2,1),complex),np.full((2,1),1e10)]):
            logs,proxy,reason = ext._cf_log_weights(y,t,np.ones(1),np.zeros(1),.125,2.,1.)
        self.assertIsNone(logs)
        self.assertGreater(proxy,1e-8)
        self.assertIn("conditioning",reason)


if __name__ == "__main__":
    unittest.main()

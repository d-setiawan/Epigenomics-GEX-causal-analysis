"""Oracle and scoring checks; no learned fit or empirical assay data."""
import itertools
import unittest
import numpy as np
import replicate_published_model as pilot
from src.scoring import score_up_to_signed_perm


class PublishedControlChecks(unittest.TestCase):
    def test_true_mixing_graph_with_signed_permutation(self):
        A=np.array([[0.,0.,0.],[.4,0.,0.],[.3,-.5,0.]])
        G=np.repeat(np.eye(3),2,axis=0)*np.array([1.,1.7,1.,1.5,1.,1.3])[:,None]
        B=G@np.linalg.inv(np.eye(3)-A)
        observed=B[:,[2,0,1]]*np.array([-1.,1.,-1.])
        learner=pilot.LinearMDCRL(alpha=.05,gamma=.2)
        learner.nr_joint=3;learner.joint_mixing=observed.copy()
        learner.get_joint_graph()
        score,aligned=learner.score_graph_param_matrix(A)
        self.assertLess(score,1e-10)
        np.testing.assert_allclose(aligned,A,atol=1e-10)
        # Independent exhaustive check confirms zero error for this oracle case.
        exact=min(np.linalg.norm(observed[:,p]*np.array(s)-B)
                  for p in itertools.permutations(range(3))
                  for s in itertools.product([-1.,1.],repeat=3))
        native,_=score_up_to_signed_perm(observed.copy(),B)
        self.assertLess(exact,1e-10);self.assertLess(native,1e-10)

    def test_zero_shared_is_unavailable_and_normalizations_differ(self):
        self.assertEqual(pilot.normalized_mixing(0.,0),(None,None))
        native,paper=pilot.normalized_mixing(9.,3)
        self.assertAlmostEqual(native,.1)
        self.assertAlmostEqual(paper,9/np.sqrt(90))


if __name__=="__main__": unittest.main(verbosity=2)

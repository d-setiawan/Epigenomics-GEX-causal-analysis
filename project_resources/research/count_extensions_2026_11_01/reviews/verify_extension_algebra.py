"""Independent count-kernel identities and code checks, without project data."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import hyp0f1, hyp1f1
from scipy.stats import nbinom, poisson

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
import extension_estimators as ex


def factorial_coordinate(y, t, order, theta, retention):
    if order == 0:
        return np.ones(len(y))
    out = np.ones(len(y))
    for r in range(order):
        out *= y-r
    multiplier = np.prod([1+r/theta for r in range(order)]) if np.isfinite(theta) else 1.
    return out*retention**(order-1)/(t**order*multiplier)


def multi_moment(y, t, indices, theta, retention):
    degree = np.bincount(indices, minlength=y.shape[1])
    out = np.ones(len(y))
    for j,r in enumerate(degree):
        out *= factorial_coordinate(y[:,j], t[:,j], int(r), theta, retention)
    return out


def main():
    checks, counts = [], {}
    # Fixed integer algebra fixture; it is not a generated benchmark world.
    y = np.array([[0,2,1],[4,0,2],[1,1,0],[2,3,1],[0,1,4],
                  [3,2,0],[7,1,2],[2,0,6]],dtype=float)
    t = np.array([[.6,1.2,.8],[1.1,.9,1.4],[.7,1.3,1.],
                  [1.5,.8,.9],[1.2,1.,.7],[.9,1.4,1.2],
                  [.8,1.1,1.3],[1.4,.7,1.1]])
    w = np.array([.2,-.3,.4])
    center = np.array([.1,.5,.2])
    whitening = np.array([[.4,-.1,.2],[.1,.3,-.2]])
    for theta, retention in itertools.product([np.inf,10.,2.,.5],[1.,.75,.5]):
        raw = [np.ones(len(y))]
        for degree in range(1,5):
            out = np.zeros(len(y))
            for indices in itertools.product(range(y.shape[1]),repeat=degree):
                out += np.prod(w[list(indices)])*multi_moment(y,t,indices,theta,retention)
            raw.append(out)
        expected = np.column_stack([sum(math.comb(degree,r)*(-w@center)**(degree-r)*raw[r]
                                        for r in range(degree+1)) for degree in range(1,5)])
        actual = ex.projected_moments(y,t,w[None,:],center,theta,True,retention)
        np.testing.assert_allclose(actual["moment_contributions"][:,0,:],expected,rtol=2e-12,atol=2e-11)
        mu = (y/t).mean(0)
        raw2 = np.array([[multi_moment(y,t,(i,j),theta,retention).mean()
                          for j in range(3)] for i in range(3)])
        raw3 = np.array([[[multi_moment(y,t,(i,j,k),theta,retention).mean()
                           for k in range(3)] for j in range(3)] for i in range(3)])
        third = raw3.copy()
        for i,j,k in itertools.product(range(3),repeat=3):
            third[i,j,k] -= mu[i]*raw2[j,k]+mu[j]*raw2[i,k]+mu[k]*raw2[i,j]-2*mu[i]*mu[j]*mu[k]
        stats = ex.count_covariance(y,t,theta,True,retention)
        np.testing.assert_allclose(stats["covariance"],raw2-np.outer(mu,mu),atol=2e-12)
        r,h2,h3 = ex._pseudo(y,t,theta,retention,3,True)
        projected = ex._third_projected(r,h2,h3,whitening,mu)
        expected_tensor = np.einsum("ai,bj,ck,ijk->abc",whitening,whitening,whitening,third)
        np.testing.assert_allclose(projected,expected_tensor,rtol=2e-12,atol=2e-12)
    checks.append("all_12_kernel_projected_moments_and_repeated_index_C2_C3_against_multinomial_factorial_expansion")
    counts["kernel_parameter_combinations"] = 12

    # A zero count coordinate and a positive noninteger constant rate have
    # nonpositive corrected variance and no cross covariance. Their numerical
    # working scale must not alter the positive subspace on other coordinates.
    augmented_y=np.column_stack([y,np.zeros(len(y)),np.ones(len(y))])
    augmented_t=np.column_stack([t,np.ones(len(y)),np.full(len(y),10.)])
    for theta,retention,corrected in [(th,q,True) for th,q in itertools.product(
            [np.inf,10.,2.,.5],[1.,.75,.5])]+[(np.inf,1.,False)]:
        base=ex.count_covariance(y,t,theta,corrected,retention)
        extended=ex.count_covariance(augmented_y,augmented_t,theta,corrected,retention)
        assert extended["constant_feature_count"]==2
        np.testing.assert_array_equal(extended["scale"][-2:],[0.,0.])
        np.testing.assert_array_equal(extended["working_scale"][-2:],[1.,1.])
        np.testing.assert_array_equal(extended["covariance"][-2:,:3],np.zeros((2,3)))
        assert np.all(np.diag(extended["covariance"])[-2:]<=0)
        e0,v0=np.linalg.eigh(base["standardized_covariance"])
        e1,v1=np.linalg.eigh(extended["standardized_covariance"])
        keep0,keep1=e0>1e-10,e1>1e-10
        np.testing.assert_allclose(e0[keep0],e1[keep1],rtol=1e-12,atol=1e-12)
        b0=base["working_scale"][:,None]*v0[:,keep0]*np.sqrt(e0[keep0])
        b1=extended["working_scale"][:,None]*v1[:,keep1]*np.sqrt(e1[keep1])
        unmix=(v1[:,keep1]/np.sqrt(e1[keep1])).T/extended["working_scale"][None,:]
        np.testing.assert_allclose(b1[-2:],0,atol=1e-12)
        np.testing.assert_allclose(b1[:3]@b1[:3].T,b0@b0.T,atol=2e-12)
        np.testing.assert_allclose(unmix@b1,np.eye(keep1.sum()),atol=2e-12)
        np.testing.assert_allclose(unmix@extended["covariance"]@unmix.T,np.eye(keep1.sum()),atol=2e-12)
    checks.append("13_kernel_constant_coordinate_positive_subspace_and_unit_scale_checks")

    # Empirical PGF Hessian: independent direct derivative calculation before
    # whitening, compared with the module's projected/log-weight computation.
    mu = (y/t).mean(0)
    matrices,metadata = ex._pgf_matrices(y,t,mu,whitening,5)
    for s,actual in zip(metadata["pgf_processing_points"],matrices):
        base = 1+s/t
        weight = np.prod(base**y,axis=1)
        transform = weight.mean()
        first = np.mean(weight[:,None]*y/(t+s),axis=0)
        second = np.empty((3,3))
        for i,j in itertools.product(range(3),repeat=2):
            numerator = y[:,i]*(y[:,j]-(i==j))
            second[i,j] = np.mean(weight*numerator/((t[:,i]+s[i])*(t[:,j]+s[j])))
        hessian = second/transform-np.outer(first,first)/transform**2
        np.testing.assert_allclose(actual,whitening@hessian@whitening.T,atol=2e-12)
        assert base.min() >= .5-1e-14
    checks.append("all_eight_PGF_matrices_against_unwhitened_direct_PGF_derivatives")

    r,h2,h3 = ex._pseudo(y,t,2.,.75,3,True)
    tensor = ex._third_projected(r,h2,h3,whitening,mu)
    shrunk,metadata = ex._shrink_tensor(r,h2,h3,whitening,mu,tensor)
    sizes,groups = metadata["shrink_group_sizes"],metadata["shrink_group_tensors"]
    np.testing.assert_allclose(np.average(groups,axis=0,weights=sizes),tensor,atol=2e-12)
    variance = sum(n*(group-tensor)**2 for n,group in zip(sizes,groups))/((len(groups)-1)*len(y))
    np.testing.assert_allclose(metadata["shrink_variance_proxy"],variance,atol=2e-12)
    np.testing.assert_allclose(shrunk,tensor**3/(tensor**2+variance),atol=2e-12)
    assert np.all(np.abs(shrunk)<=np.abs(tensor)+1e-12)
    checks.append("unequal_group_size_shrinkage_partition_mean_variance_and_amplitude")

    # Compare the implemented kernel with its independently expanded finite
    # factorial polynomial, with low-count fixtures that pass numerical checks.
    for theta,retention in itertools.product([np.inf,10.,2.,.5],[1.,.75,.5]):
        for frequency in [.125,.25,.5]:
            logs,proxy,reason = ex._cf_log_weights(y,t,w*.1,center,frequency,theta,retention)
            assert logs is not None, reason
            kernel = np.ones(y.shape,dtype=complex)
            for j in range(3):
                for order in range(1,int(y[:,j].max())+1):
                    kernel[:,j] += (1j*frequency*w[j]*.1)**order/math.factorial(order)*factorial_coordinate(
                        y[:,j],t[:,j],order,theta,retention)
            expected = np.prod(kernel,axis=1)*np.exp(-1j*frequency*(w*.1)@center)
            np.testing.assert_allclose(np.exp(logs),expected,rtol=1e-11,atol=1e-12)
            assert proxy <= 1e-8
    checks.append("36_implemented_CF_kernel_frequency_checks_against_finite_factorial_polynomial")

    # Deterministic conditional expectations, no random datasets or fitted
    # source model. The omitted NB/Poisson probability is checked explicitly.
    support = np.arange(1500,dtype=float)
    maximum_mean_error,maximum_second_error = 0.,0.
    for theta,retention in itertools.product([np.inf,10.,2.,.5],[1.,.75,.5]):
        exposure,intensity = 1.3,1.7
        active_mean = exposure*intensity/retention
        mass = retention*(poisson.pmf(support,active_mean) if np.isinf(theta) else
                          nbinom.pmf(support,theta,theta/(theta+active_mean)))
        mass[0] += 1-retention
        assert abs(mass.sum()-1)<1e-13
        for frequency in [.125,.25,.5]:
            h = ((1+1j*frequency*retention/exposure)**support if np.isinf(theta) else
                 hyp1f1(-support,theta,-1j*frequency*theta*retention/exposure))
            kernel = 1+(h-1)/retention
            expected_mean = np.exp(1j*frequency*intensity)
            error = abs(mass@kernel-expected_mean)
            assert error<2e-11
            maximum_mean_error = max(maximum_mean_error,float(error))
            # Laguerre/Hille-Hardy consequence; it quantifies the noise risk.
            base_second = (np.exp(frequency**2*retention*intensity/exposure) if np.isinf(theta) else
                           hyp0f1(theta,frequency**2*(retention*theta*intensity/exposure+intensity**2)))
            expected_second = (base_second+(1-retention)*(1-2*np.cos(frequency*intensity)))/retention
            error = abs(mass@np.abs(kernel)**2-expected_second)
            assert error<2e-10
            maximum_second_error = max(maximum_second_error,float(error))
    checks.append("36_conditional_CF_means_and_absolute_second_moments_by_probability_sums")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"status":"passed","checks":checks,"counts":counts,
              "project_data_read":False,"scientific_model_fits":0,"new_benchmark_worlds":0,
              "maximum_conditional_CF_mean_error":maximum_mean_error,
              "maximum_conditional_CF_second_moment_error":maximum_second_error,
              "source_sha256":{p.name:sha(p) for p in [STUDY/"extension_estimators.py",Path(__file__)]}}
    (Path(__file__).with_name("extension_algebra_audit.json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()

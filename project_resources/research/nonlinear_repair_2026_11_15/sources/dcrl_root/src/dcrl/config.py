from dataclasses import dataclass,field
from typing import Optional
import math


@dataclass
class BoundsConfig:
    lob: float=0.60
    upb: float=0.65
    lob2: float=0.35
    upb2: float=0.40
    lob3: float=0.77
    upb3: float=0.82
    lob4: float=0.20
    upb4: float=0.25
    lob5: float=0.65
    upb5: float=0.70


@dataclass
class ExperimentConfig:
    N: int
    K: int
    J: Optional[int]=None
    Q_type: str="2"
    DAG_type: str="Tree"
    distribution: str="Lognormal"
    algorithm: str="PSAEM"
    bounds: BoundsConfig=field(default_factory=BoundsConfig)
    tau: Optional[float]=None
    pen: Optional[float]=None
    max_iter: int=20
    tol: Optional[float]=None
    C: int=1
    epsilon: Optional[float]=None
    Q_N: Optional[int]=None
    kappa: float=1e-200
    L_final: float=0.0
    n_jobs: int=5
    # ``legacy`` reproduces the public procedure; revision changes only the
    # default Bernoulli inverse-link clipping constant.
    profile: str="legacy"
    init_cutoff_c: float=2.5
    inner_max_iter: Optional[int]=None
    gain_numerator: Optional[float]=None
    gain_offset: float=10.0
    min_outer_iter: Optional[int]=None
    coefficient_bound_scale: float=1.0
    likelihood_max_states: int=65536
    likelihood_batch_size: int=256

    def finalize(self):
        if self.profile not in {"legacy","revision"}:
            raise ValueError("profile must be 'legacy' or 'revision'")
        self.Q_type=str(self.Q_type)
        if self.J is None:
            self.J=3*self.K
        lambda_vec=[self.N**(1/8),self.N**(2/8),self.N**(3/8)]
        const=0.9
        tau_vec=2*self.N**((const*(1/8))-1/2),2*self.N**((const*(2/8))-1/2),2*self.N**((const*(3/8))-1/2)
        if self.pen is None:
            self.pen=lambda_vec[1]
        if self.tau is None:
            self.tau=tau_vec[1]
        if self.tol is None:
            self.tol=0.5 if self.distribution=="Bernoulli" else 0.05
        if self.epsilon is None:
            self.epsilon=(0.001 if self.profile=="revision" and self.distribution=="Bernoulli"
                            else 1e-50 if self.distribution=="Poisson" else 1e-5)
        if self.Q_N is None:
            self.Q_N=self.N
        for name in ("init_cutoff_c","coefficient_bound_scale","gain_offset","epsilon"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f"{name} must be finite and positive")
        if self.distribution=="Bernoulli" and self.epsilon>=0.5:
            raise ValueError("Bernoulli epsilon must be less than 0.5")
        gain=self.gain_numerator if self.gain_numerator is not None else (0.1 if self.distribution=="Bernoulli" else 0.5)
        if not math.isfinite(gain) or not 0<gain<=self.gain_offset:
            raise ValueError("Require 0 < gain_numerator <= gain_offset")
        for name in ("N","K","J","max_iter","C","likelihood_max_states","likelihood_batch_size"):
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<1:
                raise ValueError(f"{name} must be a positive integer")
        if self.inner_max_iter is not None and (not isinstance(self.inner_max_iter,int) or self.inner_max_iter<1):
            raise ValueError("inner_max_iter must be a positive integer or None")
        if self.min_outer_iter is not None and (not isinstance(self.min_outer_iter,int) or not 0<=self.min_outer_iter<=self.max_iter):
            raise ValueError("min_outer_iter must be between zero and max_iter")
        for name in ("pen","tau","tol"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f"{name} must be finite and nonnegative")
        return self

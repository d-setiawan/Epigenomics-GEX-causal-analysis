import numpy as np

from .config import BoundsConfig
from .latent_models import sample_latent_matrix
from .utils import sigmoid


class GenerateData:
    def __init__(
        self,
        N,
        J,
        K,
        Q_type,
        DAG_type,
        distribution,
        lob,
        upb,
        lob2,
        upb2,
        lob3,
        upb3,
        lob4,
        upb4,
        lob5,
        upb5,
        Q_matrix=None,
    ):
        self.N=N
        self.K=K
        self.J=J
        self.Q_type=str(Q_type)
        self.DAG_type=DAG_type
        self.distribution=distribution
        if Q_matrix is None:
            if J!=3*K:
                raise ValueError("The default Q design requires J=3*K; provide Q_matrix for other item counts")
            self._provided_Q=None
        else:
            Q_matrix=np.asarray(Q_matrix)
            if Q_matrix.shape!=(J,K) or not np.isin(Q_matrix,[0,1]).all():
                raise ValueError("Q_matrix must be binary with shape (J, K)")
            if np.any(np.sum(Q_matrix,axis=1)==0):
                raise ValueError("Q_matrix must have at least one active coefficient in every row")
            self._provided_Q=Q_matrix.astype(float,copy=True)
        self.bounds=BoundsConfig(
            lob=lob,
            upb=upb,
            lob2=lob2,
            upb2=upb2,
            lob3=lob3,
            upb3=upb3,
            lob4=lob4,
            upb4=upb4,
            lob5=lob5,
            upb5=upb5,
        )
        self.Q=self.generate_Q()
        self.B=self.generate_B()
        self.gamma=self.generate_gamma()
        self.A=None
        self.X=None

    @classmethod
    def from_config(cls,config,*,Q_matrix=None):
        config.finalize()
        b=config.bounds
        return cls(
            N=config.N,
            J=config.J,
            K=config.K,
            Q_type=config.Q_type,
            DAG_type=config.DAG_type,
            distribution=config.distribution,
            lob=b.lob,
            upb=b.upb,
            lob2=b.lob2,
            upb2=b.upb2,
            lob3=b.lob3,
            upb3=b.upb3,
            lob4=b.lob4,
            upb4=b.upb4,
            lob5=b.lob5,
            upb5=b.upb5,
            Q_matrix=Q_matrix,
        )

    def generate_Q(self):
        if self._provided_Q is not None:
            return self._provided_Q.copy()
        Q=np.vstack((np.eye(self.K),np.eye(self.K),np.eye(self.K)))
        for k in range(self.K-1):
            Q[k,k+1]=1
            Q[k+1,k]=1
        if self.Q_type=="2":
            for k in range(self.K-2):
                Q[k,k+2]=1
                Q[k+2,k]=1
        return Q

    def generate_B(self):
        if self.distribution=="Poisson":
            g=1*np.ones(self.J)
            c=3*np.ones(self.J)
        elif self.distribution in {"Lognormal","Bernoulli"}:
            g=-1*np.ones(self.J)
            c=2*np.ones(self.J)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        beta_true=np.zeros((self.J,self.K+1))
        beta_part_in=np.zeros((self.J,self.K))
        for j in range(self.J):
            beta_true[j,0]=g[j]
            beta_part_in[j,self.Q[j,:]==1]=(c[j]-g[j])/np.sum(self.Q[j,:]==1)
        beta_true[:,1:]=beta_part_in
        return beta_true

    def generate_gamma(self):
        return np.ones(self.J)

    def generate_latent_data(self,alternate_prob=True):
        self.A=sample_latent_matrix(
            N=self.N,
            K=self.K,
            dag_type=self.DAG_type,
            bounds=self.bounds,
            alternate_prob=alternate_prob,
        )
        return self.A

    def generate_data(self):
        mu_correct=np.dot(np.column_stack((np.ones(self.N).reshape(-1,1),self.A)),self.B.T)
        if self.distribution=="Poisson":
            self.X=np.random.poisson(mu_correct)
        elif self.distribution=="Lognormal":
            s=np.tile(np.sqrt(self.gamma),(self.N,1))
            self.X=np.exp(mu_correct+np.random.normal(0,s,(self.N,self.J)))
        elif self.distribution=="Bernoulli":
            self.X=np.random.binomial(1,sigmoid(mu_correct))
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        return self.X


Generate_Data=GenerateData

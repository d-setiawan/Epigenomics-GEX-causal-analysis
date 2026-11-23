import numpy as np
from collections import Counter
from time import perf_counter
from scipy.optimize import minimize
from scipy.special import expit

from .data_generator import GenerateData
from .config import BoundsConfig
from .evaluation import compute_shd_triplet
from .initialization import initialize_parameters
from .latent_law import SparseLatentLaw
from .orientation import orient_columns
from .utils import binary,TLP,thres,initialize_function


class DAGEstimator(GenerateData):
    def __init__(
        self,
        N,
        J,
        K,
        Q_type,
        DAG_type,
        distribution,
        algorithm,
        upb,
        lob,
        lob2,
        upb2,
        lob3,
        upb3,
        lob4,
        upb4,
        lob5,
        upb5,
        tau,
        pen,
        max_iter,
        tol,
        C,
        epsilon,
        Q_N,
        kappa,
        L_final,
        dense_p_output=False,
        *,
        init_cutoff_c=2.5,
        inner_max_iter=None,
        gain_numerator=None,
        gain_offset=10.0,
        min_outer_iter=None,
        coefficient_bound_scale=1.0,
        known_Q=None,
        initial_parameters=None,
        observed_X=None,
        iteration_callback=None,
        optimizer_callback=None,
        graph_options=None,
    ):
        self._observed_data=observed_X is not None
        if observed_X is None:
            super().__init__(N,J,K,Q_type,DAG_type,distribution,lob,upb,lob2,upb2,lob3,upb3,lob4,upb4,lob5,upb5)
        else:
            # The observed-data API never generates or consults simulation truth.
            self.N,self.J,self.K=N,J,K
            self.Q_type,self.DAG_type,self.distribution=str(Q_type),DAG_type,distribution
            self.bounds=BoundsConfig(lob,upb,lob2,upb2,lob3,upb3,lob4,upb4,lob5,upb5)
            self.X=np.asarray(observed_X)
            if self.X.shape!=(N,J):
                raise ValueError("observed_X must have shape (N, J)")
            self.Q=self.B=self.gamma=self.A=None
        self.tau=tau
        self.pen=pen if known_Q is None else 0.0
        self.max_iter=max_iter
        self.tol=tol
        self.C=C
        self.epsilon=epsilon
        self.p_hat=None
        self.B_hat=None
        self.gamma_hat=None
        self.A_hat=None
        self.Q_N=Q_N
        self.algorithm=algorithm
        self.kappa=kappa
        self.L_final=L_final
        self.dense_p_output=dense_p_output
        self.init_cutoff_c=init_cutoff_c
        self.inner_max_iter=max_iter if inner_max_iter is None else inner_max_iter
        self.gain_numerator=(0.1 if distribution=="Bernoulli" else 0.5) if gain_numerator is None else gain_numerator
        self.gain_offset=gain_offset
        self.min_outer_iter=(3 if distribution=="Bernoulli" else 0) if min_outer_iter is None else min_outer_iter
        self.coefficient_bound_scale=coefficient_bound_scale
        self.known_Q=None if known_Q is None else np.asarray(known_Q,dtype=bool)
        if self.known_Q is not None and self.known_Q.shape!=(J,K):
            raise ValueError("known_Q must have shape (J, K)")
        self.initial_parameters=initial_parameters
        self.iteration_callback=iteration_callback
        self.optimizer_callback=optimizer_callback
        self.graph_options={} if graph_options is None else dict(graph_options)
        self.iteration_diagnostics=[]
        self.optimizer_diagnostics=[]
        self.stopping_reason=None

    def _bounds(self,lb,ub,item):
        lower,upper=lb.copy(),ub.copy()
        lower[:self.K+1]*=self.coefficient_bound_scale
        upper[:self.K+1]*=self.coefficient_bound_scale
        if self.known_Q is not None:
            excluded=np.flatnonzero(~self.known_Q[item])+1
            lower[excluded]=upper[excluded]=0
        return list(zip(lower,upper))

    def _support_update(self,B_update):
        if self.known_Q is None:
            return np.concatenate((B_update[:,[0]],thres(B_update[:,1:],self.tau)),axis=1)
        result=B_update.copy()
        result[:,1:]*=self.known_Q
        return result

    def _minimize(self,objective,x0,*,bounds,method,options,item,iteration):
        started=perf_counter()
        try:
            result=minimize(objective,x0,bounds=bounds,method=method,options=options)
        except Exception as exc:
            record={
                "iteration": iteration+1,"item": item,"success": False,
                "status": None,"message": f"{type(exc).__name__}: {exc}",
                "exception": True,"finite": False,"fun": None,
                "finite_objective": False,"elapsed_seconds": perf_counter()-started,
            }
            self.optimizer_diagnostics.append(record)
            self.stopping_reason="optimizer_exception"
            if self.optimizer_callback is not None:
                self.optimizer_callback(dict(record))
            raise
        lower,upper=np.array(bounds).T
        objective_value=float(getattr(result,"fun",np.nan))
        record={
            "iteration": iteration+1,"item": item,
            "success": bool(getattr(result,"success",False)),"status": int(getattr(result,"status",-1)),
            "message": str(getattr(result,"message","Optimizer did not provide diagnostics")),"nit": int(getattr(result,"nit",0)),
            "nfev": int(getattr(result,"nfev",0)),
            "fun": objective_value if np.isfinite(objective_value) else None,
            "finite_objective": bool(np.isfinite(objective_value)),
            "finite": bool(np.all(np.isfinite(result.x))),
            "initial_outside_bounds": bool(np.any(x0<lower) or np.any(x0>upper)),
            "at_lower_bound": int(np.sum(np.isclose(result.x,lower,rtol=0,atol=1e-7))),
            "at_upper_bound": int(np.sum(np.isclose(result.x,upper,rtol=0,atol=1e-7))),
            "elapsed_seconds": perf_counter()-started,
        }
        self.optimizer_diagnostics.append(record)
        if self.optimizer_callback is not None:
            self.optimizer_callback(dict(record))
        return result

    def _continue(self,err,t):
        return ((t<self.min_outer_iter or abs(err)>self.tol) and t<self.max_iter)

    def _record_iteration(self,t,err,step,started,continuing,A_new):
        current=self.optimizer_diagnostics[-self.J:]
        self._likelihood_weight=(1-step)*self._likelihood_weight+step
        reason=(None if continuing else "nonfinite_parameter_change" if not np.isfinite(err)
                  else "max_iter" if t>=self.max_iter else "tolerance")
        record={
            "iteration": t,"squared_parameter_change": float(err) if np.isfinite(err) else None,
            "finite_parameter_change": bool(np.isfinite(err)),
            "gain": float(step),"accumulated_likelihood_weight": float(self._likelihood_weight),
            "optimizer_calls": len(current),
            "optimizer_failures": sum(not r["success"] for r in current),
            "optimizer_objective_sum_before_threshold": (float(sum(r["fun"] for r in current)) if all(r["fun"] is not None for r in current) else None),
            "nonzero_slopes": int(np.count_nonzero(self.B_hat[:,1:])),
            "finite_parameters": bool(np.all(np.isfinite(self.B_hat)) and np.all(np.isfinite(self.gamma_hat))),
            "elapsed_seconds": perf_counter()-started,
            "stopping_reason": reason,
        }
        self.iteration_diagnostics.append(record)
        self.stopping_reason=reason
        self.last_A=A_new.copy()
        if self.iteration_callback is not None:
            self.iteration_callback(dict(record))

    def ftn_T(self,X):
        if self.distribution=="Poisson":
            return np.atleast_2d(np.array(X))
        if self.distribution=="Lognormal":
            T_1=-(np.log(X))**2
            T_2=np.log(X)
            return np.vstack((T_1,T_2))
        if self.distribution=="Bernoulli":
            return np.atleast_2d(np.array(X))
        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def ftn_h(self,Y):
        if self.distribution=="Poisson":
            return np.atleast_2d(np.log(Y[0,:]))
        if self.distribution=="Lognormal":
            Y_1=Y[0,:]
            Y_2=np.maximum(Y[1,:],1e-100)
            return np.vstack((1/(2*Y_2),Y_1/Y_2))
        if self.distribution=="Bernoulli":
            return np.atleast_2d(Y[0,:])
        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def ftn_A(self,eta):
        if self.distribution=="Poisson":
            return np.exp(eta)
        if self.distribution=="Lognormal":
            eta_1=np.maximum(eta[:,0],1e-100)
            eta_2=eta[:,1]
            return (eta_2**2/(4*eta_1)+np.log(1/(2*eta_1))/2).reshape(len(eta_1),1)
        if self.distribution=="Bernoulli":
            return np.log1p(np.exp(eta))
        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def objective(self,phi,j):
        if self.distribution=="Poisson":
            def obj(x):
                beta_long=x.reshape(self.K+1,1)
                tmp=0
                penalty=self.pen*TLP(x,self.tau)
                for a in range(2**self.K):
                    eta=np.log(np.dot((np.insert(binary(a,self.K).flatten(),0,1)).reshape(1,1+self.K),beta_long))
                    tmp+=np.dot(eta.T,self.ftn_T(self.X[:,j]))@phi[:,a]-np.sum(self.ftn_A(eta.T))*np.sum(phi[:,a])
                return penalty-tmp
            return obj

        if self.distribution=="Lognormal":
            def obj(x):
                beta_long=x[:-1].reshape(self.K+1,1)
                gamma=np.array(x[-1]).reshape(1,1)
                tmp=0
                penalty=self.pen*TLP(x,self.tau)
                for a in range(2**self.K):
                    eta=self.ftn_h(np.vstack((np.dot((np.insert(binary(a,self.K).flatten(),0,1)).reshape(1,1+self.K),beta_long),gamma)))
                    tmp+=np.dot(eta.T,self.ftn_T(self.X[:,j]))@phi[:,a]-np.sum(self.ftn_A(eta.T))*np.sum(phi[:,a])
                return penalty-tmp
            return obj

        if self.distribution=="Bernoulli":
            def obj(x):
                beta_long=x.reshape(self.K+1,1)
                tmp=0
                penalty=self.pen*TLP(x,self.tau)
                for a in range(2**self.K):
                    eta=np.dot((np.insert(binary(a,self.K).flatten(),0,1)).reshape(1,1+self.K),beta_long)
                    tmp+=np.dot(eta.T,self.ftn_T(self.X[:,j]))@phi[:,a]-np.sum(self.ftn_A(eta.T))*np.sum(phi[:,a])
                return penalty-tmp
            return obj

        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def ftn_pen(self,beta_gamma):
        if self.distribution=="Poisson":
            return self.pen*TLP(beta_gamma[1:],self.tau)
        if self.distribution=="Lognormal":
            return self.pen*TLP(beta_gamma[1:-1],self.tau)
        if self.distribution=="Bernoulli":
            return self.pen*TLP(beta_gamma[1:],self.tau)
        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def F_1_SAEM(self,Xj,A_sample_long):
        A_sample_long=np.array(A_sample_long,dtype=np.float32)

        if self.distribution=="Poisson":
            def obj(x):
                beta_long=x.reshape(self.K+1,1)
                tmp=0
                for c in range(self.C):
                    A_beta=np.dot(A_sample_long[:,:,c],beta_long)
                    eta=self.ftn_h(A_beta.T)
                    tmp+=np.sum(eta*self.ftn_T(Xj))-np.sum(self.ftn_A(eta.T))
                return -(tmp/self.C)
            return obj

        if self.distribution=="Lognormal":
            def obj(x):
                beta_long=x[:-1].reshape(self.K+1,1)
                gamma=x[-1].reshape(1,1)
                tmp=0
                for c in range(self.C):
                    A_beta=np.dot(A_sample_long[:,:,c],beta_long)
                    eta=self.ftn_h(np.vstack((A_beta.T,np.tile(gamma,(1,self.N)))))
                    tmp+=np.sum(eta*self.ftn_T(Xj))-np.sum(self.ftn_A(eta.T))
                return -(tmp/self.C)
            return obj

        if self.distribution=="Bernoulli":
            def obj(x):
                beta_long=x.reshape(self.K+1,1)
                tmp=0
                for c in range(self.C):
                    A_beta=np.dot(A_sample_long[:,:,c],beta_long)
                    eta=self.ftn_h(A_beta.T)
                    tmp+=np.sum(eta*self.ftn_T(Xj))-np.sum(self.ftn_A(eta.T))
                return -(tmp/self.C)
            return obj

        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def _saem_finalize_p(self,counts_smooth):
        law=SparseLatentLaw.from_counts(self.K,counts_smooth,self.L_final)
        self.p_hat=law.to_dense() if self.dense_p_output else law

    def PSAEM(self,ite):
        # Old serialized estimators and lightweight subclasses can omit the
        # newly exposed knobs. Their missing values retain public defaults.
        defaults={
            "inner_max_iter": self.max_iter,"init_cutoff_c": 2.5,
            "gain_numerator": 0.1 if self.distribution=="Bernoulli" else 0.5,
            "gain_offset": 10.0,"min_outer_iter": 3 if self.distribution=="Bernoulli" else 0,
            "coefficient_bound_scale": 1.0,"known_Q": None,
            "iteration_callback": None,"optimizer_callback": None,
        }
        for name,value in defaults.items():
            if not hasattr(self,name):
                setattr(self,name,value)
        self.iteration_diagnostics=[]
        self.optimizer_diagnostics=[]
        self._likelihood_weight=0.0
        self.p_hat,self.B_hat,self.gamma_hat,self.A_hat=self.init(ite)
        state_dtype=np.int64 if self.K<=63 else object
        pow2=np.array([1<<k for k in range(self.K-1,-1,-1)],dtype=state_dtype)
        err=1
        t=0
        loglik=0
        rows_idx0=(self.A_hat.astype(np.int64)*pow2).sum(axis=1).astype(state_dtype)
        counts_smooth=Counter()
        for u,c in zip(*np.unique(rows_idx0,return_counts=True)):
            counts_smooth[int(u)]=float(c)

        options={"disp": False,"maxiter": self.inner_max_iter}

        if self.distribution=="Poisson":
            lb=np.zeros(self.K+1)
            ub=np.concatenate(([2],3*np.ones(self.K)))
            B_update=np.zeros((self.J,self.K+1))
            iter_indicator=True
            A_new=self.A_hat.copy()
            A_sample_long=np.zeros((self.N,self.K+1,self.C))
            f_old_1=[initialize_function() for _ in range(self.J)]

            while iter_indicator:
                iteration_started=perf_counter()
                A_cur=A_new.copy()
                counts_hat=Counter()
                ones_col=np.ones((self.N,1),dtype=int)

                for c in range(self.C):
                    idx_cur=(A_cur.astype(np.int64)*pow2).sum(axis=1).astype(state_dtype)
                    for i in range(self.N):
                        z_i=np.dot(np.insert(A_cur[i],0,1,axis=0),self.B_hat.T).astype(float)
                        T_i=self.ftn_T(self.X[i,:]).T
                        for k in np.random.permutation(self.K):
                            mask=int(pow2[k])
                            col=self.B_hat[:,k+1].astype(float)
                            if A_cur[i,k]==1:
                                z0,z1=z_i-col,z_i
                                idx1,idx0=idx_cur[i]|mask,idx_cur[i]&~mask
                            else:
                                z0,z1=z_i,z_i+col
                                idx1,idx0=idx_cur[i]|mask,idx_cur[i]&~mask

                            log_prior_ratio=np.log(counts_smooth.get(idx1,0.0)+self.kappa)-np.log(counts_smooth.get(idx0,0.0)+self.kappa)
                            eta1=self.ftn_h(np.maximum(z1.reshape(1,-1),1e-150))
                            eta2=self.ftn_h(np.maximum(z0.reshape(1,-1),1e-150))
                            eta1T=eta1.T
                            eta2T=eta2.T
                            dots1=np.sum(eta1T*T_i,axis=1)
                            dots0=np.sum(eta2T*T_i,axis=1)
                            A1=self.ftn_A(eta1T)[:,0]
                            A0=self.ftn_A(eta2T)[:,0]
                            pa1=float(np.sum(dots1-A1))
                            pa0=float(np.sum(dots0-A0))
                            prob1=expit(log_prior_ratio+(pa1-pa0))
                            new_bit=np.random.binomial(1,prob1)
                            if new_bit!=A_cur[i,k]:
                                A_cur[i,k]=new_bit
                                z_i=z1 if new_bit==1 else z0
                                idx_cur[i]=idx1 if new_bit==1 else idx0

                    A_sample_long[:,:,c]=np.hstack((ones_col,A_cur))
                    uniq,cnts=np.unique(idx_cur,return_counts=True)
                    for k_idx,v in zip(uniq.tolist(),cnts.tolist()):
                        counts_hat[k_idx]=counts_hat.get(int(k_idx),0.0)+float(v)

                c0,t0=self.gain_numerator,self.gain_offset
                step=c0/(t+t0)
                if self.C>1:
                    for k_idx in list(counts_hat.keys()):
                        counts_hat[k_idx]/=float(self.C)
                if counts_smooth:
                    for k_idx in list(counts_smooth.keys()):
                        counts_smooth[k_idx]*=(1.0-step)
                for k_idx,v in counts_hat.items():
                    counts_smooth[k_idx]=counts_smooth.get(k_idx,0.0)+step*float(v)

                A_new=A_cur.copy()
                for j in range(self.J):
                    Xj=self.X[:,j].copy()
                    f_loglik=self.F_1_SAEM(Xj,A_sample_long)

                    def update_f_old_1(x,f_old=f_old_1[j],f_new=f_loglik,step=step):
                        return (1-step)*f_old(x)+step*f_new(x)

                    f_old_1[j]=update_f_old_1

                    def f_j(beta_gamma):
                        return f_old_1[j](beta_gamma)+self.ftn_pen(beta_gamma)

                    opt_result=self._minimize(f_j,self.B_hat[j,:].flatten(),bounds=self._bounds(lb,ub,j),method="SLSQP",options=options,item=j,iteration=t)
                    B_update[j,:]=opt_result.x

                err=np.linalg.norm(self.B_hat-self._support_update(B_update),"fro")**2
                self.B_hat=self._support_update(B_update)
                t+=1
                print(f"SAEM Iteration {t}, Err {err:.5f}")
                iter_indicator=self._continue(err,t)
                self._record_iteration(t,err,step,iteration_started,iter_indicator,A_new)

            self._saem_finalize_p(counts_smooth)
            return self.p_hat,self.B_hat,self.A_hat,t,loglik

        if self.distribution=="Lognormal":
            lb=np.concatenate(([-2],np.zeros(self.K+1)))
            ub=np.concatenate((4*np.ones(self.K+1),[2]))
            B_update=np.zeros((self.J,self.K+1))
            gamma_update=np.zeros((self.J,1))
            iter_indicator=True
            A_new=self.A_hat.copy()
            A_sample_long=np.zeros((self.N,self.K+1,self.C))
            f_old_1=[initialize_function() for _ in range(self.J)]

            while iter_indicator:
                iteration_started=perf_counter()
                A_cur=A_new.copy()
                counts_hat=Counter()
                ones_col=np.ones((self.N,1),dtype=int)

                for c in range(self.C):
                    idx_cur=(A_cur.astype(np.int64)*pow2).sum(axis=1).astype(state_dtype)
                    for i in range(self.N):
                        z_i=np.dot(np.insert(A_cur[i],0,1,axis=0),self.B_hat.T).astype(float)
                        T_i=self.ftn_T(self.X[i,:]).T
                        for k in np.random.permutation(self.K):
                            mask=int(pow2[k])
                            col=self.B_hat[:,k+1].astype(float)
                            if A_cur[i,k]==1:
                                z0,z1=z_i-col,z_i
                                idx1,idx0=idx_cur[i]|mask,idx_cur[i]&~mask
                            else:
                                z0,z1=z_i,z_i+col
                                idx1,idx0=idx_cur[i]|mask,idx_cur[i]&~mask

                            log_prior_ratio=np.log(counts_smooth.get(idx1,0.0)+self.kappa)-np.log(counts_smooth.get(idx0,0.0)+self.kappa)
                            eta1=self.ftn_h(np.vstack((z1.reshape(1,-1),self.gamma_hat.reshape(1,-1))))
                            eta2=self.ftn_h(np.vstack((z0.reshape(1,-1),self.gamma_hat.reshape(1,-1))))
                            eta1T=eta1.T
                            eta2T=eta2.T
                            dots1=np.sum(eta1T*T_i,axis=1)
                            dots0=np.sum(eta2T*T_i,axis=1)
                            A1=self.ftn_A(eta1T)[:,0]
                            A0=self.ftn_A(eta2T)[:,0]
                            pa1=float(np.sum(dots1-A1))
                            pa0=float(np.sum(dots0-A0))
                            prob1=expit(log_prior_ratio+(pa1-pa0))
                            new_bit=np.random.binomial(1,prob1)
                            if new_bit!=A_cur[i,k]:
                                A_cur[i,k]=new_bit
                                z_i=z1 if new_bit==1 else z0
                                idx_cur[i]=idx1 if new_bit==1 else idx0

                    A_sample_long[:,:,c]=np.hstack((ones_col,A_cur))
                    uniq,cnts=np.unique(idx_cur,return_counts=True)
                    for k_idx,v in zip(uniq.tolist(),cnts.tolist()):
                        counts_hat[k_idx]=counts_hat.get(int(k_idx),0.0)+float(v)

                c0,t0=self.gain_numerator,self.gain_offset
                step=c0/(t+t0)
                if self.C>1:
                    for k_idx in list(counts_hat.keys()):
                        counts_hat[k_idx]/=float(self.C)
                if counts_smooth:
                    for k_idx in list(counts_smooth.keys()):
                        counts_smooth[k_idx]*=(1.0-step)
                for k_idx,v in counts_hat.items():
                    counts_smooth[k_idx]=counts_smooth.get(k_idx,0.0)+step*float(v)

                A_new=A_cur.copy()
                for j in range(self.J):
                    Xj=self.X[:,j]
                    f_loglik=self.F_1_SAEM(Xj,A_sample_long)

                    def update_f_old_1(x,f_old=f_old_1[j],f_new=f_loglik,step=step):
                        return (1-step)*f_old(x)+step*f_new(x)

                    f_old_1[j]=update_f_old_1

                    def f_j(beta_gamma):
                        return f_old_1[j](beta_gamma)+self.ftn_pen(beta_gamma)

                    opt_result=self._minimize(
                        f_j,
                        np.concatenate((self.B_hat[j,:].flatten(),np.array(self.gamma_hat[j]))),
                        bounds=self._bounds(lb,ub,j),
                        method="SLSQP",
                        options=options,
                        item=j,iteration=t,
                    )
                    B_update[j,:]=opt_result.x[:-1]
                    gamma_update[j]=opt_result.x[-1]

                err=(
                    np.linalg.norm(self.B_hat-self._support_update(B_update),"fro")**2
                    +np.linalg.norm(self.gamma_hat-gamma_update,"fro")**2
                )
                self.B_hat=self._support_update(B_update)
                self.gamma_hat=gamma_update.copy()
                t+=1
                print(f"SAEM Iteration {t}, Err {err:.5f}")
                iter_indicator=self._continue(err,t)
                self._record_iteration(t,err,step,iteration_started,iter_indicator,A_new)

            self._saem_finalize_p(counts_smooth)
            return self.p_hat,self.B_hat,self.gamma_hat,self.A_hat,t,loglik

        if self.distribution=="Bernoulli":
            lb=np.concatenate(([-5],-2.5*np.ones(self.K)))
            ub=5*np.ones(self.K+1)
            B_update=np.zeros((self.J,self.K+1))
            iter_indicator=True
            A_new=self.A_hat.copy()
            A_sample_long=np.zeros((self.N,self.K+1,self.C))
            f_old_1=[initialize_function() for _ in range(self.J)]

            while iter_indicator:
                iteration_started=perf_counter()
                A_cur=A_new.copy()
                counts_hat=Counter()
                ones_col=np.ones((self.N,1),dtype=int)

                for c in range(self.C):
                    idx_cur=(A_cur.astype(np.int64)*pow2).sum(axis=1).astype(state_dtype)
                    for i in range(self.N):
                        z_i=np.dot(np.insert(A_cur[i],0,1,axis=0),self.B_hat.T).astype(float)
                        T_i=self.ftn_T(self.X[i,:]).T
                        for k in np.random.permutation(self.K):
                            mask=int(pow2[k])
                            col=self.B_hat[:,k+1].astype(float)
                            if A_cur[i,k]==1:
                                z0,z1=z_i-col,z_i
                                idx1,idx0=idx_cur[i]|mask,idx_cur[i]&~mask
                            else:
                                z0,z1=z_i,z_i+col
                                idx1,idx0=idx_cur[i]|mask,idx_cur[i]&~mask

                            log_prior_ratio=np.log(counts_smooth.get(idx1,0.0)+self.kappa)-np.log(counts_smooth.get(idx0,0.0)+self.kappa)
                            eta1=self.ftn_h(z1.reshape(1,-1))
                            eta2=self.ftn_h(z0.reshape(1,-1))
                            eta1T=eta1.T
                            eta2T=eta2.T
                            dots1=np.sum(eta1T*T_i,axis=1)
                            dots0=np.sum(eta2T*T_i,axis=1)
                            A1=self.ftn_A(eta1T)[:,0]
                            A0=self.ftn_A(eta2T)[:,0]
                            pa1=float(np.sum(dots1-A1))
                            pa0=float(np.sum(dots0-A0))
                            prob1=expit(log_prior_ratio+(pa1-pa0))
                            new_bit=np.random.binomial(1,prob1)
                            if new_bit!=A_cur[i,k]:
                                A_cur[i,k]=new_bit
                                z_i=z1 if new_bit==1 else z0
                                idx_cur[i]=idx1 if new_bit==1 else idx0

                    A_sample_long[:,:,c]=np.hstack((ones_col,A_cur))
                    uniq,cnts=np.unique(idx_cur,return_counts=True)
                    for k_idx,v in zip(uniq.tolist(),cnts.tolist()):
                        counts_hat[k_idx]=counts_hat.get(int(k_idx),0.0)+float(v)

                c0,t0=self.gain_numerator,self.gain_offset
                step=c0/(t+t0)
                if self.C>1:
                    for k_idx in list(counts_hat.keys()):
                        counts_hat[k_idx]/=float(self.C)
                if counts_smooth:
                    for k_idx in list(counts_smooth.keys()):
                        counts_smooth[k_idx]*=(1.0-step)
                for k_idx,v in counts_hat.items():
                    counts_smooth[k_idx]=counts_smooth.get(k_idx,0.0)+step*float(v)

                A_new=A_cur.copy()
                for j in range(self.J):
                    Xj=self.X[:,j].copy()
                    f_loglik=self.F_1_SAEM(Xj,A_sample_long)

                    def update_f_old_1(x,f_old=f_old_1[j],f_new=f_loglik,step=step):
                        return (1-step)*f_old(x)+step*f_new(x)

                    f_old_1[j]=update_f_old_1

                    def f_j(beta_gamma):
                        return f_old_1[j](beta_gamma)+self.ftn_pen(beta_gamma)

                    opt_result=self._minimize(f_j,self.B_hat[j,:].flatten(),bounds=self._bounds(lb,ub,j),method="SLSQP",options=options,item=j,iteration=t)
                    B_update[j,:]=opt_result.x

                err=np.linalg.norm(self.B_hat-self._support_update(B_update),"fro")**2
                self.B_hat=self._support_update(B_update)
                t+=1
                print(f"SAEM Iteration {t}, Err {err:.5f}")
                iter_indicator=self._continue(err,t)
                self._record_iteration(t,err,step,iteration_started,iter_indicator,A_new)

            self._saem_finalize_p(counts_smooth)
            return self.p_hat,self.B_hat,self.A_hat,t,loglik

        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def init(self,ite):
        if getattr(self,"initial_parameters",None) is not None:
            p,B,gamma,A=self.initial_parameters
            result=(dict(p) if isinstance(p,dict) else p,B.copy(),gamma.copy(),A.copy())
        else:
            if not getattr(self,"_observed_data",False):
                self.generate_latent_data()
                self.generate_data()
            result=initialize_parameters(
                self.X,self.J,self.K,self.distribution,self.epsilon,sparse=True,
                cutoff_c=getattr(self,"init_cutoff_c",2.5),
            )
        if getattr(self,"known_Q",None) is not None:
            # Also enforce support before the FIRST Gibbs update for callers
            # using the estimator directly rather than fit_observed.
            result[1][:,1:]*=self.known_Q
        return result

    def estimate(self,ite,truth_graph):
        if self.distribution=="Poisson":
            if self.algorithm=="PSAEM":
                self.p_hat,self.B_hat,self.A_hat,_,_=self.PSAEM(ite)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

        elif self.distribution=="Lognormal":
            if self.algorithm=="PSAEM":
                self.p_hat,self.B_hat,self.gamma_hat,self.A_hat,_,_=self.PSAEM(ite)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

        elif self.distribution=="Bernoulli":
            if self.algorithm=="PSAEM":
                self.p_hat,self.B_hat,self.A_hat,_,_=self.PSAEM(ite)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        oriented=orient_columns(self.B_hat,self.p_hat,self.gamma_hat)
        self.B_hat,self.p_hat=oriented['B'],oriented['p']
        flips=oriented['flipped_columns']
        if self.A_hat is not None:
            self.A_hat[:,flips]=1-self.A_hat[:,flips]
        self.orientation=oriented
        return compute_shd_triplet(
            p_hat=self.p_hat,
            B_hat=self.B_hat,
            K=self.K,
            J=self.J,
            Q_N=self.Q_N,
            truth_graph=truth_graph,
            Q_truth=self.Q,
            **getattr(self,"graph_options",{}),
        )


DAG_Estimator=DAGEstimator

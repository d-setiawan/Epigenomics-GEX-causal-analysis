## write a header here

from ._N_coefs import COEF

import numpy as np
import scipy.linalg as la
from scipy.special import comb
from numpy.random import seed, permutation as perm

import warnings


class MEITD:
    MAX_D = len(COEF.keys())
    
    
    @staticmethod
    def _default_wlb_sched(meitd):
        return 0.1 / meitd.r
    
    
    @staticmethod
    def _default_Nb_sched(meitd):
        return -1


    def __init__(self, d = None, tau = None):
        """Initialize meitd instance, at least one of d and tau must be given. If both are given, then d is neglected.
        
        Parameters
        ----------
        d : int or array, optional
            If int, then the algorithm will use order 1,...,d. If array [d1, d2,...], then order d1, d2,... are used.
        tau : array, optional
            The weights on costs of different orders. If order k is not used, put 0 at tau[k-1]. 
            E.g., [1, 0, 2, 3] will use weights 1, 2, 3 for order 1, 3, 4, and order 5 is not used.
            If only tau is given, d = where(tau != 0) is used. 
            If None, then the normalized weights (n-i)! / n! for order i is used. By default None. 
        """

        # basic dimensionalities
        self.n = 0
        self.p = 0
        self.r = 0
        
        if d is None and tau is None:
            raise ValueError('At least one of d or tau must be given.')
        
        if tau is None:
            # initialize tau using normalized weights and d
            self.prep_tau = True
            if isinstance(d, int):
                if d > self.MAX_D:
                    raise ValueError(f'Order {d} is not supported, maximum supported order is {self.MAX_D}')
                self.maxd_to_use = d
                self.tau = np.ones(d)
            else:
                # d = list of order to use
                self.maxd_to_use = max(d)
                if self.maxd_to_use > self.MAX_D:
                    raise ValueError(f'Order {i} is not supported, maximum supported order is {self.MAX_D}')
                self.tau = np.zeros(self.maxd_to_use)
                for i in d:
                    self.tau[i-1] = 1
        else:
            # tau is given, d is neglected
            self.prep_tau = False
            i = len(tau)
            while tau[i-1] == 0:
                i -= 1
            self.tau = np.array(tau[:i])
            self.maxd_to_use = i
        
        # optimizers
        self.opt = None
        self.qpsolve = None
        self.linsch = None
        self.nbsched = self._default_Nb_sched
        self.wsched = self._default_wlb_sched

        # matrices and tensors
        self.M = None
        self.w = None 
        self.oldM = None
        self.oldw = None 
        self.VTM = None     # VT**s @ M**s
        self.MTM = None     # MT**s @ M**s
        self.mtnsw = None   # M**s W
        self.vtns = None    # V**s
        self.vmean = None   # mean vector of the full data
        self.wLHS = None    # the L matrix, d/dw = -2 (L w - r)
        self.wRHS = None    # the r vector
        self.wLHS_temp = None    
        self.wRHS_temp = None 
        self.vmean = None
        self.gamma = None   
        self.Mlb = None
        self.Mub = None

        # AA utils and status monitors
        self.m = 0
        self.AA = False
        self.dxhist = None 
        self.dfhist = None
        self.nAA = 0
        self.nlinsch = 0
        self.cursor = 0

        # convergence status monitors
        self.Mtol = None 
        self.wtol = None
        self.Mconv = False 
        self.wconv = False
        self.n_Mstep = 0
        self.n_wstep = 0
        self.costs = []  
        self.wstepsize = []
        self.Mstepsize = []
        self.niter = 0
        self.n_cost = 0
        self.n_deriv = 0


    def set_optimizer(self, qp_solver = None, wt_lb_scheduler = None, line_search_device = None, nb_scheduler = None, AA_depth = 0):
        """Set optimizers

        Parameters
        ----------
        qp_solver : Callable, optional
            Solver that solves the quadratic programming problem for the weight update, the syntax is

            >> x = qp_solver(b, W, lb)
            
            solves min ||x - b||_W, s.t. x >= lb, <x, 1> = 1
            should return the solution, or None if solve fails.
            By default None
            
        wt_lb_scheduler : Callable, optional
            Assign a lower bound on the weights for current step, syntax 
            
            >> lb = wt_lb_scheduler(meitd)
            
            returns the current weight lower bound (scalar). 
            Can access meitd.niter for the current iteration count, amoung others.
            If None is given when weight update is called, will use the default one (= 0.1/r), by default None
            
        line_search_device : Callable, optional
            The program that does line search in the AA step. If None is given when AA is called, no line search will be performed. Syntax
            
            >> x = line_search_device(f, df, x0, dx, amax, fx0 = OPTIONAL, dfx0 = OPTIONAL)
            
            f, df: Callable, the functions to evalate the cost f and gradient df
            x0, dx: starting point and the search direction
            amax: maximal search distance, i.e., will search on the segment [x0, x0 + amax * dx]
            fx0, dfx0: Optional, the function and gradient value at x0, if not provided, will be recomputed.
            By default None
            
        nb_scheduler : Callable, optional
            Assign the number of blocks to be used in the mean update ALS, syntax 
            
            >> nb = nb_scheduler(meitd)
            
            returns the requested number of blocks (scalar). 
            Can access meitd.niter and meitd.warmup for the current iteration count and check if the program is in the warmup stage.
            If None is given when mean update is called, will use the default one (block size = 1 at all time), by default None
            
        AA_depth : int, optional
            The maximal history steps to be used in AA. If <= 0, AA will be ignored, by default 0
        """
        
        if qp_solver is not None:
            self.qpsolve = qp_solver
        if line_search_device is not None:
            self.linsch = line_search_device
        if wt_lb_scheduler is not None:
            self.wsched = wt_lb_scheduler
        if nb_scheduler is not None:
            self.nbsched = nb_scheduler
        
        self.m = AA_depth
        self.AA = self.m > 0 
        

    def _setup(self, V, M0, w0, regularize_data, translate_init):
        if M0.ndim != 2:
            raise ValueError('The initial mean init_M must have shape (n, r)')
        if V.ndim != 2:
            raise ValueError('The data V must have shape (n, p)')
        if V.shape[0] != M0.shape[0]:
            raise ValueError(f'The shape of data {V.shape} and the shape of initial mean init_M {M0.shape} must be compatible')
        if w0.shape[0] != M0.shape[1]:
            raise ValueError(f'The shape of initial weight init_w {w0.shape} and the shape of initial mean init_M {M0.shape} must be compatible')
        self.n, self.r = M0.shape
        self.M = M0.copy()
        self.w = w0.copy()
        self.V = V.copy()
        self.p = V.shape[1]
        
        if self.prep_tau:
            self.tau *= self.norm_tau_weights()
        self.fact_tau = np.ones(self.maxd_to_use)
        for i in range(1, self.maxd_to_use):
            self.fact_tau[i] = self.fact_tau[i-1] * (i + 1)
    
        self.fact_tau *= self.tau  # i! * tau_i
        
        self.vmean = np.mean(V, axis = 1, keepdims = True)
        if regularize_data:
            self.delta = -self.vmean
            self.V += self.delta
            self.gamma = 1 / np.std(self.V, axis = 1, keepdims = True, ddof = 1)
            self.V *= self.gamma
            self.centered_data = True
        else:
            self.gamma = np.ones((self.n, 1)) 
            self.delta = np.zeros((self.n, 1))  
            self.centered_data = False     

        if translate_init == 'center':
            self.M -= np.mean(self.M, axis = 1, keepdims = True)
            self.M /= np.std(self.M, axis = 1, keepdims = True, ddof = 1)
        elif translate_init == 'data':
            self.M += self.delta
            self.M *= self.gamma

        self.Mlb = np.min(self.V, axis = 1, keepdims = True)
        self.Mub = np.max(self.V, axis = 1, keepdims = True)
        
        # restrict M to the circumscribing box of the data
        self.M = np.where(self.M > self.Mub, self.Mub, np.where(self.M < self.Mlb, self.Mlb, self.M))

        self.VTM = np.zeros((self.maxd_to_use, self.p, self.r))
        self.MTM = np.zeros((self.maxd_to_use, self.r, self.r))
        self.vtns = np.zeros((self.maxd_to_use, self.n, self.p))
        self.mtnsw = np.zeros((self.maxd_to_use, self.n, self.r))
        for curd in range(self.maxd_to_use):
            Md = self.M**(1 + curd)
            self.vtns[curd] = self.V**(1 + curd) 
            self.MTM[-curd-1] = (-1)**curd * np.matmul(Md.T, Md)
            self.VTM[-curd-1] = (-1)**curd * np.matmul(self.vtns[curd].T, Md)
            self.mtnsw[curd] = Md * self.w.reshape(1, -1)

        self.wLHS = np.zeros((self.r, self.r))
        self.wRHS = np.zeros(self.r)
        self.wLHS_temp = np.zeros((self.r, self.r))
        self.wRHS_temp = np.zeros(self.r)

        if self.AA:
            self.dfhist = np.zeros((self.n * self.r + self.r, self.m))
            self.dxhist = np.zeros((self.n * self.r + self.r, self.m))
            self.firstAAcall = True
            self.cursor = 0
    
    
    def norm_tau_weights(self, d = None):
        """Compute the tau weights for i = 1,...,d
        tau_i = (n-i)! / n!

        Parameters
        ----------
        d : int, optional
            The maximal d, if None then use maximal order self.maxd_to_use, by default None

        Returns
        -------
        array
            The tau weights
        """
        
        if d is None:
            d = self.maxd_to_use
        result = np.ones(d)
        for i in range(1, len(result)):
            result[i] = result[i-1] / (self.n - i) if self.n > i else 0
        return result
        
    
    def setnb(self, nb):
        if nb <= 0 or nb > self.n:
            nb = self.n
        self.perm_idx = np.arange(self.n)
        self.k = nb
        self.b_starts = np.zeros(self.k, dtype = np.int)
        self.b_ends = np.zeros(self.k, dtype = np.int)
        self.bsize = self.n // self.k
        
        if self.r > comb(np.floor((self.n - self.bsize - 1) / 2), np.floor(self.maxd_to_use / 2)):
            raise ValueError(f'block size {self.bsize} too large for rank {self.r}, training with {self.n - self.bsize} rows')
        
        for i in range(self.k):
            self.b_starts[i] = i * self.bsize
            self.b_ends[i] = (i+1) * self.bsize
        

    def update_params(self, M = None, w = None, follow_data_transform = False):
        """Update the current parameters (M, w) with new parameters (M', w')

        Parameters
        ----------
        M : (n, r) array, optional
            The new mean matrix. If None, self.M is used. By default None
        w : (r,) array, optional
            The new weight array. If None, self.w is used. By default None
        follow_data_transform : bool, optional
            If the data is regularized, whether (M, w) should be translated as well, by default False
        """

        M_need_update = w_need_update = False
        
        w_need_update = w is not None and not np.allclose(w, self.w)
        M_need_update = False
        if M is not None:
            M = M.reshape(self.n, self.r)
            if follow_data_transform:
                M = M + self.delta
                M = M * self.gamma
            M_need_update = not np.allclose(M, self.M)

        if w_need_update:
            w = w / np.sum(w)
            ratio = w / self.w
            self.w = w.copy()
            if not M_need_update:
                for curd in range(self.maxd_to_use):
                    self.mtnsw[curd] *= ratio.reshape(1, -1)

        if M_need_update:
            self.M = M.copy()
            for curd in range(self.maxd_to_use):
                Md = self.M**(1 + curd)
                self.VTM[-curd-1] = (-1)**curd * np.matmul(self.vtns[curd].T, Md)
                self.MTM[-curd-1] = (-1)**curd * np.matmul(Md.T, Md)
                self.mtnsw[curd] = Md * self.w.reshape(1, -1)
        

    def _eval_deriv_at_s(self, s, coef):
        MM = np.ones((self.r, self.r))
        VM = np.ones((self.p, self.r))
        minus1power = 0
        for _, (j, nj) in enumerate(s):
            minus1power += nj * (j-1)
            MM *= self.MTM[-j] ** nj
            VM *= self.VTM[-j] ** nj
            
        ys = np.mean(VM, axis = 0) 
        self.wLHS_temp += ((-1)**minus1power * coef) * MM
        self.wRHS_temp += ((-1)**minus1power * coef) * ys
        
        VMderiv = np.zeros((self.n, self.r))
        MMderiv = np.zeros((self.n, self.r))
        for _, (i, ni) in enumerate(s):
            with np.errstate(divide = 'ignore'):
                MMi = np.nan_to_num(np.true_divide(MM, self.MTM[-i]))
                VMi = np.nan_to_num(np.true_divide(VM, self.VTM[-i]))
            VMtemp = ni * i * (np.matmul(self.vtns[i-1], VMi)) 
            MMtemp = ni * i * (np.matmul(self.mtnsw[i-1], MMi)) 
            if i > 1:
                VMtemp *= self.mtnsw[i-2]
                MMtemp *= self.mtnsw[i-2]
            else:
                VMtemp *= self.w.reshape(1, -1)
                MMtemp *= self.w.reshape(1, -1)
            VMderiv += VMtemp * ((-1)**(minus1power + i-1) / self.p)
            MMderiv += MMtemp * (-1)**(minus1power + i-1)
        deriv = (2 * coef) * (MMderiv - VMderiv)
        return deriv
        
    
    def eval_deriv(self, M = None, w = None, follow_data_transform = False):
        """compute derivatives at (M, w)

        Parameters
        ----------
        M : (n, r) array, optional
            The mean matrix, if None, use current mean, by default None
        w : (r,) array, optional
            The weight, if None, use current weight, by default None
        follow_data_transform : bool, optional
            If True, and the data is regularized, the regularization translation will be applied to the input mean M, by default False

        Returns
        -------
        dw : (r,) array
        dM : (n, r) array
            The derivative at (M, w)
        """
        
        dM = self._eval_deriv_and_norm_eqn(M, w, follow_data_transform)
        dw = 2 * (self.wLHS @ w - self.wRHS)
        self.n_deriv -= 1
        return dw, dM.reshape(self.n, self.r)
    
    
    def _eval_deriv_and_norm_eqn(self, M = None, w = None, follow_data_transform = False, return_all = False, get_one_drop = False):
        self.update_params(M, w, follow_data_transform)
        dM = np.zeros((self.n, self.r))
        if return_all or get_one_drop:
            dMs = np.zeros((self.maxd_to_use, self.n, self.r))
        
        self.wLHS *= 0
        self.wRHS *= 0

        for curd in range(self.maxd_to_use):
            wd = self.tau[curd]
            if wd == 0:
                continue  # skip this d if the cost does not contain it.

            coef_dict = COEF[curd + 1]
            dM_d = np.zeros((self.n, self.r))
            
            self.wLHS_temp *= 0
            self.wRHS_temp *= 0
            
            for _, jnj_ns in enumerate(coef_dict.items()):
                jnj, ns = jnj_ns
                dM_jnj = self._eval_deriv_at_s(jnj, ns) 
                dM_d += dM_jnj
            dM += wd * dM_d
            if return_all or get_one_drop:
                dMs[curd] = wd * dM_d
            self.wLHS += wd * self.wLHS_temp
            self.wRHS += wd * self.wRHS_temp
        
        self.n_deriv += 1
        
        if get_one_drop:
            temp = np.zeros(len(dMs) + 1)
            temp[-1] = la.norm(dM.reshape(-1))
            for i in range(len(dMs)):
                temp[i] = la.norm((dM - dMs[i]).reshape(-1))
            drop = np.argmax(temp) if np.max(temp) > 1.00 * temp[-1] else None
        
        if get_one_drop:
            if return_all:
                return dM.reshape(-1), dMs, drop
            else:
                return dM.reshape(-1), drop
        elif return_all:
            return dM.reshape(-1), dMs
        return dM.reshape(-1)
        

    def _prep_norm_eqn(self, pi = None, ret_r = False):
        # get the highest order we need
        maxlen = self.maxd_to_use
        while not self.fact_tau[maxlen - 1]:
            maxlen -= 1
             
        EtnsVTM = np.ones((maxlen+1, self.p, self.r))
        EtnsMTM = np.ones((maxlen+1, self.r, self.r))
        for i in range(1, maxlen+1):
            EtnsVTM[i] = 1/i * np.sum(EtnsVTM[:i] * self.VTM[-i:], axis = 0)
            EtnsMTM[i] = 1/i * np.sum(EtnsMTM[:i] * self.MTM[-i:], axis = 0)
        
        idx1 = (self.fact_tau > 0)
        idx2 = np.zeros(maxlen + 1, dtype = np.bool_)
        idx2[1:] = idx1[:maxlen]
        
        self.wLHS = np.tensordot(self.fact_tau[idx1], EtnsMTM[idx2], axes = ([0], [0]))
        y = np.tensordot(self.fact_tau[idx1], EtnsVTM[idx2], axes = ([0], [0]))
            
        if not ret_r:
            if pi is None:
                self.wRHS = np.mean(y, axis = 0)
            else:
                self.wRHS = np.tensordot(y, pi, axes = ([0], [0]))
        else:
            if pi is None:
                r = np.mean(y, axis = 0)
            else:
                r = np.tensordot(y, pi, axes = ([0], [0]))
            return r
    
    
    def eval_cost(self, M = None, w = None, follow_data_transform = False):
        """compute cost at (M, w)

        Parameters
        ----------
        M : (n, r) array, optional
            The mean matrix, if None, use current mean, by default None
        w : (r,) array, optional
            The weight, if None, use current weight, by default None
        follow_data_transform : bool, optional
            If True, and the data is regularized, the regularization translation will be applied to the input mean M, by default False

        Returns
        -------
        cost : float
            The cost value at (M, w)
        """
        
        ret = self._eval_cost_and_norm_eqn(M, w, follow_data_transform)
        self.n_cost -= 1
        return ret


    def _eval_cost_and_norm_eqn(self, M = None, w = None, follow_data_transform = False):
        self.update_params(M, w, follow_data_transform)  
        self._prep_norm_eqn()
        result = (-2 * self.wRHS + self.wLHS @ self.w) @ self.w
        self.n_cost += 1
        return result
                
    
    def _update_M_block(self, idx):

        ai = self.M[idx]
        vi = self.V[idx]
        
        for j in range(1, self.maxd_to_use+1):
            aij = ai**j
            self.MTM[-j] -= ((-1)**(j-1) * aij).T @ aij
            self.VTM[-j] -= ((-1)**(j-1) * (vi**j)).T @ aij
        
        rhs = self._prep_norm_eqn(pi = vi.T, ret_r = True) / self.p
        
        if self.tau[0] != 0:
            self.wLHS += self.tau[0]
            if not self.centered_data:
                rhs += self.tau[0] * np.outer(np.ones(self.r), self.vmean[idx] + self.delta[idx])
        
        soln = (la.solve(self.wLHS, rhs, assume_a = 'sym').reshape(self.r, -1) / self.w.reshape(-1, 1)).T
        soln = np.where(soln > self.Mub[idx], self.Mub[idx], np.where(soln < self.Mlb[idx], self.Mlb[idx], soln))
        self.M[idx] = soln
        
        for j in range(1, self.maxd_to_use+1):
            aij = soln**j
            self.MTM[-j] += ((-1)**(j-1) * aij).T @ aij
            self.VTM[-j] += ((-1)**(j-1) * (vi**j)).T @ aij
        for curd in range(self.maxd_to_use):
            Md = self.M[idx]**(1 + curd)
            self.mtnsw[curd, idx] = Md * self.w.reshape(1, -1)

        return soln

    
    def _Mstep(self):
        temp = self.fact_tau.copy()
        self.fact_tau[:-1] = self.fact_tau[1:]
        self.fact_tau[-1] = 0
        
        nb = self.nbsched(self)
        self.setnb(nb)
        if nb != -1:
            self.perm_idx = perm(self.perm_idx)
        for i in range(self.k):
            idx_update = self.perm_idx[self.b_starts[i]:self.b_ends[i]]
            self._update_M_block(idx_update)
        
        self.n_Mstep += 1
        self.fact_tau = temp

    
    def _wstep(self):
        rhs = la.solve(self.wLHS, self.wRHS, sym_pos = True)
        wsoln = self.qpsolve(rhs, self.wLHS, self.wtlb)
        
        if wsoln is None:
            warnings.warn('optimal update to w not found in w-step.')
            badidx = rhs < self.wtlb
            nbadidx = np.sum(badidx)
            goodidx = 1 - badidx
            sum_of_good_idx = 1 - nbadidx * self.wtlb
            rhs[goodidx] *= sum_of_good_idx / np.sum(rhs[goodidx]) 
            rhs[badidx] = self.wtlb
            wsoln = rhs
        
        self.w = wsoln
        ratio = self.w / self.oldw
        for curd in range(self.maxd_to_use):
            self.mtnsw[curd] *= ratio.reshape(1, -1)
        self.n_wstep += 1
    
    
    def _cost_helper(self, wM):
        w = wM[:self.r]
        M = wM[self.r:]
        return self._eval_cost_and_norm_eqn(w = w, M = M)
    

    def _deriv_helper(self, wM):
        w = wM[:self.r]
        M = wM[self.r:]
        dM = self._eval_deriv_and_norm_eqn(w = w, M = M)
        dw = 2 * (self.wLHS @ w - self.wRHS)
        return np.hstack((dw, dM))
    

    def _AAstep(self, dM, dw, Mcost, iplb):
        # update history
        newgrad = np.hstack((dw, dM))
        newx = np.hstack((self.w, self.M.reshape(-1)))
        if self.firstAAcall:
            self.firstAAcall = False
            self.dfhist[:, 0] = newgrad
            self.dxhist[:, 0] = newx
            return self.M
            
        self.dfhist[:, self.cursor] = newgrad - self.dfhist[:, self.cursor]
        self.dxhist[:, self.cursor] = newx - self.dxhist[:, self.cursor]
        
        # find optimal coefs
        U, s, VT = la.svd(self.dfhist[:, :self.cursor+1], full_matrices = False)
        c_opt = VT.T @ (s**(-1) * (U.T @ newgrad))
        
        # compute optimal x
        dx = - self.dxhist[:, :self.cursor+1] @ c_opt
        x_opt = newx + dx
        
        # check if delta M agrees with the gradient on M
        ip = np.dot(dx, newgrad) / la.norm(dx) / la.norm(newgrad)
        if ip > -iplb:
            # restart AA
            self._restart_AA(newgrad, newx)
            return self.M
        else:
            self.nAA += 1
            if self.linsch is not None:
                neww = x_opt[:self.r]
                dw = dx[:self.r]
                bnds = np.zeros(self.r)
                bnds[dw < 0] = self.wtlb
                bnds[dw >= 0] = 1  
                # maximal step to stay in simplex
                amax = np.min((bnds - neww) / dw)
                x_opt = self.linsch(self._cost_helper, self._deriv_helper, newx, dx, amax, fx0 = Mcost, dfx0 = newgrad)
            result = x_opt[self.r:]
        
        self._update_AA_history(newgrad, newx)
        return result.reshape(self.n, self.r)
    

    def _update_AA_history(self, newgrad, newx):
        if self.cursor < self.m - 1:
            self.cursor += 1
            self.dxhist[:, self.cursor] = newx
            self.dfhist[:, self.cursor] = newgrad
        else:
            self.dxhist[:, :-1] = self.dxhist[:, 1:]
            self.dxhist[:, -1] = newx
            self.dfhist[:, :-1] = self.dfhist[:, 1:]
            self.dfhist[:, -1] = newgrad
        

    def _restart_AA(self, newgrad, newx):
        self.cursor = 0
        self.dxhist[:, 0] = newx
        self.dfhist[:, 0] = newgrad
    

    def solve_mean_weight(self, V, init_M, init_w, maxiter, 
                           Mtol = 1E-4, wtol = 1E-4, ftol = 1E-6,
                           translate_init = None, regularize_data = True,
                           update_weights = True,
                           warmup = 0, 
                           qp_solver = None, 
                           wt_lb_scheduler = None,
                           nb_scheduler = None,
                           line_search_device = None, 
                           AA_depth = 0, AAtol = 1E-4,
                           monit = 0.1):
        """Solve the mean and weights using ALS++

        Parameters
        ----------
        V : (n, p) array
            The data matrix, comprised of column data vectors
        init_M : (n, r) array
            Initial guess of the mean matrix, comprised of columns of mean vectors
        init_w : (r,) array
            Initial guess of the weight
        maxiter : int
            Maximum number of iterations
        Mtol : float, optional
            Termination tolerance on relative stepsize on M (Frobenius norm), by default 1E-4
        wtol : float, optional
            Termination tolerance on relative stepsize on w (2 norm), by default 1E-4
        ftol : float, optional
            Termination tolerance on relative change in cost values, by default 1E-6
        translate_init : str, optional
            Whether the initial guess on mean should be translated
            if 'center', then shift the means vectors to have a mean 0
            if 'data', then the mean vectors will follow the same translation that regularizes the data (see regularize_data),
            else no translation is made.
            by default None
        regularize_data : bool, optional
            If True, translate the data to mean 0 and row variance 1, by default True
        update_weights : bool, optional
            If True, then the weights are updated to minimize costs, if False, the weights are fixed and not optimized, by default True
        warmup : int, optional
            Number of warmup iterations, by default 0
        qp_solver : Callable, optional
            See set_optimizer, by default None
        wt_lb_scheduler : Callable, optional
            See set_optimizer, by default None
        nb_scheduler : Callable, optional
            See set_optimizer, by default None
        line_search_device : Callable, optional
            See set_optimizer, by default None
        AA_depth : int, optional
            The maximal history size used in AA, if <= 0, AA is ignored, by default 0
        AAtol : float, optional
            The tolerance epsilon_{AA}, by default 1E-4
        monit : float or int, optional
            The frequency of printing the status. If int >= 1, then print the status every monit iterations. 
            If float k in (0, 1), then print status every k * maxiter iterations, by default 0.1

        Returns
        -------
        M : (n, r) array
        w : (r,) array
            Solved means and weights
        converged : bool
            True if the termination tolerance is met within maxiter iterations, False otherwise
        """

        print()
        print('Algorithm Start.')
        print()

        self.set_optimizer(qp_solver=qp_solver, 
                          wt_lb_scheduler=wt_lb_scheduler, 
                          line_search_device=line_search_device, AA_depth=AA_depth, nb_scheduler=nb_scheduler)
        
        if len(init_w) < 3:
            regularize_data = False 
        self._setup(V, init_M, init_w, regularize_data, translate_init)
        self.maxiter = maxiter
        
        status = self._step(maxiter, Mtol, wtol, ftol, update_weights, warmup, AAtol, monit)
        M, w = self.get_soln()
        return M, w, status

    
    def _step(self, maxiter, Mtol, wtol, ftol, update_weights, warmup, AAtol, monit):
        self.Mtol = Mtol
        self.wtol = wtol
        self.ftol = ftol
        fconv = False
        df = -1
        converged = False
        niter = 0
        drop = None
        cost = np.inf
        
        seed(12345)  # random state for permuting rows in M in blocked ALS

        if monit > 0:
            if monit < 1:
                monit = max(1, np.floor(monit * maxiter))

        while not converged and niter < maxiter:
            self.warmup = (niter < warmup)
            self.Mconv = False
            self.wconv = False
            
            self.wtlb = self.wsched(self)
            self.oldM = self.M.copy()
            oldcost = cost
            
            # do drop
            if self.warmup and drop is not None:
                dropped = self.fact_tau[drop]
                self.fact_tau[drop] = 0
            
            # M step
            self._Mstep()
            
            # undo the drop
            if drop is not None:
                self.fact_tau[drop] = dropped
            
            if self.AA and not self.warmup:
                dM = self._eval_deriv_and_norm_eqn()
                dw = 2 * (self.wLHS @ self.w - self.wRHS)
                newM = self._AAstep(dM, dw, cost, AAtol)
                cost = self._eval_cost_and_norm_eqn(M = newM)
            else:
                cost = self._eval_cost_and_norm_eqn()
            
            # check M update
            Mstepsize = la.norm(self.M - self.oldM, 'fro') / la.norm(self.oldM, 'fro')
            self.Mconv = Mstepsize < Mtol
            self.Mstepsize.append(Mstepsize)
            
            # check f update
            if self.ftol is not None and niter > 0:
                df = np.abs((cost - oldcost) / oldcost)
            
            # wstep
            if update_weights:
                self.oldw = self.w.copy()
                self._wstep()
                wstepsize = la.norm(self.w - self.oldw)
                self.wstepsize.append(wstepsize / la.norm(self.oldw))
                self.wconv = wstepsize < self.wtol
            else:
                self.wstepsize.append(0)
                self.wconv = True
            
            # get drop if warmup
            if self.warmup:
                dM, drop = self._eval_deriv_and_norm_eqn(get_one_drop = True)
            else:
                drop = None
            
            # print info
            if monit > 0 and (niter - 1) % monit == 0:
                pstr = ''
                fstr = f'delta f = {df:.2e} '
                Mstr = f'delta M = {Mstepsize:.2e} '
                wstr = f'delta w = {wstepsize:.2e} '
                if self.ftol is not None:
                    pstr += fstr
                if self.Mtol is not None:
                    pstr += Mstr
                if self.wtol is not None:
                    pstr += wstr
                print(f'Iteration {self.niter}: cost = {cost:.5f} ' + pstr)

            # check termination    
            converged = (self.Mconv and self.wconv) or fconv
            self.niter += 1
            niter += 1
            
            if not converged:
                self.costs.append(cost)
            else:
                if self.warmup:
                    warmup = -1  # will exit warmup stage immediately
                    drop = None
                    converged = False  # wait until next convergence
                    
                else:  
                    # converged, find final cost after the last w step
                    cost = self._eval_cost_and_norm_eqn() 
                    self.costs.append(cost) 
                    print(f'Converged at iteration {self.niter}.')

        # if not converged
        if not converged:
            print(f'Fail to converge in {maxiter} iterations.')
            
        print('Done.')
        print()
            
        self.M = self.M.reshape(self.n, self.r)
        return converged
    

    def get_soln(self):
        """Extract the solution computed by solve_mean_weight

        Returns
        -------
        M : (n, r) array
        w : (r,) array
            The solved mean matrix and weights
        """
        
        M = self.M / self.gamma - self.delta 
        return M, self.w
    

    def get_data(self):
        """Undo the regularization map on the data to recover the original dataset

        Returns
        -------
        V : (n, p) array
            The untranslated data matrix
        """
        
        return self.V / self.gamma - self.delta
    

    def solve_std(self, qp_solver, reg = None):
        """Solve standard deviation after solving the means and weights

        Parameters
        ----------
        qp_solver : Callable
            Same as qp_solver in set_optimizer.
        reg : float, optional
            Regularization constant to prevent negative standard deviation, if not None, will impose std >= reg, by default None

        Returns
        -------
        std : (n, r) array
            The standard deviation vectors, the kth column is the standard deviation for class k
        """

        result = np.zeros((self.n, self.r))
        temp = self.fact_tau.copy()
        self.fact_tau[:-1] = self.fact_tau[1:]
        self.fact_tau[-1] = 0
            
        if reg is not None:
            lb_temp = self.w * (self.gamma**2 * reg**2 + self.M**2)
        else:
            lb_temp = None
            
            
        for i in range(self.n):
            # r * k matrix, kth col is the ith row if the order[k] moment matrix
            ai = self.M[i]
            vi = self.V[i]
            for j in range(1, self.maxd_to_use+1):
                aij = ai**j
                self.MTM[-j] -= np.outer(aij, (-1)**(j-1) * aij)
                self.VTM[-j] += np.outer((-vi)**j, aij)
            solni = self._solve_M2(vi**2, qp_solver, lb_temp[i]).T
            for j in range(1, self.maxd_to_use+1):
                aij = ai**j
                self.MTM[-j] += np.outer(aij, (-1)**(j-1) * aij)
                self.VTM[-j] -= np.outer((-vi)**j, aij)
            result[i, :] = np.sqrt(solni - self.M[i]**2) / self.gamma[i] 
        
        self.fact_tau = temp
        return result
    

    def _solve_M2(self, v2, qp_solver, reg):
        ys = self._prep_norm_eqn(pi = v2, ret_r = True) / self.p
        self.wLHS += self.tau[0]
        ys += np.mean(v2) * self.tau[0]

        soln = la.solve(self.wLHS, ys, assume_a = 'sym')
        soln = qp_solver(soln, self.wLHS, lb = reg)
        
        with np.errstate(divide = 'ignore'):
            soln = np.nan_to_num(np.true_divide(soln, self.w))
        return soln

    
    def solve_gen_mean(self, fn, k = 1, *args, **kwargs):
        """Solve general mean E g(X)

        Parameters
        ----------
        fn : Callable
            Encodes a list of k coordinatewise functions whose expectation is asked.
            Takes in a (p,) vector x, and returns (p, k) array (or (p,) if k = 1), where 
            each column corresponds to the output of a coordinatewise function acting on x
        k : int, optional
            The number of the coordinatewise functions encoded in fn (i.e. shape[1] of fn(x)), by default 1
        *args, **kwargs : 
            Additional args put into fn

        Returns
        -------
        result : (k, n, r) array, (n, r) if k = 1
            The general means, result[i] is the (n, r) matrix whose columns are the expectation if the ith function on mixture classes
        """
        
        temp = self.fact_tau.copy()
        self.fact_tau[:-1] = self.fact_tau[1:]
        self.fact_tau[-1] = 0
        
        result = np.zeros((k, self.n, self.r))
            
        for i in range(self.n):
            # r * k matrix, kth col is the ith row if the kth general moment matrix (E f_k)
            ai = self.M[i]
            vi = self.V[i]
            for j in range(1, self.maxd_to_use+1):
                aij = ai**j
                self.MTM[-j] -= np.outer(aij, (-1)**(j-1) * aij)
                self.VTM[-j] -= np.outer(vi**j, (-1)**(j-1) * aij)
            
            # center and normalize the ith row of f(V)
            org_vi = self.V[i] / self.gamma[i] - self.delta[i]
            fvi = np.array(fn(org_vi, *args, **kwargs)).reshape(self.p, k)  # p by k
            ave = np.mean(fvi, axis = 0, keepdims = True)
            scale = np.std(fvi, axis = 0, ddof = 1, keepdims = True)
            fvi = ((fvi - ave) / scale)  # p by k

            solni = self._solve_gmean(fvi)  # k by r
            result[:, i, :] = ave.T + solni * scale.T
            
            for j in range(1, self.maxd_to_use+1):
                aij = ai**j
                self.MTM[-j] += np.outer(aij, (-1)**(j-1) * aij)
                self.VTM[-j] += np.outer(vi**j, (-1)**(j-1) * aij)
        
        self.fact_tau = temp
        if k == 1:
            result = result[0]
        return result
        
    
    def _solve_gmean(self, fvi):
        ys = self._prep_norm_eqn(pi = fvi, ret_r = True) / self.p
        self.wLHS += self.tau[0]

        soln = la.solve(self.wLHS, ys, assume_a = 'sym').T 
        
        with np.errstate(divide = 'ignore'):
            soln = np.nan_to_num(np.true_divide(soln, self.w))
        
        return soln

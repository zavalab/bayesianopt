import numpy as np
from matplotlib import pyplot as pyp
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from cyipopt import minimize_ipopt
import numdifftools as nd
from joblib import Parallel, delayed
import sklearn.gaussian_process as gpr
from collections import OrderedDict
#import torch
import time


class LCB_AF():
    
    def __init__(self, model, dim, exp_w, descale,
                 refmod = None, args = (), eps = 1e-3):
        self.model = model
        self.dim = dim
        self.exp_w = exp_w
        self.descale = descale
        self.args = args
        self.eps = eps
        
        if refmod:
            self.refmod = refmod
            
        else:
            def zr(x):
                return 0
            
            self.refmod = zr
            
            
    def fun(self, x):
        x = np.array([x]).reshape(-1,1)
        x = x.reshape(int(x.shape[0]/self.dim), self.dim)
        
        mu, std = self.model.predict(x, return_std=True);
        mu = mu.flatten()
        
        yref = self.refmod(self.descale(x), *self.args)
            
        lcb = (yref+mu-self.exp_w*std).flatten()
        
        if len(lcb) == 1:
            lcb = lcb[0]
            
        return lcb
    
    
    def grad(self, x):
        f_prime = nd.Gradient(self.fun, step = self.eps)(x)
        
        return f_prime
    
    
    def hess(self, x):
        f_hess = nd.Hessian(self.fun, step = self.eps)(x)
        
        return f_hess
        


class qLCB():
    
    def __init__(self, model, q, dim, exp_w, samps, eps = 1e-3):
        self.model = model
        self.q = q
        self.dim = dim
        self.exp_w = exp_w
        self.n = samps
        self.eps = eps
        
    
    def fun(self, x):
        x = x.reshape(self.q, self.dim)
        
        if np.unique(np.round(x, 4), axis = 0).shape[0] < self.q:
            return np.max(self.model.predict(x))
        
        else:
            mu, Sigma = self.model.predict(x, return_cov = True)
            L = np.linalg.cholesky(Sigma)
            S = 0
            
            for i in range(self.n):
                z = np.random.normal(np.zeros(mu.shape), np.ones(mu.shape), mu.shape)
                s = mu-self.exp_w*np.abs(L@z)
                S += np.min(s)
                
            S = S/(self.n)
            
            if len(S) == 1:
                S = S[0]
            
            return S
        
        
        def grad(self, x):
            f_prime = nd.Gradient(self.fun, step = self.eps)(x)
            
            return f_prime
        
        
        def hess(self, x):
            f_hess = nd.Hessian(self.fun, step = self.eps)(x)
            
            return f_hess


def hyperspace_partitions(n, phi):
    binary_combinations = np.array([list(np.binary_repr(i, width = n)) for i in range(2**n)], dtype = int)
    partitions = np.ones((n, 2, 2**n))
    
    for i, binary_combination in enumerate(binary_combinations):
        lower_boundaries = np.array(tuple(bit / 2.0 for bit in binary_combination))
        lower_boundaries[lower_boundaries == 0.5] = 0.5-phi/2
        upper_boundaries = np.array(tuple((bit+1) / 2.0 for bit in binary_combination))
        upper_boundaries[upper_boundaries == 0.5] = 0.5+phi/2
        partitions[:, :, i] = np.hstack([lower_boundaries.reshape(-1, 1), upper_boundaries.reshape(-1, 1)])
        
    return partitions


def LCB_nmc(x, LCB_fant, y_s):
    af = 0
    
    for j in range(y_s.shape[0]):
        af += LCB_fant[str(j+1)].LCB(x)
    af = af/(j+1)
    
    return af


class BO():
    
    def __init__(self, ub, lb, dim, exp_w, kernel, system, bounds,
                 args = (), ref_args = (), shift_exp_w = [], **aux_mods):
        
        self.ub = ub
        self.lb = lb
        self.dim = dim
        self.exp_w = exp_w
        self.kernel = kernel
        self.system = system
        self.bounds = bounds
        self.args = args
        self.ref_args = ref_args
        self.shift_exp_w = shift_exp_w
        self.dist_ref = {}
        
        if aux_mods:
            self.refmod = aux_mods['refmod']
            
            if len(aux_mods) > 1:
                self.distmod = aux_mods['distmod']
                self.dist_ref['distrefmod'] = aux_mods['ref_distmod']
                
                for i in range(3,len(aux_mods)):
                    self.dist_ref['distrefmod'+str(i-2)] = aux_mods['ref_distmod'+str(i-2)]
                    
                self.dist_ref = OrderedDict(self.dist_ref)
        
        self.refbo_optim = False
        self.lsbo_optim = False
        self.hsbo_optim = False
        self.exbo_optim = False
        self.nmcbo_optim = False
        self.qbo_optim = False
        self.vpbo_optim = False
           
         
    def descale(self, x,
                use_self = True,
                lb = None, ub = None,
                scale_lb = None, scale_ub = None):
        
        if use_self:
            m = (self.ub-self.lb)/(self.bounds.ub-self.bounds.lb)
            b = self.ub-m*self.bounds.ub
            
        else:
            m = (ub-lb)/(scale_ub-scale_lb)
            b = ub-m*scale_ub
        
        return m*x+b
    
    
    def scale(self, x,
              use_self = True,
              lb = None, ub = None,
              scale_lb = None, scale_ub = None):
        
        if use_self:    
            m = (self.bounds.ub-self.bounds.lb)/(self.ub-self.lb)
            b = self.bounds.ub-m*self.ub
            
        else:
            m = (scale_ub-scale_lb)/(ub-lb)
            b = scale_ub-m*ub
            
        return m*x+b
    
    
    def optimizer_sbo(self, trials,
                      x_init = None, init_pts = 1,
                      af_cores = 1, af_restarts = 128,
                      af_solver = 'L-BFGS-B', af_solver_options = None):
        
        print('Standard BO run...')
        start = time.time()
        self.trials_sbo = trials
        self.time_sbo = np.ones(self.trials_sbo)
        self.time_fsbo = np.ones(self.trials_sbo)
        
        if x_init is None:
            x = np.random.uniform(self.bounds.lb,
                                  self.bounds.ub,
                                  (init_pts, self.dim))
            
        else:
            x = x_init.reshape(-1, self.dim)
            init_pts = len(x)
        
        start_f = time.time()
        y = self.system(self.descale(x), *self.args).reshape(-1, 1)
        end_f = time.time()
        
        model_sbo = gpr.GaussianProcessRegressor(self.kernel,
                                               alpha = 1e-6,
                                               normalize_y = True,
                                               n_restarts_optimizer = 20)
        model_sbo.fit(x, y)
        
        LCBgp = LCB_AF(model_sbo, self.dim, self.exp_w, self.descale)
        
        end = time.time()
        
        for i in range(init_pts):
            self.time_sbo[i] = (i+1)*(end-start)/init_pts
            self.time_fsbo[i] = (i+1)*(end_f-start_f)/init_pts
        
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_sbo[init_pts-1]:.1f}')
              
        for i in range(self.trials_sbo-init_pts):
            x0 = np.random.uniform(self.bounds.lb,
                                   self.bounds.ub,
                                   (af_restarts, self.dim))
            
            if af_solver == 'IPOPT':
                opt = Parallel(n_jobs = af_cores)(delayed(minimize_ipopt)(LCBgp.fun,
                                                                          x0 = x_0,
                                                                          jac = LCBgp.grad,
                                                                          hess = LCBgp.hess,
                                                                          bounds = self.bounds,
                                                                          options = af_solver_options)
                                                  for x_0 in x0)
                
            else:
                opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCBgp.fun,
                                                                    x0 = x_0,
                                                                    method = af_solver,
                                                                    bounds = self.bounds)
                                                  for x_0 in x0)
            
            x_nxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            x_nxt = x_nxts[np.argmin(funs)].reshape(1, self.dim)
            
            start_f = time.time()
            y_nxt = self.system(self.descale(x_nxt), *self.args).reshape(-1, 1)
            end_f = time.time()
            self.time_fsbo[i+init_pts] = self.time_fsbo[i+init_pts-1]+(end_f-start_f)
            
            x = np.vstack([x, x_nxt])
            y = np.vstack([y, y_nxt])
            model_sbo.fit(x, y)

            end = time.time()
            self.time_sbo[i+init_pts] = end-start
            
            print('ITERATION COUNT IS AT 'f'{init_pts+i+1};\
                  TOTAL ELAPSED TIME: 'f'{self.time_sbo[i+init_pts]:.1f}')
                  
        self.model_sbo = model_sbo
        self.x_sbo = self.descale(x)
        self.y_sbo = y
        
        
    def optimizer_refbo(self, trials,
                        x_init = None, init_pts = 1,
                        af_cores = 1, af_restarts = 128,
                        af_solver = 'L-BFGS-B', af_solver_options = None):
        
        print('BO with reference model run...')
        start = time.time()
        self.trials_ref = trials
        self.time_ref = np.ones(self.trials_ref)
        self.time_fref = np.ones(self.trials_ref)
        
        if x_init is None:
            x = np.random.uniform(self.bounds.lb,
                                  self.bounds.ub,
                                  (init_pts, self.dim))
            
        else:
            x = x_init.reshape(-1, self.dim)
            init_pts = len(x)
        
        start_f = time.time()
        y = self.system(self.descale(x), *self.args)
        eps = y - self.refmod(self.descale(x), *self.ref_args)
        eps = eps.reshape(-1, 1)
        end_f = time.time()
        
        model_ref = gpr.GaussianProcessRegressor(self.kernel,
                                                 alpha = 1e-6,
                                                 normalize_y = True,
                                                 n_restarts_optimizer = 20)
        model_ref.fit(x, eps)
        
        LCBref = LCB_AF(model_ref, self.dim, self.exp_w, self.descale, self.refmod, self.ref_args)
        
        end = time.time()
        
        for i in range(init_pts):
            self.time_ref[i] = (i+1)*(end-start)/init_pts
            self.time_fref[i] = (i+1)*(end_f-start_f)/init_pts
        
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_ref[init_pts-1]:.1f}')
        
        for i in range(self.trials_ref-init_pts):
            x0 = np.random.uniform(self.bounds.lb,
                                   self.bounds.ub,
                                   (af_restarts, self.dim))
            
            if af_solver == 'IPOPT':
                opt = Parallel(n_jobs = af_cores)(delayed(minimize_ipopt)(LCBref.fun,
                                                                          x0 = x_0,
                                                                          jac = LCBref.grad,
                                                                          hess = LCBref.hess,
                                                                          bounds = self.bounds,
                                                                          options = af_solver_options)
                                                  for x_0 in x0)
                
            else:
                opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCBref.fun,
                                                                    x0 = x_0,
                                                                    method = af_solver,
                                                                    tol = 1e-6,
                                                                    bounds = self.bounds)
                                                  for x_0 in x0)
                
            x_nxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            x_nxt = x_nxts[np.argmin(funs)].reshape(1, self.dim)
            
            start_f = time.time()
            y_nxt = self.system(self.descale(x_nxt), *self.args)
            eps_nxt = y_nxt - self.refmod(self.descale(x_nxt), *self.ref_args)
            eps_nxt = eps_nxt.reshape(-1, 1)
            end_f = time.time()
            self.time_fref[i+init_pts] = self.time_fref[i+init_pts-1]+(end_f-start_f)
            
            x = np.vstack([x, x_nxt])
            y = np.vstack([y, y_nxt])
            eps = np.vstack([eps, eps_nxt])
            model_ref.fit(x, eps)
            
            end = time.time()
            self.time_ref[i+init_pts] = end-start
            
            print('ITERATION COUNT IS AT 'f'{init_pts+i+1};\
                  TOTAL ELAPSED TIME: 'f'{self.time_ref[i+init_pts]:.1f}')
        
        self.refbo_optim = True
        self.model_ref = model_ref
        self.x_ref = self.descale(x)
        self.y_ref = y
        self.eps = eps
    
    
    def optimizer_lsbo(self, trials, partition_number,
                       x_init = None, partitions = None,
                       repartition_intervals = [], x_samps = [],
                       f_cores = 1, ref_cores = 1, af_cores = 1, af_restarts = 128,
                       af_solver = 'SLSQP', af_solver_options = None):
        
        """
        > partition_number is the number of desired partitions
        
        > repartition_intervals is the iteration at which repartitioning is done,
          if repartitioning is not wanted, enter empty list '[]'
        
        > f_cores and af_cores are the cores used for sampling f and optimizing the AF
        
        > x_samps contains the points at which samples of \hat{f} are collected
          during repartitioning in order to determine the median, if repartitioning
          is not desired, enter empty array, 'np.array([])', or list '[]'
        
        > partitions should be a numerically indexed dictionary with
          each entry containing a list of the constraints (linear or nonlinear)
          required to set up the desired space partition
          
        > if reference model is not available, make it a function that returns 0
         
        > x_init is an array containing the intial points at which to sample
        """
        
        print('Level-set partitioned BO Run...')
        start = time.time()
        self.trials_ls = trials
        splits = partition_number
        re_part = repartition_intervals
        self.cons_ref = partitions
        self.time_ls = np.ones(self.trials_ls)
        self.time_fls = np.ones(self.trials_ls)
        
        def intpts(x, i):
            l = str(i+1)
            res = 0
            for cons in self.cons_ref[l]:
                if type(cons) == dict:
                    if cons['fun'](x) > 0:
                        res += 0
                    else:
                        res += 1e6
                        
                else:
                    if cons.lb < cons.fun(x) < cons.ub:
                        res += 0
                    else:
                        res += 1e6
                        
            return res
                
        model_ls = gpr.GaussianProcessRegressor(self.kernel,
                                                alpha = 1e-6,
                                                n_restarts_optimizer = 10,
                                                normalize_y = True) 
        
        cons_fun = lambda x: self.refmod(self.descale(x), *self.ref_args)+model_ls.predict(x.reshape(1, 2))
        
        if self.cons_ref is None:
            self.cons_ref = {}
            y_samps = self.refmod(x_samps, *self.ref_args)
            delta = np.linspace(np.min(y_samps), np.max(y_samps), splits+1)
            
            self.cons_ref['1'] = [NonlinearConstraint(cons_fun, -np.inf, delta[1])]
            
            for i in range(1, splits):
                self.cons_ref[str(i+1)] = [NonlinearConstraint(cons_fun, delta[i], delta[i+1])]
        
        x = np.array([]).reshape(0, self.dim)
        switch = True
        
        for i in range(splits):
            n = 0
            
            if x_init is not None and switch == True:
                for cons in self.cons_ref[str(i+1)]:
                    if type(cons) == dict:
                        if cons['fun'](x_init) > 0:
                            n += 1
                    
                    else:
                        if cons.fun(x_init) < cons.ub and cons.fun(x_init) > cons.lb:
                                n += 1
                            
                if n == len(self.cons_ref[str(i+1)]):
                    x0 = x_init.reshape(1, self.dim)
                    switch = False
                    
            if n != len(self.cons_ref[str(i+1)]):
                x0 = np.random.uniform(self.bounds.lb,
                                       self.bounds.ub,
                                       (10, self.dim))
                
                opt = Parallel(n_jobs = 1)(delayed(minimize)(intpts,
                                                             x_0,
                                                             args = (i,),
                                                             method = 'SLSQP',
                                                             bounds = self.bounds,
                                                             tol = 1e-6,
                                                             constraints = self.cons_ref[str(i+1)])
                                           for x_0 in x0)
                
                x0s = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                x0 = x0s[np.argmin(funs)]
                
            x = np.vstack([x, x0.reshape(1, self.dim)])
        
        init_pts = int(len(x)/splits)
        splt = int(x.shape[0]/f_cores)
        x_bs = np.array(np.ones(f_cores), dtype = tuple)
        
        if f_cores == 1:
            x_bs[0] = x
            
        else:
            for i in range(f_cores-1):
                x_bs[i] = x[i*splt:(i+1)*splt, :]
                
            x_bs[-1] = x[(i+1)*splt:, :]
        
        start_f = time.time()
        y = Parallel(n_jobs = f_cores)(delayed(self.system)(self.descale(x_s), *self.args)
                                       for x_s in x_bs)
        y_ref = Parallel(n_jobs = ref_cores)(delayed(self.refmod)(self.descale(x_s), *self.ref_args)
                                             for x_s in x_bs)
        end_f = time.time()
        
        y = np.hstack(y[:]).T.reshape(-1,1)
        y_ref = np.hstack(y_ref[:]).T.reshape(-1,1)
        eps = y-y_ref
        y_bst = min(y).reshape(-1,1)
        
        model_ls.fit(x, eps)
        
        LCB = LCB_AF(model_ls, self.dim, self.exp_w, self.descale, self.refmod, self.ref_args)
            
        
        restarts = max(1, af_restarts//splits)
        x_nxt = np.ones((splits, self.dim))
        x_nxtbs = np.array(np.ones(f_cores), dtype = tuple)
        
        end = time.time()
        
        for i in range(init_pts):
            self.time_ls[i] = (i+1)*(end-start)/init_pts
            self.time_fls[i] = (i+1)*(end_f-start_f)/init_pts
        
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_ls[init_pts-1]:.1f}')
        
        J = 0
        
        for i in range(self.trials_ls-init_pts):
            x0 = np.random.uniform(self.bounds.lb,
                                   self.bounds.ub,
                                   (restarts, self.dim))
            
            for j in range(splits):
                if af_solver == 'IPOPT': 
                    if af_solver_options and af_solver_options.get('hessian_approximation') == 'exact':
                        lcb_hess = LCB.hess
                        
                    else: 
                        lcb_hess = None
                    
                    opt = Parallel(n_jobs = af_cores)(delayed(minimize_ipopt)(LCB.fun,
                                                                              x0 = x_0,
                                                                              jac = LCB.grad,
                                                                              hess = lcb_hess,
                                                                              bounds = self.bounds,
                                                                              constraints = self.cons_ref[str(j+1)],
                                                                              options = af_solver_options)
                                                      for x_0 in x0)
                    
                else:
                    opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCB.fun,
                                                                        x0 = x_0,
                                                                        method = 'SLSQP',
                                                                        bounds = self.bounds,
                                                                        tol = 1e-6,
                                                                        constraints = self.cons_ref[str(j+1)])
                                                      for x_0 in x0)
                    
                x_nxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                sts = np.array([res.success for res in opt])
                funs[np.where(sts == False)] = max(1e6, np.max(funs))
                x_nxt[j] = x_nxts[np.argmin(funs)]
            
            if f_cores == 1:
                x_nxtbs[0] = x_nxt
                
            else:
                for j in range(f_cores-1):
                    x_nxtbs[j] = x_nxt[j*splt:(j+1)*splt, :]
                    
                x_nxtbs[-1] = x_nxt[(j+1)*splt:, :]
            
            start_f = time.time()
            y_nxt =  Parallel(n_jobs = f_cores)(delayed(self.system) (self.descale(x_s), *self.args)
                                                for x_s in x_nxtbs)
            y_ref =  Parallel(n_jobs = ref_cores)(delayed(self.refmod)(self.descale(x_s), *self.ref_args)
                                                  for x_s in x_nxtbs)
            end_f = time.time()
            self.time_fls[i+init_pts] = self.time_fls[i+init_pts-1]+(end_f-start_f)
            
            x = np.vstack([x, x_nxt])
            y_nxt = np.hstack(y_nxt[:]).T.reshape(-1, 1)
            y = np.vstack([y, y_nxt])
            y_ref = np.hstack(y_ref[:]).T.reshape(-1, 1)
            eps_nxt = y_nxt-y_ref
            eps = np.vstack([eps, eps_nxt])
            y_bst = np.vstack([y_bst, min(y_nxt).reshape(-1, 1)])
            model_ls.fit(x, eps)
            
            if i+init_pts+1 in re_part:
                J+=1
                y_samps = self.refmod(x_samps, *self.ref_args)+model_ls.predict(self.scale(x_samps))
                med = np.median(y_samps)
                
                for j in range(J):
                    idx = np.where(y_samps <= med)
                    y_samps = y_samps[idx]
                    med = np.median(y_samps)
                delta = np.linspace(np.min(y_samps), np.max(y_samps), splits+1)
                self.cons_ref['1'] = [NonlinearConstraint(cons_fun, -np.inf, delta[1])]
                
                for j in range(1, splits):
                    self.cons_ref[str(j+1)] = [NonlinearConstraint(cons_fun, delta[j], delta[j+1])]

            end = time.time()
            self.time_ls[i+init_pts] = end-start
            
            print('ITERATION COUNT IS AT 'f'{init_pts+i+1};\
                  TOTAL ELAPSED TIME: 'f'{self.time_ls[i+init_pts]:.1f}')
            
        self.lsbo_optim = True
        self.model_ls = model_ls
        self.x_ls = self.descale(x)
        self.y_ls = y
        self.y_lsbst = y_bst
        self.eps_ls = eps
    
    
    def optimizer_hsbo(self, trials, phi, f_cores = 1, af_cores = 1, x_init = None):
        
        """
        parallelization of BO based on HyperSpace algorithm of Young et al. (2018);
        overlap parameter is denoted by phi and the partition number is set to 2^d_x;
        partitions are set up internally by the algorithms as equal-sized blocks;
        each partition (hyperspace) is optimized using its own surrogate model
        """
        
        print('Hyperspace BO run...')
        start = time.time()
        self.trials_hyp = trials
        splits = 2**self.dim
        phi = phi**(1/self.dim)
        self.time_hyp = np.ones(self.trials_hyp)
        self.time_fhyp = np.ones(self.trials_hyp)
        
        def intpts(x):
            return 0
        
        partitions = hyperspace_partitions(self.dim, phi)
        self.bounds_hyp = {}
        
        for i in range(partitions.shape[2]):
            partitions[:, 0, i] = self.scale(partitions[:, 0, i], use_self = False, lb = 0, ub = 1)
            partitions[:, 1, i] = self.scale(partitions[:, 1, i], use_self = False, lb = 0, ub = 1)
            self.bounds_hyp[f'{i+1}'] = Bounds(lb = partitions[:, 0, i], ub = partitions[:, 1, i])
        
        x = np.array([]).reshape(0, self.dim)
        switch = False
        
        for i in range(splits):
            n = 0
            if x_init is not None and not switch:    
                if (x_init <= self.bounds_hyp[str(i+1)].ub).all()\
                    and (x_init >= self.bounds_hyp[str(i+1)].lb).all():
                        x0 = x_init.reshape(1, self.dim)
                        switch = True
                        n = 1
            if n == 0:
                x0 = np.random.uniform(self.bounds.lb,
                                       self.bounds.ub,
                                       (10, self.dim))                
                opt = Parallel(n_jobs = 1)(delayed(minimize)(intpts,
                                                             x_0,
                                                             method = 'L-BFGS-B',
                                                             bounds = self.bounds_hyp[f'{i+1}'],
                                                             tol = 1e-6)
                                           for x_0 in x0)
                x0s = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                x0 = x0s[np.argmin(funs)]
            x = np.vstack([x, x0.reshape(1, self.dim)])
        
        init_pts = int(len(x)/splits)
        splt = int(x.shape[0]/f_cores)
        x_bs = np.array(np.ones(f_cores), dtype = tuple)
        
        if f_cores == 1:
            x_bs[0] = x
        else:
            for i in range(f_cores-1):
                x_bs[i] = x[i*splt:(i+1)*splt, :]
                x_bs[-1] = x[(i+1)*splt:, :]
        
        start_f = time.time()
        y = Parallel(n_jobs = f_cores)(delayed(self.system)(self.descale(x_s), *self.args)
                                       for x_s in x_bs)
        end_f = time.time()
        self.time_fhyp[0] = end_f-start_f
        
        y = np.hstack(y[:]).T.reshape(-1,1)
        y_bst = min(y).reshape(-1,1)
        
        model_hyp = {}
        LCB = {}        
        for i in range(splits):
            model_hyp[str(i)] = gpr.GaussianProcessRegressor(self.kernel,
                                                            alpha = 1e-6,
                                                            n_restarts_optimizer = 10,
                                                            normalize_y = True)
            model_hyp[str(i)].fit(x[i::splits], y[i::splits])
            LCB[str(i)] = LCB_AF(model_hyp[str(i)], self.dim, self.exp_w, self.descale).LCB
        
        restarts = max(1, int(round(128/splits, 0)))
        x_nxt = np.ones((splits, self.dim))
        x_nxtbs = np.array(np.ones(f_cores), dtype = tuple)
        
        end = time.time()
        for i in range(init_pts):
            self.time_hyp[i] = (i+1)*(end-start)/init_pts
            self.time_fhyp[i] = (i+1)*(end_f-start_f)/init_pts
        
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_hyp[init_pts-1]:.1f}')
        
        for i in range(self.trials_hyp-init_pts):
            x0 = np.random.uniform(self.bounds.lb,
                                   self.bounds.ub,
                                   (restarts, self.dim))
            
            for j in range(splits):
                opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCB[str(j)],
                                                                    x_0,
                                                                    method = 'L-BFGS-B',
                                                                    bounds = self.bounds_hyp[str(j+1)])
                                                  for x_0 in x0)
                x_nxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                sts = np.array([res.success for res in opt])
                funs[np.where(sts == False)] = max(1e6, np.max(funs))
                x_nxt[j] = x_nxts[np.argmin(funs)]            
            
            if f_cores == 1:
                x_nxtbs[0] = x_nxt
            else:
                for j in range(f_cores-1):
                    x_nxtbs[j] = x_nxt[j*splt:(j+1)*splt, :]
                x_nxtbs[-1] = x_nxt[(j+1)*splt:, :]
                
            start_f = time.time()
            y_nxt =  Parallel(n_jobs = f_cores)(delayed(self.system) (self.descale(x_s), *self.args)
                                                for x_s in x_nxtbs)
            end_f = time.time()
            self.time_fhyp[i+init_pts] = self.time_fhyp[i+init_pts-1]+(end_f-start_f)
            
            x = np.vstack([x, x_nxt])
            y_nxt = np.hstack(y_nxt[:]).T.reshape(-1, 1)
            y = np.vstack([y, y_nxt])
            y_bst = np.vstack([y_bst, min(y_nxt).reshape(-1, 1)])
            
            for j in range(splits):
                model_hyp[str(j)].fit(x[j::splits], y[j::splits])
        
            end = time.time()
            self.time_hyp[i+init_pts] = end-start
            
            print('ITERATION COUNT IS AT 'f'{init_pts+i+1};\
                  TOTAL ELAPSED TIME: 'f'{self.time_hyp[i+init_pts]:.1f}')
            
        self.hsbo_optim = True
        self.model_hyp = model_hyp
        self.x_hyp = self.descale(x)
        self.y_hyp = y
        self.y_hypbst = y_bst
    
    
    def optimizer_exbo(self, trials, num_weights, lam = 1, f_cores = 1, af_cores = 1, x_init = None):
        
        """
        parallelization of BO based on the algorithm of Hutter et al. (2012);
        different values for the exploratory parameter are drawn from the
        expontial distribution with length scale parameter (lam) equal to 1;
        new samples are drawn up at every iteration and num_weights sets the
        number of samples (and parallel experiments) drawn
        """
        
        print('Variable κ BO run...')
        start = time.time()
        self.trials_expw = trials
        self.time_expw = np.zeros(self.trials_expw)
        self.time_fexpw = np.zeros(self.trials_expw)
        splits = num_weights
        kappas = np.random.exponential(lam, splits)
        
        if x_init is None:
            x = np.random.uniform(self.bounds.lb,
                                  self.bounds.ub,
                                  (splits, self.dim))
        else:
            x = x_init.reshape(-1, self.dim)
            if x.shape[0] < splits:
                x = np.vstack([x,
                               np.random.uniform(self.bounds.lb,
                                                 self.bounds.ub,
                                                 (splits-x.shape[0], self.dim))])
        
        start_f = time.time()
        y = Parallel(n_jobs = f_cores)(delayed(self.system)(self.descale(x_s), *self.args)
                                       for x_s in x)
        end_f = time.time()
        self.time_fexpw [0] = end_f-start_f
        
        y = np.hstack(y[:]).T.reshape(-1, 1)
        y_bst = min(y)
        
        model_expw = gpr.GaussianProcessRegressor(self.kernel,
                                                  alpha = 1e-6,
                                                  normalize_y = True,
                                                  n_restarts_optimizer = 10)
        model_expw.fit(x, y)
        LCB = LCB_AF(model_expw, self.dim, kappas, self.descale)
        
        init_pts = int(len(x)/splits)
        x_nxt = np.ones((splits, self.dim))
        restarts = max(1, int(round(128/splits, 0)))
        
        end = time.time()
        for i in range(init_pts):
            self.time_expw[i] = (i+1)*(end-start)/init_pts
            self.time_fexpw[i] = (i+1)*(end_f-start_f)/init_pts
        
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_expw[init_pts-1]:.1f}')
        
        for i in range(self.trials_expw-init_pts):
            x0 = np.random.uniform(self.bounds.lb,
                                   self.bounds.ub,
                                   (restarts, self.dim))
            
            for j in range(splits):
                LCB.exp_w = kappas[j]
                opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCB.LCB,
                                                                    x_0,
                                                                    method = 'L-BFGS-B',
                                                                    bounds = self.bounds)
                                                  for x_0 in x0)
                x_nxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                x_nxt[j] = x_nxts[np.argmin(funs)]
            
            start_f = time.time()
            y_nxt = Parallel(n_jobs = f_cores)(delayed(self.system)(self.descale(x_s), *self.args)
                                               for x_s in x_nxt)
            end_f = time.time()
            self.time_fexpw[i+init_pts] = self.time_fexpw[i+init_pts-1]+(end_f-start_f)
            
            x = np.vstack([x, x_nxt])
            y_nxt = np.hstack(y_nxt[:]).T.reshape(-1, 1)
            y = np.vstack([y, y_nxt])
            y_bst = np.vstack([y_bst, min(y_nxt)])
            
            model_expw.fit(x, y)
            kappas = np.random.exponential(lam, splits)
            
            end = time.time()
            self.time_expw[i+init_pts] = end-start
            
            print('ITERATION COUNT IS AT 'f'{init_pts+i+1};\
                  TOTAL ELAPSED TIME: 'f'{self.time_expw[i+init_pts]:.1f}')
        
        self.exbo_optim = True
        self.x_expw = self.descale(x)
        self.y_expw = y
        self.model_expw = model_expw
        self.y_expwbst = y_bst
    
    
    def optimizer_nmcbo(self, trials, parallel_exps, sample_num, f_cores = 1, af_cores = 1, x_init = None):
        
        """
        parallelization of BO based on N times Monte Carlo algorithm as described
        in Snoek et al. (2012); algorithm uses fantasy samples to generate a batch
        of experiments of size parallel_exps; generating fantasy points is done by
        sampling from the GP models sample_num times
        
        """
    
        print('N times Monte Carlo BO run...')
        start = time.time()
        self.trials_nmc = trials
        self.time_nmc = np.zeros(self.trials_nmc)
        self.time_fnmc = np.zeros(self.trials_nmc)
        splits = parallel_exps
        samples = int(sample_num)
        
        if x_init is None:
            x = np.random.uniform(self.bounds.lb,
                                  self.bounds.ub,
                                  (splits, self.dim))
        else:
            x = x_init.reshape(-1, self.dim)
            if x.shape[0] < splits:
                x = np.vstack([x,
                               np.random.uniform(self.bounds.lb,
                                                 self.bounds.ub,
                                                 (splits-x.shape[0], self.dim))])
                
        start_f = time.time()
        y = Parallel(n_jobs = f_cores)(delayed(self.system)(self.descale(x_s), *self.args)
                                       for x_s in x)
        end_f = time.time()
        self.time_fnmc[0] = end_f-start_f
        
        y = np.hstack(y[:]).T.reshape(-1, 1)
        y_bst = min(y)
        
        model_nmc = gpr.GaussianProcessRegressor(self.kernel,
                                                 alpha = 1e-6,
                                                 normalize_y = True,
                                                 n_restarts_optimizer = 10)
        model_nmc.fit(x, y)
        LCB = LCB_AF(model_nmc, self.dim, self.exp_w, self.descale).LCB
        
        init_pts = int(len(x)/splits)
        x_nxt = np.ones((splits, self.dim))
        restarts = max(1, int(round(128/splits, 0)))
        
        end = time.time()
        for i in range(init_pts):
            self.time_nmc[i] = (i+1)*(end-start)/init_pts
            self.time_fnmc[i] = (i+1)*(end_f-start_f)/init_pts
        
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_nmc[init_pts-1]:.1f}')
        
        
        for i in range(self.trials_nmc-init_pts):
            x0 = np.random.uniform(self.bounds.lb,
                                   self.bounds.ub,
                                   (restarts, self.dim))
            
            opt = Parallel(n_jobs = 1)(delayed(minimize)(LCB,
                                                         x_0,
                                                         method = 'L-BFGS-B',
                                                         bounds = self.bounds)
                                       for x_0 in x0)
            x_nxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            x_nxt[0] = x_nxts[np.argmin(funs)]
            
            for k in range(1, splits):
                x_s = x_nxt[:k]
                y_s = model_nmc.sample_y(x_s, n_samples = samples, random_state = np.random.randint(0, 1e6))
                y_s = np.vstack(y_s[:]).T
                x_f = np.vstack([x, x_s])
                
                fant_mod = {}
                LCB_fant = {}
                for j in range(y_s.shape[0]):
                    y_f = np.vstack([y, y_s[j].reshape(-1, 1)])
                    fant_mod[str(j+1)] = gpr.GaussianProcessRegressor(self.kernel,
                                                                      alpha = 1e-6,
                                                                      normalize_y = True,
                                                                      n_restarts_optimizer = 10)
                    fant_mod[str(j+1)].fit(x_f, y_f)
                    LCB_fant[str(j+1)] = LCB_AF(fant_mod[str(j+1)], self.dim, self.exp_w, self.descale)
                
                opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCB_nmc,
                                                                    x_0,
                                                                    method = 'L-BFGS-B',
                                                                    bounds = self.bounds,
                                                                    args = (LCB_fant, y_s))
                                                  for x_0 in x0)
                x_nxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                x_nxt[k] = x_nxts[np.argmin(funs)]
            
            start_f = time.time()
            y_nxt = Parallel(n_jobs = f_cores)(delayed(self.system) (self.descale(x_s), *self.args)
                                             for x_s in x_nxt)
            end_f = time.time()
            self.time_fnmc[i+init_pts] = self.time_fnmc[i+init_pts-1]+(end_f-start_f)
            
            x = np.vstack([x, x_nxt])
            y_nxt = np.hstack(y_nxt[:]).T.reshape(-1, 1)
            y = np.vstack([y, y_nxt])
            y_bst = np.vstack([y_bst, min(y_nxt)])
            model_nmc.fit(x, y)
                
            end = time.time()
            self.time_nmc[i+init_pts] = end-start
            
            print('ITERATION COUNT IS AT 'f'{init_pts+i+1};\
                  TOTAL ELAPSED TIME: 'f'{self.time_nmc[i+init_pts]:.1f}')
                  
        self.nmcbo_optim = True
        self.x_nmc = self.descale(x)
        self.y_nmc = y
        self.model_nmc = model_nmc
        self.y_nmcbst = y_bst


    def optimizer_qBO(self, trials, q, n_samps, f_cores = 1, af_cores = 1, x_init = None):
        
        """
        parallelization of BO based on q-LCB AF proposed in Wislon et al.(2017);
        a batch of q experiments is run in parallel by calculating the joint value
        of entire batch of experiments; analytical expression for resulting AF is
        not available and Monte Carlo is used to estimate its value using n_samps
        
        """
        
        print('q-BO Run...')
        start = time.time()
        self.trials_qbo = trials
        self.time_qbo = np.ones(self.trials_qbo)
        self.time_fqbo = np.ones(self.trials_qbo)
        
        if x_init is None:
            x = np.random.uniform(self.bounds.lb,
                                  self.bounds.ub,
                                  (q, self.dim))
        else:
            x = x_init.reshape(-1, self.dim)
            if x.shape[0] < q:
                x = np.vstack([x,
                               np.random.uniform(self.bounds.lb,
                                                 self.bounds.ub,
                                                 (q-x.shape[0], self.dim))])
        
        init_pts = int(len(x)/q)
        splt = int(x.shape[0]/f_cores)
        x_bs = np.array(np.ones(f_cores), dtype = tuple)
        
        if f_cores == 1:
            x_bs[0] = x
        else:
            for i in range(f_cores-1):
                x_bs[i] = x[i*splt:(i+1)*splt, :]
            x_bs[-1] = x[(i+1)*splt:, :]
            
        start_f = time.time()
        y = Parallel(n_jobs = f_cores)(delayed(self.system)(self.descale(x_s), *self.args)
                                       for x_s in x_bs)
        end_f = time.time()
        self.time_fqbo[0] = end_f-start_f
        
        y = np.hstack(y[:]).T.reshape(-1,1)
        y_bst = min(y).reshape(-1,1)
        
        model_qbo = gpr.GaussianProcessRegressor(self.kernel,
                                                 alpha = 1e-6,
                                                 normalize_y = True,
                                                 n_restarts_optimizer = 10)
        model_qbo.fit(x, y)
        LCB_qBO = qLCB(model_qbo, q, self.dim, self.exp_w, n_samps).LCB
            
        restarts = max(1, int(round(128/q, 0)))
        x_nxtbs = np.array(np.ones(f_cores), dtype = tuple)
        bnds_qBO = Bounds(np.zeros((q*self.dim)), np.ones((q*self.dim)))
        
        end = time.time()
        for i in range(init_pts):
            self.time_qbo[i] = (i+1)*(end-start)/init_pts
            self.time_fqbo[i] = (i+1)*(end_f-start_f)/init_pts
        
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_qbo[init_pts-1]:.1f}')
        
        for i in range(self.trials_qbo-init_pts):
            x0 = np.random.uniform(np.array([self.bounds.lb]*q).flatten(),
                                   np.array([self.bounds.ub]*q).flatten(),
                                   (restarts, self.dim*q))
            
            opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCB_qBO,
                                                                x_0,
                                                                method = 'L-BFGS-B',
                                                                bounds = bnds_qBO)
                                              for x_0 in x0)
            x_nxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            x_nxt = x_nxts[np.argmin(funs)].reshape(q, self.dim)
            
            if f_cores == 1:
                x_nxtbs[0] = x_nxt
            else:
                for j in range(f_cores-1):
                    x_nxtbs[j] = x_nxt[j*splt:(j+1)*splt, :]
                x_nxtbs[-1] = x_nxt[(j+1)*splt:, :]
            
            start_f = time.time()
            y_nxt = Parallel(n_jobs = f_cores)(delayed(self.system)(self.descale(x_s), *self.args)
                                               for x_s in x_nxtbs)
            end_f = time.time()
            self.time_fqbo[i+init_pts] = self.time_fqbo[i+init_pts-1]+(end_f-start_f)
            
            x = np.vstack([x, x_nxt])
            y_nxt = np.hstack(y_nxt[:]).T.reshape(-1,1)
            y = np.vstack([y, y_nxt])
            y_bst = np.vstack([y_bst, min(y_nxt).reshape(-1, 1)])
            
            model_qbo.fit(x, y)
            
            end = time.time()
            self.time_qbo[i+init_pts] = end-start
            
            print('ITERATION COUNT IS AT 'f'{init_pts+i+1};\
                  TOTAL ELAPSED TIME: 'f'{self.time_qbo[i+init_pts]:.1f}')
            
        self.qbo_optim = True
        self.model_qbo = model_qbo
        self.x_qbo = self.descale(x)
        self.y_qbo = y
        self.y_qbobst = y_bst
    
    
    def optimizer_vpbo(self, trials, split_num, lim_init,
                         f_cores = 1, af_cores = 1, ref_cores = 1, x_init  = None):
        
        """
        > split_number is the number of splits the input space is being separated into
        
        > lim_init is the initial point that is being used to sample (minimum of the reference model)
          and these serve as the value limits along the axes that are NOT being optimized
          in each respective partition; it is updated as the algorithm progresses
          
        > lim_init should be the same size as x and it can be randomly selected if not available
        
        > f_cores and af_cores are the cores used for sampling f and optimizing the AF
        
        > if reference model is not available, make them functions that return 0
         
        > x_init is an array containing the intial points at which to sample
        """
        
        # Split partition using variables as split point
        print('Variable Partitioned BO run...')
        start = time.time()
        self.trials_vp = trials
        splits = split_num
        self.time_vp = np.zeros(self.trials_vp)
        self.time_fvp = np.zeros(self.trials_vp)
        div = int(self.dim/splits)        
        ref_mod = self.dist_ref['distrefmod']
        
        x = lim_init*np.ones((splits, self.dim))
        lwr = x.copy()
        upr = x.copy()+1e-6
        for i in range(splits):
            if x_init is None:
                x[i, i*div:(i+1)*div] = np.random.uniform(self.bounds.lb,
                                                          self.bounds.ub,
                                                          (1, div))
            else:
                x_init = x_init.reshape(1, self.dim)
                x[i, i*div:(i+1)*div] = x_init[0, i*div:(i+1)*div]
            lwr[i, i*div:(i+1)*div] = self.bounds.lb[i]
            upr[i, i*div:(i+1)*div] = self.bounds.ub[i]
        x = np.vstack([x, lim_init])
        
        init_pts = int(len(x)/splits)
        splt = int(x.shape[0]/f_cores)
        x_bs = np.array(np.ones(f_cores), dtype = tuple)
        
        if f_cores == 1:
            x_bs[0] = x
        else:
            for i in range(f_cores-1):
                x_bs[i] = x[i*splt:(i+1)*splt, :]
            x_bs[-1] = x[(i+1)*splt:, :]
        
        start_f = time.time()
        y = Parallel(n_jobs = f_cores)(delayed(self.distmod)(self.descale(x_s), *self.args)
                                       for x_s in x_bs)
        if str(type(ref_mod))=="<class '__main__.Network'>":
            y_ref = Parallel(n_jobs = ref_cores)(delayed(ref_mod)(x_s, *self.ref_args)
                                                 for x_s in torch.from_numpy(x).float())
            y_ref = torch.hstack(y_ref[:]).T.reshape(-1, 1).data.numpy()
            end_f = time.time()
        else:
            y_ref = Parallel(n_jobs = ref_cores)(delayed(ref_mod)(self.descale(x_s), *self.ref_args)
                                                 for x_s in x_bs)
            y_ref = np.vstack(y_ref[:])
            end_f = time.time()            
        self.time_fvp[0] = end_f-start_f
        
        y = np.vstack(y[:])
        eps = y-y_ref
        y_bst = np.min(y, axis = 0).reshape(-1, 1).T
        
        bnds_var = {}
        model_vp = {}
        LCB = {}
        
        for i in range(splits):
            model_vp[str(i+1)] = gpr.GaussianProcessRegressor(self.kernel,
                                                              alpha = 1e-6,
                                                              n_restarts_optimizer = 10,
                                                              normalize_y = True)
            model_vp[str(i+1)].fit(x, eps[:, i])

            lwr[i] = x[np.argmin(y[:, i])]
            lwr[i, i*div:(i+1)*div] = self.bounds.lb[i]
            upr[i] = x[np.argmin(y[:, i])]+1e-6
            upr[i, i*div:(i+1)*div] = self.bounds.ub[i]
            bnds_var[str(i+1)] = Bounds(lwr[i], upr[i])
            
            LCB[str(i+1)] = LCB_AF(model_vp[str(i+1)],
                                   self.dim,
                                   self.exp_w,
                                   self.descale,
                                   self.dist_ref['distrefmod'+str(i+1)],
                                   self.ref_args).LCB
        
        restarts = int(round(128/(splits+1), 0))
        x_nxt = x.copy()
        x_nxtbs = np.array(np.ones(f_cores), dtype = tuple)
        
        end = time.time()
        for i in range(init_pts):
            self.time_vp[i] = (i+1)*(end-start)/init_pts
            self.time_fvp[i] = (i+1)*(end_f-start_f)/init_pts
        
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_vp[init_pts-1]:.1f}')
        
        end = time.time()
        self.time_vp[0] = end-start
        
        for i in range(self.trials_vp-init_pts):
            x0 = np.random.uniform(self.bounds.lb,
                                   self.bounds.ub,
                                   (restarts, self.dim))
            
            for j in range(splits):
                opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCB[str(j+1)],
                                                                    x_0,
                                                                    method = 'L-BFGS-B',
                                                                    bounds = bnds_var[str(j+1)])
                                                  for x_0 in x0)
                x_nxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                x_nxt[j] = x_nxts[np.argmin(funs)]
                x_nxt[-1, j*div:(j+1)*div] = x_nxts[np.argmin(funs), j*div:(j+1)*div]
            
            if f_cores == 1:
                x_nxtbs[0] = x_nxt
            else:
                for j in range(f_cores-1):
                    x_nxtbs[j] = x_nxt[j*splt:(j+1)*splt, :]
                x_nxtbs[-1] = x_nxt[(j+1)*splt:, :]
            
            start_f = time.time()
            y_nxt = Parallel(n_jobs = f_cores)(delayed(self.distmod)(self.descale(x_s), *self.args)
                                             for x_s in x_nxtbs)
            if str(type(ref_mod))=="<class '__main__.Network'>":
                y_ref = Parallel(n_jobs = ref_cores)(delayed(ref_mod)(x_s, *self.ref_args)
                                                 for x_s in torch.from_numpy(x).float())
                y_ref = torch.hstack(y_ref[:]).T.reshape(-1, 1).data.numpy()
                end_f = time.time()
            else:
                y_ref = Parallel(n_jobs = ref_cores)(delayed(ref_mod)(self.descale(x_s), *self.ref_args)
                                                 for x_s in x_nxtbs)
                y_ref = np.vstack(y_ref[:])
                end_f = time.time()
            self.time_fvp[i+init_pts] = self.time_fvp[i+init_pts-1]+(end_f-start_f)
            
            y_nxt = np.vstack(y_nxt[:])

            for j in range(splits):
                if any(y_nxt[:, j] < min(y[:, j])):
                    lwr[j] = x_nxt[np.argmin(y_nxt[:, j])]
                    lwr[j, j*div:(j+1)*div] = self.bounds.lb[j]
                    upr[j] = x_nxt[np.argmin(y_nxt[:, j])]+1e-6
                    upr[j, j*div:(j+1)*div] = self.bounds.ub[j]
                    
            x = np.vstack([x, x_nxt])
            y = np.vstack([y, y_nxt])
            eps_nxt = y_nxt-y_ref
            eps = np.vstack([eps, eps_nxt])
            y_bst = np.vstack([y_bst, np.min(y_nxt, axis = 0).reshape(-1,1).T])
                
            for j in range(splits):
                model_vp[str(j+1)].fit(x, eps[:, j])
                bnds_var[str(j+1)] = Bounds(lwr[j], upr[j])

            end = time.time()
            self.time_vp[i+init_pts] = end-start
            
            print('ITERATION COUNT IS AT 'f'{init_pts+i+1};\
                  TOTAL ELAPSED TIME: 'f'{self.time_vp[i+init_pts]:.1f}')
            
        self.vpbo_optim = True
        self.model_vp = model_vp
        self.x_vp = self.descale(x)
        self.y_vp = y
        self.y_vpbst = y_bst
    
    
    def plots(self, figure_name):
        itr = np.linspace(1, self.trials_sbo, self.trials_sbo)
        ylim_l = min(self.y_sbo)-0.01*abs(min(self.y_sbo))
        ylim_u = max(self.y_sbo)+0.01*abs(max(self.y_sbo))
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlim((1, self.trials_sbo))
        ax1.set_xlabel('Sample Number', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel('Operating cost (MM USD/yr)', fontsize = 24)
        pyp.yticks(fontsize = 24);
        ax1.scatter(itr, self.y_sbo, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200);
        ax1.plot(itr, self.y_sbo, color = 'black', linewidth = 3, label = 'S-BO');
        
        if self.refbo_optim:
            itr = np.linspace(1, self.trials_ref, self.trials_ref)
            ax1.scatter(itr, self.y_ref, marker = 'o', color = 'white', edgecolor = 'blue',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.y_ref, color = 'blue', linewidth = 3, label = 'Ref-BO');
            ylim_l = min(ylim_l, min(self.y_ref)-0.01*abs(min(self.y_ref)))
            ylim_u = max(ylim_u, max(self.y_ref)+0.01*abs(max(self.y_ref)))
        
        if self.lsbo_optim:
            itr = np.linspace(1, self.trials_ls, self.trials_ls)
            ax1.scatter(itr, self.y_lsbst, marker = 'o', color = 'white', edgecolor = 'gray',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.y_lsbst, color = 'gray', linewidth = 3, label = 'LS-BO');
            ylim_l = min(ylim_l, min(self.y_lsbst)-0.01*abs(min(self.y_lsbst)))
            ylim_u = max(ylim_u, max(self.y_lsbst)+0.01*abs(max(self.y_lsbst)))
        
        if self.hsbo_optim:
            itr = np.linspace(1, self.trials_hyp, self.trials_hyp)
            ax1.scatter(itr, self.y_hypbst, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.y_hypbst, color = 'green', linewidth = 3, label = 'HS-BO');
            ylim_l = min(ylim_l, min(self.y_hypbst)-0.01*abs(min(self.y_hypbst)))
            ylim_u = max(ylim_u, max(self.y_hypbst)+0.01*abs(max(self.y_hypbst)))
        
        if self.exbo_optim:
            itr = np.linspace(1, self.trials_expw, self.trials_expw)
            ax1.scatter(itr, self.y_expwbst, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.y_expwbst, color = 'red', linewidth = 3, label = 'HP-BO');
            ylim_l = min(ylim_l, min(self.y_expwbst)-0.01*abs(min(self.y_expwbst)))
            ylim_u = max(ylim_u, max(self.y_expwbst)+0.01*abs(max(self.y_expwbst)))
        
        if self.nmcbo_optim:
            itr = np.linspace(1, self.trials_nmc, self.trials_nmc)
            ax1.scatter(itr, self.y_nmcbst, marker = 'o', color = 'white', edgecolor = 'pink',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.y_nmcbst, color = 'pink', linewidth = 3, label = 'MC-BO');
            ylim_l = min(ylim_l, min(self.y_nmcbst)-0.01*abs(min(self.y_nmcbst)))
            ylim_u = max(ylim_u, max(self.y_nmcbst)+0.01*abs(max(self.y_nmcbst)))
        
        if self.qbo_optim:
            itr = np.linspace(1, self.trials_qbo, self.trials_qbo)
            ax1.scatter(itr, self.y_qbobst, marker = 'o', color = 'white', edgecolor = 'gold',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.y_qbobst, color = 'gold', linewidth = 3, label = 'q-BO');
            ylim_l = min(ylim_l, min(self.y_qbobst)-0.01*abs(min(self.y_qbobst)))
            ylim_u = max(ylim_u, max(self.y_qbobst)+0.01*abs(max(self.y_qbobst)))
        
        if self.vpbo_optim:
            itr = np.linspace(1, self.trials_vp, self.trials_vp)
            ax1.scatter(itr, self.y_vpbst[:, -1], marker = 'o', color = 'white', edgecolor = 'brown',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.y_vpbst[:, -1], color = 'brown', linewidth = 3, label = 'VP-BO');
            ylim_l = min(ylim_l, min(self.y_vpbst[:, -1])-0.01*abs(min(self.y_vpbst[:, -1])))
            ylim_u = max(ylim_u, max(self.y_vpbst[:, -1])+0.01*abs(max(self.y_vpbst[:, -1])))
        
        ax1.set_ylim(ylim_l, ylim_u)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);
        
        
    def plots_time(self, figure_name):
        ylim_l = min(self.y_sbo)-0.01*abs(min(self.y_sbo))
        ylim_u = max(self.y_sbo)+0.01*abs(max(self.y_sbo))
        xlim_u = round(self.time_sbo[-1]+1, 0)
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlabel('Time (s)', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel('Operating cost (MM USD/hr)', fontsize = 24)
        pyp.yticks(fontsize = 24);
        ax1.scatter(self.time_sbo, self.y_sbo, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200);
        ax1.plot(self.time_sbo, self.y_sbo, color = 'black', linewidth = 3, label = 'S-BO');
        
        if self.refbo_optim:
            ax1.scatter(self.time_ref, self.y_ref, marker = 'o', color = 'white', edgecolor = 'blue',
                    zorder = 3, s = 200);
            ax1.plot(self.time_ref, self.y_ref, color = 'blue', linewidth = 3, label = 'Ref-BO');
            ylim_l = min(ylim_l, min(self.y_ref)-0.01*abs(min(self.y_ref)))
            ylim_u = max(ylim_u, max(self.y_ref)+0.01*abs(max(self.y_ref)))
            xlim_u = max(xlim_u, self.time_ref[-1]+1)
        
        if self.lsbo_optim:
            ax1.scatter(self.time_ls, self.y_lsbst, marker = 'o', color = 'white', edgecolor = 'gray',
                    zorder = 3, s = 200);
            ax1.plot(self.time_ls, self.y_lsbst, color = 'gray', linewidth = 3, label = 'LS-BO');
            ylim_l = min(ylim_l, min(self.y_lsbst)-0.01*abs(min(self.y_lsbst)))
            ylim_u = max(ylim_u, max(self.y_lsbst)+0.01*abs(max(self.y_lsbst)))
            xlim_u = max(xlim_u, self.time_ls[-1]+1)
        
        if self.hsbo_optim:
            ax1.scatter(self.time_hyp, self.y_hypbst, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(self.time_hyp, self.y_hypbst, color = 'green', linewidth = 3, label = 'HS-BO');
            ylim_l = min(ylim_l, min(self.y_hypbst)-0.01*abs(min(self.y_hypbst)))
            ylim_u = max(ylim_u, max(self.y_hypbst)+0.01*abs(max(self.y_hypbst)))
            xlim_u = max(xlim_u, self.time_hyp[-1]+1)
        
        if self.exbo_optim:
            ax1.scatter(self.time_expw, self.y_expwbst, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(self.time_expw, self.y_expwbst, color = 'red', linewidth = 3, label = 'HP-BO');
            ylim_l = min(ylim_l, min(self.y_expwbst)-0.01*abs(min(self.y_expwbst)))
            ylim_u = max(ylim_u, max(self.y_expwbst)+0.01*abs(max(self.y_expwbst)))
            xlim_u = max(xlim_u, self.time_expw[-1]+1)
        
        if self.nmcbo_optim:
            ax1.scatter(self.time_nmc, self.y_nmcbst, marker = 'o', color = 'white', edgecolor = 'pink',
                    zorder = 3, s = 200);
            ax1.plot(self.time_nmc, self.y_nmcbst, color = 'pink', linewidth = 3, label = 'MC-BO');
            ylim_l = min(ylim_l, min(self.y_nmcbst)-0.01*abs(min(self.y_nmcbst)))
            ylim_u = max(ylim_u, max(self.y_nmcbst)+0.01*abs(max(self.y_nmcbst)))
            xlim_u = max(xlim_u, self.time_nmc[-1]+1)
        
        if self.qbo_optim:
            ax1.scatter(self.time_qbo, self.y_qbobst, marker = 'o', color = 'white', edgecolor = 'gold',
                    zorder = 3, s = 200);
            ax1.plot(self.time_qbo, self.y_qbobst, color = 'gold', linewidth = 3, label = 'q-BO');
            ylim_l = min(ylim_l, min(self.y_qbobst)-0.01*abs(min(self.y_qbobst)))
            ylim_u = max(ylim_u, max(self.y_qbobst)+0.01*abs(max(self.y_qbobst)))
            xlim_u = max(xlim_u, self.time_qbo[-1]+1)
        
        if self.vpbo_optim:
            ax1.scatter(self.time_vp, self.y_vpbst[:, -1], marker = 'o', color = 'white', edgecolor = 'brown',
                    zorder = 3, s = 200);
            ax1.plot(self.time_vp, self.y_vpbst[:, -1], color = 'brown', linewidth = 3, label = 'VP-BO');
            ylim_l = min(ylim_l, min(self.y_vpbst[:, -1])-0.01*abs(min(self.y_vpbst[:, -1])))
            ylim_u = max(ylim_u, max(self.y_vpbst[:, -1])+0.01*abs(max(self.y_vpbst[:, -1])))
            xlim_u = max(xlim_u, self.time_vp[-1]+1)
                
        ax1.set_ylim(ylim_l, ylim_u)
        ax1.set_xlim(0, xlim_u)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'_time.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);
        
        
    def plot_exptime(self, figure_name):
        ylim_l = min(self.y_sbo)-0.01*abs(min(self.y_sbo))
        ylim_u = max(self.y_sbo)+0.01*abs(max(self.y_sbo))
        xlim_u = round(self.time_fsbo[-1]+1, 0)
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlabel('Time (s)', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel('Operating cost (MM USD/hr)', fontsize = 24)
        pyp.yticks(fontsize = 24);
        ax1.scatter(self.time_fsbo, self.y_sbo, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200);
        ax1.plot(self.time_fsbo, self.y_sbo, color = 'black', linewidth = 3, label = 'S-BO');
        
        if self.refbo_optim:
            ax1.scatter(self.time_fref, self.y_ref, marker = 'o', color = 'white', edgecolor = 'blue',
                    zorder = 3, s = 200);
            ax1.plot(self.time_fref, self.y_ref, color = 'blue', linewidth = 3, label = 'Ref-BO');
            ylim_l = min(ylim_l, min(self.y_ref)-0.01*abs(min(self.y_ref)))
            ylim_u = max(ylim_u, max(self.y_ref)+0.01*abs(max(self.y_ref)))
            xlim_u = max(xlim_u, self.time_fref[-1]+1)
        
        if self.lsbo_optim:
            ax1.scatter(self.time_fls, self.y_lsbst, marker = 'o', color = 'white', edgecolor = 'gray',
                    zorder = 3, s = 200);
            ax1.plot(self.time_fls, self.y_lsbst, color = 'gray', linewidth = 3, label = 'LS-BO');
            ylim_l = min(ylim_l, min(self.y_lsbst)-0.01*abs(min(self.y_lsbst)))
            ylim_u = max(ylim_u, max(self.y_lsbst)+0.01*abs(max(self.y_lsbst)))
            xlim_u = max(xlim_u, self.time_fls[-1]+1)
        
        if self.hsbo_optim:
            ax1.scatter(self.time_fhyp, self.y_hypbst, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(self.time_fhyp, self.y_hypbst, color = 'green', linewidth = 3, label = 'HS-BO');
            ylim_l = min(ylim_l, min(self.y_hypbst)-0.01*abs(min(self.y_hypbst)))
            ylim_u = max(ylim_u, max(self.y_hypbst)+0.01*abs(max(self.y_hypbst)))
            xlim_u = max(xlim_u, self.time_fhyp[-1]+1)
        
        if self.exbo_optim:
            ax1.scatter(self.time_fexpw, self.y_expwbst, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(self.time_fexpw, self.y_expwbst, color = 'red', linewidth = 3, label = 'HP-BO');
            ylim_l = min(ylim_l, min(self.y_expwbst)-0.01*abs(min(self.y_expwbst)))
            ylim_u = max(ylim_u, max(self.y_expwbst)+0.01*abs(max(self.y_expwbst)))
            xlim_u = max(xlim_u, self.time_fexpw[-1]+1)
        
        if self.nmcbo_optim:
            ax1.scatter(self.time_fnmc, self.y_nmcbst, marker = 'o', color = 'white', edgecolor = 'pink',
                    zorder = 3, s = 200);
            ax1.plot(self.time_fnmc, self.y_nmcbst, color = 'pink', linewidth = 3, label = 'MC-BO');
            ylim_l = min(ylim_l, min(self.y_nmcbst)-0.01*abs(min(self.y_nmcbst)))
            ylim_u = max(ylim_u, max(self.y_nmcbst)+0.01*abs(max(self.y_nmcbst)))
            xlim_u = max(xlim_u, self.time_fnmc[-1]+1)
        
        if self.qbo_optim:
            ax1.scatter(self.time_fqbo, self.y_qbobst, marker = 'o', color = 'white', edgecolor = 'gold',
                    zorder = 3, s = 200);
            ax1.plot(self.time_fqbo, self.y_qbobst, color = 'gold', linewidth = 3, label = 'q-BO');
            ylim_l = min(ylim_l, min(self.y_qbobst)-0.01*abs(min(self.y_qbobst)))
            ylim_u = max(ylim_u, max(self.y_qbobst)+0.01*abs(max(self.y_qbobst)))
            xlim_u = max(xlim_u, self.time_fqbo[-1]+1)
        
        if self.vpbo_optim:
            ax1.scatter(self.time_fvp, self.y_vpbst[:, -1], marker = 'o', color = 'white', edgecolor = 'brown',
                    zorder = 3, s = 200);
            ax1.plot(self.time_fvp, self.y_vpbst[:, -1], color = 'brown', linewidth = 3, label = 'VP-BO');
            ylim_l = min(ylim_l, min(self.y_vpbst[:, -1])-0.01*abs(min(self.y_vpbst[:, -1])))
            ylim_u = max(ylim_u, max(self.y_vpbst[:, -1])+0.01*abs(max(self.y_vpbst[:, -1])))
            xlim_u = max(xlim_u, self.time_fvp[-1]+1)
                
        ax1.set_ylim(ylim_l, ylim_u)
        ax1.set_xlim(0, xlim_u)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'_exptime.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);

import numpy as np
from matplotlib import pyplot as pyp, cm
from scipy.optimize import minimize, Bounds, approx_fprime
from scipy.misc import derivative
from joblib import Parallel, delayed
import sklearn.gaussian_process as gpr
from collections import OrderedDict
import torch
import time

class LCB_AF():
    def __init__(self, model, dim, exp_w, descale, **refmod):
        self.model = model
        self.dim = dim
        self.exp_w = exp_w
        self.descale = descale
        if refmod:
            self.refmod = refmod['refmod']
        else:
            def zr(x):
                return 0
            self.refmod = zr
    def LCB(self, x):
        x = np.array([x]).reshape(-1,1);
        x = x.reshape(int(x.shape[0]/self.dim), self.dim)
        mu, std = self.model.predict(x, return_std=True);
        mu = mu.flatten()
        if str(type(self.refmod))=="<class '__main__.Network'>":
            yref = self.refmod(torch.from_numpy(x).float()).data.numpy()  
        else:
            yref = self.refmod(self.descale(x))
        return (yref+mu-self.exp_w*std).flatten()

class LCB_EMBD():
    def __init__(self, model, var_num, dim, exp_w, fun, descale, include_x, **refmod):
        self.model = model
        self.var_num = var_num
        self.dim = dim
        self.exp_w = exp_w
        self.fun = fun
        self.descale = descale
        self.include_x = include_x
        if refmod:
            self.refmod = refmod['refmod']
        else:
            def zr(x):
                return np.zeros(x.shape)
            self.refmod = zr
    def LCB(self, x):
        x = x.reshape(-1, 1)
        x = x.reshape((int(x.shape[0]/self.dim), self.dim))
        mu = np.ones((x.shape[0], self.var_num))
        std = mu.copy()
        b = np.ones((x.shape[0], self.var_num+1))
        sigma = mu.copy()
        yref = self.refmod(self.descale(x))
        for i in range(self.var_num):
            mu[:, i], std[:, i] = self.model[str(i+1)].predict(x, return_std = True)
        eps = (yref+mu)*1e-3
        y0 = yref+mu+eps
        if self.include_x:
            y0 = np.hstack([y0, self.descale(x)])
        fp = np.ones(y0.shape)
        for i in range(x.shape[0]):
            fp[i] = approx_fprime(y0[i], self.fun, eps[i])
        b[:, 0] = self.fun(y0)
        b[:, 1:] = -fp*y0
        sigma = (fp**2*std**2)**0.5
        MU = np.sum(b, axis = 1)+np.sum(fp*mu, axis = 1)
        SIGMA = np.sum(sigma**2, axis = 1)**0.5
        return MU-self.exp_w*SIGMA

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

class BO():
    def __init__(self, ub, lb, dim, exp_w, kernel, system, bounds, **aux_mods):
        self.ub = ub
        self.lb = lb
        self.dim = dim
        self.exp_w = exp_w
        self.kernel = kernel
        self.system = system
        self.bounds = bounds
        self.dist_ref = {}
        if aux_mods:
            self.refmod = {'refmod': list(aux_mods.values())[0]}
            if len(aux_mods) > 1:
                self.distmod = aux_mods['distmod']
                self.dist_ref['distrefmod'] = aux_mods['ref_distmod']
                for i in range(3,len(aux_mods)):
                    self.dist_ref['distrefmod'+str(i-2)] = aux_mods['ref_distmod'+str(i-2)]
                self.dist_ref = OrderedDict(self.dist_ref)
                    

    def descale(self, x, scaling_factor = 1):
        # b = (self.ub+self.lb)/2
        # m = (b-self.lb)/scaling_factor
        m = (self.ub-self.lb)/scaling_factor
        b = self.lb
        return m*x+b
    
    def scale(self, x, scaling_factor = 1):
        # m = 2*scaling_factor/(self.ub-self.lb)
        # b = scaling_factor-m*self.ub
        m = scaling_factor/(self.ub-self.lb)
        b = -self.lb/(self.ub-self.lb)
        return m*x+b
    
    def optimizergp(self, trials, scaling_factor, cores = 4, xinit = None):
        print('Vanilla BO Run...')
        start = time.time()
        self.trialsgp = trials
        self.timegp = np.ones(self.trialsgp+1)
        self.timefgp = np.ones(self.trialsgp+1)
        sf = scaling_factor
        if xinit is None:
            # x = np.random.uniform(-sf, sf, (1, self.dim))
            x = np.random.uniform(0, sf, (1, self.dim))
        else:
            x = xinit.reshape(1, self.dim)
        startf = time.time()
        y = self.system(self.descale(x))
        endf = time.time()
        self.timefgp[0] = endf-startf
        modelgp = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, 
                                                  normalize_y = True,
                                                  n_restarts_optimizer = 10)
        modelgp.fit(x, y)
        # x0 = np.random.uniform(-sf, sf, (100, self.dim))
        x0 = np.random.uniform(0, sf, (128, self.dim))
        LCBgp = LCB_AF(modelgp, self.dim, self.exp_w, self.descale).LCB
        opt = Parallel(n_jobs = cores)(delayed(minimize)(LCBgp, x0 = start_point,
                                                      method = 'L-BFGS-B',
                                                      bounds = self.bounds)
                                    for start_point in x0)
        xnxts = np.array([res.x for res in opt], dtype  = 'float')
        funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        end = time.time()
        self.timegp[0] = end-start
        for i in range(self.trialsgp):
            xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
            startf = time.time()
            ynxt = self.system(self.descale(xnxt))
            endf = time.time()
            self.timefgp[i+1] = self.timefgp[i]+(endf-startf)
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            modelgp.fit(x, y)
            # x0 = np.random.uniform(-sf, sf, (100, self.dim))
            x0 = np.random.uniform(0, sf, (128, self.dim))
            opt = Parallel(n_jobs = cores)(delayed(minimize)(LCBgp, x0 = start_point,
                                                      method = 'L-BFGS-B',
                                                      bounds = self.bounds)
                                    for start_point in x0)
            xnxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            end = time.time()
            self.timegp[i+1] = end-start
        self.modelgp = modelgp
        self.xgp = self.descale(x)
        self.ygp = y
        self.ref_optim = False
        self.dist_optim = False
        self.distref_optim = False
        self.splt_optim = False
        self.splt_optimref = False
        self.spltvar_optim = False
        self.hyp_optim = False
        self.expwl_optim = False
        self.nmc_optim = False
        self.embd_optim = False
        
    def optimizeref(self, trials, scaling_factor, cores = 4, xinit = None):
        print('BO with Reference Model Run...')
        start = time.time()
        self.trialsref = trials
        self.timeref = np.ones(self.trialsref+1)
        self.timefref = np.ones(self.trialsref+1)
        refmod = self.refmod['refmod']
        sf = scaling_factor
        if xinit is None:
            # x = np.random.uniform(-sf, sf, (1, self.dim))
            x = np.random.uniform(0, sf, (1, self.dim))
        else:
            x = xinit.reshape(1, self.dim)
        startf = time.time()
        y = self.system(self.descale(x))
        if str(type(refmod))=="<class '__main__.Network'>":
            eps = y - refmod(torch.from_numpy(x).float()).data.numpy()
        else:
            eps = y - refmod(self.descale(x))
        endf = time.time()
        self.timefref[0] = endf-startf
        self.modeleps = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, 
                                                  normalize_y = True,
                                                  n_restarts_optimizer = 10)
        self.modeleps.fit(x, eps)
        # x0 = np.random.uniform(-sf, sf, (100, self.dim))
        x0 = np.random.uniform(0, sf, (128, self.dim))
        LCBref = LCB_AF(self.modeleps, self.dim, self.exp_w, self.descale, **self.refmod).LCB
        opt = Parallel(n_jobs = cores)(delayed(minimize)(LCBref, x0 = start_point,
                                                      method = 'L-BFGS-B', tol = 1e-6,
                                                      bounds = self.bounds)
                                    for start_point in x0)
        xnxts = np.array([res.x for res in opt], dtype  = 'float')
        funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        end = time.time()
        self.timeref[0] = end-start
        for i in range(self.trialsref):
            xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
            startf = time.time()
            ynxt = self.system(self.descale(xnxt))
            if str(type(refmod))=="<class '__main__.Network'>":
                epsnxt = ynxt - refmod(torch.from_numpy(xnxt).float()).data.numpy()
            else:
                epsnxt = ynxt - refmod(self.descale(xnxt))
            endf = time.time()
            self.timefref[i+1] = self.timefref[i]+(endf-startf)
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            eps = np.vstack([eps, epsnxt])
            self.modeleps.fit(x, eps)
            # x0 = np.random.uniform(-sf, sf, (100, self.dim))
            x0 = np.random.uniform(0, sf, (128, self.dim))
            opt = Parallel(n_jobs = cores)(delayed(minimize)(LCBref, x0 = start_point,
                                                      method = 'L-BFGS-B', tol = 1e-6,
                                                      bounds = self.bounds)
                                    for start_point in x0)
            xnxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            end = time.time()
            self.timeref[i+1] = end-start
        self.ref_optim = True
        self.xref = self.descale(x)
        self.ytru = y
        self.eps = eps
    
    def optimizerdist(self, trials, distributions, cores = 4, xinit = None):
        print('Distirbuted BO Run...')
        self.trialsdist = trials
        funcs={}
        y = {}
        LCBdist = {}
        if xinit is None:
            # x = np.random.uniform(-1, 1, (1, self.dim))
            x = np.random.uniform(0, 1, (1, self.dim))
        else:
            x = xinit.reshape(1, self.dim)
        for i in range(distributions):
            idx = 'model'+str(i+1)+'dist'
            funcs[idx] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                      normalize_y = True,
                                                      n_restarts_optimizer = 10)
            y['dist'+str(i+1)] = self.distmod(self.descale(x))[i]
            funcs[idx].fit(x, y['dist'+str(i+1)])
            LCBdist['dist'+str(i)] = LCB_AF(funcs[idx], self.dim, self.exp_w, self.descale).LCB
        funcs['modeldist'] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, 
                                                  normalize_y = True,
                                                  n_restarts_optimizer = 10)
        y['dist'] = self.distmod(self.descale(x))[-1]
        funcs = OrderedDict(funcs)
        y = OrderedDict(y)
        LCBdist = OrderedDict(LCBdist)
        # x0 = np.random.uniform(-1, 1, (100, self.dim))
        x0 = np.random.uniform(0, 1, (128, self.dim))
        opt = Parallel(n_jobs = cores)(delayed(minimize)(LCBdist['dist1'], x0 = start_point,
                                                      method = 'L-BFGS-B',
                                                      bounds = self.bounds)
                                    for start_point in x0)
        xnxts = np.array([res.x for res in opt], dtype  = 'float')
        funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        for i in range(self.trialsdist):
            xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
            x = np.vstack([x, xnxt])
            for j, k, l in zip(funcs.keys(), y.keys(), range(len(funcs))):
                ynxt = self.distmod(self.descale(xnxt))[l]
                y[k] = np.vstack([y[k], ynxt])
                funcs[j].fit(x, y[k])
            LCBiter = list(LCBdist.values())[(i+1)%len(LCBdist)]
            # x0 = np.random.uniform(-1, 1, (100, self.dim))
            x0 = np.random.uniform(0, 1, (128, self.dim))
            opt = Parallel(n_jobs = 4)(delayed(minimize)(LCBiter, x0 = start_point,
                                                      method = 'L-BFGS-B',
                                                      bounds = self.bounds)
                                    for start_point in x0)
            xnxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        self.dist_optim = True
        self.modeldist = funcs
        self.xdist = self.descale(x)
        self.ydist = y
    
    def optimizerdistref(self, trials, distributions, xinit = None):
        print('Distributed BO with Reference Model Run...')
        self.trialsdr = trials
        epsmod={}
        y = {}
        eps = {}
        LCBdr = {}
        if xinit is None:
            # x = np.random.uniform(-1, 1, (1, self.dim))
            x = np.random.uniform(0, 1, (1, self.dim))
        else:
            x = xinit.reshape(1, self.dim)
        for i in range(distributions):
            idx = 'dist_eps_model'+str(i+1)
            idx2 = 'dist'+str(i+1)
            epsmod[idx] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                      normalize_y = True,
                                                      n_restarts_optimizer = 10)
            y[idx2] = self.distmod(self.descale(x))[i]
            eps[idx2] = y[idx2]-list(self.dist_ref.values())[i](self.descale(x))
            epsmod[idx].fit(x, eps[idx2])
            refmod = {'refmod': list(self.dist_ref.values())[i]}
            LCBdr[idx2] = LCB_AF(epsmod[idx], self.dim, self.exp_w, self.descale, **refmod).LCB
        epsmod['dist_eps_model'] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, 
                                                  normalize_y = True,
                                                  n_restarts_optimizer = 10)
        y['dist'] = self.distmod(self.descale(x))[-1]
        eps['dist'] = y['dist']-list(self.dist_ref.values())[-1](self.descale(x))
        epsmod['dist_eps_model'].fit(x, eps['dist'])
        epsmod = OrderedDict(epsmod)
        y = OrderedDict(y)
        eps = OrderedDict(eps)
        LCBdr = OrderedDict(LCBdr)
        # x0 = np.random.uniform(-1, 1, (100, self.dim))
        x0 = np.random.uniform(0, 1, (128, self.dim))
        opt = Parallel(n_jobs = -1)(delayed(minimize)(LCBdr['dist1'], x0 = start_point,
                                                      method = 'L-BFGS-B',
                                                      bounds = self.bounds)
                                    for start_point in x0)
        xnxts = np.array([res.x for res in opt], dtype  = 'float')
        funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        for i in range(self.trialsdr):
            xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
            x = np.vstack([x, xnxt])
            for j, k, l in zip(epsmod.keys(), y.keys(), range(len(epsmod))):
                ynxt = self.distmod(self.descale(xnxt))[l]
                epsnxt = ynxt-list(self.dist_ref.values())[l](self.descale(xnxt))
                y[k] = np.vstack([y[k], ynxt])
                eps[k] = np.vstack([eps[k], epsnxt])
                epsmod[j].fit(x, eps[k])
            LCBiter = list(LCBdr.values())[(i+1)%len(LCBdr)]
            # x0 = np.random.uniform(-1, 1, (100, self.dim))
            x0 = np.random.uniform(0, 1, (128, self.dim))
            opt = Parallel(n_jobs = -1)(delayed(minimize)(LCBiter, x0 = start_point,
                                                      method = 'L-BFGS-B',
                                                      bounds = self.bounds)
                                    for start_point in x0)
            xnxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        self.distref_optim = True
        self.modeldistref = epsmod
        self.xdistref = self.descale(x)
        self.ydistref = y
        self.epsdistref = eps
        
    def optimizersplt(self, trials, partition_num, partition_cons, scaling_factor, fcores = 4, afcores = 4, xinit = None):
        """
        partition_cons should be a dictionary of constraints numerically
        indexed and contained in lists that set up the partitions of the
        feature space according to whatever partition shape is desired;
        bounds can be linear or nonlinear
        """
        print('Partitioned Domain BO Run...')
        start = time.time()
        self.trialsplt = trials
        self.split = partition_num
        self.cons = partition_cons
        self.timesplt = np.ones(self.trialsplt+1)
        self.timefsplt = np.ones(self.trialsplt+1)
        sf = scaling_factor
        x = np.array([]).reshape(0, self.dim)
        switch = True
        def intpts(x, i):
            l = str(i+1)
            for j in range(len(self.cons[l])):
                if self.cons[l][j].lb < self.cons[l][j].fun(x) < self.cons[l][j].ub:
                    res = 0
                else:
                    res = 1e6
            return res
        for i in range(self.split):
            n = 0
            if xinit is not None and switch == True:
                for j in range(len(self.cons[str(i+1)])):    
                    if self.cons[str(i+1)][j].fun(xinit) < self.cons[str(i+1)][j].ub\
                        and self.cons[str(i+1)][j].fun(xinit) > self.cons[str(i+1)][j].lb:
                            n+=1
                if n == len(self.cons[str(i+1)]):
                    x0 = xinit.reshape(1, self.dim)
                    switch = False
            if n!=len(self.cons[str(i+1)]):
                x0 = np.random.uniform(0, sf, (10, self.dim))
                opt = Parallel(n_jobs = 1)(delayed(minimize)(intpts, x0 = x, args = (i,),
                                                             method = 'SLSQP',
                                                             bounds = self.bounds, tol = 1e-6,
                                                             constraints = self.cons[str(i+1)])
                                           for x in x0)
                x0s = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                x0 = x0s[np.argmin(funs)]
            x = np.vstack([x, x0.reshape(1, self.dim)])
        splt = int(x.shape[0]/fcores)
        xbs = np.array(np.ones(fcores), dtype = tuple)
        if fcores == 1:
            xbs[0] = x
        else:
            for i in range(fcores-1):
                xbs[i] = x[i*splt:(i+1)*splt, :]
            xbs[-1] = x[(i+1)*splt:, :]
        startf = time.time()
        y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xbs)
        endf = time.time()
        self.timefsplt[0] = endf-startf
        y = np.hstack(y[:]).T.reshape(-1,1)
        ybst = min(y).reshape(-1,1)
        modelsplt = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                 n_restarts_optimizer = 10,
                                                 normalize_y = True)
        modelsplt.fit(x, y)
        xnxt = np.ones((self.split, self.dim))
        LCB = LCB_AF(modelsplt, self.dim, self.exp_w, self.descale).LCB
        init_pts = max(1, int(round(128/self.split, 0)))
        x0 = np.random.uniform(0, sf, (init_pts, self.dim))
        for i in range(self.split):
            # x0 = np.random.uniform(-sf, sf, (100, self.dim))
            opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB, x0 = start_points,
                                                          method = 'SLSQP',
                                                          bounds = self.bounds,
                                                          constraints = self.cons[str(i+1)])
                                        for start_points in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            sts = np.array([res.success for res in opt])
            funs[np.where(sts==False)] = max(1e6, np.max(funs))
            xnxt[i] = xnxts[np.argmin(funs)]
        xnxtbs = np.array(np.ones(fcores), dtype = tuple)
        end = time.time()
        self.timesplt[0] = end-start
        for i in range(self.trialsplt):
            if fcores == 1:
                xnxtbs[0] = xnxt
            else:
                for j in range(fcores-1):
                    xnxtbs[j] = xnxt[j*splt:(j+1)*splt, :]
                xnxtbs[-1] = xnxt[(j+1)*splt:, :]
            startf = time.time()
            ynxt =  Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xnxtbs)
            endf = time.time()
            self.timefsplt[i+1] = self.timefsplt[i]+(endf-startf)
            ynxt = np.hstack(ynxt[:]).T.reshape(-1, 1)
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            ybst = np.vstack([ybst, min(ynxt).reshape(-1, 1)])
            modelsplt.fit(x, y)
            x0 = np.random.uniform(0, sf, (init_pts, self.dim))
            for j in range(self.split):
                opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB, x0 = start_points,
                                                              method = 'SLSQP',
                                                              bounds = self.bounds,
                                                              constraints = self.cons[str(j+1)])
                                            for start_points in x0)
                xnxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                sts = np.array([res.success for res in opt])
                funs[np.where(sts==False)] = max(1e6, np.max(funs))
                xnxt[j] = xnxts[np.argmin(funs)]
            end = time.time()
            self.timesplt[i+1] = end-start
        self.splt_optim = True
        self.modelsplt = modelsplt
        self.xsplt = self.descale(x)
        self.ysplt = y
        self.yspltbst = ybst
    
    def optimizerspltref(self, trials, partition_num, partition_cons, scaling_factor, fcores = 4, afcores = 4, xinit = None):
        """
        partition_cons should be a dictionary of constraints numerically
        indexed and contained in lists that set up the partitions of the
        feature space according to whatever partition shape is desired;
        bounds can be linear or nonlinear; base these off reference model
        """
        print('Partitioned Domain with Reference Model BO Run...')
        start = time.time()
        self.trialspltref = trials
        self.splitref = partition_num
        self.consref = partition_cons
        self.timespltref = np.ones(self.trialspltref+1)
        self.timefspltref = np.ones(self.trialspltref+1)
        sf = scaling_factor
        x = np.array([]).reshape(0, self.dim)
        refmod = self.refmod['refmod']
        switch = True
        def intpts(x, i):
            l = str(i+1)
            for j in range(len(self.consref[l])):
                if self.consref[l][j].lb < self.consref[l][j].fun(x) < self.consref[l][j].ub:
                    res = 0
                else:
                    res = 1e6
            return res
        for i in range(self.splitref):
            n = 0
            if xinit is not None and switch == True:
                for j in range(len(self.consref[str(i+1)])):    
                    if self.consref[str(i+1)][j].fun(xinit) < self.consref[str(i+1)][j].ub\
                        and self.consref[str(i+1)][j].fun(xinit) > self.consref[str(i+1)][j].lb:
                            n+=1
                if n == len(self.consref[str(i+1)]):
                    x0 = xinit.reshape(1, self.dim)
                    switch = False
            if n!=len(self.consref[str(i+1)]):
                x0 = np.random.uniform(0, sf, (10, self.dim))
                opt = Parallel(n_jobs = 1)(delayed(minimize)(intpts, x0 = x, args = (i,),
                                                             method = 'SLSQP',
                                                             bounds = self.bounds, tol = 1e-6,
                                                             constraints = self.consref[str(i+1)])
                                           for x in x0)
                x0s = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                x0 = x0s[np.argmin(funs)]
            x = np.vstack([x, x0.reshape(1, self.dim)])
        splt = int(x.shape[0]/fcores)
        xbs = np.array(np.ones(fcores), dtype = tuple)
        if fcores == 1:
            xbs[0] = x
        else:
            for i in range(fcores-1):
                xbs[i] = x[i*splt:(i+1)*splt, :]
            xbs[-1] = x[(i+1)*splt:, :]
        startf = time.time()
        y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xbs)
        if str(type(refmod))=="<class '__main__.Network'>":
            yref = Parallel(n_jobs = fcores)(delayed(refmod)(start_point) for start_point in torch.from_numpy(x).float())
            endf = time.time()
            yref = torch.hstack(yref[:]).T.reshape(-1, 1).data.numpy()
        else:
            yref = Parallel(n_jobs = fcores)(delayed(refmod)(self.descale(start_point)) for start_point in xbs)
            endf = time.time()
            yref = np.hstack(yref[:]).T.reshape(-1,1)
        y = np.hstack(y[:]).T.reshape(-1,1)
        eps = y-yref
        ybst = min(y).reshape(-1,1)
        self.timefspltref[0] = endf-startf
        modelsplteps = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                 n_restarts_optimizer = 10,
                                                 normalize_y = True)
        modelsplteps.fit(x, eps)
        xnxt = np.ones((self.splitref, self.dim))
        LCB = LCB_AF(modelsplteps, self.dim, self.exp_w, self.descale, **self.refmod).LCB
        init_pts = max(1, int(round(128/self.splitref, 0)))
        x0 = np.random.uniform(0, sf, (init_pts, self.dim))
        for i in range(self.splitref):
            # x0 = np.random.uniform(-sf, sf, (100, self.dim))
            opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB, x0 = start_points,
                                                          method = 'SLSQP',
                                                          bounds = self.bounds, tol = 1e-6,
                                                          constraints = self.consref[str(i+1)])
                                        for start_points in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            sts = np.array([res.success for res in opt])
            funs[np.where(sts==False)] = max(1e6, np.max(funs))
            xnxt[i] = xnxts[np.argmin(funs)]
        xnxtbs = np.array(np.ones(fcores), dtype = tuple)
        end = time.time()
        self.timespltref[0] = end-start
        for i in range(self.trialspltref):
            if fcores == 1:
                xnxtbs[0] = xnxt
            else:
                for j in range(fcores-1):
                    xnxtbs[j] = xnxt[j*splt:(j+1)*splt, :]
                xnxtbs[-1] = xnxt[(j+1)*splt:, :]
            startf = time.time()
            ynxt =  Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xnxtbs)
            if str(type(refmod))=="<class '__main__.Network'>":
                yref = Parallel(n_jobs = fcores)(delayed(refmod)(start_point) for start_point in torch.from_numpy(xnxt).float())
                endf = time.time()
                yref = torch.hstack(yref[:]).T.reshape(-1, 1).data.numpy()
            else:
                yref =  Parallel(n_jobs = fcores)(delayed(refmod)(self.descale(start_point)) for start_point in xnxtbs)
                endf = time.time()
                yref = np.hstack(yref[:]).T.reshape(-1, 1)
            ynxt = np.hstack(ynxt[:]).T.reshape(-1,1)
            epsnxt = ynxt-yref
            self.timefspltref[i+1] = self.timefspltref[i]+(endf-startf)
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            eps = np.vstack([eps, epsnxt])
            ybst = np.vstack([ybst, min(ynxt).reshape(-1, 1)])
            modelsplteps.fit(x, eps)
            x0 = np.random.uniform(0, sf, (init_pts, self.dim))
            for j in range(self.splitref):
                opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB, x0 = start_points,
                                                              method = 'SLSQP',
                                                              bounds = self.bounds, tol = 1e-6,
                                                              constraints = self.consref[str(j+1)])
                                            for start_points in x0)
                xnxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                sts = np.array([res.success for res in opt])
                funs[np.where(sts==False)] = max(1e6, np.max(funs))
                xnxt[j] = xnxts[np.argmin(funs)]
            end = time.time()
            self.timespltref[i+1] = end-start
        self.splt_optimref = True
        self.modelsplteps = modelsplteps
        self.xspltref = self.descale(x)
        self.yspltref = y
        self.yspltbstref = ybst
        self.epssplt = eps
    
    def hyperspace(self, trials, partition_num, partition_cons, scaling_factor, fcores = 4, afcores = 4, xinit = None):
        """
        parallelization of BO based on HyperSpace algorithm as described in
        literature; overlap parameter is based using constraints and each
        partition (hyperspace) is optimized using its own model
        """
        print('Hyperspace Run...')
        start = time.time()
        self.trialshyp = trials
        self.splithyp = partition_num
        self.conshyp = partition_cons
        self.timehyp = np.ones(self.trialshyp+1)
        self.timefhyp = np.ones(self.trialshyp+1)
        sf = scaling_factor
        x = np.array([]).reshape(0, self.dim)
        switch = True
        def intpts(x):
            return 0
        for i in range(self.splithyp):
            n = 0
            if xinit is not None and switch == True:
                for j in range(len(self.conshyp[str(i+1)])):    
                    if self.conshyp[str(i+1)][j].fun(xinit) < self.conshyp[str(i+1)][j].ub\
                        and self.conshyp[str(i+1)][j].fun(xinit) > self.conshyp[str(i+1)][j].lb:
                            n+=1
                if n == len(self.conshyp[str(i+1)]):
                    x0 = xinit.reshape(1, self.dim)
                    switch = False
            if n!=len(self.conshyp[str(i+1)]):
                # x0 = np.random.uniform(-sf, sf, (self.dim,))
                x0 = np.random.uniform(0, sf, (self.dim,))
                x0 = minimize(intpts, x0, bounds = self.bounds, constraints = self.conshyp[str(i+1)]).x
            x = np.vstack([x, x0.reshape(1, self.dim)])
        splt = int(x.shape[0]/fcores)
        xbs = np.array(np.ones(fcores), dtype = tuple)
        if fcores == 1:
            xbs[0] = x
        else:
            for i in range(fcores-1):
                xbs[i] = x[i*splt:(i+1)*splt, :]
            xbs[-1] = x[(i+1)*splt:, :]
        startf = time.time()
        y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xbs)
        endf = time.time()
        self.timefhyp[0] = endf-startf
        y = np.hstack(y[:]).T.reshape(-1,1)
        ybst = min(y).reshape(-1,1)
        modelhyp = {}
        LCB = {}
        for i in range(self.splithyp):
            modelhyp[str(i)] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                     n_restarts_optimizer = 10,
                                                     normalize_y = True)
            modelhyp[str(i)].fit(x[i::self.splithyp], y[i::self.splithyp])
            LCB[str(i)] = LCB_AF(modelhyp[str(i)], self.dim, self.exp_w, self.descale).LCB
        xnxt = np.ones((self.splithyp, self.dim))
        init_pts = max(1, int(round(128/self.splithyp, 0)))
        x0 = np.random.uniform(0, sf, (init_pts, self.dim))
        for i in range(self.splithyp):
            # x0 = np.random.uniform(-sf, sf, (100, self.dim))
            opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB[str(i)], x0 = start_points,
                                                          method = 'SLSQP',
                                                          bounds = self.bounds,
                                                          constraints = self.conshyp[str(i+1)])
                                        for start_points in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            sts = np.array([res.success for res in opt])
            funs[np.where(sts == False)] = max(1e6, np.max(funs))
            xnxt[i] = xnxts[np.argmin(funs)]
        xnxtbs = np.array(np.ones(fcores), dtype = tuple)
        end = time.time()
        self.timehyp[0] = end-start 
        for i in range(self.trialshyp):
            if fcores == 1:
                xnxtbs[0] = xnxt
            else:
                for j in range(fcores-1):
                    xnxtbs[j] = xnxt[j*splt:(j+1)*splt, :]
                xnxtbs[-1] = xnxt[(j+1)*splt:, :]
            startf = time.time()
            ynxt =  Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xnxtbs)
            endf = time.time()
            self.timefhyp[i+1] = self.timefhyp[i]+(endf-startf)
            ynxt = np.hstack(ynxt[:]).T.reshape(-1, 1)
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            ybst = np.vstack([ybst, min(ynxt).reshape(-1, 1)])
            x0 = np.random.uniform(0, sf, (init_pts, self.dim))
            for j in range(self.splithyp):
                modelhyp[str(j)].fit(x[j::self.splithyp], y[j::self.splithyp])
                opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB[str(j)], x0 = start_points,
                                                              method = 'SLSQP',
                                                              bounds = self.bounds,
                                                              constraints = self.conshyp[str(j+1)])
                                            for start_points in x0)
                xnxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                sts = np.array([res.success for res in opt])
                funs[np.where(sts==False)] = max(1e6, np.max(funs))
                xnxt[j] = xnxts[np.argmin(funs)]
            end = time.time()
            self.timehyp[i+1] = end-start
        self.hyp_optim = True
        self.modelhyp = modelhyp
        self.xhyp = self.descale(x)
        self.yhyp = y
        self.yhypbst = ybst
    
    def optimizerexpw(self, trials, num_weights, scaling_factor, fcores, afcores, xinit = None):
        print('Variable κ BO Run...')
        start = time.time()
        self.trialsexpw = trials
        self.timexpw = np.zeros(self.trialsexpw+1)
        self.timefexpw = np.zeros(self.trialsexpw+1)
        sf = scaling_factor
        spltexpw = num_weights
        exp_wl = np.random.exponential(1, spltexpw)
        if xinit is None:
            # x = np.random.uniform(-sf, sf, (1, self.dim))
            x = np.random.uniform(0, sf, (spltexpw, self.dim))
        else:
            x = xinit.reshape(-1, 1)
            rw = int(x.shape[0]/self.dim)
            x = xinit.reshape(rw, self.dim)
            if x.shape[0]<spltexpw:
                x = np.vstack([x, np.random.uniform(0, sf, (spltexpw-x.shape[0], self.dim))])
        startf = time.time()
        y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(t)) for t in x)
        endf = time.time()
        y = np.hstack(y[:]).T.reshape(-1, 1)
        self.timefexpw [0] = endf-startf
        ybst = min(y)
        modelexpw = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, normalize_y = True, n_restarts_optimizer = 10)
        modelexpw.fit(x, y)
        init_pts = max(1, int(round(128/1, 0)))
        x0 = np.random.uniform(0, sf, (init_pts, self.dim))
        LCB = {}
        xnxt = np.ones((spltexpw, self.dim))
        for i in range(spltexpw):
            LCB[str(i+1)] = LCB_AF(modelexpw, self.dim, exp_wl[i], self.descale)
            opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB[str(i+1)].LCB, x0 = start_points,
                                                         method = 'L-BFGS-B',
                                                         bounds = self.bounds)
                                       for start_points in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            xnxt[i] = xnxts[np.argmin(funs)]
        end = time.time()
        self.timexpw[0] = end-start
        for i in range(self.trialsexpw):
            startf = time.time()
            ynxt = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(t)) for t in xnxt)
            endf = time.time()
            self.timefexpw[i+1] = self.timefexpw[i]+(endf-startf)
            ynxt = np.hstack(ynxt[:]).T.reshape(-1, 1)
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            ybst = np.vstack([ybst, min(ynxt)])
            modelexpw.fit(x, y)
            x0 = np.random.uniform(0, sf, (init_pts, self.dim))
            exp_wl = np.random.exponential(1, spltexpw)
            for j in range(spltexpw):
                LCB[str(j+1)].exp_w = exp_wl[j]
                opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB[str(j+1)].LCB, x0 = start_points,
                                                                   method = 'L-BFGS-B',
                                                                   bounds = self.bounds)
                                                 for start_points in x0)
                xnxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                xnxt[j] = xnxts[np.argmin(funs)]
            end = time.time()
            self.timexpw[i+1] = end-start
        self.expwl_optim = True
        self.xexpw = self.descale(x)
        self.yexpw = y
        self.modelexpw = modelexpw
        self.ybstexpw = ybst
    
    def optimizernmc(self, trials, parallel_num, sample_num, scaling_factor, fcores = 4, afcores = 4, xinit = None):
        """
        parallelization of BO based on N times Monte Carlo algorithm as described
        in literature; parallel_num is number of parallel sample points desired
        """
        print('N times MC BO Run...')
        start = time.time()
        self.trialsnmc = trials
        self.timenmc = np.zeros(self.trialsnmc+1)
        self.timefnmc = np.zeros(self.trialsnmc+1)
        sf = scaling_factor
        spltnmc = parallel_num
        samp_num = int(sample_num)
        if xinit is None:
            # x = np.random.uniform(-sf, sf, (1, self.dim))
            x = np.random.uniform(0, sf, (spltnmc, self.dim))
        else:
            x = xinit.reshape(-1, 1)
            rw = int(x.shape[0]/self.dim)
            x = xinit.reshape(rw, self.dim)
            if x.shape[0]<spltnmc:
                x = np.vstack([x, np.random.uniform(0, sf, (spltnmc-x.shape[0], self.dim))])
        startf = time.time()
        y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(t)) for t in x)
        endf = time.time()
        self.timefnmc[0] = endf-startf
        y = np.hstack(y[:]).T.reshape(-1, 1)
        ybst = min(y)
        modelnmc = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                     normalize_y = True,
                                                     n_restarts_optimizer = 10)
        modelnmc.fit(x, y)
        init_pts = max(1, int(round(128/1, 0)))
        x0 = np.random.uniform(0, 1, (init_pts, self.dim))
        LCB = LCB_AF(modelnmc, self.dim, self.exp_w, self.descale).LCB
        xnxt = np.ones((spltnmc, self.dim))
        opt = Parallel(n_jobs = 1)(delayed(minimize)(LCB, x0 = start_points,
                                                     method = 'L-BFGS-B',
                                                     bounds = self.bounds)
                                   for start_points in x0)
        xnxts = np.array([res.x for res in opt], dtype = 'float')
        funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        xnxt[0] = xnxts[np.argmin(funs)]
        for i in range(1, spltnmc):
            xs = xnxt[:i]
            ys = modelnmc.sample_y(xs, n_samples = samp_num, random_state = np.random.randint(0, 1e6))
            ys = np.vstack(ys[:]).T
            x_f = np.vstack([x, xs])
            fant_mod = {}
            LCB_fant = {}
            for j in range(ys.shape[0]):
                y_f = np.vstack([y, ys[j].reshape(-1, 1)])
                fant_mod[str(j+1)] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                            normalize_y = True,
                                            n_restarts_optimizer = 10)
                fant_mod[str(j+1)].fit(x_f, y_f)
                LCB_fant[str(j+1)] = LCB_AF(fant_mod[str(j+1)], self.dim, self.exp_w, self.descale)
            def LCB_nmc(x, LCB_fant):
                af = 0
                for j in range(ys.shape[0]):
                    af += LCB_fant[str(j+1)].LCB(x)
                af = af/(j+1)
                return af
            opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB_nmc, x0 = start_points,
                                                               method = 'L-BFGS-B',
                                                               bounds = self.bounds,
                                                               args = (LCB_fant))
                                             for start_points in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            xnxt[i] = xnxts[np.argmin(funs)]
        end = time.time()
        self.timenmc[0] = end-start
        for i in range(self.trialsnmc):
            startf = time.time()
            ynxt = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(t)) for t in xnxt)
            endf = time.time()
            self.timefnmc[i+1] = self.timefnmc[i]+(endf-startf)
            ynxt = np.hstack(ynxt[:]).T.reshape(-1, 1)
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            ybst = np.vstack([ybst, min(ynxt)])
            modelnmc.fit(x, y)
            xnxt = np.ones((spltnmc, self.dim))
            opt = Parallel(n_jobs = 1)(delayed(minimize)(LCB, x0 = start_points,
                                                         method = 'L-BFGS-B',
                                                         bounds = self.bounds)
                                       for start_points in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            xnxt[0] = xnxts[np.argmin(funs)]
            for k in range(1, spltnmc):
                xs = xnxt[:k]
                ys = modelnmc.sample_y(xs, n_samples = samp_num, random_state = np.random.randint(0, 1e6))
                ys = np.vstack(ys[:]).T
                x_f = np.vstack([x, xs])
                fant_mod = {}
                LCB_fant = {}
                for j in range(ys.shape[0]):
                    y_f = np.vstack([y, ys[j].reshape(-1, 1)])
                    fant_mod[str(j+1)] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                normalize_y = True,
                                                n_restarts_optimizer = 10)
                    fant_mod[str(j+1)].fit(x_f, y_f)
                    LCB_fant[str(j+1)] = LCB_AF(fant_mod[str(j+1)], self.dim, self.exp_w, self.descale)
                def LCB_nmc(x, LCB_fant):
                    af = 0
                    for j in range(ys.shape[0]):
                        af += LCB_fant[str(j+1)].LCB(x)
                    af = af/(j+1)
                    return af
                opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB_nmc, x0 = start_points,
                                                                   method = 'L-BFGS-B',
                                                                   bounds = self.bounds,
                                                                   args = (LCB_fant))
                                                 for start_points in x0)
                xnxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                xnxt[k] = xnxts[np.argmin(funs)]
            end = time.time()
            self.timenmc[i+1] = end-start
        self.nmc_optim = True
        self.xnmc = self.descale(x)
        self.ynmc = y
        self.modelnmc = modelnmc
        self.ybstnmc = ybst
    
    def optimizerspltvar(self, trials, split_num, liminit, scaling_factor, fcores, afcores, xinit  = None):
        # Split partition using variables as split point
        print('Partitioned Variables BO Run...')
        start = time.time()
        self.trialspltvar = trials
        self.splitvar = split_num
        self.timespltvar = np.zeros(self.trialspltvar+1)
        self.timefspltvar = np.zeros(self.trialspltvar+1)
        div = int(self.dim/self.splitvar)
        sf = scaling_factor
        refmod = self.dist_ref['distrefmod']
        # x = np.random.uniform(-sf, sf, (self.splitvar, self.dim))
        x = liminit*np.ones((self.splitvar, self.dim))
        lwr = x.copy()
        upr = x.copy()+1e-6
        for i in range(self.splitvar):
            if xinit is None:
                x[i, i*div:(i+1)*div] = np.random.uniform(0, 1, (1, div))
            else:
                xinit = xinit.reshape(1, self.dim)
                x[i, i*div:(i+1)*div] = xinit[0, i*div:(i+1)*div]
            lwr[i, i*div:(i+1)*div] = 0
            upr[i, i*div:(i+1)*div] = sf
        x = np.vstack([x, liminit])
        splt = int(x.shape[0]/fcores)
        xbs = np.array(np.ones(fcores), dtype = tuple)
        if fcores == 1:
            xbs[0] = x
        else:
            for i in range(fcores-1):
                xbs[i] = x[i*splt:(i+1)*splt, :]
            xbs[-1] = x[(i+1)*splt:, :]
        startf = time.time()
        y = Parallel(n_jobs = fcores)(delayed(self.distmod)(self.descale(start_point)) for start_point in xbs)
        if str(type(refmod))=="<class '__main__.Network'>":
            yref = Parallel(n_jobs = fcores)(delayed(refmod)(start_point) for start_point in torch.from_numpy(x).float())
            endf = time.time()
            yref = torch.hstack(yref[:]).T.reshape(-1, 1).data.numpy()
        else:
            yref = Parallel(n_jobs = fcores)(delayed(refmod)(self.descale(start_point)) for start_point in xbs)
            endf = time.time()
            yref = np.vstack(yref[:])
        self.timefspltvar[0] = endf-startf
        y = np.vstack(y[:])
        eps = y-yref
        ybst = np.min(y, axis = 0).reshape(-1, 1).T
        modelspltvar = {}
        bndsvar = {}
        LCB = {}
        xnxt = x.copy()
        init_pts = int(round(128**(div/self.dim)))
        x0 = np.random.uniform(0, sf, (init_pts, self.dim))
        for i in range(self.splitvar):
            modelspltvar[str(i+1)] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                                  n_restarts_optimizer = 10,
                                                                  normalize_y = True)
            modelspltvar[str(i+1)].fit(x, eps[:, i])
            bndsvar[str(i+1)] = Bounds(lwr[i], upr[i])
            LCB[str(i+1)] = LCB_AF(modelspltvar[str(i+1)], self.dim, self.exp_w,
                                   self.descale, **{'refmod': self.dist_ref['distrefmod'+str(i+1)]}).LCB
            opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB[str(i+1)], x0 = start_point,
                                                              method = 'L-BFGS-B',
                                                              bounds = bndsvar[str(i+1)])
                                            for start_point in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            xnxt[i] = xnxts[np.argmin(funs)]
            xnxt[-1, i*div:(i+1)*div] = xnxts[np.argmin(funs), i*div:(i+1)*div]
        xnxtbs = np.array(np.ones(fcores), dtype = tuple)
        end = time.time()
        self.timespltvar[0] = end-start
        for i in range(self.trialspltvar):
            if fcores == 1:
                xnxtbs[0] = xnxt
            else:
                for j in range(fcores-1):
                    xnxtbs[j] = xnxt[j*splt:(j+1)*splt, :]
                xnxtbs[-1] = xnxt[(j+1)*splt:, :]
            startf = time.time()
            ynxt = Parallel(n_jobs = fcores)(delayed(self.distmod)(self.descale(start_point)) for start_point in xnxtbs)
            if str(type(refmod))=="<class '__main__.Network'>":
                yref = Parallel(n_jobs = fcores)(delayed(refmod)(start_point) for start_point in torch.from_numpy(x).float())
                endf = time.time()
                yref = torch.hstack(yref[:]).T.reshape(-1, 1).data.numpy()
            else:
                yref = Parallel(n_jobs = fcores)(delayed(refmod)(self.descale(start_point)) for start_point in xnxtbs)
                endf = time.time()
                yref = np.vstack(yref[:])
            ynxt = np.vstack(ynxt[:])
            epsnxt = ynxt-yref
            self.timefspltvar[i+1] = self.timefspltvar[i]+(endf-startf)
            for j in range(self.splitvar):
                if any(ynxt[:, j] < min(y[:, j])):
                    lwr[j] = xnxt[np.argmin(ynxt[:, j])]
                    lwr[j, j*div:(j+1)*div] = 0
                    upr[j] = xnxt[np.argmin(ynxt[:, j])]+1e-6
                    upr[j, j*div:(j+1)*div] = sf
                # lwr[j] = xnxt[-1]
                # lwr[j, j*div:(j+1)*div] = 0
                # upr[j] = xnxt[-1]+1e-6
                # upr[j, j*div:(j+1)*div] = sf
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            eps = np.vstack([eps, epsnxt])
            ybst = np.vstack([ybst, np.min(ynxt, axis = 0).reshape(-1,1).T])
            # x0 = np.random.uniform(-sf, sf, (100, self.dim))
            x0 = np.random.uniform(0, sf, (init_pts, self.dim))
            for j in range(self.splitvar):
                modelspltvar[str(j+1)].fit(x, eps[:, j])
                bndsvar[str(j+1)] = Bounds(lwr[j], upr[j])
                opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB[str(j+1)], x0 = start_point,
                                                                  method = 'L-BFGS-B',
                                                                  bounds = bndsvar[str(j+1)])
                                                for start_point in x0)
                xnxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                xnxt[j] = xnxts[np.argmin(funs)]
                xnxt[-1, j*div:(j+1)*div] = xnxts[np.argmin(funs), j*div:(j+1)*div]
            end = time.time()
            self.timespltvar[i+1] = end-start
        self.spltvar_optim = True
        self.modelspltvar = modelspltvar
        self.xspltvar = self.descale(x)
        self.yspltvar = y
        self.yspltvarbst = ybst
        
    def optimizerembd(self, trials, var_num, include_x, fun, scaling_factor, fcores, afcores, xinit = None):
        print('Embedded function BO Run...')
        start = time.time()
        self.trialsembd = trials
        self.var_num = var_num
        self.include_x = include_x
        self.fun = fun
        sf = scaling_factor
        self.timembd = np.zeros(self.trialsembd+1)
        self.timefembd = np.zeros(self.trialsembd+1)
        if xinit is None:
            x = np.random.uniform(0, sf, (1, self.dim))
        else:
            x = xinit.reshape(1, self.dim)
        startf = time.time()
        d = self.system(self.descale(x))
        endf = time.time()
        self.timefembd[0] = endf-startf
        modembd = {}
        for i in range(self.var_num):
            modembd[str(i+1)] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                             n_restarts_optimizer = 10,
                                                             normalize_y = True)
            modembd[str(i+1)].fit(x, d[:, i].reshape(-1, 1))
        LCB = LCB_EMBD(modembd, self.var_num, self.dim, self.exp_w, self.fun, self.descale, self.include_x).LCB
        x0 = np.random.uniform(0, 1, (100, self.dim))
        opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB, x0 = start_point,
                                                           method = 'L-BFGS-B',
                                                           bounds = self.bounds)
                                         for start_point in x0)
        xnxts = np.array([res.x for res in opt], dtype = 'float')
        funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
        end = time.time()
        self.timembd[0] = end-start
        for i in range(self.trialsembd):
            startf = time.time()
            dnxt = self.system(self.descale(xnxt))
            endf = time.time()
            self.timefembd[i+1] = self.timefembd[i]+(endf-startf)
            x = np.vstack([x, xnxt])
            d = np.vstack([d, dnxt])
            for j in range(self.var_num):
                modembd[str(j+1)].fit(x, d[:, j])
            x0 = np.random.uniform(0, 1, (100, self.dim))
            opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB, x0 = start_point,
                                                               method = 'L-BFGS-B',
                                                               bounds = self.bounds)
                                             for start_point in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
            end = time.time()
            self.timembd[i+1] = end-start
        self.embd_optim = True
        self.xembd = self.descale(x)
        self.yembd = d
        if self.include_x:
            self.yembd = np.hstack([self.yembd, self.xembd])
        self.fembd = self.fun(self.yembd)
        self.modembd = modembd
    
    def plots(self, figure_name):
        itr = np.arange(1, self.trialsgp+2, 1)
        yliml = min(self.ygp)-0.01*abs(min(self.ygp))
        ylimu = max(self.ygp)+0.01*abs(max(self.ygp))
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlim((1, self.trialsgp+1))
        ax1.set_xlabel('Sample Number', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel('Operating cost (MM USD/yr)', fontsize = 24)
        pyp.yticks(fontsize = 24);
        ax1.scatter(itr, self.ygp, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200);
        ax1.plot(itr, self.ygp, color = 'black', linewidth = 3, label = 'BO');
        if self.ref_optim:
            itr = np.arange(1, self.trialsref+2, 1)
            ax1.scatter(itr, self.ytru, marker = 'o', color = 'white', edgecolor = 'blue',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.ytru, color = 'blue', linewidth = 3, label = 'Ref-BO');
            yliml = min(yliml, min(self.ytru)-0.01*abs(min(self.ytru)))
            ylimu = max(ylimu, max(self.ytru)+0.01*abs(max(self.ytru)))
        if self.dist_optim:
            itr = np.arange(1, self.trialsdist+2, 1)
            yplot = list(self.ydist.values())[-1]
            ax1.scatter(itr, yplot, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(itr, yplot, color = 'red', linewidth = 3, label = 'Dist BO');
            yliml = min(yliml, min(yplot)-0.01*abs(min(yplot)))
            ylimu = max(ylimu, max(yplot)+0.01*abs(max(yplot)))
        if self.distref_optim:
            itr = np.arange(1, self.trialsdr+2, 1)
            yplot = list(self.ydistref.values())[-1]
            ax1.scatter(itr, yplot, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(itr, yplot, color = 'green', linewidth = 3, label = 'Dist BO+Ref');
            yliml = min(yliml, min(yplot)-0.01*abs(min(yplot)))
            ylimu = max(ylimu, max(yplot)+0.01*abs(max(yplot)))
        if self.splt_optim:
            itr = np.arange(1, self.trialsplt+2, 1)
            ax1.scatter(itr, self.yspltbst, marker = 'o', color = 'white', edgecolor = 'purple',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.yspltbst, color = 'purple', linewidth = 3, label = 'Partioned BO');
            yliml = min(yliml, min(self.yspltbst)-0.01*abs(min(self.yspltbst)))
            ylimu = max(ylimu, max(self.yspltbst)+0.01*abs(max(self.yspltbst)))
        if self.splt_optimref:
            itr = np.arange(1, self.trialspltref+2, 1)
            ax1.scatter(itr, self.yspltbstref, marker = 'o', color = 'white', edgecolor = 'gray',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.yspltbstref, color = 'gray', linewidth = 3, label = 'LS-BO');
            yliml = min(yliml, min(self.yspltbstref)-0.01*abs(min(self.yspltbstref)))
            ylimu = max(ylimu, max(self.yspltbstref)+0.01*abs(max(self.yspltbstref)))
        if self.spltvar_optim:
            itr = np.arange(1, self.trialspltvar+2, 1)
            ax1.scatter(itr, self.yspltvarbst[:, -1], marker = 'o', color = 'white', edgecolor = 'brown',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.yspltvarbst[:, -1], color = 'brown', linewidth = 3, label = 'VP-BO');
            yliml = min(yliml, min(self.yspltvarbst[:, -1])-0.01*abs(min(self.yspltvarbst[:, -1])))
            ylimu = max(ylimu, max(self.yspltvarbst[:, -1])+0.01*abs(max(self.yspltvarbst[:, -1])))
        if self.hyp_optim:
            itr = np.arange(1, self.trialshyp+2, 1)
            ax1.scatter(itr, self.yhypbst, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.yhypbst, color = 'green', linewidth = 3, label = 'HS-BO');
            yliml = min(yliml, min(self.yhypbst)-0.01*abs(min(self.yhypbst)))
            ylimu = max(ylimu, max(self.yhypbst)+0.01*abs(max(self.yhypbst)))
        if self.expwl_optim:
            itr = np.arange(1, self.trialsexpw+2, 1)
            ax1.scatter(itr, self.ybstexpw, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.ybstexpw, color = 'red', linewidth = 3, label = 'HP-BO');
            yliml = min(yliml, min(self.ybstexpw)-0.01*abs(min(self.ybstexpw)))
            ylimu = max(ylimu, max(self.ybstexpw)+0.01*abs(max(self.ybstexpw)))
        if self.nmc_optim:
            itr = np.arange(1, self.trialsnmc+2, 1)
            ax1.scatter(itr, self.ybstnmc, marker = 'o', color = 'white', edgecolor = 'pink',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.ybstnmc, color = 'pink', linewidth = 3, label = 'MC-BO');
            yliml = min(yliml, min(self.ybstnmc)-0.01*abs(min(self.ybstnmc)))
            ylimu = max(ylimu, max(self.ybstnmc)+0.01*abs(max(self.ybstnmc)))
        if self.embd_optim:
            itr = np.arange(1, self.trialsembd+2, 1)
            ax1.scatter(itr, self.fembd, marker = 'o', color = 'white', edgecolor = 'lime',
                        zorder = 3, s = 200);
            ax1.plot(itr, self.fembd, color = 'lime', linewidth = 3, label = 'Embedded f BO')
            yliml = min(yliml, min(self.fembd)-0.01*abs(min(self.fembd)))
            ylimu = max(ylimu, max(self.fembd)+0.01*abs(max(self.fembd)))
        
        ax1.set_ylim(yliml, ylimu)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);
        
    def plotstime(self, figure_name):
        yliml = min(self.ygp)-0.01*abs(min(self.ygp))
        ylimu = max(self.ygp)+0.01*abs(max(self.ygp))
        xlimu = round(self.timegp[-1]+1, 0)
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlabel('Time (s)', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel('Operating cost (MM USD/hr)', fontsize = 24)
        pyp.yticks(fontsize = 24);
        ax1.scatter(self.timegp, self.ygp, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200);
        ax1.plot(self.timegp, self.ygp, color = 'black', linewidth = 3, label = 'BO');
        if self.ref_optim:
            ax1.scatter(self.timeref, self.ytru, marker = 'o', color = 'white', edgecolor = 'blue',
                    zorder = 3, s = 200);
            ax1.plot(self.timeref, self.ytru, color = 'blue', linewidth = 3, label = 'Ref-BO');
            yliml = min(yliml, min(self.ytru)-0.01*abs(min(self.ytru)))
            ylimu = max(ylimu, max(self.ytru)+0.01*abs(max(self.ytru)))
            xlimu = max(xlimu, self.timeref[-1]+1)
        if self.splt_optim:
            ax1.scatter(self.timesplt, self.yspltbst, marker = 'o', color = 'white', edgecolor = 'purple',
                    zorder = 3, s = 200);
            ax1.plot(self.timesplt, self.yspltbst, color = 'purple', linewidth = 3, label = 'Partioned BO');
            yliml = min(yliml, min(self.yspltbst)-0.01*abs(min(self.yspltbst)))
            ylimu = max(ylimu, max(self.yspltbst)+0.01*abs(max(self.yspltbst)))
            xlimu = max(xlimu, self.timesplt[-1]+1)
        if self.splt_optimref:
            ax1.scatter(self.timespltref, self.yspltbstref, marker = 'o', color = 'white', edgecolor = 'gray',
                    zorder = 3, s = 200);
            ax1.plot(self.timespltref, self.yspltbstref, color = 'gray', linewidth = 3, label = 'LS-BO');
            yliml = min(yliml, min(self.yspltbstref)-0.01*abs(min(self.yspltbstref)))
            ylimu = max(ylimu, max(self.yspltbstref)+0.01*abs(max(self.yspltbstref)))
            xlimu = max(xlimu, self.timespltref[-1]+1)
        if self.spltvar_optim:
            ax1.scatter(self.timespltvar, self.yspltvarbst[:, -1], marker = 'o', color = 'white', edgecolor = 'brown',
                    zorder = 3, s = 200);
            ax1.plot(self.timespltvar, self.yspltvarbst[:, -1], color = 'brown', linewidth = 3, label = 'VP-BO');
            yliml = min(yliml, min(self.yspltvarbst[:, -1])-0.01*abs(min(self.yspltvarbst[:, -1])))
            ylimu = max(ylimu, max(self.yspltvarbst[:, -1])+0.01*abs(max(self.yspltvarbst[:, -1])))
            xlimu = max(xlimu, self.timespltvar[-1]+1)
        if self.hyp_optim:
            ax1.scatter(self.timehyp, self.yhypbst, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(self.timehyp, self.yhypbst, color = 'green', linewidth = 3, label = 'HS-BO');
            yliml = min(yliml, min(self.yhypbst)-0.01*abs(min(self.yhypbst)))
            ylimu = max(ylimu, max(self.yhypbst)+0.01*abs(max(self.yhypbst)))
            xlimu = max(xlimu, self.timehyp[-1]+1)
        if self.expwl_optim:
            ax1.scatter(self.timexpw, self.ybstexpw, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(self.timexpw, self.ybstexpw, color = 'red', linewidth = 3, label = 'HP-BO');
            yliml = min(yliml, min(self.ybstexpw)-0.01*abs(min(self.ybstexpw)))
            ylimu = max(ylimu, max(self.ybstexpw)+0.01*abs(max(self.ybstexpw)))
            xlimu = max(xlimu, self.timexpw[-1]+1)
        if self.nmc_optim:
            ax1.scatter(self.timenmc, self.ybstnmc, marker = 'o', color = 'white', edgecolor = 'pink',
                    zorder = 3, s = 200);
            ax1.plot(self.timenmc, self.ybstnmc, color = 'pink', linewidth = 3, label = 'MC-BO');
            yliml = min(yliml, min(self.ybstnmc)-0.01*abs(min(self.ybstnmc)))
            ylimu = max(ylimu, max(self.ybstnmc)+0.01*abs(max(self.ybstnmc)))
            xlimu = max(xlimu, self.timenmc[-1]+1)
        if self.embd_optim:
            ax1.scatter(self.timembd, self.fembd, marker = 'o', color = 'white', edgecolor = 'lime',
                        zorder = 3, s = 200);
            ax1.plot(self.timembd, self.fembd, color = 'lime', linewidth = 3, label = 'Embedded f BO')
            yliml = min(yliml, min(self.fembd)-0.01*abs(min(self.fembd)))
            ylimu = max(ylimu, max(self.fembd)+0.01*abs(max(self.fembd)))
            xlimu = max(xlimu, self.timembd[-1]+1)
                
        ax1.set_ylim(yliml, ylimu)
        ax1.set_xlim(0, xlimu)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'_time.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);
        
    def plotexptime(self, figure_name):
        yliml = min(self.ygp)-0.01*abs(min(self.ygp))
        ylimu = max(self.ygp)+0.01*abs(max(self.ygp))
        xlimu = round(self.timefgp[-1]+1, 0)
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlabel('Time (s)', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel('Operating cost (MM USD/hr)', fontsize = 24)
        pyp.yticks(fontsize = 24);
        ax1.scatter(self.timefgp, self.ygp, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200);
        ax1.plot(self.timefgp, self.ygp, color = 'black', linewidth = 3, label = 'BO');
        if self.ref_optim:
            ax1.scatter(self.timefref, self.ytru, marker = 'o', color = 'white', edgecolor = 'blue',
                    zorder = 3, s = 200);
            ax1.plot(self.timefref, self.ytru, color = 'blue', linewidth = 3, label = 'Ref-BO');
            yliml = min(yliml, min(self.ytru)-0.01*abs(min(self.ytru)))
            ylimu = max(ylimu, max(self.ytru)+0.01*abs(max(self.ytru)))
            xlimu = max(xlimu, self.timefref[-1]+1)
        if self.splt_optim:
            ax1.scatter(self.timefsplt, self.yspltbst, marker = 'o', color = 'white', edgecolor = 'purple',
                    zorder = 3, s = 200);
            ax1.plot(self.timefsplt, self.yspltbst, color = 'purple', linewidth = 3, label = 'Partioned BO');
            yliml = min(yliml, min(self.yspltbst)-0.01*abs(min(self.yspltbst)))
            ylimu = max(ylimu, max(self.yspltbst)+0.01*abs(max(self.yspltbst)))
            xlimu = max(xlimu, self.timefsplt[-1]+1)
        if self.splt_optimref:
            ax1.scatter(self.timefspltref, self.yspltbstref, marker = 'o', color = 'white', edgecolor = 'gray',
                    zorder = 3, s = 200);
            ax1.plot(self.timefspltref, self.yspltbstref, color = 'gray', linewidth = 3, label = 'LS-BO');
            yliml = min(yliml, min(self.yspltbstref)-0.01*abs(min(self.yspltbstref)))
            ylimu = max(ylimu, max(self.yspltbstref)+0.01*abs(max(self.yspltbstref)))
            xlimu = max(xlimu, self.timefspltref[-1]+1)
        if self.spltvar_optim:
            ax1.scatter(self.timefspltvar, self.yspltvarbst[:, -1], marker = 'o', color = 'white', edgecolor = 'brown',
                    zorder = 3, s = 200);
            ax1.plot(self.timefspltvar, self.yspltvarbst[:, -1], color = 'brown', linewidth = 3, label = 'VP-BO');
            yliml = min(yliml, min(self.yspltvarbst[:, -1])-0.01*abs(min(self.yspltvarbst[:, -1])))
            ylimu = max(ylimu, max(self.yspltvarbst[:, -1])+0.01*abs(max(self.yspltvarbst[:, -1])))
            xlimu = max(xlimu, self.timefspltvar[-1]+1)
        if self.hyp_optim:
            ax1.scatter(self.timefhyp, self.yhypbst, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(self.timefhyp, self.yhypbst, color = 'green', linewidth = 3, label = 'HS-BO');
            yliml = min(yliml, min(self.yhypbst)-0.01*abs(min(self.yhypbst)))
            ylimu = max(ylimu, max(self.yhypbst)+0.01*abs(max(self.yhypbst)))
            xlimu = max(xlimu, self.timefhyp[-1]+1)
        if self.expwl_optim:
            ax1.scatter(self.timefexpw, self.ybstexpw, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(self.timefexpw, self.ybstexpw, color = 'red', linewidth = 3, label = 'HP-BO');
            yliml = min(yliml, min(self.ybstexpw)-0.01*abs(min(self.ybstexpw)))
            ylimu = max(ylimu, max(self.ybstexpw)+0.01*abs(max(self.ybstexpw)))
            xlimu = max(xlimu, self.timefexpw[-1]+1)
        if self.nmc_optim:
            ax1.scatter(self.timefnmc, self.ybstnmc, marker = 'o', color = 'white', edgecolor = 'pink',
                    zorder = 3, s = 200);
            ax1.plot(self.timefnmc, self.ybstnmc, color = 'pink', linewidth = 3, label = 'MC-BO');
            yliml = min(yliml, min(self.ybstnmc)-0.01*abs(min(self.ybstnmc)))
            ylimu = max(ylimu, max(self.ybstnmc)+0.01*abs(max(self.ybstnmc)))
            xlimu = max(xlimu, self.timefnmc[-1]+1)
        if self.embd_optim:
            ax1.scatter(self.timefembd, self.fembd, marker = 'o', color = 'white', edgecolor = 'lime',
                        zorder = 3, s = 200);
            ax1.plot(self.timefembd, self.fembd, color = 'lime', linewidth = 3, label = 'Embedded f BO')
            yliml = min(yliml, min(self.fembd)-0.01*abs(min(self.fembd)))
            ylimu = max(ylimu, max(self.fembd)+0.01*abs(max(self.fembd)))
            xlimu = max(xlimu, self.timefembd[-1]+1)
                
        ax1.set_ylim(yliml, ylimu)
        ax1.set_xlim(0, xlimu)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'_exptime.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);

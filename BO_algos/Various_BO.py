import numpy as np
from matplotlib import pyplot as pyp,cm
from scipy.optimize import minimize, Bounds, fsolve, LinearConstraint
from joblib import Parallel, delayed
import sklearn.gaussian_process as gpr
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D

class LCB_AF():
    def __init__(self, model, exp_w, descale, **refmod):
        self.model = model
        self.exp_w = exp_w
        self.descale = descale
        if refmod:
            self.refmod = refmod['refmod']
        else:
            def zr(x):
                return 0
            self.refmod = zr
    def LCB(self, x):
        x = np.array([x]).reshape(-1,1); x=x.reshape(1,x.shape[0])
        mu, std = self.model.predict(x, return_std=True); std=std.reshape(-1,1)
        if str(type(self.refmod))=="<class '__main__.Network'>":
            yref = self.refmod(torch.from_numpy(self.descale(x)).float()).data.numpy()  
        else:
            yref = self.refmod(self.descale(x))
        return (yref+mu-self.exp_w*std).flatten()

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
                self.distmod = list(aux_mods.values())[1]
                for i in range(2,len(aux_mods)):
                    self.dist_ref['distrefmod'+str(i-1)] = list(aux_mods.values())[i]
                self.dist_ref['distrefmod'] = self.refmod['refmod']
                self.dist_ref = OrderedDict(self.dist_ref)
                    

    def descale(self, x, scaling_factor = 1):
        # b = (self.ub+self.lb)/2
        # m = (b-self.lb)/scaling_factor
        m = (self.ub-self.lb)/scaling_factor
        b = self.lb
        return m*x+b
    
    def scale(self, x, scaling_factor = 1):
        m = 2*scaling_factor/(self.ub-self.lb)
        b = scaling_factor-m*self.ub
        # m = scaling_factor/(self.ub-self.lb)
        # b = -self.lb/(self.ub-self.lb)
        return m*x+b
    
    def optimizergp(self, trials, xinit = None):
        print('Vanilla BO Run...')
        self.trialsgp = trials
        if xinit is None:
            # x = np.random.uniform(-1, 1, (1, self.dim))
            x = np.random.uniform(0, 1, (1, self.dim))
        else:
            x = xinit.reshape(1, self.dim)
        y = self.system(self.descale(x))
        modelgp = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, 
                                                  normalize_y = True,
                                                  n_restarts_optimizer = 10)
        modelgp.fit(x, y)
        # x0 = np.random.uniform(-1, 1, (100, self.dim))
        x0 = np.random.uniform(0, 1, (100, self.dim))
        LCBgp = LCB_AF(modelgp, self.exp_w, self.descale).LCB
        opt = Parallel(n_jobs = -1)(delayed(minimize)(LCBgp, x0 = start_point,
                                                      method = 'L-BFGS-B',
                                                      bounds = self.bounds)
                                    for start_point in x0)
        xnxts = np.array([res.x for res in opt], dtype  = 'float')
        funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        for i in range(self.trialsgp):
            xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
            ynxt = self.system(self.descale(xnxt))
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            modelgp.fit(x, y)
            # x0 = np.random.uniform(-1, 1, (100, self.dim))
            x0 = np.random.uniform(0, 1, (100, self.dim))
            opt = Parallel(n_jobs = -1)(delayed(minimize)(LCBgp, x0 = start_point,
                                                      method = 'L-BFGS-B',
                                                      bounds = self.bounds)
                                    for start_point in x0)
            xnxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        self.modelgp = modelgp
        self.xgp = self.descale(x)
        self.ygp = y
        self.ref_optim = False
        self.dist_optim = False
        self.distref_optim = False
        self.splt_optim = False
        self.spltvar_optim = False
        
    def optimizeref(self, trials, xinit = None):
        print('BO with Reference Model Run...')
        self.trialsref = trials
        refmod = self.refmod['refmod']
        if xinit is None:
            # x = np.random.uniform(-1, 1, (1, self.dim))
            x = np.random.uniform(0, 1, (1, self.dim))
        else:
            x = xinit.reshape(1, self.dim)
        y = self.system(self.descale(x))
        eps = y - refmod(self.descale(x))
        self.modeleps = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, 
                                                  normalize_y = True,
                                                  n_restarts_optimizer = 10)
        self.modeleps.fit(x, eps)
        # x0 = np.random.uniform(-1, 1, (100, self.dim))
        x0 = np.random.uniform(0, 1, (100, self.dim))
        LCBref = LCB_AF(self.modeleps, self.exp_w, self.descale, **self.refmod).LCB
        opt = Parallel(n_jobs = -1)(delayed(minimize)(LCBref, x0 = start_point,
                                                      method = 'L-BFGS-B',
                                                      bounds = self.bounds)
                                    for start_point in x0)
        xnxts = np.array([res.x for res in opt], dtype  = 'float')
        funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        for i in range(self.trialsref):
            xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
            ynxt = self.system(self.descale(xnxt))
            epsnxt = ynxt - refmod(self.descale(xnxt))
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            eps = np.vstack([eps, epsnxt])
            self.modeleps.fit(x, eps)
            # x0 = np.random.uniform(-1, 1, (100, self.dim))
            x0 = np.random.uniform(0, 1, (100, self.dim))
            opt = Parallel(n_jobs = -1)(delayed(minimize)(LCBref, x0 = start_point,
                                                      method = 'L-BFGS-B',
                                                      bounds = self.bounds)
                                    for start_point in x0)
            xnxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        self.ref_optim = True
        self.xref = self.descale(x)
        self.ytru = y
        self.eps = eps
    
    def optimizerdist(self, trials, distributions, xinit = None):
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
            LCBdist['dist'+str(i)] = LCB_AF(funcs[idx], self.exp_w, self.descale).LCB
        funcs['modeldist'] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, 
                                                  normalize_y = True,
                                                  n_restarts_optimizer = 10)
        y['dist'] = self.distmod(self.descale(x))[-1]
        funcs = OrderedDict(funcs)
        y = OrderedDict(y)
        LCBdist = OrderedDict(LCBdist)
        # x0 = np.random.uniform(-1, 1, (100, self.dim))
        x0 = np.random.uniform(0, 1, (100, self.dim))
        opt = Parallel(n_jobs = -1)(delayed(minimize)(LCBdist['dist1'], x0 = start_point,
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
            x0 = np.random.uniform(0, 1, (100, self.dim))
            opt = Parallel(n_jobs = -1)(delayed(minimize)(LCBiter, x0 = start_point,
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
            LCBdr[idx2] = LCB_AF(epsmod[idx], self.exp_w, self.descale, **refmod).LCB
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
        x0 = np.random.uniform(0, 1, (100, self.dim))
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
            x0 = np.random.uniform(0, 1, (100, self.dim))
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
        
    def optimizersplt(self, trials, partition_num, partition_mtrx, high, scaling_factor, xinit = None):
        # partition_bnds should be a set of matrices, A, that set up a set of
        # inequalities so that A*x < ub this constrains feasibilty region and
        # sets up effective partition in desired shape
        print('Partitioned Domain BO Run...')
        self.trialsplt = trials
        self.split = partition_num
        self.A = partition_mtrx
        self.high = high
        self.lincon = {}
        sf = scaling_factor
        x = np.array([]).reshape(0, self.dim)
        switch = True
        def intpts(x):
            return 0
        for i in range(self.split):
            lwr = -self.dim*np.ones((self.high[i].shape[0]))
            self.lincon['partition'+str(i+1)] = LinearConstraint(self.A[i], lwr, self.high[i])
            if xinit is not None and sum(self.A[i]@xinit.flatten() <= self.high[i]) == 2 and switch == True:
                x0 = xinit.reshape(1, self.dim)
                switch = False
            else:
                # x0 = np.random.uniform(-sf, sf, (self.dim,))
                x0 = np.random.uniform(0, sf, (self.dim,))
                x0 = minimize(intpts, x0, bounds = self.bounds, constraints = self.lincon['partition'+str(i+1)]).x
            x = np.vstack([x, x0.reshape(1, self.dim)])
        y = Parallel(n_jobs = -1)(delayed(self.system)(start_point) for start_point in self.descale(x))
        y = np.hstack(y[:]).T.reshape(-1,1)
        ybst = min(y).reshape(-1,1)
        modelsplt = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                 n_restarts_optimizer = 10,
                                                 normalize_y = True)
        modelsplt.fit(x, y)
        xnxt = np.ones((self.split, self.dim))
        LCB = LCB_AF(modelsplt, self.exp_w, self.descale).LCB
        x0 = np.random.uniform(0, sf, (100, self.dim))
        for i in range(self.split):
            # x0 = np.random.uniform(-sf, sf, (100, self.dim))
            opt = Parallel(n_jobs = -1)(delayed(minimize)(LCB, x0 = start_points,
                                                          method = 'SLSQP',
                                                          bounds = self.bounds,
                                                          constraints = self.lincon['partition'+str(i+1)])
                                        for start_points in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            xnxt[i] = xnxts[np.argmin(funs)]
        for i in range(self.trialsplt):
            ynxt =  Parallel(n_jobs = -1)(delayed(self.system)(start_point) for start_point in self.descale(xnxt))
            ynxt = np.hstack(ynxt[:]).T.reshape(-1,1)
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            ybst = np.vstack([ybst, min(ynxt).reshape(-1,1)])
            modelsplt.fit(x, y)
            x0 = np.random.uniform(0, sf, (100, self.dim))
            for i in range(self.split):
                # x0 = np.random.uniform(-sf, sf, (100, self.dim))
                opt = Parallel(n_jobs = -1)(delayed(minimize)(LCB, x0 = start_points,
                                                              method = 'SLSQP',
                                                              bounds = self.bounds,
                                                              constraints = self.lincon['partition'+str(i+1)])
                                            for start_points in x0)
                xnxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                xnxt[i] = xnxts[np.argmin(funs)]
        self.splt_optim = True
        self.modelsplt = modelsplt
        self.xsplt = self.descale(x)
        self.ysplt = y
        self.yspltbst = ybst
    
    def optimizerspltvar(self, trials, split_num, scaling_factor):
        # Split partition using variables as split point
        print('Partitioned Variables BO Run...')
        self.trialspltvar = trials
        sf = scaling_factor
        self.splitvar = split_num
        # x = np.random.uniform(-sf, sf, (self.splitvar, self.dim))
        x = np.random.uniform(0, sf, (self.splitvar, self.dim))
        y = Parallel(n_jobs = -1)(delayed(self.system)(start_point) for start_point in self.descale(x))
        y = np.hstack(y).T.reshape(-1, 1)
        ybst = min(y).reshape(-1, 1)
        modelspltvar = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, 
                                                    n_restarts_optimizer = 10,
                                                    normalize_y = True)
        modelspltvar.fit(x, y)
        xnxt = x.copy()
        self.bndsvar = {}
        LCB = LCB_AF(modelspltvar, self.exp_w, self.descale).LCB
        J = []
        x0 = np.random.uniform(0, sf, (100, self.dim))
        for i in range(self.splitvar):
            j = np.random.randint(0, x.shape[0])
            while j in J:
                j = np.random.randint(0, x.shape[0])
            J.append(j)
            lwr = x[j].flatten()
            upr = x[j].flatten()+1e-6
            # lwr[i%self.dim] = -sf
            lwr[i%self.dim] = 0
            upr[i%self.dim] = sf
            self.bndsvar['prt'+str(i+1)] = Bounds(lwr, upr)
            # x0 = np.random.uniform(-sf, sf, (100, self.dim))
            opt = Parallel(n_jobs = -1)(delayed(minimize)(LCB, x0 = start_point,
                                                          method = 'L-BFGS-B',
                                                          bounds = self.bndsvar['prt'+str(i+1)])
                                        for start_point in x0)
            xnxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            xnxt[i] = xnxts[np.argmin(funs)]
        for i in range(self.trialspltvar):
            ynxt = Parallel(n_jobs = -1)(delayed(self.system)(start_point) for start_point in self.descale(xnxt))
            ynxt = np.hstack(ynxt[:]).T.reshape(-1, 1)
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            ybst = np.vstack([ybst, min(ynxt).reshape(-1,1)])
            modelspltvar.fit(x, y)
            J = []
            # x0 = np.random.uniform(-sf, sf, (100, self.dim))
            x0 = np.random.uniform(0, sf, (100, self.dim))
            for i in range(self.splitvar):
                j = np.random.randint(0, x.shape[0])
                while j in J:
                    j = np.random.randint(0, x.shape[0])
                J.append(j)
                lwr = x[j].flatten()
                upr = x[j].flatten()+1e-6
                # lwr[i%self.dim] = -sf
                lwr[i%self.dim] = 0
                upr[i%self.dim] = sf
                self.bndsvar['prt'+str(i+1)] = Bounds(lwr, upr)
                # x0 = np.random.uniform(-sf, sf, (100, self.dim))
                opt = Parallel(n_jobs = -1)(delayed(minimize)(LCB, x0 = start_point,
                                                              method = 'L-BFGS-B',
                                                              bounds = self.bndsvar['prt'+str(i+1)])
                                            for start_point in x0)
                xnxts = np.array([res.x for res in opt], dtype  = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                xnxt[i] = xnxts[np.argmin(funs)]
        self.spltvar_optim = True
        self.modelspltvar = modelspltvar
        self.xspltvar = self.descale(x)
        self.yspltvar = y
        self.yspltvarbst = ybst
    
    def plots(self, figure_name):
        itr = np.arange(1, self.trialsgp+2, 1)
        yliml = min(self.ygp)-0.1*abs(min(self.ygp))
        ylimu = max(self.ygp)+0.1*abs(max(self.ygp))
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlim((1, self.trialsgp+1))
        ax1.set_xlabel(r'$Sample\ Number$', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel(r'$Operating\ cost\ (10k\ USD/hr)$', fontsize = 24)
        pyp.yticks(fontsize = 24);
        ax1.scatter(itr, self.ygp, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200);
        ax1.plot(itr, self.ygp, color = 'black', linewidth = 3, label = 'BO');
        if self.ref_optim:
            itr = np.arange(1, self.trialsref+2, 1)
            ax1.scatter(itr, self.ytru, marker = 'o', color = 'white', edgecolor = 'blue',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.ytru, color = 'blue', linewidth = 3, label = 'BO+Ref');
            yliml = min(yliml, min(self.ytru)-0.1*abs(min(self.ytru)))
            ylimu = max(ylimu, max(self.ytru)+0.1*abs(max(self.ytru)))
        if self.dist_optim:
            itr = np.arange(1, self.trialsdist+2, 1)
            yplot = list(self.ydist.values())[-1]
            ax1.scatter(itr, yplot, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(itr, yplot, color = 'red', linewidth = 3, label = 'Dist BO');
            yliml = min(yliml, min(yplot)-0.1*abs(min(yplot)))
            ylimu = max(ylimu, max(yplot)+0.1*abs(max(yplot)))
        if self.distref_optim:
            itr = np.arange(1, self.trialsdr+2, 1)
            yplot = list(self.ydistref.values())[-1]
            ax1.scatter(itr, yplot, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(itr, yplot, color = 'green', linewidth = 3, label = 'Dist BO+Ref');
            yliml = min(yliml, min(yplot)-0.1*abs(min(yplot)))
            ylimu = max(ylimu, max(yplot)+0.1*abs(max(yplot)))
        if self.splt_optim:
            itr = np.arange(1, self.trialsplt+2, 1)
            ax1.scatter(itr, self.yspltbst, marker = 'o', color = 'white', edgecolor = 'purple',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.yspltbst, color = 'purple', linewidth = 3, label = 'Partioned BO');
            yliml = min(yliml, min(self.yspltbst)-0.1*abs(min(self.yspltbst)))
            ylimu = max(ylimu, max(self.yspltbst)+0.1*abs(max(self.yspltbst)))
        if self.spltvar_optim:
            itr = np.arange(1, self.trialspltvar+2, 1)
            ax1.scatter(itr, self.yspltvarbst, marker = 'o', color = 'white', edgecolor = 'brown',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.yspltvarbst, color = 'brown', linewidth = 3, label = 'Partioned Variable BO');
            yliml = min(yliml, min(self.yspltvarbst)-0.1*abs(min(self.yspltvarbst)))
            ylimu = max(ylimu, max(self.yspltvarbst)+0.1*abs(max(self.yspltvarbst)))
            
        ax1.set_ylim(yliml, ylimu)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);
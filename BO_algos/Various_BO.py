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
        ax1.set_ylim(yliml, ylimu)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);

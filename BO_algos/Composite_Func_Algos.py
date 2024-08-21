import numpy as np
from matplotlib import pyplot as pyp
from scipy.optimize import minimize, Bounds, approx_fprime, NonlinearConstraint
from joblib import Parallel, delayed
import sklearn.gaussian_process as gpr
from collections import OrderedDict
import time


class LCB_AF():
    def __init__(self, model, dim, exp_w, descale, refmod = None, args = ()):
        self.model = model
        self.dim = dim
        self.exp_w = exp_w
        self.descale = descale
        self.args = args
        
        if refmod:
            self.refmod = refmod
        else:
            def zr(x):
                return 0
            self.refmod = zr
            
    def LCB(self, x):
        x = np.array([x]).reshape(-1,1);
        x = x.reshape(int(x.shape[0]/self.dim), self.dim)
        
        mu, std = self.model.predict(x, return_std=True);
        mu = mu.flatten()
        
        yref = self.refmod(self.descale(x), *self.args)
            
        return (yref+mu-self.exp_w*std).flatten()


class LCB_BOIS():
    def __init__(self, model, gp_sim, f, exp_w, descale, eps, idx,
                 gp_args = (), f_args = (), args = (),
                 f_prime_regularizer = None,
                 feasibility_check = False,
                 lb = None, ub = None, clip_to_bounds = False,
                 norm_xdat = False, refmod = None):
        self.model = model
        self.gp_sim = gp_sim
        self.f = f
        self.exp_w = exp_w
        self.descale = descale
        self.eps = eps
        self.idx = idx
        self.gp_args = gp_args
        self.f_args = f_args
        self.args = args
        self.f_prime_regularizer = f_prime_regularizer
        self.feasibility_check = feasibility_check
        self.lb = lb
        self.ub = ub
        self.clip_to_bounds = clip_to_bounds
        self.norm_xdat = norm_xdat
        
        if refmod:
            self.refmod = refmod
        else:
            def zr(x):
                return 0
            self.refmod = zr
    
    def transform_Y(self, Y):
        self.Y = []
        for i, y in enumerate(Y):
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            self.Y.append(y[:, self.idx[1][i]])
        self.Y = np.hstack(self.Y)
        
    def f_prime(self):
        y0 = (1+self.eps)*self.Y
        if self.idx[0] != None:
            fp = [approx_fprime(i, self.f, i*self.eps, *(j, *self.f_args)) for (i, j) in zip(y0, self.x)]
            fp = np.array(fp)
            f_args = (self.x,)+self.f_args
        else:
            fp = [approx_fprime(i, self.f, i*self.eps, *self.f_args) for i in y0]
            fp = np.array(fp)
            f_args = self.f_args
        return fp, f_args
         
    def LCB(self, x, mu_Y, sigma_Y, mu_x = None, sigma_x = None, moments = False):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1).T
        
        mu, sigma = self.gp_sim(x, self.model, mu_Y, sigma_Y, *self.gp_args)
        self.transform_Y(sigma)
        sigma = self.Y
        self.transform_Y(mu)
        mu = self.Y
        
        if self.feasibility_check:
            if self.clip_to_bounds:
                row_l, col_l = np.where(mu<self.lb)
                row_u, col_u = np.where(mu>self.ub)
                mu[row_l, col_l] = self.lb[col_l]
                mu[row_u, col_u] = self.ub[col_u]
            
            else:
                mu_dummy = mu.copy()
                
                for i, u in enumerate(mu_dummy):
                    while any(u<self.lb) or any(u>self.ub):
                        row_l = np.where(u<self.lb)[0]
                        row_u = np.where(u>self.ub)[0]
                            
                        u[row_l] = np.random.normal(mu[i, row_l], sigma[i, row_l])
                        u[row_u] = np.random.normal(mu[i, row_u], sigma[i, row_u])
    
                        row_l = np.where(u<self.lb)[0]
                        row_u = np.where(u>self.ub)[0]
    
                        cutoff_l = (self.lb[row_l]-u[row_l])/sigma[i, row_l]
                        cutoff_u = (u[row_u]-self.ub[row_u])/sigma[i, row_u]
    
                        fix_l = np.where(cutoff_l>3)[0]
                        fix_u = np.where(cutoff_u>3)[0]
    
                        u[row_l[fix_l]] = mu[i, row_l][fix_l]+cutoff_l[fix_l]*sigma[i, row_l][fix_l]
                        u[row_u[fix_u]] = mu[i, row_u][fix_u]-cutoff_u[fix_u]*sigma[i, row_u][fix_u]
                            
                    mu_dummy[i, :] = u
                    
                mu = mu_dummy.copy()
                del mu_dummy
                
            self.Y = mu
        
        if self.norm_xdat:
            self.x = self.descale(unnormalize(x, mu_x, sigma_x))
        else:
            self.x = self.descale(x)
            
        y0 = (1+self.eps)*mu
        fp, f_args = self.f_prime()
        f0 = self.f(y0, *f_args)
        b = [i-j.T@k for i, j, k, in zip(f0, fp, y0)]
        b = np.array(b)
        
        mu_f = [i.T@j+k for i, j, k in zip(mu, fp, b)]
        mu_f = np.array(mu_f)
        
        if self.f_prime_regularizer is not None:
            if type(self.f_prime_regularizer) == np.ndarray:
                fp = fp*self.f_prime_regularizer
                
            elif type(self.f_prime_regularizer) == float:
                self.f_prime_regularizer = self.f_prime_regularizer*np.ones(len(fp))
                fp = fp*self.f_prime_regularizer
                
            else:
                raise Exception("f_prime_regularizer must be a float or numpy array")
        
        sigma_f = [(i**2).T@(j**2) for i, j in zip(sigma, fp)]
        sigma_f = np.array(sigma_f)**0.5
        
        yref = self.refmod(self.x, *self.args)
        
        if moments:
            return mu_f, sigma_f
        else:
            return (yref+mu_f-self.exp_w*sigma_f).flatten()


class LCB_MCBO():
    def __init__(self, model, gp_sim, f, exp_w, descale, n_samples, idx,
                 gp_args = (), f_args = (), args = (),
                 feasibility_check = False,
                 lb = None, ub = None, clip_to_bounds = False,
                 refmod = None, norm_xdat = False):
        self.model = model
        self.gp_sim = gp_sim
        self.f = f
        self.exp_w = exp_w
        self.descale = descale
        self.n_samples = n_samples
        self.idx = idx
        self.gp_args = gp_args
        self.f_args = f_args
        self.args = args
        self.feasibility_check = feasibility_check
        self.lb = lb
        self.ub = ub
        self.clip_to_bounds = clip_to_bounds
        self.norm_xdat = norm_xdat
        
        if refmod:
            self.refmod = refmod
        else:
            def zr(x):
                return 0
            self.refmod = zr
    
    def transform_Y(self, Y):
        self.Y = []
        for i, y in enumerate(Y):
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            self.Y.append(y[:, self.idx[1][i]])
        self.Y = np.hstack(self.Y)
         
    def LCB(self, x, mu_Y, sigma_Y, mu_x = None, sigma_x = None, moments = False):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1).T
        
        mu, sigma = self.gp_sim(x, self.model, mu_Y, sigma_Y, *self.gp_args)
        self.transform_Y(sigma)
        sigma = self.Y
        self.transform_Y(mu)
        mu = self.Y
        f = np.zeros((x.shape[0], self.n_samples))
        
        if self.feasibility_check:
            mu_dummy = mu.copy()
            
            for i, u in enumerate(mu_dummy):
                while any(u<self.lb) or any(u>self.ub):
                    row_l = np.where(u<self.lb)[0]
                    row_u = np.where(u>self.ub)[0]
                    
                    if self.clip_to_bounds:
                        u[row_l] = self.lb[row_l]
                        u[row_u] = self.ub[row_u]
                        
                    else:
                        u[row_l] = np.random.normal(mu[i, row_l], sigma[i, row_l])
                        u[row_u] = np.random.normal(mu[i, row_u], sigma[i, row_u])
                        
                        row_l = np.where(u<self.lb)[0]
                        row_u = np.where(u>self.ub)[0]
                        
                        cutoff_l = (self.lb[row_l]-u[row_l])/sigma[i, row_l]
                        cutoff_u = (u[row_l]-self.ub[row_u])/sigma[i, row_u]
                        
                        fix_l = np.where(cutoff_l>3)[0]
                        fix_u = np.where(cutoff_u>3)[0]
                        
                        u[row_l[fix_l]] = mu[i, row_l][fix_l]+cutoff_l[fix_l]*sigma[i, row_l][fix_l]
                        u[row_u[fix_u]] = mu[i, row_u][fix_u]-cutoff_u[fix_u]*sigma[i, row_u][fix_u]
                        
                mu_dummy[i, :] = u
                
            mu = mu_dummy.copy()
            self.Y = mu
            del mu_dummy
        
        if self.norm_xdat:
            self.x = self.descale(unnormalize(x, mu_x, sigma_x))
        else:
            self.x = self.descale(x)
        
        for j in range(len(self.x)):
            Sigma = np.diag((sigma[j]**2).flatten())
            L = np.linalg.cholesky(Sigma)
        
            for i in range(self.n_samples):
                z = np.random.normal(np.zeros(mu[j].shape), np.ones(mu[j].shape), mu[j].shape)
                y = mu[j]+(L@z)
            
                if self.idx[0] != None:
                    f[j, i] = self.f(y, self.x[j], *self.f_args)
            
                else:
                    f[j, i] = self.f(y, *self.f_args)
        
        mu_f = np.mean(f, axis = 1)
        sigma_f = np.std(f, axis = 1, ddof = 1)
        yref = self.refmod(self.descale(self.x), *self.args)
        
        if moments:
            return mu_f, sigma_f
        else:
            return (yref+mu_f-self.exp_w*sigma_f).flatten()
        

class OPTBO_AF():
    def __init__(self, f, descale, dim, idx,
                 f_args = (), norm_xdat = False):
        self.f = f
        self.descale = descale
        self.dim = dim
        self.idx = idx
        self.f_args = f_args
        self.norm_xdat = norm_xdat
    
    
    def transform_Y(self, Y):
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        self.Y = Y[:, self.idx[1]]
            
         
    def AF(self, X, mu_x = None, sigma_x = None):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1).T
        
        x = X[:, :self.dim]
        y = X[:, self.dim:]
        self.transform_Y(y)
        
        if self.norm_xdat:
            self.x = self.descale(unnormalize(x, mu_x, sigma_x))
        else:
            self.x = self.descale(x)
        
        if self.idx[0] is None:
            f = self.f(self.Y, *self.f_args)
        
        else:
            f = self.f(self.Y, self.x, *self.f_args)
        
        return f
    

def gp_sim_bounds(X, gp_sim, model, mu_Y, sigma_Y, dim, exp_w, feasible_bound, lcb, *gp_args):
    x = X[:dim]
    y = X[dim:]
    mu, sigma = gp_sim(x.reshape(-1, dim), model, mu_Y, sigma_Y, *gp_args)
    
    mu = np.hstack(mu).flatten()
    sigma = np.hstack(sigma).flatten()
    
    if lcb:
        bound = mu-exp_w*sigma
        row_l = np.where(bound<feasible_bound)[0]
        bound[row_l] = feasible_bound[row_l]
        return y-bound
    
    else:
        bound = mu+exp_w*sigma
        row_u = np.where(bound>feasible_bound)[0]
        bound[row_u] = feasible_bound[row_u]
        return bound-y
    

def transform_X(x, Y, x_idx, y_idx):
    x_train = []
    
    for index, index1, index2 in zip(x_idx, y_idx[0], y_idx[1]):
        
        if index is not None and index1 is not None:
            Yt = [Y[i][:, j] for i, j in zip(index1, index2)]
            x_train.append(np.hstack([x[:, index], np.hstack(Yt)]))
            
        elif index1 is None:
            x_train.append(x[:, index])
            
        elif index is None:
            Yt = [Y[i][:, j] for i, j in zip(index1, index2)]
            x_train.append(np.hstack(Yt))
            
    return x_train


def normalize(x, axis = None, mu = None, sigma = None):
    if mu is None:
        mu = np.mean(x, axis = axis)
        sigma = np.std(x, axis = axis, ddof = 1)
        return (x-mu)/sigma, mu, sigma
    
    else:
        return (x-mu)/sigma


def unnormalize(x, mu, sigma):
    return sigma*x+mu


class BO():
    def __init__(self, ub, lb, dim, exp_w, kernel, system, bounds, args = (),
                 ref_args = (), shift_exp_w = [], **aux_mods):
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
            self.refmod = list(aux_mods.values())[0]
            if len(aux_mods) > 1:
                self.distmod = aux_mods['distmod']
                self.dist_ref['distrefmod'] = aux_mods['ref_distmod']
                for i in range(3,len(aux_mods)):
                    self.dist_ref['distrefmod'+str(i-2)] = aux_mods['ref_distmod'+str(i-2)]
                self.dist_ref = OrderedDict(self.dist_ref)
        
        self.bois_optim = False
        self.mcbo_optim = False
        self.opbo_optim = False
           
         
    def descale(self, x):
        m = (self.ub-self.lb)/(self.bounds.ub-self.bounds.lb)
        b = self.ub-m*self.bounds.ub
        return m*x+b
    
    
    def scale(self, x, use_self = True, lb = None, ub = None):
        if use_self:    
            m = (self.bounds.ub-self.bounds.lb)/(self.ub-self.lb)
            b = self.bounds.ub-m*self.ub
        else:
            m = (self.bounds.ub-self.bounds.lb)/(ub-lb)
            b = self.bounds.ub-m*ub
        return m*x+b
    
    
    def optimizer_sbo(self, trials,
                      af_cores = 1, x_init = None, init_pts = 1, restarts = 128):
        
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
        LCBgp = LCB_AF(model_sbo, self.dim, self.exp_w, self.descale).LCB
        
        end = time.time()
        for i in range(init_pts):
            self.time_sbo[i] = (i+1)*(end-start)/init_pts
            self.time_fsbo[i] = (i+1)*(end_f-start_f)/init_pts
        
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_sbo[init_pts-1]:.1f}')
              
        for i in range(self.trials_sbo-init_pts):
            x0 = np.random.uniform(self.bounds.lb,
                                   self.bounds.ub,
                                   (restarts, self.dim))
            opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCBgp,
                                                                x_0,
                                                                method = 'L-BFGS-B',
                                                                bounds = self.bounds)
                                              for x_0 in x0)
            x_nxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            self.x_nxt = x_nxts[np.argmin(funs)].reshape(1, self.dim)
            
            start_f = time.time()
            y_nxt = self.system(self.descale(self.x_nxt), *self.args).reshape(-1, 1)
            end_f = time.time()
            self.time_fsbo[i+init_pts] = self.time_fsbo[i+init_pts-1]+(end_f-start_f)
            
            x = np.vstack([x, self.x_nxt])
            y = np.vstack([y, y_nxt])
            model_sbo.fit(x, y)

            end = time.time()
            self.time_sbo[i+init_pts] = end-start
            
            print('ITERATION COUNT IS AT 'f'{init_pts+i+1};\
                  TOTAL ELAPSED TIME: 'f'{self.time_sbo[i+init_pts]:.1f}')
                  
        self.model_sbo = model_sbo
        self.x_sbo = self.descale(x)
        self.y_sbo = y
        
        
    def optimizer_bois(self, trials, init_pts, eps,
                       idx, x_idx, y_idx,
                       gp_sim, cost_fun,
                       restarts = 100, af_cores = 1,
                       f_args = (), gp_args = (),
                       x_init = None,
                       kernel_length_scale_bnds = None, nu = None,
                       split_gps = False, norm_xdat = False,
                       f_prime_regularizer = None,
                       feasibility_check = False, 
                       feasible_lb = None, feasible_ub = None,
                       clip_to_bounds = False):
        
        """
        BOIS algorithm 
        """
        
        print('BOIS run...')
        start = time.time()
        k_bnds = kernel_length_scale_bnds
        self.trials_bois = trials
        self.time_bois = np.zeros(trials)
        self.time_fbois = np.zeros(trials)
        
        if x_init is None:
            x = np.random.uniform(self.bounds.lb,
                                  self.bounds.ub,
                                  (init_pts, self.dim))
        
        else:
            x = np.vstack([x_init.reshape(-1, self.dim), np.random.uniform(self.bounds.lb,
                                                                           self.bounds.ub,
                                                                           (init_pts-len(x_init), self.dim))])
        
        start_f = time.time()
        
        Y_o = [(self.system(self.descale(i), *self.args)) for i in x]
        Y = list(Y_o[0])
        
        for i in range(1, len(Y_o)):
            for j, (y, y_nxt) in enumerate(zip(Y, Y_o[i])):
                Y[j] = np.vstack([y, y_nxt])
        
        if idx[0] != None:    
            F = cost_fun(Y, self.descale(x), *f_args).reshape(-1, 1)

        else:
            F = cost_fun(Y, *f_args).reshape(-1, 1) 
            
        end_f = time.time()
        
        mu_Y = [np.mean(y, axis = 0) for y in Y]
        sig_Y = [np.std(y, axis = 0, ddof = 1) for y in Y]
        Y_scale = [(y-mu)/sig for y, mu, sig in zip(Y, mu_Y, sig_Y)]
        
        if norm_xdat:
            x_scale, mu_x, sig_x = normalize(x, axis = 0)
            x_train = transform_X(x_scale, Y_scale, x_idx, y_idx)
            af_args = (mu_Y, sig_Y, mu_x, sig_x)
            af_bounds = Bounds(normalize(self.bounds.lb, mu = mu_x, sigma = sig_x),
                               normalize(self.bounds.ub, mu = mu_x, sigma = sig_x))
        
        else:    
            x_train = transform_X(x, Y_scale, x_idx, y_idx)
            af_args = (mu_Y, sig_Y)
            af_bounds = self.bounds

        if nu is None:
            nu = np.array([2.5]*len(Y))
        
        if k_bnds is None:
            k_bnds = np.array([[1e-2, 1e2]]*len(Y))    
            
        model_bois = {}
        
        if split_gps:
            for i, (x_t, y) in enumerate(zip(x_train, Y)):
                sz = x_t.shape[1]
                kernel = gpr.kernels.Matern(np.ones(sz),
                                            np.array([k_bnds[i]]*sz), 
                                            nu = nu[i])
                outputs = y.shape[1]
                model_bois[f'{i+1}'] = []
                
                for j in range(outputs):
                    model_bois[f'{i+1}'].append(gpr.GaussianProcessRegressor(kernel,
                                                                             n_restarts_optimizer = 20,
                                                                             normalize_y = True))
                    model_bois[f'{i+1}'][j].fit(x_t, y[:, j].reshape(-1, 1))
        
        else:
            for i, (x_t, y) in enumerate(zip(x_train, Y)):
                sz = x_t.shape[1]
                kernel = gpr.kernels.Matern(np.ones(sz),
                                            np.array([k_bnds[i]]*sz),
                                            nu = nu[i])
                model_bois[f'{i+1}'] = gpr.GaussianProcessRegressor(kernel,
                                                                    n_restarts_optimizer = 20,
                                                                    normalize_y = True)
                model_bois[f'{i+1}'].fit(x_t, y)
        
        LCB_bois = LCB_BOIS(model = model_bois,
                            gp_sim = gp_sim, f = cost_fun,
                            exp_w = self.exp_w[0],
                            descale = self.descale,
                            eps = eps,
                            idx = idx,
                            gp_args = gp_args, f_args = f_args,
                            f_prime_regularizer = f_prime_regularizer,
                            feasibility_check = feasibility_check,
                            lb = feasible_lb, ub = feasible_ub,
                            clip_to_bounds = clip_to_bounds,
                            norm_xdat = norm_xdat)
        
        end = time.time()
        
        for i in range(init_pts):
            self.time_bois[i] = (i+1)*(end-start)/init_pts
            self.time_fbois[i] = (i+1)*(end_f-start_f)/init_pts
            
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_bois[init_pts-1]:.1f}')
        
        for i in range(self.trials_bois-init_pts):
            x0 = np.random.uniform(af_bounds.lb,
                                   af_bounds.ub,
                                   (restarts, self.dim))
            opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCB_bois.LCB,
                                                                x_0,
                                                                method = 'SLSQP',
                                                                bounds = af_bounds,
                                                                args = af_args)
                                              for x_0 in x0)
            
            x_nxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            x_nxt = x_nxts[np.argmin(funs)].reshape(-1, self.dim)

            if norm_xdat:
                x_nxt = unnormalize(x_nxt, mu_x, sig_x).reshape(-1, self.dim)
            
            start_f = time.time()
            
            Y_nxt = list(self.system(self.descale(x_nxt), *self.args))
            
            for j, (y, y_nxt) in enumerate(zip(Y, Y_nxt)):
                Y[j] = np.vstack([y, y_nxt])
            
            if idx[0] != None:    
                F_nxt = cost_fun(Y_nxt, self.descale(x_nxt), *f_args).reshape(-1, 1)
            
            else:
                F_nxt = cost_fun(Y_nxt, *f_args).reshape(-1, 1)
            
            #print(self.descale(x_nxt), F_nxt)
            end_f = time.time()
            self.time_fbois[i+init_pts] = self.time_fbois[i+init_pts-1]+end_f-start_f 
            
            x = np.vstack([x, x_nxt])
            F = np.vstack([F, F_nxt])
            
            mu_Y = [np.mean(y, axis = 0) for y in Y]
            sig_Y = [np.std(y, axis = 0, ddof = 1) for y in Y]
            Y_scale = [(y-mu)/sig for y, mu, sig in zip(Y, mu_Y, sig_Y)]
            
            if norm_xdat:
                x_scale, mu_x, sig_x = normalize(x, axis = 0)
                x_train = transform_X(x_scale, Y_scale, x_idx, y_idx)
                af_args = (mu_Y, sig_Y, mu_x, sig_x)
                af_bounds = Bounds(normalize(self.bounds.lb, mu = mu_x, sigma = sig_x),
                                   normalize(self.bounds.ub, mu = mu_x, sigma = sig_x))
            
            else:    
                x_train = transform_X(x, Y_scale, x_idx, y_idx)
                af_args = (mu_Y, sig_Y)
                af_bounds = self.bounds
            
            if split_gps:
                for (key, x_t, y) in zip(model_bois, x_train, Y):
                    for j, model in enumerate(model_bois[key]):
                        model.fit(x_t, y[:, j].reshape(-1, 1))
            
            else:
                for (key, x_t, y) in zip(model_bois, x_train, Y):
                    model_bois[key].fit(x_t, y)
            
            if i+1+init_pts in self.shift_exp_w:
                k = self.shift_exp_w.index(i+1+init_pts)+1
                LCB_bois.exp_w = self.exp_w[k]
            
            end = time.time()
            self.time_bois[i+init_pts] = end-start
            print('ITERATION COUNT IS AT 'f'{i+1+init_pts};\
                  TOTAL ELAPSED TIME: 'f'{self.time_bois[i+init_pts]:.1f}')
            
        self.bois_optim = True
        self.model_bois = model_bois
        self.x_bois = self.descale(x)
        self.y_bois = Y
        self.f_bois = F

    
    def optimizer_mcbo(self, trials, init_pts, n_samples,
                       idx, x_idx, y_idx,
                       gp_sim, cost_fun,
                       restarts = 100, af_cores = 1,
                       f_args = (), gp_args = (),
                       x_init = None,
                       kernel_length_scale_bnds = None, nu = None,
                       split_gps = False, norm_xdat = False,
                       feasibility_check = False,
                       feasible_lb = None, feasible_ub = None,
                       clip_to_bounds = False):
        
        print('Monte Carlo BO run...')
        start = time.time()
        k_bnds = kernel_length_scale_bnds
        self.trials_mc = trials
        self.time_mcbo = np.zeros(trials)
        self.time_fmcbo = np.zeros(trials)
        
        if x_init is None:
            x = np.random.uniform(self.bounds.lb,
                                  self.bounds.ub,
                                  (init_pts, self.dim))
        
        else:
            x = np.vstack([x_init.reshape(-1, self.dim), np.random.uniform(self.bounds.lb,
                                                                           self.bounds.ub,
                                                                           (init_pts-len(x_init), self.dim))])
        
        start_f = time.time()
        
        Y_o = [(self.system(self.descale(i), *self.args)) for i in x]
        Y = list(Y_o[0])
        
        for i in range(1, len(Y_o)):
            for j, (y, y_nxt) in enumerate(zip(Y, Y_o[i])):
                Y[j] = np.vstack([y, y_nxt])
        
        if idx[0] != None:
            F = cost_fun(Y, self.descale(x), *f_args).reshape(-1, 1)
            
        else:
            F = cost_fun(Y, *f_args).reshape(-1, 1)
            
        end_f = time.time()
        
        mu_Y = [np.mean(y, axis = 0) for y in Y]
        sig_Y = [np.std(y, axis = 0, ddof = 1) for y in Y]
        Y_scale = [(y-mu)/sig for y, mu, sig in zip(Y, mu_Y, sig_Y)]
        
        if norm_xdat:
            x_scale, mu_x, sig_x = normalize(x, axis = 0)
            x_train = transform_X(x_scale, Y_scale, x_idx, y_idx)
            af_args = (mu_Y, sig_Y, mu_x, sig_x)
            af_bounds = Bounds(normalize(self.bounds.lb, mu = mu_x, sigma = sig_x),
                               normalize(self.bounds.ub, mu = mu_x, sigma = sig_x))
            
        else:
            x_train = transform_X(x, Y_scale, x_idx, y_idx)
            af_args = (mu_Y, sig_Y)
            af_bounds = self.bounds

        if nu is None:
            nu = np.array([2.5]*len(Y))
        
        if k_bnds is None:
            k_bnds = np.array([[1e-2, 1e2]]*len(Y))    
        
        model_mcbo = {}
        
        if split_gps:
            for i, (x_t, y) in enumerate(zip(x_train, Y)):
                sz = x_t.shape[1]
                kernel = gpr.kernels.Matern(np.ones(sz),
                                            np.array([k_bnds[i]]*sz),
                                            nu = nu[i])
                outputs = y.shape[1]
                model_mcbo[f'{i+1}'] = []
                
                for j in range(outputs):
                    model_mcbo[f'{i+1}'].append(gpr.GaussianProcessRegressor(kernel = kernel,
                                                                           n_restarts_optimizer = 20,
                                                                           normalize_y = True))
                    model_mcbo[f'{i+1}'][j].fit(x_t, y[:, j].reshape(-1, 1))
        
        else:
            for i, (x_t, y) in enumerate(zip(x_train, Y)):
                sz = x_t.shape[1]
                kernel = gpr.kernels.Matern(np.ones(sz),
                                            np.array([k_bnds[i]]*sz),
                                            nu = nu[i])
                model_mcbo[f'{i+1}'] = gpr.GaussianProcessRegressor(kernel,
                                                                  n_restarts_optimizer = 20,
                                                                  normalize_y = True)
                model_mcbo[f'{i+1}'].fit(x_t, y)
        
        LCB_mc = LCB_MCBO(model = model_mcbo,
                          gp_sim = gp_sim, f = cost_fun,
                          exp_w = self.exp_w[0],
                          descale = self.descale,
                          n_samples = n_samples,
                          idx = idx,
                          gp_args = gp_args, f_args = f_args,
                          feasibility_check = feasibility_check,
                          lb = feasible_lb, ub = feasible_ub,
                          clip_to_bounds = clip_to_bounds,
                          norm_xdat = norm_xdat)
        
        end = time.time()
        
        for i in range(init_pts):
            self.time_mcbo[i] = (i+1)*(end-start)/init_pts
            self.time_fmcbo[i] = (i+1)*(end_f-start_f)/init_pts
        
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_mcbo[init_pts-1]:.1f}')
        
        for i in range(self.trials_mc-init_pts):
            x0 = np.random.uniform(af_bounds.lb,
                                   af_bounds.ub,
                                   (restarts, self.dim))
            opt = Parallel(n_jobs = af_cores)(delayed(minimize)(LCB_mc.LCB,
                                                                x_0,
                                                                method = 'SLSQP',
                                                                bounds = af_bounds,
                                                                args = af_args)
                                              for x_0 in x0)
            
            x_nxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            x_nxt = x_nxts[np.argmin(funs)].reshape(-1, self.dim)
            
            if norm_xdat:
                x_nxt = unnormalize(x_nxt, mu_Y, sig_Y).reshape(-1, self.dim)
            
            start_f = time.time()
            
            Y_nxt = list(self.system(self.descale(x_nxt), *self.args))
            
            for j, (y, y_nxt) in enumerate(zip(Y, Y_nxt)):
                Y[j] = np.vstack([y, y_nxt])
            
            if idx[0] != None:
                F_nxt = cost_fun(Y_nxt, self.descale(x_nxt), *f_args).reshape(-1, 1)

            else:
                F_nxt = cost_fun(Y_nxt, *f_args).reshape(-1, 1)

            end_f = time.time()
            self.time_fmcbo[i+init_pts] = self.time_fmcbo[i+init_pts-1]+end_f-start_f 
            
            x = np.vstack([x, x_nxt])
            F = np.vstack([F, F_nxt])
            
            mu_Y = [np.mean(y, axis = 0) for y in Y]
            sig_Y = [np.std(y, axis = 0, ddof = 1) for y in Y]
            Y_scale = [(y-mu)/sig for y, mu, sig in zip(Y, mu_Y, sig_Y)]
            
            if norm_xdat:
                x_scale, mu_x, sig_x = normalize(x, axis = 0)
                x_train = transform_X(x_scale, Y_scale, x_idx, y_idx)
                af_args = (mu_Y, sig_Y, mu_x, sig_x)
                af_bounds = Bounds(normalize(self.bounds.lb, mu = mu_x, sigma = sig_x),
                                   normalize(self.bounds.ub, mu = mu_x, sigma = sig_x))
                
            else:
                x_train = transform_X(x, Y_scale, x_idx, y_idx)
                af_args = (mu_Y, sig_Y)
                af_bounds = self.bounds
                
            if split_gps:
                for (key, x_t, y) in zip(model_mcbo, x_train, Y):
                    for j, model in enumerate(model_mcbo[key]):
                        model.fit(x_t, y[:, j].reshape(-1, 1))
                        
            else:
                for (key, x_t, y) in zip(model_mcbo, x_train, Y):
                    model_mcbo[key].fit(x_t, y)
            
            if i+1+init_pts in self.shift_exp_w:
                k = self.shift_exp_w.index(i+1+init_pts)+1
                LCB_mc.exp_w = self.exp_w[k]
            
            end = time.time()
            self.time_mcbo[i+init_pts] = end-start
            print('ITERATION COUNT IS AT 'f'{i+1+init_pts};\
                  TOTAL ELAPSED TIME: 'f'{self.time_mcbo[i+init_pts]:.1f}')
            
        self.mcbo_optim = True
        self.model_mcbo = model_mcbo
        self.x_mcbo = self.descale(x)
        self.y_mcbo = Y
        self.f_mcbo = F   
    
    
    def optimizer_optimism_bo(self, trials, init_pts,
                              idx, x_idx, y_idx,
                              gp_sim, cost_fun,
                              feasible_lb, feasible_ub,
                              restarts = 100, af_cores = 1,
                              f_args = (), gp_args = (),
                              x_init = None,
                              kernel_length_scale_bnds = None, nu = None,
                              split_gps = False, norm_xdat = False):
        
        """
        based on the work of Xu, et al: "Bayesian optimization of expensive nested grey-box functions"
        """
        
        print('Optimisim-driven BO run...')
        start = time.time()
        k_bnds = kernel_length_scale_bnds
        self.trials_opbo = trials
        self.time_opbo = np.zeros(trials)
        self.time_fopbo = np.zeros(trials)
        
        if x_init is None:
            x = np.random.uniform(self.bounds.lb,
                                  self.bounds.ub,
                                  (init_pts, self.dim))
        
        else:
            x = np.vstack([x_init.reshape(-1, self.dim), np.random.uniform(self.bounds.lb,
                                                                           self.bounds.ub,
                                                                           (init_pts-len(x_init), self.dim))])
        
        start_f = time.time()
        
        Y_o = [(self.system(self.descale(i), *self.args)) for i in x]
        Y = list(Y_o[0])
        
        for i in range(1, len(Y_o)):
            for j, (y, y_nxt) in enumerate(zip(Y, Y_o[i])):
                Y[j] = np.vstack([y, y_nxt])
        
        if idx[0] != None:    
            F = cost_fun(Y, self.descale(x), *f_args).reshape(-1, 1)

        else:
            F = cost_fun(Y, *f_args).reshape(-1, 1) 
            
        end_f = time.time()
        
        mu_Y = [np.mean(y, axis = 0) for y in Y]
        sig_Y = [np.std(y, axis = 0, ddof = 1) for y in Y]
        Y_scale = [(y-mu)/sig for y, mu, sig in zip(Y, mu_Y, sig_Y)]
        
        if norm_xdat:
            x_scale, mu_x, sig_x = normalize(x, axis = 0)
            x_train = transform_X(x_scale, Y_scale, x_idx, y_idx)
            af_args = (mu_x, sig_x)
            af_bounds = Bounds(normalize(self.bounds.lb, mu = mu_x, sigma = sig_x),
                               normalize(self.bounds.ub, mu = mu_x, sigma = sig_x))
        
        else:    
            x_train = transform_X(x, Y_scale, x_idx, y_idx)
            af_args = ()
            af_bounds = self.bounds
        
        augmented_bounds = Bounds(np.hstack([af_bounds.lb, feasible_lb]), np.hstack([af_bounds.ub, feasible_ub]))

        if nu is None:
            nu = np.array([2.5]*len(Y))
        
        if k_bnds is None:
            k_bnds = np.array([[1e-2, 1e2]]*len(Y))    
            
        model_opbo = {}
        
        if split_gps:
            for i, (x_t, y) in enumerate(zip(x_train, Y)):
                sz = x_t.shape[1]
                kernel = gpr.kernels.Matern(np.ones(sz),
                                            np.array([k_bnds[i]]*sz), 
                                            nu = nu[i])
                outputs = y.shape[1]
                model_opbo[f'{i+1}'] = []
                
                for j in range(outputs):
                    model_opbo[f'{i+1}'].append(gpr.GaussianProcessRegressor(kernel,
                                                                             n_restarts_optimizer = 20,
                                                                             normalize_y = True))
                    model_opbo[f'{i+1}'][j].fit(x_t, y[:, j].reshape(-1, 1))
        
        else:
            for i, (x_t, y) in enumerate(zip(x_train, Y)):
                sz = x_t.shape[1]
                kernel = gpr.kernels.Matern(np.ones(sz),
                                            np.array([k_bnds[i]]*sz),
                                            nu = nu[i])
                model_opbo[f'{i+1}'] = gpr.GaussianProcessRegressor(kernel,
                                                                    n_restarts_optimizer = 20,
                                                                    normalize_y = True)
                model_opbo[f'{i+1}'].fit(x_t, y)
        
        OPT_AF = OPTBO_AF(f = cost_fun,
                          descale = self.descale,
                          dim = self.dim,
                          idx = idx,
                          f_args = f_args,
                          norm_xdat = norm_xdat)
        
        end = time.time()
        
        for i in range(init_pts):
            self.time_opbo[i] = (i+1)*(end-start)/init_pts
            self.time_fopbo[i] = (i+1)*(end_f-start_f)/init_pts
            
        print('ITERATION COUNT IS AT 'f'{init_pts};\
              TOTAL ELAPSED TIME: 'f'{self.time_opbo[init_pts-1]:.1f}')
        
        for i in range(self.trials_opbo-init_pts):
            lower_bound = lambda x: gp_sim_bounds(x, gp_sim, model_opbo, mu_Y, sig_Y, self.dim,
                                                  self.exp_w[0], feasible_lb, True, *gp_args)
            nlc1 = NonlinearConstraint(lower_bound, 0, np.inf)
            
            upper_bound = lambda x: gp_sim_bounds(x, gp_sim, model_opbo, mu_Y, sig_Y, self.dim,
                                                  self.exp_w[0], feasible_ub, False, *gp_args)
            nlc2 = NonlinearConstraint(upper_bound, 0, np.inf)
            
            x0 = np.random.uniform(af_bounds.lb,
                                   af_bounds.ub,
                                   (restarts, self.dim))
            mu0, sig0 = gp_sim(x0, model_opbo, mu_Y, sig_Y, *gp_args)
            mu0 = np.hstack(mu0)
            sig0 = np.hstack(sig0)
            y0 = mu0+np.random.uniform(0, 1, (mu0.shape))*sig0
            z0 = np.hstack([x0, y0])
            
            opt = Parallel(n_jobs = af_cores)(delayed(minimize)(OPT_AF.AF,
                                                                z_0,
                                                                method = 'SLSQP',
                                                                bounds = augmented_bounds,
                                                                tol = 1e-9,
                                                                constraints = [nlc1, nlc2],
                                                                args = af_args)
                                              for z_0 in z0)
            
            x_nxts = np.array([res.x for res in opt], dtype  = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            
            for j, x_nxt in enumerate(x_nxts):
                mu_xnxt, _ = gp_sim(x_nxt[:self.dim], model_opbo, mu_Y, sig_Y, *gp_args)
                mu_xnxt = np.hstack(mu_xnxt).flatten()
                cutoff = 1e-3*np.abs(mu_xnxt)
                
                if any(gp_sim_bounds(x_nxt, gp_sim, model_opbo, mu_Y, sig_Y, self.dim,
                                     self.exp_w[0], feasible_lb, True, *gp_args) < -cutoff):
                    funs[j] = 1e6 
                
                if any(gp_sim_bounds(x_nxt, gp_sim, model_opbo, mu_Y, sig_Y, self.dim,
                                     self.exp_w[0], feasible_ub, False, *gp_args) < -cutoff):
                    funs[j] = 1e6
            
            print(np.where(funs != 1e6)[0].shape)
            x_nxt = x_nxts[np.argmin(funs)][:self.dim].reshape(-1, self.dim)

            if norm_xdat:
                x_nxt = unnormalize(x_nxt, mu_x, sig_x).reshape(-1, self.dim)
            
            start_f = time.time()
            
            Y_nxt = list(self.system(self.descale(x_nxt), *self.args))
            
            for j, (y, y_nxt) in enumerate(zip(Y, Y_nxt)):
                Y[j] = np.vstack([y, y_nxt])
            
            if idx[0] != None:    
                F_nxt = cost_fun(Y_nxt, self.descale(x_nxt), *f_args).reshape(-1, 1)
            
            else:
                F_nxt = cost_fun(Y_nxt, *f_args).reshape(-1, 1)
            
            #print(self.descale(x_nxt), F_nxt)
            end_f = time.time()
            self.time_fopbo[i+init_pts] = self.time_fopbo[i+init_pts-1]+end_f-start_f 
            
            x = np.vstack([x, x_nxt])
            F = np.vstack([F, F_nxt])
            
            mu_Y = [np.mean(y, axis = 0) for y in Y]
            sig_Y = [np.std(y, axis = 0, ddof = 1) for y in Y]
            Y_scale = [(y-mu)/sig for y, mu, sig in zip(Y, mu_Y, sig_Y)]
            
            if norm_xdat:
                x_scale, mu_x, sig_x = normalize(x, axis = 0)
                x_train = transform_X(x_scale, Y_scale, x_idx, y_idx)
                af_args = (mu_Y, sig_Y, mu_x, sig_x)
                af_bounds = Bounds(normalize(self.bounds.lb, mu = mu_x, sigma = sig_x),
                                   normalize(self.bounds.ub, mu = mu_x, sigma = sig_x))
            
            else:    
                x_train = transform_X(x, Y_scale, x_idx, y_idx)
                af_args = (mu_Y, sig_Y)
                af_bounds = self.bounds
                
            augmented_bounds = Bounds(np.hstack([af_bounds.lb, feasible_lb]), np.hstack([af_bounds.ub, feasible_ub]))
            
            if split_gps:
                for (key, x_t, y) in zip(model_opbo, x_train, Y):
                    for j, model in enumerate(model_opbo[key]):
                        model.fit(x_t, y[:, j].reshape(-1, 1))
            
            else:
                for (key, x_t, y) in zip(model_opbo, x_train, Y):
                    model_opbo[key].fit(x_t, y)
            
            if i+1+init_pts in self.shift_exp_w:
                k = self.shift_exp_w.index(i+1+init_pts)+1
                OPT_AF.exp_w = self.exp_w[k]
            
            end = time.time()
            self.time_opbo[i+init_pts] = end-start
            print('ITERATION COUNT IS AT 'f'{i+1+init_pts};\
                  TOTAL ELAPSED TIME: 'f'{self.time_opbo[i+init_pts]:.1f}')
            
        self.opbo_optim = True
        self.model_opbo = model_opbo
        self.x_opbo = self.descale(x)
        self.y_opbo = Y
        self.f_opbo = F
        
        
    def plots(self, figure_name):
        itr = np.linspace(1, self.trials_sbo, self.trials_sbo)
        ylim_l = min(self.y_sbo)-0.01*abs(min(self.y_sbo))
        ylim_u = max(self.y_sbo)+0.01*abs(max(self.y_sbo))
        xlim_u = self.trials_sbo
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlabel('Sample Number', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel('Operating cost (MM USD/yr)', fontsize = 24)
        pyp.yticks(fontsize = 24)
        ax1.scatter(itr, self.y_sbo, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200)
        ax1.plot(itr, self.y_sbo, color = 'black', linewidth = 3, label = 'S-BO');
        
        if self.bois_optim:
            itr = np.linspace(1, self.trials_bois, self.trials_bois)
            ax1.scatter(itr, self.f_bois, marker = 'o', color = 'white', edgecolor = 'lime',
                        zorder = 3, s = 200)
            ax1.plot(itr, self.f_bois, color = 'lime', linewidth = 3, label = 'BOIS')
            ylim_l = min(ylim_l, min(self.f_bois)-0.01*abs(min(self.f_bois)))
            ylim_u = max(ylim_u, max(self.f_bois)+0.01*abs(max(self.f_bois)))
            xlim_u = max(xlim_u, self.trials_bois)
        
        if self.mc_optim:
            itr = np.linspace(1, self.trials_mc, self.trials_mc)
            ax1.scatter(itr, self.f_mcbo, marker = 'o', color = 'white', edgecolor = 'orange',
                        zorder = 3, s = 200)
            ax1.plot(itr, self.f_mcbo, color = 'orange', linewidth = 3, label = 'MC-BO')
            ylim_l = min(ylim_l, min(self.f_mcbo)-0.01*abs(min(self.f_mcbo)))
            ylim_u = max(ylim_u, max(self.f_mcbo)+0.01*abs(max(self.f_mcbo)))
            xlim_u = max(xlim_u, self.trials_mc)
            
        if self.opbo_optim:
            itr = np.linspace(1, self.trials_opbo, self.trials_opbo)
            ax1.scatter(itr, self.f_opbo, marker = 'o', color = 'white', edgecolor = 'cyan',
                        zorder = 3, s = 200)
            ax1.plot(itr, self.f_opbo, color = 'orange', linewidth = 3, label = 'Optimism-BO')
            ylim_l = min(ylim_l, min(self.f_opbo)-0.01*abs(min(self.f_opbo)))
            ylim_u = max(ylim_u, max(self.f_opbo)+0.01*abs(max(self.f_opbo)))
            xlim_u = max(xlim_u, self.trials_opbo)
        
        ax1.set_ylim(ylim_l, ylim_u)
        ax1.set_xlim(1, xlim_u)
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
        pyp.yticks(fontsize = 24)
        ax1.scatter(self.time_sbo, self.y_sbo, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200)
        ax1.plot(self.time_sbo, self.y_sbo, color = 'black', linewidth = 3, label = 'S-BO');
        
        if self.bois_optim:
            ax1.scatter(self.time_bois, self.f_bois, marker = 'o', color = 'white', edgecolor = 'lime',
                        zorder = 3, s = 200)
            ax1.plot(self.time_bois, self.f_bois, color = 'lime', linewidth = 3, label = 'BOIS')
            ylim_l = min(ylim_l, min(self.f_bois)-0.01*abs(min(self.f_bois)))
            ylim_u = max(ylim_u, max(self.f_bois)+0.01*abs(max(self.f_bois)))
            xlim_u = max(xlim_u, self.time_bois[-1]+1)
        
        if self.mc_optim:
            ax1.scatter(self.time_mcbo, self.f_mcbo, marker = 'o', color = 'white', edgecolor = 'orange',
                        zorder = 3, s = 200)
            ax1.plot(self.time_mcbo, self.f_mcbo, color = 'orange', linewidth = 3, label = 'MC-BO')
            ylim_l = min(ylim_l, min(self.f_mcbo)-0.01*abs(min(self.f_mcbo)))
            ylim_u = max(ylim_u, max(self.f_mcbo)+0.01*abs(max(self.f_mcbo)))
            xlim_u = max(xlim_u, self.time_mcbo[-1]+1)
        
        if self.opbo_optim:
            ax1.scatter(self.time_opbo, self.f_opbo, marker = 'o', color = 'white', edgecolor = 'cyan',
                        zorder = 3, s = 200)
            ax1.plot(self.time_opbo, self.f_opbo, color = 'orange', linewidth = 3, label = 'Optimism-BO')
            ylim_l = min(ylim_l, min(self.f_opbo)-0.01*abs(min(self.f_opbo)))
            ylim_u = max(ylim_u, max(self.f_opbo)+0.01*abs(max(self.f_opbo)))
            xlim_u = max(xlim_u, self.time_opbo[-1]+1)
                
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
        pyp.yticks(fontsize = 24)
        ax1.scatter(self.time_fsbo, self.y_sbo, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200)
        ax1.plot(self.time_fsbo, self.y_sbo, color = 'black', linewidth = 3, label = 'S-BO');
        
        if self.bois_optim:
            ax1.scatter(self.time_fbois, self.f_bois, marker = 'o', color = 'white', edgecolor = 'lime',
                        zorder = 3, s = 200)
            ax1.plot(self.time_fbois, self.f_bois, color = 'lime', linewidth = 3, label = 'BOIS')
            ylim_l = min(ylim_l, min(self.f_bois)-0.01*abs(min(self.f_bois)))
            ylim_u = max(ylim_u, max(self.f_bois)+0.01*abs(max(self.f_bois)))
            xlim_u = max(xlim_u, self.time_fbois[-1]+1)
        
        if self.mc_optim:
            ax1.scatter(self.time_fmcbo, self.f_mcbo, marker = 'o', color = 'white', edgecolor = 'orange',
                        zorder = 3, s = 200)
            ax1.plot(self.time_fmcbo, self.f_mcbo, color = 'orange', linewidth = 3, label = 'MC-BO')
            ylim_l = min(ylim_l, min(self.f_mcbo)-0.01*abs(min(self.f_mcbo)))
            ylim_u = max(ylim_u, max(self.f_mcbo)+0.01*abs(max(self.f_mcbo)))
            xlim_u = max(xlim_u, self.time_fmcbo[-1]+1)
        
        if self.opbo_optim:
            ax1.scatter(self.time_fopbo, self.f_opbo, marker = 'o', color = 'white', edgecolor = 'cyan',
                        zorder = 3, s = 200)
            ax1.plot(self.time_fopbo, self.f_opbo, color = 'orange', linewidth = 3, label = 'Optimism-BO')
            ylim_l = min(ylim_l, min(self.f_opbo)-0.01*abs(min(self.f_opbo)))
            ylim_u = max(ylim_u, max(self.f_opbo)+0.01*abs(max(self.f_opbo)))
            xlim_u = max(xlim_u, self.time_fopbo[-1]+1)
                
        ax1.set_ylim(ylim_l, ylim_u)
        ax1.set_xlim(0, xlim_u)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'_exptime.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1)
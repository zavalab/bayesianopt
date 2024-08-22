import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint
import sklearn.gaussian_process as gpr
import time
import RXTR_SYSTEM
from joblib import Parallel, delayed

import sys
sys.path.append('./../../BO_algos')
import Parallel_Algos as BO_algos

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

# Define Parameters
exp_w = 2.6
C0varf = np.loadtxt('C_var_gp.txt')
FR = np.array([0.100, 0.075, 0.075])
R_Frac = 1e-6
ub = np.array([423, 423])
lb = np.array([303, 303])
bounds = Bounds((0, 0), (1, 1))
dim = len(ub)
kernel = gpr.kernels.Matern((1, 1), ((0.06, 5), (0.06, 5)), nu = 2.5)


# Set system and reference functions
SYST_RECYCLE = RXTR_SYSTEM.SYST_RECYCLE
SYST_RECYCLE_REF = RXTR_SYSTEM.RXTR_REG()


# Train reference model (GP)
def scale(x, ub = 423, lb = 303, sf = 1):
    m = sf/(ub - lb)
    b = -lb*m
    return m*x+b

TT = np.arange(303, 424, 1)
TT = np.meshgrid(TT, TT)
TT = np.hstack([TT[0].reshape(-1, 1), TT[1].reshape(-1, 1)])

print('Create Statistical Reference Model...')
start = time.time()

Tmod = np.linspace(303, 423, 13)
Tmod = np.meshgrid(Tmod, Tmod)
Tmod = np.hstack([Tmod[0].reshape(-1, 1), Tmod[1].reshape(-1, 1)])
CTREF = np.ones((Tmod.shape[0], 3))

Ctref = Parallel(n_jobs = 5)(delayed(SYST_RECYCLE_REF)(Tmod, FR, R_Frac, Cdist)
                             for Cdist in C0varf)

for i in range(3):
    C = np.vstack(Ctref[:][:]).T[:, i::3]
    C = 8*np.sum(C, axis = 1)
    CTREF[:, i] = C

kergp = gpr.kernels.Matern((5, 5), ((1, 10), (1, 10)), nu = 2.5)
gprefmod = gpr.GaussianProcessRegressor(kernel = kergp,
                                        alpha = 1e-6,
                                        n_restarts_optimizer = 10,
                                        normalize_y = True)
gprefmod.fit(scale(Tmod), CTREF)

end = time.time()
mobdtm = end-start
print(mobdtm)

def SYST_C(T):
    CtR = 0
    for i in range(C0varf.shape[0]):
        Ctr = 8*SYST_RECYCLE(T, FR, R_Frac, C0varf[i])[-1]
        CtR += Ctr
    return CtR

def SYST_C_DIST(T):
    CtR = 0
    for i in range(C0varf.shape[0]):
        Ctr = SYST_RECYCLE(T, FR, R_Frac, C0varf[i])
        Ctr = np.vstack(Ctr[:]).T
        CtR += 8*Ctr
    return CtR

def SYST_C_REFGP(T):
    T = T.flatten()
    T = T.reshape(int(T.shape[0]/2), 2)
    T = scale(T)
    return gprefmod.predict(T)[:, -1]

def SYST_C_DISTGP(T):
    T = T.flatten()
    T = T.reshape(int(T.shape[0]/2), 2)
    T = scale(T)
    return gprefmod.predict(T)

def SYST_C_REF1(T):
    T = T.flatten()
    T = T.reshape(int(T.shape[0]/2), 2)
    T = scale(T)
    return gprefmod.predict(T)[:, 0]

def SYST_C_REF2(T):
    T = T.flatten()
    T = T.reshape(int(T.shape[0]/2), 2)
    T = scale(T)
    return gprefmod.predict(T)[:, 1]

def zr(x):
    x = x.reshape(-1, 1)
    x = x.reshape(int(x.shape[0]/2), 2)
    return np.zeros((x.shape[0], 3))


# Setup BO class
REACOPTIM = BO_algos.BO(ub = ub,
                        lb = lb,
                        dim = 2,
                        exp_w = exp_w,
                        kernel = kernel,
                        system = SYST_C,
                        bounds = bounds,
                        **{'refmod': SYST_C_REFGP,
                           'distmod': SYST_C_DIST,
                           'ref_distmod': SYST_C_DISTGP,
                           'ref_distmod1': SYST_C_REF1,
                           'ref_distmod2': SYST_C_REF2})


## Generate level sets
gpparts = gpr.GaussianProcessRegressor(kernel = kergp,
                                       alpha = 1e-6,
                                       n_restarts_optimizer = 10,
                                       normalize_y = True)
gpparts.fit(scale(Tmod), (CTREF[:, -1]).reshape(-1, 1))

parts = np.array([-461, -383]) # ref mod with g_1(T_1) and g_2(T_1, T_2)

con1 = lambda x: (gpparts.predict(x.reshape(1, 2))).flatten()
con2 = lambda x: x[0]

nlc1 = NonlinearConstraint(con1, -1e4, parts[0])
nlc21 = NonlinearConstraint(con1, parts[0], parts[1])
nlc22 = NonlinearConstraint(con2, 0.4, 1.1)
nlc31 = NonlinearConstraint(con1, parts[0], parts[1])
nlc32 = NonlinearConstraint(con2, 0, 0.38)
nlc4 = NonlinearConstraint(con1, parts[1], 1e4)

cons = {'1': [nlc1], '2': [nlc21, nlc22], '3': [nlc31, nlc32], '4': [nlc4]}

## VP-BO intial points
lim_init = REACOPTIM.scale(np.array([336, 380]))


# Run_BO
trials_seq = 100
trials_par1 = 25
trials_par2 = 33

x_init = np.linspace(bounds.lb, bounds.ub, 5)
x_init = np.meshgrid(*x_init.T)
x_init = np.reshape(x_init, (dim, -1)).T

MET_SBO = np.zeros((trials_seq, 4))
PARAMS_SBO = np.ones(x_init.shape)
DIST_SBO = np.array([]).reshape(0, dim)
RES_SBO = np.ones((MET_SBO.shape[0], x_init.shape[0]))

MET_REFBO = np.zeros((trials_seq, 4))
PARAMS_REFBO = np.ones(x_init.shape)
DIST_REFBO = np.array([]).reshape(0, dim)
RES_REFBO = np.ones((MET_REFBO.shape[0], x_init.shape[0]))

MET_LSBO1 = np.zeros((trials_par1, 4))
PARAMS_LSBO1 = np.ones(x_init.shape)
DIST_LSBO1 = np.array([]).reshape(0, dim)
RES_LSBO1 = np.ones((MET_LSBO1.shape[0], x_init.shape[0]))

MET_LSBO2 = np.zeros((trials_par1, 4))
PARAMS_LSBO2 = np.ones(x_init.shape)
DIST_LSBO2 = np.array([]).reshape(0, dim)
RES_LSBO2 = np.ones((MET_LSBO2.shape[0], x_init.shape[0]))

MET_VPBO1 = np.zeros((trials_par2, 4))
PARAMS_VPBO1 = np.ones(x_init.shape)
DIST_VPBO1 = np.array([]).reshape(0, dim)
RES_VPBO1 = np.ones((MET_VPBO1.shape[0], x_init.shape[0]))

MET_VPBO2 = np.zeros((trials_par2, 4))
PARAMS_VPBO2 = np.ones(x_init.shape)
DIST_VPBO2 = np.array([]).reshape(0, dim)
RES_VPBO2 = np.ones((MET_VPBO2.shape[0], x_init.shape[0]))

MET_HSBO = np.zeros((trials_par1, 4))
PARAMS_HSBO = np.ones(x_init.shape)
DIST_HSBO = np.array([]).reshape(0, dim)
RES_HSBO = np.ones((MET_HSBO.shape[0], x_init.shape[0]))

MET_EXBO = np.zeros((trials_par1, 4))
PARAMS_EXBO = np.ones(x_init.shape)
DIST_EXBO = np.array([]).reshape(0, dim)
RES_EXBO = np.ones((MET_EXBO.shape[0], x_init.shape[0]))

MET_NMCBO = np.zeros((trials_par1, 4))
PARAMS_NMCBO = np.ones(x_init.shape)
DIST_NMCBO = np.array([]).reshape(0, dim)
RES_NMCBO = np.ones((MET_NMCBO.shape[0], x_init.shape[0]))

MET_QBO = np.zeros((trials_par1, 4))
PARAMS_QBO = np.ones(x_init.shape)
DIST_QBO = np.array([]).reshape(0, dim)
RES_QBO = np.ones((MET_QBO.shape[0], x_init.shape[0]))


## S-BO
for i, x_0 in enumerate(x_init):  
    start = time.time()
    REACOPTIM.optimizer_sbo(trials = trials_seq, x_init = x_0)
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    MET_SBO[:, 0] += REACOPTIM.time_sbo.flatten()
    MET_SBO[:, 1] += REACOPTIM.time_fsbo.flatten()
    MET_SBO[:, 2] += MET_SBO[:, 2]+REACOPTIM.y_sbo.flatten()
    MET_SBO[:, 3] += MET_SBO[:, 3]+np.min(REACOPTIM.y_sbo)
    print('Best S-BO value is '+str(np.min(REACOPTIM.y_sbo)))
    PARAMS_SBO[i] = REACOPTIM.x_sbo[np.argmin(REACOPTIM.y_sbo)]
    DIST_SBO = np.vstack([DIST_SBO, REACOPTIM.x_sbo])
    RES_SBO[:, i] = REACOPTIM.y_sbo[:, 0]
MET_SBO[:, 0] = MET_SBO[:, 0]/(i+1)
MET_SBO[:, 1] = MET_SBO[:, 1]/(i+1)
MET_SBO[:, 2] = MET_SBO[:, 2]/(i+1)
MET_SBO[:, 3] = MET_SBO[:, 3]/(i+1)
REACOPTIM.y_sbo = MET_SBO[:, 2].flatten()
REACOPTIM.time_sbo = MET_SBO[:, 0].flatten()
REACOPTIM.time_fsbo = MET_SBO[:, 1].flatten()
    

## Ref-BO
for i, x_0 in enumerate(x_init):  
    start = time.time()
    REACOPTIM.optimizer_refbo(trials = trials_seq, x_init = x_0)
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    MET_REFBO[:, 0] += REACOPTIM.time_ref.flatten()
    MET_REFBO[:, 1] += REACOPTIM.time_fref.flatten()
    MET_REFBO[:, 2] += REACOPTIM.y_ref.flatten()
    MET_REFBO[:, 3] += np.min(REACOPTIM.y_ref)
    print('Best Ref-BO value is '+str(np.min(REACOPTIM.y_ref)))
    PARAMS_REFBO[i] = REACOPTIM.x_ref[np.argmin(REACOPTIM.y_ref)]
    DIST_REFBO = np.vstack([DIST_REFBO, REACOPTIM.x_ref])
    RES_REFBO[:, i] = REACOPTIM.y_ref[:, 0]
MET_REFBO[:, 0] = MET_REFBO[:, 0]/(i+1)
MET_REFBO[:, 1] = MET_REFBO[:, 1]/(i+1)
MET_REFBO[:, 2] = MET_REFBO[:, 2]/(i+1)
MET_REFBO[:, 3] = MET_REFBO[:, 3]/(i+1)
REACOPTIM.y_ref = MET_REFBO[:, 2].flatten()
REACOPTIM.time_ref = MET_REFBO[:, 0].flatten()
REACOPTIM.time_fref = MET_REFBO[:, 1].flatten()


## LS-BO
for i, x_0 in enumerate(x_init):
    start = time.time()
    REACOPTIM.optimizer_lsbo(trials = trials_par1,
                             partition_number = 4,
                             repartition_intervals = [],
                             x_samps = [],
                             f_cores = 4,
                             af_cores = 1,
                             ref_cores = 1,
                             partitions = cons, 
                             x_init = x_0)
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    MET_LSBO1[:, 0] += REACOPTIM.time_ls.flatten()
    MET_LSBO1[:, 1] += REACOPTIM.time_fls.flatten()
    MET_LSBO1[:, 2] += REACOPTIM.y_lsbst.flatten()
    MET_LSBO1[:, 3] += np.min(REACOPTIM.y_lsbst)
    print('Best LS-BO value is '+str(np.min(REACOPTIM.y_lsbst)))
    PARAMS_LSBO1[i] = REACOPTIM.x_ls[np.argmin(REACOPTIM.y_ls)]
    DIST_LSBO1 = np.vstack([DIST_LSBO1, REACOPTIM.x_ls])
    RES_LSBO1[:, i] = REACOPTIM.y_lsbst[:, 0]
MET_LSBO1[:, 0] = MET_LSBO1[:, 0]/(i+1)
MET_LSBO1[:, 1] = MET_LSBO1[:, 1]/(i+1)
MET_LSBO1[:, 2] = MET_LSBO1[:, 2]/(i+1)
MET_LSBO1[:, 3] = MET_LSBO1[:, 3]/(i+1)
REACOPTIM.y_lsbst = MET_LSBO1[:, 2].flatten()
REACOPTIM.time_ls = MET_LSBO1[:, 0].flatten()
REACOPTIM.time_fls = MET_LSBO1[:, 1].flatten()


## VP-BO 
for i, x_0 in enumerate(x_init):
    start = time.time()
    REACOPTIM.optimizer_vpbo(trials = trials_par2,
                             split_num = 2,
                             lim_init = lim_init,
                             f_cores = 3,
                             af_cores = 1,
                             ref_cores = 1,
                             x_init  = x_0)
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    MET_VPBO1[:, 0] += REACOPTIM.time_vp.flatten()
    MET_VPBO1[:, 1] += REACOPTIM.time_fvp.flatten()
    MET_VPBO1[:, 2] += REACOPTIM.y_vpbst[:, -1].flatten()
    MET_VPBO1[:, 3] += np.min(REACOPTIM.y_vpbst[:, -1])
    print('Best VP-BO value is '+str(np.min(REACOPTIM.y_vpbst[:, -1])))
    PARAMS_VPBO1[i] = REACOPTIM.x_vp[np.argmin(REACOPTIM.y_vp[:, -1])]
    DIST_VPBO1 = np.vstack([DIST_VPBO1, REACOPTIM.x_vp])
    RES_VPBO1[:, i] = REACOPTIM.y_vpbst[:, -1]
MET_VPBO1[:, 0] = MET_VPBO1[:, 0]/(i+1)
MET_VPBO1[:, 1] = MET_VPBO1[:, 1]/(i+1)
MET_VPBO1[:, 2] = MET_VPBO1[:, 2]/(i+1)
MET_VPBO1[:, 3] = MET_VPBO1[:, 3]/(i+1)
REACOPTIM.y_vpbst = MET_VPBO1[:, 2].reshape(-1, 1)
REACOPTIM.time_vp = MET_VPBO1[:, 0].flatten()
REACOPTIM.time_fvp = MET_VPBO1[:, 1].flatten()


## HS-BO
for i, x_0 in enumerate(x_init):
    start = time.time()
    REACOPTIM.optimizer_hsbo(trials = trials_par1,
                             phi = 0.45,
                             f_cores = 4,
                             af_cores = 1,
                             x_init = x_0)
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    MET_HSBO[:, 0] += REACOPTIM.time_hyp.flatten()
    MET_HSBO[:, 1] += REACOPTIM.time_fhyp.flatten()
    MET_HSBO[:, 2] += REACOPTIM.y_hypbst.flatten()
    MET_HSBO[:, 3] += np.min(REACOPTIM.y_hypbst)
    print('Best HS-BO value is '+str(np.min(REACOPTIM.y_hypbst)))
    PARAMS_HSBO[i] =  REACOPTIM.x_hyp[np.argmin(REACOPTIM.y_hyp)]
    DIST_HSBO = np.vstack([DIST_HSBO, REACOPTIM.x_hyp])
    RES_HSBO[:, i] = REACOPTIM.y_hypbst[:, 0]
MET_HSBO[:, 0] = MET_HSBO[:, 0]/(i+1)
MET_HSBO[:, 1] = MET_HSBO[:, 1]/(i+1)
MET_HSBO[:, 2] = MET_HSBO[:, 2]/(i+1)
MET_HSBO[:, 3] = MET_HSBO[:, 3]/(i+1)
REACOPTIM.y_hypbst = MET_HSBO[:, 2].flatten()
REACOPTIM.time_hyp = MET_HSBO[:, 0].flatten()
REACOPTIM.time_fhyp = MET_HSBO[:, 1].flatten()


## HP-BO
for i, x_0 in enumerate(x_init):
    start = time.time()
    REACOPTIM.optimizer_exbo(trials = trials_par1,
                             num_weights = 4,
                             lam = 1,
                             f_cores = 4,
                             af_cores = 1,
                             x_init = x_0)
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    MET_EXBO[:, 0] += REACOPTIM.time_expw.flatten()
    MET_EXBO[:, 1] += REACOPTIM.time_fexpw.flatten()
    MET_EXBO[:, 2] += REACOPTIM.y_expwbst.flatten()
    MET_EXBO[:, 3] += np.min(REACOPTIM.y_expwbst)
    print('Best HP-BO value is '+str(np.min(REACOPTIM.y_expwbst)))
    PARAMS_EXBO[i] = REACOPTIM.x_expw[np.argmin(REACOPTIM.y_expw)]
    DIST_EXBO = np.vstack([DIST_EXBO, REACOPTIM.x_expw])
    RES_EXBO[:, i] = REACOPTIM.y_expwbst[:, 0]
MET_EXBO[:, 0] = MET_EXBO[:, 0]/(i+1)
MET_EXBO[:, 1] = MET_EXBO[:, 1]/(i+1)
MET_EXBO[:, 2] = MET_EXBO[:, 2]/(i+1)
MET_EXBO[:, 3] = MET_EXBO[:, 3]/(i+1)
REACOPTIM.y_expwbst = MET_EXBO[:, 2].flatten()
REACOPTIM.time_expw = MET_EXBO[:, 0].flatten()
REACOPTIM.time_fexpw = MET_EXBO[:, 1].flatten()


## NMC_BO
for i, x_0 in enumerate(x_init):
    start = time.time()
    REACOPTIM.optimizer_nmcbo(trials = trials_par1,
                              parallel_exps = 4,
                              sample_num = 20,
                              f_cores = 4,
                              af_cores = 4,
                              x_init = x_0)
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    MET_NMCBO[:, 0] += REACOPTIM.time_nmc.flatten()
    MET_NMCBO[:, 1] += REACOPTIM.time_fnmc.flatten()
    MET_NMCBO[:, 2] += REACOPTIM.y_nmcbst.flatten()
    MET_NMCBO[:, 3] += np.min(REACOPTIM.y_nmcbst)
    print('Best NMC-BO value is '+str(np.min(REACOPTIM.y_nmcbst)))
    PARAMS_NMCBO[i] = REACOPTIM.x_nmc[np.argmin(REACOPTIM.y_nmc)]
    DIST_NMCBO = np.vstack([DIST_NMCBO, REACOPTIM.x_nmc])
    RES_NMCBO[:, i] = REACOPTIM.y_nmcbst[:, 0]
MET_NMCBO[:, 0] = MET_NMCBO[:, 0]/(i+1)
MET_NMCBO[:, 1] = MET_NMCBO[:, 1]/(i+1)
MET_NMCBO[:, 2] = MET_NMCBO[:, 2]/(i+1)
MET_NMCBO[:, 3] = MET_NMCBO[:, 3]/(i+1)
REACOPTIM.y_nmcbst = MET_NMCBO[:, 2].flatten()
REACOPTIM.time_nmc = MET_NMCBO[:, 0].flatten()
REACOPTIM.time_fnmc = MET_NMCBO[:, 1].flatten()


## Q_BO
for i, x_0 in enumerate(x_init):
    start = time.time()
    REACOPTIM.optimizer_qBO(trials = trials_par1,
                            q = 4,
                            n_samps = 20,
                            f_cores = 4,
                            af_cores = 1,
                            x_init = x_0)
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    MET_QBO[:, 0] += REACOPTIM.time_qbo.flatten()
    MET_QBO[:, 1] += REACOPTIM.time_fqbo.flatten()
    MET_QBO[:, 2] += REACOPTIM.y_qbobst.flatten()
    MET_QBO[:, 3] += np.min(REACOPTIM.y_qbobst)
    print('Best q-BO value is '+str(np.min(REACOPTIM.y_qbobst)))
    PARAMS_QBO[i] = REACOPTIM.x_qbo[np.argmin(REACOPTIM.y_qbo)]
    DIST_QBO = np.vstack([DIST_QBO, REACOPTIM.x_qbo])
    RES_QBO[:, i] = REACOPTIM.y_qbobst[:, 0]
MET_QBO[:, 0] = MET_QBO[:, 0]/(i+1)
MET_QBO[:, 1] = MET_QBO[:, 1]/(i+1)
MET_QBO[:, 2] = MET_QBO[:, 2]/(i+1)
MET_QBO[:, 3] = MET_QBO[:, 3]/(i+1)
REACOPTIM.y_qbobst = MET_QBO[:, 2].flatten()
REACOPTIM.time_qbo = MET_QBO[:, 0].flatten()
REACOPTIM.time_fqbo = MET_QBO[:, 1].flatten()


# Plot convergence plots
REACOPTIM.plots('R1')
REACOPTIM.plots_time('R1')
REACOPTIM.plot_exptime('R1')
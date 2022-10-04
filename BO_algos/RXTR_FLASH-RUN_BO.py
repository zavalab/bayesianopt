import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint
import sklearn.gaussian_process as gpr
import Various_BO
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
from matplotlib import pyplot as pyp, cm
import sys
sys.path.append('./../Models')
import RECYL_SYST_MODS4
# Disable Covergence and Runtime warnings that arise during BO runs due to parameter bounds
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

exp_w = 2.6
C0varf = np.loadtxt('./../Models/C_var2.txt')
# C0var = np.ones((1, 3))
FR = np.array([0.100, 0.075, 0.075])
R_Frac = 1e-6
SYST_RECYCLE = RECYL_SYST_MODS4.SYST_RECYCLE
SYST_RECYCLE_REF = RECYL_SYST_MODS4.RXTR_REG()
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
gprefmod = gpr.GaussianProcessRegressor(kergp, alpha = 1e-6, n_restarts_optimizer = 10,
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

def SYST_C_REF(T):
    CtR = 0
    for i in range(C0varf.shape[0]):
        Ctr = 8*SYST_RECYCLE_REF(T, FR, R_Frac, C0varf[i])[-1]
        CtR += Ctr
    return CtR

def SYST_C_REFGP(T):
    T = T.flatten()
    T = T.reshape(int(T.shape[0]/2), 2)
    T = scale(T)
    return gprefmod.predict(T)[:, -1]

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

def SYST_C_DISTGP(T):
    T = T.flatten()
    T = T.reshape(int(T.shape[0]/2), 2)
    T = scale(T)
    return gprefmod.predict(T)

def SYST_C_DIST(T):
    CtR = 0
    for i in range(C0varf.shape[0]):
        Ctr = SYST_RECYCLE(T, FR, R_Frac, C0varf[i])
        Ctr = np.vstack(Ctr[:]).T
        CtR += 8*Ctr
    return CtR

def SYST_C_DIST_REF(T):
    CtR = 0
    for i in range(C0varf.shape[0]):
        Ctr = SYST_RECYCLE_REF(T, FR, R_Frac, C0varf[i])
        Ctr = np.vstack(Ctr[:]).T
        CtR += 8*Ctr
    return CtR

# SETUP and build partitions
ub = np.array([423, 423])
lb = np.array([303, 303])
bnds2 = Bounds((0, 0), (1, 1))
kernel = gpr.kernels.Matern((1, 1), ((0.06, 5), (0.06, 5)), nu = 2.5)
REACOPTIM = Various_BO.BO(ub, lb, 2, exp_w, kernel, SYST_C, bnds2, **{'refmod': SYST_C_REFGP, 'distmod': SYST_C_DIST,
'ref_distmod': SYST_C_DISTGP, 'ref_distmod1': SYST_C_REF1, 'ref_distmod2': SYST_C_REF2})
blks = 3
dim = 2
consb = {}
funsb = {}
funsb['1'] = lambda x: x[0]
funsb['2'] = lambda x: x[1]
lwrx = 0
uprx = 1/blks
lwry = 1-(1/blks)
upry = 1
j = 1
for i in range(blks**2):
    nlc1 = NonlinearConstraint(funsb['1'], lwrx, uprx)
    nlc2 = NonlinearConstraint(funsb['2'], lwry, upry)
    consb[str(i+1)] = [nlc1, nlc2]
    lwrx = lwrx+(1/blks)
    uprx = uprx+(1/blks)
    if (i+1)%blks == 0:
        lwrx = 0
        uprx = 1/blks
        lwry = lwry-(1/blks)
        upry = upry-(1/blks)
# Learning the bounds (using GP we already made)
gpparts = gpr.GaussianProcessRegressor(kergp, alpha = 1e-6, n_restarts_optimizer = 10,
                                       normalize_y = True)
gpparts.fit(scale(Tmod), (CTREF[:, -1]).reshape(-1, 1))
parts = np.array([-365, -345, -330]) # ref mod with g_1(T_1) and g_2(T_2)
con1 = lambda x: (gpparts.predict(x.reshape(1, 2))).flatten()
con2 = lambda x: x[0]
nlc1 = NonlinearConstraint(con1, -1e4, parts[0])
nlc21 = NonlinearConstraint(con1, parts[0], parts[1])
nlc22 = NonlinearConstraint(con2, 0.25, 1)
nlc31 = NonlinearConstraint(con1, parts[1], parts[2])
nlc32 = NonlinearConstraint(con2, 0, 0.25)
nlc4 = NonlinearConstraint(con1, parts[2], 1e4)
# nlc5 = NonlinearConstraint(con1, parts[2], 0)
cons = {'1': [nlc1], '2': [nlc21], '3': [nlc31], '4': [nlc4]}#, '5': [nlc5]}
# Using Curves we generate by analyzing the ref model
# con1 = lambda x: 13*(x[0]-0.066)**2+x[1]
# nlc1 = NonlinearConstraint(con1, -100, 0.7)
# con21 = lambda x: 13*(x[0]-0.066)**2+x[1]
# nlc21 = NonlinearConstraint(con21, 0.7, 100)
# con22 = lambda x: 6.0*(x[0]-0.07)**2+x[1]
# nlc22 = NonlinearConstraint(con22, -100, 0.87)
# con31 = lambda x: 6.0*(x[0]-0.07)**2+x[1]
# nlc31 = NonlinearConstraint(con31, 0.87, 100)
# con32 = lambda x: x[1]-2.583*x[0]
# nlc32 = NonlinearConstraint(con32, -1.1, 100)
# con4 = lambda x: x[1]-2.583*x[0]
# nlc4 = NonlinearConstraint(con4, -100, -1.1)
# cons = {'1': [nlc1], '2': [nlc21, nlc22], '3': [nlc31, nlc32], '4': [nlc4]}
# Variable split initial x1, x2
# liminit = REACOPTIM.scale(np.array([336, 318]))
liminit = REACOPTIM.scale(np.array([336, 380]))

# Start BO Runs
x0 = np.linspace(0, 1, 3)
x0 = np.meshgrid(x0, x0)
x0 = np.hstack([x0[0].reshape(-1, 1), x0[1].reshape(-1, 1)])

METGP = np.zeros((50, 4))
PARAMSGP = np.ones(x0.shape)
DISTGP = np.array([]).reshape(0, dim)
RESGP = np.ones((METGP.shape[0], x0.shape[0]))

METREF = np.zeros((50, 4))
PARAMSREF = np.ones(x0.shape)
DISTREF = np.array([]).reshape(0, dim)
RESREF = np.ones((METREF.shape[0], x0.shape[0]))

METSPLT = np.zeros((13, 4))
PARAMSPLT = np.ones(x0.shape)
DISTSPLT = np.array([]).reshape(0, dim)
RESPLT = np.ones((METSPLT.shape[0], x0.shape[0]))

METSPLTREF = np.zeros((13, 4))
PARAMSPLTREF = np.ones(x0.shape)
DISTSPLTREF = np.array([]).reshape(0, dim)
RESPLTREF = np.ones((METSPLTREF.shape[0], x0.shape[0]))

METSPLTVAR = np.zeros((16, 4))
PARAMSPLTVAR = np.ones(x0.shape)
DISTSPLTVAR = np.array([]).reshape(0, dim)
RESPLTVAR = np.ones((METSPLTVAR.shape[0], x0.shape[0]))

METSPLTVAR2 = np.zeros((16, 4))
PARAMSPLTVAR2 = np.ones(x0.shape)
DISTSPLTVAR2 = np.array([]).reshape(0, dim)
RESPLTVAR2 = np.ones((METSPLTVAR2.shape[0], x0.shape[0]))

METHYP = np.zeros((13, 4))
PARAMSHYP = np.ones(x0.shape)
DISTHYP = np.array([]).reshape(0, dim)
RESHYP = np.ones((METHYP.shape[0], x0.shape[0]))

METEXPW = np.zeros((13, 4))
PARAMSEXPW = np.ones(x0.shape)
DISTEXPW = np.array([]).reshape(0, dim)
RESEXPW = np.ones((METEXPW.shape[0], x0.shape[0]))

METNMC = np.zeros((13, 4))
PARAMSNMC = np.ones(x0.shape)
DISTNMC = np.array([]).reshape(0, dim)
RESNMC = np.ones((METNMC.shape[0], x0.shape[0]))

for i in range(x0.shape[0]):
    start = time.time()
    REACOPTIM.optimizergp(49, 1, cores = 1, xinit = x0[i])
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    METGP[:, 0] = METGP[:, 0]+REACOPTIM.timegp.flatten()
    METGP[:, 1] = METGP[:, 1]+REACOPTIM.timefgp.flatten()
    METGP[:, 2] = METGP[:, 2]+REACOPTIM.ygp.flatten()
    METGP[:, 3] = METGP[:, 3]+np.min(REACOPTIM.ygp)
    print('Best gp value is '+str(np.min(REACOPTIM.ygp)))
    PARAMSGP[i] = REACOPTIM.xgp[np.argmin(REACOPTIM.ygp)]
    DISTGP = np.vstack([DISTGP, REACOPTIM.xgp])
    RESGP[:, i] = REACOPTIM.ygp[:, 0]
METGP[:, 0] = METGP[:, 0]/(i+1)
METGP[:, 1] = METGP[:, 1]/(i+1)
METGP[:, 2] = METGP[:, 2]/(i+1)
METGP[:, 3] = METGP[:, 3]/(i+1)
REACOPTIM.ygp = METGP[:, 2].flatten()
REACOPTIM.timegp = METGP[:, 0].flatten()
REACOPTIM.timefgp = METGP[:, 1].flatten()

for i in range(1):#x0.shape[0]):
    start = time.time()
    REACOPTIM.optimizeref(49, 1, cores = 1, xinit = x0[i])
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    METREF[:, 0] = METREF[:, 0]+REACOPTIM.timeref.flatten()
    METREF[:, 1] = METREF[:, 1]+REACOPTIM.timefref.flatten()
    METREF[:, 2] = METREF[:, 2]+REACOPTIM.ytru.flatten()
    METREF[:, 3] = METREF[:, 3]+np.min(REACOPTIM.ytru)
    print('Best ref value is '+str(np.min(METREF[:, 3])/(i+1)))
    PARAMSREF[i] = REACOPTIM.xref[np.argmin(REACOPTIM.ytru)]
    DISTREF = np.vstack([DISTREF, REACOPTIM.xref])
    RESREF[:, i] = REACOPTIM.ytru[:, 0]
METREF[:, 0] = METREF[:, 0]/(i+1)
METREF[:, 1] = METREF[:, 1]/(i+1)
METREF[:, 2] = METREF[:, 2]/(i+1)
METREF[:, 3] = METREF[:, 3]/(i+1)
REACOPTIM.ytru = METREF[:, 2].flatten()
REACOPTIM.timeref = METREF[:, 0].flatten()
REACOPTIM.timefref = METREF[:, 1].flatten()

for i in range(x0.shape[0]):
    start = time.time()
    REACOPTIM.optimizersplt(12, 4, cons, 1, fcores = 4, afcores = 1, xinit = x0[i])
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    METSPLT[:, 0] = METSPLT[:, 0]+REACOPTIM.timesplt.flatten()
    METSPLT[:, 1] = METSPLT[:, 1]+REACOPTIM.timefsplt.flatten()
    METSPLT[:, 2] = METSPLT[:, 2]+REACOPTIM.yspltbst.flatten()
    METSPLT[:, 3] = METSPLT[:, 3]+np.min(REACOPTIM.yspltbst)
    print('Best split value is '+str(np.min(REACOPTIM.yspltbst)))
    PARAMSPLT[i] = REACOPTIM.xsplt[np.argmin(REACOPTIM.ysplt)]
    DISTSPLT = np.vstack([DISTSPLT, REACOPTIM.xsplt])
    RESPLT[:, i] = REACOPTIM.yspltbst[:, 0]
METSPLT[:, 0] = METSPLT[:, 0]/(i+1)
METSPLT[:, 1] = METSPLT[:, 1]/(i+1)
METSPLT[:, 2] = METSPLT[:, 2]/(i+1)
METSPLT[:, 3] = METSPLT[:, 3]/(i+1)
REACOPTIM.yspltbst = METSPLT[:, 2].flatten()
REACOPTIM.timesplt = METSPLT[:, 0].flatten()
REACOPTIM.timefsplt = METSPLT[:, 1].flatten()

for i in range(x0.shape[0]):
    start = time.time()
    REACOPTIM.optimizerspltref(12, 4, cons, 1, fcores = 4, afcores = 1, xinit = x0[i])
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    METSPLTREF[:, 0] = METSPLTREF[:, 0]+REACOPTIM.timespltref.flatten()
    METSPLTREF[:, 1] = METSPLTREF[:, 1]+REACOPTIM.timefspltref.flatten()
    METSPLTREF[:, 2] = METSPLTREF[:, 2]+REACOPTIM.yspltbstref.flatten()
    METSPLTREF[:, 3] = METSPLTREF[:, 3]+np.min(REACOPTIM.yspltbstref)
    print('Best split-ref value is '+str(np.min(REACOPTIM.yspltbstref)))
    PARAMSPLTREF[i] = REACOPTIM.xspltref[np.argmin(REACOPTIM.yspltref)]
    DISTSPLTREF = np.vstack([DISTSPLTREF, REACOPTIM.xspltref])
    RESPLTREF[:, i] = REACOPTIM.yspltbstref[:, 0]
METSPLTREF[:, 0] = METSPLTREF[:, 0]/(i+1)
METSPLTREF[:, 1] = METSPLTREF[:, 1]/(i+1)
METSPLTREF[:, 2] = METSPLTREF[:, 2]/(i+1)
METSPLTREF[:, 3] = METSPLTREF[:, 3]/(i+1)
REACOPTIM.yspltbstref = METSPLTREF[:, 2].flatten()
REACOPTIM.timespltref = METSPLTREF[:, 0].flatten()
REACOPTIM.timefspltref = METSPLTREF[:, 1].flatten()

for i in range(x0.shape[0]):
    start = time.time()
    REACOPTIM.optimizerspltvar(15, 2, liminit, 1, fcores = 3, afcores = 1, xinit = x0[i])
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    METSPLTVAR[:, 0] = METSPLTVAR[:, 0]+REACOPTIM.timespltvar.flatten()
    METSPLTVAR[:, 1] = METSPLTVAR[:, 1]+REACOPTIM.timefspltvar.flatten()
    METSPLTVAR[:, 2] = METSPLTVAR[:, 2]+REACOPTIM.yspltvarbst[:, -1].flatten()
    METSPLTVAR[:, 3] = METSPLTVAR[:, 3]+np.min(REACOPTIM.yspltvarbst[:, -1])
    print('Best split-var value is '+str(np.min(REACOPTIM.yspltvarbst)/8760))
    PARAMSPLTVAR[i] = REACOPTIM.xspltvar[np.argmin(REACOPTIM.yspltvar[:, -1])]
    DISTSPLTVAR = np.vstack([DISTSPLTVAR, REACOPTIM.xspltvar])
    RESPLTVAR[:, i] = REACOPTIM.yspltvarbst[:, -1]
METSPLTVAR[:, 0] = METSPLTVAR[:, 0]/(i+1)
METSPLTVAR[:, 1] = METSPLTVAR[:, 1]/(i+1)
METSPLTVAR[:, 2] = METSPLTVAR[:, 2]/(i+1)
METSPLTVAR[:, 3] = METSPLTVAR[:, 3]/(i+1)
REACOPTIM.yspltvarbst = METSPLTVAR[:, 2].reshape(-1, 1)
REACOPTIM.timespltvar = METSPLTVAR[:, 0].flatten()
REACOPTIM.timefspltvar = METSPLTVAR[:, 1].flatten()
    
def zr(x):
    x = x.reshape(-1, 1)
    x = x.reshape(int(x.shape[0]/2), 2)
    return np.zeros((x.shape[0], 3))
REACOPTIM.dist_ref['distrefmod'] = zr
for i in range(x0.shape[0]):
    start = time.time()
    REACOPTIM.optimizerspltvar(15, 2, liminit, 1, fcores = 3, afcores = 1, xinit = x0[i])
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    METSPLTVAR2[:, 0] = METSPLTVAR2[:, 0]+REACOPTIM.timespltvar.flatten()
    METSPLTVAR2[:, 1] = METSPLTVAR2[:, 1]+REACOPTIM.timefspltvar.flatten()
    METSPLTVAR2[:, 2] = METSPLTVAR2[:, 2]+REACOPTIM.yspltvarbst[:, -1].flatten()
    METSPLTVAR2[:, 3] = METSPLTVAR2[:, 3]+np.min(REACOPTIM.yspltvarbst[:, -1])
    print('Best split-var value is '+str(np.min(REACOPTIM.yspltvarbst)/8760))
    PARAMSPLTVAR2[i] = REACOPTIM.xspltvar[np.argmin(REACOPTIM.yspltvar[:, -1])]
    DISTSPLTVAR2 = np.vstack([DISTSPLTVAR2, REACOPTIM.xspltvar])
    RESPLTVAR2[:, i] = REACOPTIM.yspltvarbst[:, -1]
METSPLTVAR2[:, 0] = METSPLTVAR2[:, 0]/(i+1)
METSPLTVAR2[:, 1] = METSPLTVAR2[:, 1]/(i+1)
METSPLTVAR2[:, 2] = METSPLTVAR2[:, 2]/(i+1)
METSPLTVAR2[:, 3] = METSPLTVAR2[:, 3]/(i+1)

blks = 2
dim = 2
ϕ = 0.45
consb = {}
funsb = {}
funsb['1'] = lambda x: x[0]
funsb['2'] = lambda x: x[1]
lwrx = 0
uprx = 1-ϕ
lwry = ϕ
upry = 1
j = 1
for i in range(blks**2):
    nlc1 = NonlinearConstraint(funsb['1'], lwrx, uprx)
    nlc2 = NonlinearConstraint(funsb['2'], lwry, upry)
    consb[str(i+1)] = [nlc1, nlc2]
    lwrx = lwrx+ϕ
    uprx = uprx+ϕ
    if (i+1)%blks == 0:
        lwrx = 0
        uprx = 1-ϕ
        lwry = lwry-ϕ
        upry = upry-ϕ
expwl = np.array([0.5, 1, 2, 3])

for i in range(x0.shape[0]):
    start = time.time()
    REACOPTIM.hyperspace(12, 4, consb, 1, fcores = 4, afcores = 1, xinit = x0[i])
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    METHYP[:, 0] = METHYP[:, 0]+REACOPTIM.timehyp.flatten()
    METHYP[:, 1] = METHYP[:, 1]+REACOPTIM.timefhyp.flatten()
    METHYP[:, 2] = METHYP[:, 2]+REACOPTIM.yhypbst.flatten()
    METHYP[:, 3] = METHYP[:, 3]+np.min(REACOPTIM.yhypbst)
    print('Best hyperspace value is '+str(np.min(REACOPTIM.yhypbst)))
    PARAMSHYP[i] =  REACOPTIM.xhyp[np.argmin(REACOPTIM.yhyp)]
    DISTHYP = np.vstack([DISTHYP, REACOPTIM.xhyp])
    RESHYP[:, i] = REACOPTIM.yhypbst[:, 0]
METHYP[:, 0] = METHYP[:, 0]/(i+1)
METHYP[:, 1] = METHYP[:, 1]/(i+1)
METHYP[:, 2] = METHYP[:, 2]/(i+1)
METHYP[:, 3] = METHYP[:, 3]/(i+1)
REACOPTIM.yhypbst = METHYP[:, 2].flatten()
REACOPTIM.timehyp = METHYP[:, 0].flatten()
REACOPTIM.timefhyp = METHYP[:, 1].flatten()

for i in range(x0.shape[0]):
    start = time.time()
    REACOPTIM.optimizerexpw(12, 4, 1, fcores = 4, afcores = 1, xinit = x0[i])
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    METEXPW[:, 0] = METEXPW[:, 0]+REACOPTIM.timexpw.flatten()
    METEXPW[:, 1] = METEXPW[:, 1]+REACOPTIM.timefexpw.flatten()
    METEXPW[:, 2] = METEXPW[:, 2]+REACOPTIM.ybstexpw.flatten()
    METEXPW[:, 3] = METEXPW[:, 3]+np.min(REACOPTIM.ybstexpw)
    print('Best κ-varying value is '+str(np.min(REACOPTIM.ybstexpw)/8760))
    PARAMSEXPW[i] = REACOPTIM.xexpw[np.argmin(REACOPTIM.yexpw)]
    DISTEXPW = np.vstack([DISTEXPW, REACOPTIM.xexpw])
    RESEXPW[:, i] = REACOPTIM.ybstexpw[:, 0]
METEXPW[:, 0] = METEXPW[:, 0]/(i+1)
METEXPW[:, 1] = METEXPW[:, 1]/(i+1)
METEXPW[:, 2] = METEXPW[:, 2]/(i+1)
METEXPW[:, 3] = METEXPW[:, 3]/(i+1)
REACOPTIM.ybstexpw = METEXPW[:, 2].flatten()
REACOPTIM.timexpw = METEXPW[:, 0].flatten()
REACOPTIM.timefexpw = METEXPW[:, 1].flatten()

for i in range(x0.shape[0]):
    start = time.time()
    REACOPTIM.optimizernmc(12, 4, 20, 1, fcores = 4, afcores = 5, xinit = x0[i])
    end = time.time()
    print('Run time '+str(end-start)+'s')
    print('iteration '+str(i+1))
    METNMC[:, 0] = METNMC[:, 0]+REACOPTIM.timenmc.flatten()
    METNMC[:, 1] = METNMC[:, 1]+REACOPTIM.timefnmc.flatten()
    METNMC[:, 2] = METNMC[:, 2]+REACOPTIM.ybstnmc.flatten()
    METNMC[:, 3] = METNMC[:, 3]+np.min(REACOPTIM.ybstnmc)
    print('Best NxMCMC value is '+str(np.min(METNMC[:, 3])/(i+1)/8760))
    PARAMSNMC[i] = REACOPTIM.xnmc[np.argmin(REACOPTIM.ynmc)]
    DISTNMC = np.vstack([DISTNMC, REACOPTIM.xnmc])
    RESNMC[:, i] = REACOPTIM.ybstnmc[:, 0]
METNMC[:, 0] = METNMC[:, 0]/(i+1)
METNMC[:, 1] = METNMC[:, 1]/(i+1)
METNMC[:, 2] = METNMC[:, 2]/(i+1)
METNMC[:, 3] = METNMC[:, 3]/(i+1)
REACOPTIM.ybstnmc = METNMC[:, 2].flatten()
REACOPTIM.timenmc = METNMC[:, 0].flatten()
REACOPTIM.timefnmc = METNMC[:, 1].flatten()

# ## GENERTATE FIGURES
# # Convergence plots
# REACOPTIM.plots('R1')
# REACOPTIM.plotstime('R1')
# REACOPTIM.plotexptime('R1')
# Partition Plots
TRgp = np.linspace(303, 423, 451)
TRgp = np.meshgrid(TRgp, TRgp)
TRgp = np.hstack([TRgp[0].reshape(-1, 1), TRgp[1].reshape(-1, 1)])
Ctrefgp = gpparts.predict(REACOPTIM.scale(TRgp)).flatten()
xxgp = REACOPTIM.scale(TRgp)
# 1st partition
idx1 = np.where(Ctrefgp<=parts[0]+5e-1)[0]
idx2 = np.where(Ctrefgp>=parts[0]-5e-1)[0]
idx = np.array([], dtype = 'int')
for i in range(idx2.shape[0]):
    if idx2[i] in idx1:
        idx = np.hstack([idx, idx2[i]])
xf1 = xxgp[idx, 0]
yf1 = xxgp[idx, 1]
# 2nd and 3rd partition
idx1 = np.where(Ctrefgp<=parts[1]+7e-1)[0]
idx2 = np.where(Ctrefgp>=parts[1]-7e-1)[0]
idx = np.array([], dtype = 'int')
for i in range(idx2.shape[0]):
    if idx2[i] in idx1 and xxgp[idx2[i]][0]<0.5:
        idx = np.hstack([idx, idx2[i]])
xf2 = xxgp[idx, 0]
yf2 = xxgp[idx, 1]

idx1 = np.where(Ctrefgp<=parts[1]+2e-1)[0]
idx2 = np.where(Ctrefgp>=parts[1]-2e-1)[0]
idx = np.array([], dtype = 'int')
for i in range(idx2.shape[0]):
    if idx2[i] in idx1 and xxgp[idx2[i]][0]>0.5:
        idx = np.hstack([idx, idx2[i]])
xf3 = xxgp[idx, 0]
yf3 = xxgp[idx, 1]
# xf3 = xf2[np.where(xf2<0.25)]
# yf3 = yf2[np.where(xf2<0.25)]
# yf2 = yf2[np.where(xf2>0.25)]
# xf2 = xf2[np.where(xf2>0.25)]
# 4th partition
idx1 = np.where(Ctrefgp<=parts[2]+6e-1)[0]
idx2 = np.where(Ctrefgp>=parts[2]-6e-1)[0]
idx = np.array([], dtype = 'int')
for i in range(idx2.shape[0]):
    if idx2[i] in idx1:
        idx = np.hstack([idx, idx2[i]])
xf4 = xxgp[idx, 0]
yf4 = xxgp[idx, 1]

idx1 = np.where(Ctrefgp<=parts[2]+3e-0)[0]
idx2 = np.where(Ctrefgp>=parts[2]-3e-0)[0]
idx = np.array([], dtype = 'int')
for i in range(idx2.shape[0]):
    if idx2[i] in idx1 and xxgp[idx2[i]][0]<0.2:
        idx = np.hstack([idx, idx2[i]])
xf5 = xxgp[idx, 0]
yf5 = xxgp[idx, 1]


Tf1 = REACOPTIM.descale(np.hstack([xf1.reshape(-1, 1), yf1.reshape(-1, 1)]))
Tf2 = REACOPTIM.descale(np.hstack([xf2.reshape(-1, 1), yf2.reshape(-1, 1)]))
Tf3 = REACOPTIM.descale(np.hstack([xf3.reshape(-1, 1), yf3.reshape(-1, 1)]))
Tf4 = REACOPTIM.descale(np.hstack([xf4.reshape(-1, 1), yf4.reshape(-1, 1)]))
Tf5 = REACOPTIM.descale(np.hstack([xf5.reshape(-1, 1), yf5.reshape(-1, 1)]))

fig3D=pyp.figure(figsize=[11,8.5])
ax3D1=fig3D.add_subplot(111);
ax3D1.grid(color='gray',axis='both',alpha=0.25); ax3D1.set_axisbelow(True);
fig1=ax3D1.contourf(xxgp[:, 0].reshape(451,451),xxgp[:, 1].reshape(451,451), Ctrefgp.reshape(451,451),
                    cmap=cm.jet, levels = np.linspace(-400, -200, 51));
cbar=pyp.colorbar(fig1);
cbar.set_label(r'$Operating\ cost\ (1000\ USD/yr)$',fontsize=24,labelpad=15);
cbar.ax.tick_params(labelsize=20)
pyp.scatter(xf1, yf1, color = 'w', s = 10, marker = '+')
pyp.scatter(xf2, yf2, color = 'r', s = 10, marker = '+')
pyp.scatter(xf3, yf3, color = 'r', s = 10, marker = '+')
pyp.scatter(xf4, yf4, color = 'g', s = 10, marker = '+')
pyp.scatter(xf5, yf5, color = 'g', s = 10, marker = '+')
cbar.ax.tick_params(labelsize=20)
pyp.xlabel(r'$T_1$',fontsize=24);
pyp.xticks([0, 0.2, 0.4, 0.6, 0.8, 1]);
pyp.ylabel(r'$T_1$',fontsize=24);
pyp.yticks([0, 0.2, 0.4, 0.6, 0.8, 1]);
pyp.xlim((0, 1)); pyp.ylim((0, 1));
ax3D1.tick_params(axis='both',which='major',labelsize=20);
# pyp.savefig('GP_REF_MOD.png', dpi=300,edgecolor='white', bbox_inches='tight', pad_inches=0.1)

fig3D=pyp.figure(figsize=[11,8.5])
ax3D1=fig3D.add_subplot(111);
ax3D1.grid(color='gray',axis='both',alpha=0.25); ax3D1.set_axisbelow(True);
fig1=ax3D1.contourf(TRgp[:, 0].reshape(451,451),TRgp[:, 1].reshape(451,451), Ctrefgp.reshape(451,451),
                    cmap=cm.jet, levels = np.linspace(-400, -200, 51));
cbar=pyp.colorbar(fig1);
cbar.set_label(r'$Operating\ cost\ (1000\ USD/yr)$',fontsize=24,labelpad=15);
cbar.ax.tick_params(labelsize=20)
cbar.ax.tick_params(labelsize=20)
pyp.xlabel(r'$T_1$',fontsize=24);
pyp.scatter(Tf1[:, 0], Tf1[:, 1], color = 'w', s = 10, marker = '+')
pyp.scatter(Tf2[:, 0], Tf2[:, 1], color = 'r', s = 10, marker = '+')
pyp.scatter(Tf3[:, 0], Tf3[:, 1], color = 'r', s = 10, marker = '+')
pyp.scatter(Tf4[:, 0], Tf4[:, 1], color = 'g', s = 10, marker = '+')
pyp.xticks([320, 340, 360, 380, 400, 420]);
pyp.ylabel(r'$T_1$',fontsize=24);
pyp.yticks([320, 340, 360, 380, 400, 420]);
pyp.xlim((303, 423)); pyp.ylim((303, 423));
ax3D1.tick_params(axis='both',which='major',labelsize=20);
# pyp.savefig('REF_MOD_PARTS.png', dpi=300,edgecolor='white', bbox_inches='tight', pad_inches=0.1)
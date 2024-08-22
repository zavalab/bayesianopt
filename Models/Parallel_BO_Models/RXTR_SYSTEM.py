import numpy as np
from scipy.optimize import minimize, Bounds, fsolve
from joblib import Parallel, delayed
import time
import warnings
warnings.simplefilter('ignore', RuntimeWarning)

################# Initial Conditions and Physical Parameters #################
## RECYCLE REACTOR NETWORK
# Physicanl, Kinetic, and Thermodynamic parameters
R = 8.314
k0R = np.array([2.3283e8, 3.6556e9, 1.5234e12, 5.5640e8])
EaR = np.array([61200, 71800, 96700, 64500])
HR = np.array([ -450000., -300000., -250000.,  -500000.])
pR = np.array([850, 950, 875])
CpR=np.array([3000, 4000, 2900]);
CpH2O = 4000
TinH2O = 285
alpha = np.array([1000, 1, 500, 50, 5e4, 250])
Hvap = np.array([35, 50, 39, 48, 27, 42])
K_init = 1e-3

# Inlet Specs and Reactor Size
TinR = np.array([323, 323, 323]);
VR = np.array([1, 1.5]);
C0F1 = np.array([5000, 0, 0, 0, 0, 0])
C0F2 = np.array([0, 0, 0, 1000, 0, 0])
C0F3 = np.array([0, 0, 0, 0, 0, 1000])
C0var = np.loadtxt('C_var_rxtrs.txt')
FR = np.array([0.100, 0.075, 0.075])
R_Frac = 1e-12
rev_ratio = np.array([100, 100, 100, 100])

# Economic Parameters
cB = -0.48
cE = -0.15
cA = 0.12
cC = 0.075
cD = 0.15
cF = 0.05
cT = 0.010
c3 = np.array([cA, cB, cC, cD, cE, cF, cT])

## Initial Guesses
Cinit3=np.array([80, 1300, 200, 10, 130, 120])


#%%###########################################################################
# T range is 303 to 423
# Units are all in MKH (Hours) 

## REACTOR CLASS 
class CSTR():
    def __init__(self, T, F, C0, k0, Ea, Cinit, p, Cp, H, Tin, c, V, rev_ratio = rev_ratio):
        self.T = T.reshape(-1,1)
        self.F = F.reshape(-1,1)
        self.Cin = C0
        self.k0 = k0
        self.Ea = Ea
        self.Cinit = Cinit.reshape(Cinit.shape[0])
        self.p = np.array([p]).flatten()
        self.Cp = np.array([Cp]).flatten()
        self.H = H
        self.Tin = Tin
        self.c = c
        self.V = V
        self.rev_ratio = rev_ratio
        self.rxn_num = self.k0.shape[0]
        self.Cf = np.ones((self.T.shape[0],self.Cinit.shape[0]))
        self.rxn = np.ones((self.T.shape[0],self.rxn_num))
        self.mH2O = np.ones((self.T.shape[0]))
    
    def MandE_bal(self, m_bal, q_bal):
        for i in range(self.T.shape[0]):
            self.k = {}
            
            for j in range(self.rxn_num):
                self.k['k'+str(j+1)] = self.k0[j]*np.exp(-self.Ea[j]/(R*(self.T[i])))
                self.k['k'+str(j+1)+'r'] = self.k['k'+str(j+1)]/self.rev_ratio[j];
            
            def C(C0):
                self.C = m_bal(self.F[i], C0, self.Cin[i], self.k, self.V).reshape(self.Cinit.shape[0])
                return self.C-C0
            
            self.soln = fsolve(C, self.Cinit)
            self.Cf[i, :] = self.soln
            
            self.soln = q_bal(float(self.T[i, 0]), float(self.F[i, 0]),
                              self.Cf[i, :], self.k,
                              self.p[i], self.Cp[i], self.H,
                              self.Tin[i], self.V)
            
            self.rxn[i, :] = self.soln[0:self.rxn_num]
            self.mH2O[i] = self.soln[-1]
    
    def Econ(self, econ):
        self.Ct = econ(self.c, self.F, self.Cf, self.Cin, self.mH2O)/1e3
        return self.Ct


# Reference reactor model class
class CSTR_REF():
    def __init__(self, T1, T2, F, C0, p, Cp, H, Tin, c, V, thetam, rev_ratio = 100, rxt_num = 1):
        self.T1 = T1.reshape(-1,1)
        self.F = F.reshape(-1,1)
        self.T2 = T2.reshape(-1, 1)
        self.Cin = C0
        self.p = np.array([p]).flatten()
        self.Cp = np.array([Cp]).flatten()
        self.H = H
        self.Tin = Tin
        self.c = c
        self.V = V
        self.thetam = thetam
        self.rev_ratio = rev_ratio
        self.rxt_num = rxt_num
        self.rxn_num = self.thetam.shape[0]
    
    def MandE_bal(self, m_bal, q_bal):
        self.Cf = m_bal(np.hstack([self.T1, self.T2]), self.F[:,0], self.Cin, self.V, self.thetam, self.rxt_num)
        self.Cf = np.transpose(self.Cf)
        
        self.soln = q_bal(np.hstack([self.T1, self.T2]), self.F[:,0],
                          self.p, self.Cp, self.H,
                          self.Tin, self.V,
                          self.thetam, self.rxt_num)
        
        self.rxn = np.hstack(self.soln[0:self.rxn_num])
        self.mH2O = self.soln[-1]
    
    def Econ(self, econ):
        self.Ct=econ(self.c, self.F, self.Cf, self.Cin, self.mH2O)/1e3
        return self.Ct


# MASS BALANCE
def mass_bal_recycle(F, C, C0, k, V):
    Ca = C0[0]-2*(k['k1']*C[0]**2-k['k1r']*C[1]-k['k4']*C[2]*C[5])*V/F
    
    Cb = C0[1]+(k['k1']*C[0]**2-k['k1r']*C[1]-k['k2']*C[1]+k['k2r']*C[2]**2)*V/F
    
    Cc = C0[2]+(2*(k['k2']*C[1]-k['k2r']*C[2]**2)-k['k3']*C[2]*C[3]+\
            k['k3r']*C[4]-k['k4']*C[2]*C[5])*V/F
        
    Cd = C0[3]-(k['k3']*C[2]*C[3]-k['k3r']*C[4])*V/F
    
    Ce = C0[4]+(k['k3']*C[2]*C[3]-k['k3r']*C[4])*V/F
    
    Cf = C0[5]-(k['k4']*C[2]*C[5])*V/F
    
    return np.concatenate([Ca, Cb, Cc, Cd, Ce, Cf])


# ENERGY BALANCE
def heat_bal_recycle(T, F, C, k, ρ, Cp, H, Tin, V):
    r1 = k['k1']*C[0]**2-k['k1r']*C[1]
    
    r2 = k['k2']*C[1]-k['k2r']*C[2]**2
    
    r3 = k['k3']*C[2]*C[3]-k['k3r']*C[4]
    
    r4 = k['k4']*C[2]*C[5]
    
    Qdot = -(r1*H[0]+r2*H[1]+r3*H[2]+r4*H[3])*V
    
    ToH2O = min(T-10,423) #313
    
    mH2O = (ρ*Cp*F*(Tin-T)+Qdot)/(CpH2O*(ToH2O-TinH2O))
    
    mH2O[mH2O<0] = 0

    return np.concatenate([r1, r2, r3, r4, mH2O])


# ECONOMICS
def econ1(c, F, C, C0, mH2O):
    Ct=c[0]*(C[:,0]+C0[:, 0])*F[:,0]+c[1]*C[:,1]*F[:,0]+c[2]*C[:,2]*F[:,0]+c[3]*mH2O
    return Ct

def econ2(c, F, C, C0, mH2O):
    Ct=c[0]*C[:,0]*F[:,0]+c[1]*C[:,1]*F[:,0]+c[2]*C[:,2]*F[:,0]+\
        c[3]*(C[:,3]+C0[:,3])*F[:,0]+c[4]*C[:,4]*F[:,0]+c[5]*mH2O
    return Ct

def econ_recycle1(c, F, C, C0, mH2O):
    Ct=c[0]*(C[:,0]+C0[:, 0])*F[:,0]+c[1]*C[:,1]*F[:,0]+c[2]*C[:,2]*F[:,0]+\
        c[3]*(C[:,3])*F[:,0]+c[4]*(C[:,4])*F[:,0]+\
        c[5]*(C[:,5]+C0[:, 5])*F[:,0]+c[6]*mH2O
    return Ct

def econ_recycle2(c, F, C, C0, mH2O):
    Ct=c[0]*(C[:,0])*F[:,0]+c[1]*C[:,1]*F[:,0]+c[2]*C[:,2]*F[:,0]+\
        c[3]*(C[:,3]+C0[:, 3])*F[:,0]+c[4]*(C[:,4])*F[:,0]+\
        c[5]*(C[:,5])*F[:,0]+c[6]*mH2O
    return Ct


## MIXER CLASS
class MIXER():
    def __init__(self, Cin, Fin, Tin, p, Cp, Tref = 298, all_C0 = False):
        self.Cin = Cin
        self.Fin = Fin
        self.Tin = Tin
        self.p = p
        self.Cp = Cp
        self.Tref = Tref
        self.m_in = p*Fin
        self.xin = self.m_in/sum(self.m_in)
        self.Cpmix = sum(self.Cp*self.xin)
        self.pmix = sum(self.p*self.xin)
        self.all_C0 = all_C0
    
    def mass_bal(self):
        self.Nin = {}
        
        for i in range(self.Fin.shape[0]):
            self.Nin['N'+str(i+1)] = (self.Fin[i]*self.Cin[i])
        
        if not self.all_C0:
            self.Nin = np.array(list(self.Nin.values()),dtype=tuple)
            self.Nin = np.hstack(self.Nin[:]).astype(float)
        else:
            self.Nin = sum(np.array(list(self.Nin.values())))
        
        self.mout = sum(self.p*self.Fin)
        self.Fout = self.mout/self.pmix
        self.Cout = self.Nin/self.Fout
        return self.Cout
    
    def heat_bal(self):
        self.Tout = self.Tref+sum(self.xin*self.Cp*(self.Tin-self.Tref))/self.Cpmix
        return self.Tout
    
    
## SPLITTER CLASS
class SPLITTER():
    def __init__(self, Fin, split, split_frac = True):
        self.Fin = Fin
        if split_frac:
            self.split_frac = split
        else:
            self.split_frac = split/self.Fin
        self.split_orig = 1-sum(self.split_frac)
    
    def split(self):
        self.Fout = np.hstack([(self.split_frac*self.Fin).reshape(-1, 1),
                               (self.split_orig*self.Fin).reshape(-1, 1)])
        return self.Fout


## FLASH CLASS
class FLASH():
    def __init__(self, Nin, VtoF, xin, alpha, Hvap, Hstm = 2080):
        self.Nin = Nin
        self.f = VtoF
        self.V = self.f*self.Nin
        self.L = (1-self.f)*self.Nin
        self.xin = xin
        self.alpha = alpha
        self.Hvap = Hvap # (kJ/mol)
        self.n = xin.shape[1]
        self.Hstm = Hstm # By default set to saturated steam at 160degC (kJ/kg)
        
    def mass_bal(self, K):
        x = np.zeros(self.n)
        self.K_ref = np.ones(self.xin.shape[0])
        self.x_liq = np.zeros(self.xin.shape)
        self.y_vap = np.zeros(self.xin.shape)
        
        def C(K, xin, f):
            for i in range(self.n):
                x[i] = xin[i]/(f*(K*self.alpha[i]-1)+1)
            return(np.sum(x)-1)
        
        for j in range(self.K_ref.shape[0]):    
            self.K_ref[j] = fsolve(C, K[j], args = (self.xin[j], self.f[j]))
        
        for i in range(self.n):
            self.x_liq[:, i] = self.xin[:, i]/(self.f*(self.K_ref*self.alpha[i]-1)+1)
            self.y_vap[:, i] = self.K_ref*self.alpha[i]*self.x_liq[:, i]
    
    def heat_bal(self):
        V = (self.f*self.Nin).reshape(-1, 1)
        self.Q = self.Hvap*V*self.y_vap
        self.Q = np.sum(self.Q, axis = 1)
        self.m_steam = self.Q/self.Hstm # in kg of steam


#%% RECYCLE SYSTEM MODEL as f(T1, T2)

def Rxtr_Recycle(FR, TR, R_Frac, Cdist = np.ones(3),  rxn_reg = False):
    # Recycle setup
    err = 1e6
    FinR3 = FR[-1]
    
    while err > 1e-6:
        pR3 = (FinR3/(FinR3+FR[2]))*pR[1]+(FR[2]/(FinR3+FR[2]))*pR[2]
        Fr = R_Frac*(pR[0]*FR[0]+pR[1]*FR[1]+pR[2]*FR[2])/((1-R_Frac)*pR3)
        err = abs(Fr-FinR3)
        FinR3 = Fr*1
    
    pR3 = pR3*np.ones(TR.shape[0])
    C0R3 = C0F3*np.ones(TR[:, 0].reshape(-1, 1).shape)
    CpR3 = (FinR3/(FinR3+FR[2]))*CpR[1]+(FR[2]/(FinR3+FR[2]))*CpR[2]
    CpR3 = CpR3*np.ones(TR.shape[0])
    TinR3 = (FR[2]*CpR[2]*TinR[-1]+FinR3*CpR3*TR[:, -1])/(FinR3*CpR3+FR[2]*CpR[2])
    err = np.array([1e6])
    FinR3 = (FinR3+FR[2])*np.ones(TR.shape[0])
    
    j = 0
    while all(err > 1e-6):
        # Reactor 1
        C0R1 = np.ones((TR.shape[0], Cinit3.shape[0]))
        TinR1 = np.ones(TR.shape[0])
        pR1 = np.ones(TR.shape[0])
        CpR1 = np.ones(TR.shape[0])
        FinR1 = np.ones(TR.shape[0])
        
        for i in range(TR.shape[0]):
            CinR1 = np.array([C0R3[i], C0F1*Cdist[0]], dtype = tuple)
            Tmix1 = np.array([TinR3[i], TinR[0]]) # make this an array
            Fmix1 = np.array([FinR3[i], FR[0]])
            pmix1 = np.array([pR3[i], pR[0]])
            Cpmix1 = np.array([CpR3[i], CpR[0]])
            MixR1 = MIXER(CinR1, Fmix1, Tmix1, pmix1, Cpmix1, all_C0 = True)
            C0R1[i] = MixR1.mass_bal()
            TinR1[i] = MixR1.heat_bal()
            pR1[i] = MixR1.pmix
            CpR1[i] = MixR1.Cpmix
            FinR1[i] = MixR1.Fout
        
        ReacR1 = CSTR(TR[:, 0], FinR1, C0R1, k0R, EaR, Cinit3, pR1, CpR1, HR, TinR1, c3, VR[0])
        ReacR1.MandE_bal(mass_bal_recycle, heat_bal_recycle)
        
        # Reactor 2
        C0R2 = np.ones((TR.shape[0], Cinit3.shape[0]))
        TinR2 = np.ones(TR.shape[0])
        pR2 = np.ones(TR.shape[0])
        CpR2 = np.ones(TR.shape[0])
        FinR2 = np.ones(TR.shape[0])
        
        for i in range(TR.shape[0]):
            CinR2 = np.array([ReacR1.Cf[i], C0F2*Cdist[1]], dtype = tuple)
            Tmix2 = np.array([ReacR1.T[i, 0], TinR[1]])
            Fmix2 = np.array([ReacR1.F[i, 0], FR[1]])
            pmix2 = np.array([ReacR1.p[i], pR[1]])
            Cpmix2 = np.array([ReacR1.Cp[i], CpR[1]])
            MixR2 = MIXER(CinR2, Fmix2, Tmix2, pmix2, Cpmix2, all_C0 = True)
            C0R2[i] = MixR2.mass_bal()
            TinR2[i] = MixR2.heat_bal()
            pR2[i] = MixR2.pmix
            CpR2[i] = MixR2.Cpmix
            FinR2[i] = MixR2.Fout
            
        ReacR2 = CSTR(TR[:, 1], FinR2, C0R2, k0R, EaR, Cinit3, pR2, CpR2, HR, TinR2, c3, VR[1])
        ReacR2.MandE_bal(mass_bal_recycle, heat_bal_recycle)
        
        # Flash 1
        Nsep = np.sum(ReacR2.Cf*ReacR2.F, axis = 1)
        xsep = ReacR2.Cf*ReacR2.F/Nsep.reshape(-1, 1)
        VtoF = xsep[:, -2].flatten()
        Fl1 = FLASH(Nsep, VtoF, xsep, alpha, Hvap)
        Fl1.mass_bal(K_init*np.ones(TR.shape[0]))
        Fl1.heat_bal()
        
        # Flash 2
        Fl2 = FLASH(Fl1.L, (1-Fl1.x_liq[:, 1]).flatten(), Fl1.x_liq, alpha, Hvap)
        Fl2.mass_bal(K_init*np.ones(TR.shape[0]))
        Fl2.heat_bal()
        
        # Product flows
        Fpd1 = (Fl1.f.reshape(-1, 1)*ReacR2.F).flatten()
        Npd1 = Fl1.V.reshape(-1, 1)*Fl1.y_vap
        Fpd2 = ((1-Fl2.f).reshape(-1, 1)*(ReacR2.F-Fpd1.reshape(-1, 1))).flatten()
        Npd2 = Fl2.L.reshape(-1, 1)*Fl2.x_liq
        V = Fl2.f.reshape(-1, 1)*(ReacR2.F-Fpd1.reshape(-1, 1))
        CfFl2 = Fl2.y_vap*Fl2.V.reshape(-1, 1)/V
        TFl2 = 423*np.ones(TR.shape[0])
        
        # Recycle
        SplitR1 = SPLITTER(V, np.array([R_Frac]))
        Fr, Fprg = SplitR1.split().T
        C0R3 = np.ones((TR.shape[0], Cinit3.shape[0]))
        TinR3 = np.ones(TR.shape[0])
        ρR3 = np.ones(TR.shape[0])
        CpR3 = np.ones(TR.shape[0])
        FinR3 = np.ones(TR.shape[0])
        
        for i in range(TR.shape[0]):
            CinR3 = np.array([CfFl2[i], C0F3*Cdist[2]], dtype = tuple)
            Tmix3 = np.array([TFl2[i], TinR[-1]])
            Fmix3 = np.array([Fr[i], FR[-1]])
            pmix3 = np.array([ReacR2.p[i], pR[-1]])
            Cpmix3 = np.array([ReacR2.Cp[i], CpR[-1]])
            MixR3 = MIXER(CinR3, Fmix3, Tmix3, pmix3, Cpmix3, all_C0 = True)
            C0R3[i] = MixR3.mass_bal()
            TinR3[i] = MixR3.heat_bal()
            ρR3[i] = MixR3.pmix
            CpR3[i] = MixR3.Cpmix
            FinR3[i] = MixR3.Fout
            
        err = ((Fpd1*ReacR2.p+Fpd2*ReacR2.p+Fprg*ReacR2.p-
                (FR[0]*pR[0]+FR[1]*pR[1]+FR[2]*pR[2]))**2)
        j += 1
    
    if rxn_reg:
        R1_rxn = ReacR1.soln[:-1]
        R2_rxn = ReacR2.soln[:-1]
        
        return R1_rxn, R2_rxn
    
    else:
        CtR1 = ReacR1.Econ(econ_recycle1)
        CtR2 = ReacR2.Econ(econ_recycle2)
        CtR = Npd1@c3[:-1]+Npd2@c3[:-1]+c3[-1]*(ReacR1.mH2O+ReacR2.mH2O)+\
            100*c3[-1]*(Fl1.m_steam+Fl2.m_steam)+FR[0]*C0F1@c3[:-1]
        CtR = CtR/1e3
        
        return CtR1, CtR2, CtR

def SYST_RECYCLE(T, FR, R_Frac, Cdist, rxn_reg = False):
    T = T.flatten()
    T = T.reshape(int(T.shape[0]/2), 2)
    
    if rxn_reg:
        R1_rxn, R2_rxn = Rxtr_Recycle(FR, T, R_Frac, Cdist, rxn_reg = rxn_reg)
        return np.hstack([R1_rxn, R2_rxn]).reshape(-1,1).T
    else:
        CtR1, CtR2, CtR = Rxtr_Recycle(FR, T, R_Frac, Cdist)
        return CtR1, CtR2, CtR


#%% RECYCLE LOOP REFERENCE MODEL FORMULATION
# Reaction Regressions
def scale(x, ub, lb, sf = 1):
    m = 2*sf/(ub-lb)
    b = sf-m*ub
    return m*x+b

def descale(x, ub, lb, sf = 1):
    b = (ub+lb)/2
    m = (b-lb)/sf
    return m*x+b

def RXTR_REG():
    print('Calculation of reaction model weights...')
    start = time.time()
    TTr = np.linspace(303, 423, 10)
    TTr = np.meshgrid(TTr, TTr)
    TTr = np.hstack([TTr[0].reshape(-1, 1), TTr[1].reshape(-1, 1)])
    INPTS = np.hstack([TTr, C0var[:TTr.shape[0]]])
    
    RXNS = Parallel(n_jobs = 4)(delayed(SYST_RECYCLE)(INPTS[i, :2], FR, R_Frac,
                                                      INPTS[i, 2:], rxn_reg = True)
                                for i in np.arange(0, INPTS.shape[0], 1))
    
    TTr = scale(1/TTr, ub = 1/303, lb = 1/423)
    TTr = np.hstack([TTr, TTr**2, TTr**3, np.ones((TTr.shape[0], 1))])
    RXNS = np.vstack(RXNS[:][:])
    lnRXNS = np.log(abs(RXNS))
    
    lo = np.min(lnRXNS, axis = 0)
    hi = np.max(lnRXNS, axis = 0)
    sf = np.std(lnRXNS, axis = 0)
    n_params = 7
    n_eqns = RXNS.shape[1]
    idx = np.random.uniform(-10, 10, size = 100)
    
    lb = -100*np.ones((n_params*n_eqns))
    lb = lb.reshape(n_eqns, n_params)
    lb[:4, 1] = 0
    lb[:4, 3] = 0
    lb[:4, 5] = 0
    
    ub = 100*np.ones((n_params*n_eqns))
    ub = ub.reshape(n_eqns, n_params)
    ub[:4, 1] = 1e-9
    ub[:4, 3] = 1e-9
    ub[:4, 5] = 1e-9
    
    thetaR = np.ones(shape = (n_eqns, n_params))
    theta = np.ones(n_params)
    
    for j in range(n_eqns):
        sf[j] = max(sf[j], 1)
        lnRXNSn = scale(lnRXNS[:, j], hi[j], lo[j], sf = sf[j])
        
        def fit(theta):
            lam = 0.05
            lnrn = TTr@theta
            return sum((lnRXNSn-lnrn)**2)+lam*theta.T@theta
        
        bndsm = Bounds(lb[j], ub[j])
        
        soln = Parallel(n_jobs = 4)(delayed(minimize)(fit, i*theta, method = 'L-BFGS-B', 
                                                       bounds = bndsm) for i in idx)
        
        loss = np.array([np.atleast_1d(res.fun)[0] for res in soln])
        theta = np.array([res.x for res in soln],dtype = 'float')
        theta = theta[np.argmin(loss)]
        thetaR[j] = theta
    
    r = np.ones(RXNS.shape)
    
    for i in range(8):
        r[:, i] = np.exp(descale(TTr@thetaR[i], hi[i], lo[i], sf = sf[i]))
    
    hi = np.hstack([hi[:4].reshape(-1, 1), hi[4:].reshape(-1, 1)])
    lo = np.hstack([lo[:4].reshape(-1, 1), lo[4:].reshape(-1, 1)])
    sf = np.hstack([sf[:4].reshape(-1, 1), sf[4:].reshape(-1, 1)])
        
    def mass_bal_reg_recycle(T, F, C0, V, theta, rxt_num):
        rn = rxt_num-1
        invT = scale(1/T, ub = 1/303, lb = 1/423)
        invT = np.hstack([invT, invT**2, invT**3, np.ones((T.shape[0], 1))])
        
        r = np.exp(descale(invT@theta.T, hi[:, rn], lo[:, rn], sf[:, rn])).T
        
        Ca = C0[:, 0]-2*(r[0]-r[3])*V/F
        
        Cb = C0[:, 1]+(r[0]-r[1])*V/F
        
        Cc = C0[:, 2]+(2*r[1]-r[2]-r[3])*V/F
        
        Cd = C0[:, 3]-r[2]*V/F
        
        Ce = C0[:, 4]+r[2]*V/F
        
        Cf = C0[:, 5]-r[3]*V/F
        
        return np.array([Ca, Cb, Cc, Cd, Ce, Cf])
        
    def heat_bal_reg_recycle(T, F, p, Cp, H, Tin, V, theta, rxt_num):
        rn = rxt_num-1
        T1 = T[:, rn]
        invT = scale(1/T, ub = 1/303, lb = 1/423)
        invT = np.hstack([invT, invT**2, invT**3, np.ones((T.shape[0], 1))])
        
        r = np.exp(descale(invT@theta.T, hi[:,rn], lo[:,rn], sf[:,rn])).T
        
        Qdot=-(r[0]*H[0]+r[1]*H[1]+r[2]*H[2]+r[3]*H[3])*V
        
        ToH2O=T1-10
        
        ToH2O[ToH2O>423] = 423
        
        mH2O=(p*Cp*F*(Tin-T1)+Qdot)/(CpH2O*(ToH2O-TinH2O))
        
        mH2O[mH2O<0] = 0

        return r[0], r[1], r[2], r[3], mH2O

    def Rxtr_Recycle_Ref(FR, TR, R_Frac, theta, Cdist = np.ones(3)):
        # Recycle setup
        err = 1e6
        FinR3 = FR[-1]
        
        while err > 1e-6:
            pR3 = (FinR3/(FinR3+FR[2]))*pR[1]+(FR[2]/(FinR3+FR[2]))*pR[2]
            Fr = R_Frac*(pR[0]*FR[0]+pR[1]*FR[1]+pR[2]*FR[2])/((1-R_Frac)*pR3)
            err = abs(Fr-FinR3)
            FinR3 = Fr*1
            
        pR3 = pR3*np.ones(TR.shape[0])
        C0R3 = C0F3*np.ones(TR[:, 0].reshape(-1, 1).shape)
        CpR3 = (FinR3/(FinR3+FR[2]))*CpR[1]+(FR[2]/(FinR3+FR[2]))*CpR[2]
        CpR3 = CpR3*np.ones(TR.shape[0])
        TinR3 = (FR[2]*CpR[2]*TinR[-1]+FinR3*CpR3*TR[:, -1])/(FinR3*CpR3+FR[2]*CpR[2])
        err = np.array([1e6])
        FinR3 = FinR3+FR[2]
        
        j = 0
        while all(err > 1e-6):
            # Reactor 1
            C0R1 = np.ones((TR.shape[0], Cinit3.shape[0]))
            TinR1 = np.ones(TR.shape[0])
            ρR1 = np.ones(TR.shape[0])
            CpR1 = np.ones(TR.shape[0])
            FinR1 = np.ones(TR.shape[0])
            
            for i in range(TR.shape[0]):
                CinR1 = np.array([C0R3[i], C0F1*Cdist[0]], dtype = tuple)
                Tmix1 = np.array([TinR3[i], TinR[0]])
                Fmix1 = np.array([FinR3, FR[0]])
                pmix1 = np.array([pR3[i], pR[0]])
                Cpmix1 = np.array([CpR3[i], CpR[0]])
                MixR1 = MIXER(CinR1, Fmix1, Tmix1, pmix1, Cpmix1, all_C0 = True)
                C0R1[i] = MixR1.mass_bal()
                TinR1[i] = MixR1.heat_bal()
                ρR1[i] = MixR1.pmix
                CpR1[i] = MixR1.Cpmix
                FinR1[i] = MixR1.Fout
                
            ReacR1_REF = CSTR_REF(TR[:, 0], TR[:, 1], FinR1, C0R1, ρR1, CpR1, HR, TinR1, c3, VR[0],
                                  theta[:4])
            ReacR1_REF.MandE_bal(mass_bal_reg_recycle, heat_bal_reg_recycle)
            
            # Reactor 2
            C0R2 = np.ones((TR.shape[0], Cinit3.shape[0]))
            TinR2 = np.ones(TR.shape[0])
            pR2 = np.ones(TR.shape[0])
            CpR2 = np.ones(TR.shape[0])
            FinR2 = np.ones(TR.shape[0])
            
            for i in range(TR.shape[0]):
                CinR2 = np.array([ReacR1_REF.Cf[i], C0F2*Cdist[1]], dtype = tuple)
                Tmix2 = np.array([ReacR1_REF.T1[i, 0], TinR[1]])
                Fmix2 = np.array([ReacR1_REF.F[i, 0], FR[1]])
                pmix2 = np.array([ReacR1_REF.p[i], pR[1]])
                Cpmix2 = np.array([ReacR1_REF.Cp[i], CpR[1]])
                MixR2 = MIXER(CinR2, Fmix2, Tmix2, pmix2, Cpmix2, all_C0 = True)
                C0R2[i] = MixR2.mass_bal()
                TinR2[i] = MixR2.heat_bal()
                pR2[i] = MixR2.pmix
                CpR2[i] = MixR2.Cpmix
                FinR2[i] = MixR2.Fout
                
            ReacR2_REF = CSTR_REF(TR[:, 0], TR[:, 1], FinR2, C0R2, pR2, CpR2, HR, TinR2, c3, VR[1],
                                  theta[4:], rxt_num=2)
            ReacR2_REF.MandE_bal(mass_bal_reg_recycle, heat_bal_reg_recycle)
            
            # Flash 1
            Nsep = np.sum(ReacR2_REF.Cf*ReacR2_REF.F, axis = 1)
            xsep = ReacR2_REF.Cf*ReacR2_REF.F/Nsep.reshape(-1, 1)
            VtoF = xsep[:, -2].flatten()
            Fl1 = FLASH(Nsep, VtoF, xsep, alpha, Hvap)
            Fl1.mass_bal(K_init*np.ones(TR.shape[0]))
            Fl1.heat_bal()
            
            # Flash 2
            Fl2 = FLASH(Fl1.L, (1-Fl1.x_liq[:, 1]).flatten(), Fl1.x_liq, alpha, Hvap)
            Fl2.mass_bal(K_init*np.ones(TR.shape[0]))
            Fl2.heat_bal()
            
            # Product Flows
            Fpd1 = (Fl1.f.reshape(-1, 1)*ReacR2_REF.F).flatten()
            Npd1 = Fl1.V.reshape(-1, 1)*Fl1.y_vap
            Fpd2 = ((1-Fl2.f).reshape(-1, 1)*(ReacR2_REF.F-Fpd1.reshape(-1, 1))).flatten()
            Npd2 = Fl2.L.reshape(-1, 1)*Fl2.x_liq
            V = Fl2.f.reshape(-1, 1)*(ReacR2_REF.F-Fpd1.reshape(-1, 1))
            CfFl2 = Fl2.y_vap*Fl2.V.reshape(-1, 1)/V
            TFl2 = 423*np.ones(TR.shape[0])
            
            # Reycle
            SplitR1 = SPLITTER(V, np.array([R_Frac]))
            Fr, Fprg = SplitR1.split().T
            C0R3 = np.ones((TR.shape[0], Cinit3.shape[0]))
            TinR3 = np.ones(TR.shape[0])
            ρR3 = np.ones(TR.shape[0])
            CpR3 = np.ones(TR.shape[0])
            
            for i in range(TR.shape[0]):
                CinR3 = np.array([CfFl2[i], C0F3*Cdist[2]], dtype = tuple)
                Tmix3 = np.array([TFl2[i], TinR[-1]])
                Fmix3 = np.array([Fr[i], FR[-1]])
                pmix3 = np.array([ReacR2_REF.p[i], pR[-1]])
                Cpmix3 = np.array([ReacR2_REF.Cp[i], CpR[-1]])
                MixR3 = MIXER(CinR3, Fmix3, Tmix3, pmix3, Cpmix3, all_C0 = True)
                C0R3[i] = MixR3.mass_bal()
                TinR3[i] = MixR3.heat_bal()
                ρR3[i] = MixR3.pmix
                CpR3[i] = MixR3.Cpmix
                FinR3 = MixR3.Fout
                
            err = (Fpd1*ReacR2_REF.p+Fpd2*ReacR2_REF.p+Fprg*ReacR2_REF.p-
                   (FR[0]*pR[0]+FR[1]*pR[1]+FR[2]*pR[2]))**2
            j += 1
            
        CtR1 = ReacR1_REF.Econ(econ_recycle1)
        CtR2 = ReacR2_REF.Econ(econ_recycle2)
        
        CtR = Npd1@c3[:-1]+Npd2@c3[:-1]+c3[-1]*(ReacR1_REF.mH2O+ReacR2_REF.mH2O)+\
            100*c3[-1]*(Fl1.m_steam+Fl2.m_steam)+FR[0]*C0F1@c3[:-1]
        CtR = CtR/1e3
        
        return CtR1, CtR2, CtR

    def SYST_RECYCLE_REF(T, FR, R_Frac, Cdist):
        T = T.flatten()
        T = T.reshape(int(T.shape[0]/2), 2)
        
        CtR1, CtR2, CtR = Rxtr_Recycle_Ref(FR, T, R_Frac, thetaR, Cdist)
        return CtR1, CtR2, CtR
    
    end = time.time()
    print(end-start)
    return SYST_RECYCLE_REF

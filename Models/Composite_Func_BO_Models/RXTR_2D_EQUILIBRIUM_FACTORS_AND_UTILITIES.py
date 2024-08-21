import numpy as np
from scipy.optimize import Bounds, fsolve
import sklearn.gaussian_process as gpr

import sys
sys.path.append('./../../BO_algos')
import Composite_Func_Algos as BO_algos

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)

##################Initial Conditions and Physical Parameters##################
## RECYCLE REACTOR NETWORK
# Physicanl, Kinetic, and Thermodynamic parameters
R = 8.314
k0R = np.array([2.3283e8, 3.6556e9, 1.5234e12, 5.5640e8])
EaR = np.array([61200, 71800, 96700, 64500])
HR = np.array([ -450000., -300000., -250000.,  -500000.])
ρR = np.array([850,950,875])
CpR = np.array([3000,4000,2900])
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
FR = np.array([0.100, 0.075, 0.075])
R_Frac = 1e-12
rev_ratio = np.array([100, 100, 100, 100])

## Economic Parameter
cB = -0.48
cE = -0.15
cA = 0.12
cC = 0.075
cD = 0.15
cF = 0.05
cT = 0.010
c3=np.array([cA,cB,cC,cD,cE,cF,cT])

## Initial Guesses
Cinit3 = np.array([80, 1300, 200, 10, 130, 120])
##############################################################################
#T range is 323 to 423
# Units are all in MKH (Hours) 

## REACTOR CLASS 
class CSTR():
    def __init__(self, T, F, C0, k0, Ea, Cinit, p, Cp, H, Tin, c, V,
                 rev_ratio=rev_ratio):
        self.T = T.reshape(-1, 1)
        self.F = F.reshape(-1, 1)
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
        self.Cf = np.ones((self.T.shape[0], self.Cinit.shape[0]))
        self.rxn = np.ones((self.T.shape[0], self.rxn_num))
        self.mH2O = np.ones((self.T.shape[0]))

    def MandE_bal(self, m_bal, q_bal):
        for i in range(self.T.shape[0]):
            self.k = {}
            for j in range(self.rxn_num):
                self.k['k'+str(j+1)] = self.k0[j]*np.exp(-self.Ea[j]/(R*(self.T[i])))
                self.k['k'+str(j+1)+'r'] = self.k['k'+str(j+1)]/self.rev_ratio[j]
            
            def C(C0):
                self.C=m_bal(self.F[i], C0, self.Cin[i],
                             self.k,self.V).reshape(self.Cinit.shape[0])
                return self.C-C0
            
            self.soln = fsolve(C,self.Cinit)
            self.Cf[i, :] = self.soln
            self.soln = q_bal(float(self.T[i, 0]), float(self.F[i, 0]),
                              self.Cf[i, :], self.k,self.p[i], self.Cp[i],
                              self.H, self.Tin[i], self.V)
            self.rxn[i, :] = self.soln[0:self.rxn_num]
            self.mH2O[i] = self.soln[-1]
    
    def Econ(self, econ):
        self.Ct=econ(self.c, self.F, self.Cf, self.Cin, self.mH2O)/1e3
        return self.Ct


class CSTR_REF():
    def __init__(self, T1, T2, F, C0, p, Cp, H, Tin, c, V, thetam,
                 rev_ratio=100, rxt_num=1):
        self.T1 = T1.reshape(-1, 1)
        self.F = F.reshape(-1, 1)
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
    
    def MandE_bal(self,m_bal,q_bal):
        self.Cf = m_bal(np.hstack([self.T1, self.T2]), self.F[:,0], self.Cin,
                        self.V, self.thetam, self.rxt_num)
        self.Cf = np.transpose(self.Cf)
        self.soln = q_bal(np.hstack([self.T1,self.T2]), self.F[:,0], self.p,
                          self.Cp, self.H, self.Tin, self.V, self.thetam, self.rxt_num)
        self.rxn = np.hstack(self.soln[0:self.rxn_num])
        self.mH2O = self.soln[-1]
    
    def Econ(self, econ):
        self.Ct = econ(self.c, self.F, self.Cf, self.Cin, self.mH2O)/1e3
        return self.Ct


# MASS BALANCES
def mass_bal_recycle(F, C, C0, k, V):
    Ca = C0[0]-2*(k['k1']*C[0]**2-k['k1r']*C[1]-k['k4']*C[2]*C[5])*V/F
    Cb = C0[1]+(k['k1']*C[0]**2-k['k1r']*C[1]-k['k2']*C[1]+k['k2r']*C[2]**2)*V/F
    Cc = C0[2]+(2*(k['k2']*C[1]-k['k2r']*C[2]**2)-k['k3']*C[2]*C[3]+\
            k['k3r']*C[4]-k['k4']*C[2]*C[5])*V/F
    Cd = C0[3]-(k['k3']*C[2]*C[3]-k['k3r']*C[4])*V/F
    Ce = C0[4]+(k['k3']*C[2]*C[3]-k['k3r']*C[4])*V/F
    Cf = C0[5]-(k['k4']*C[2]*C[5])*V/F
    return np.concatenate([Ca, Cb, Cc, Cd, Ce, Cf])


# ENERGY BALANCES
def heat_bal_recycle(T, F, C, k, p, Cp, H, Tin, V):
    r1 = k['k1']*C[0]**2-k['k1r']*C[1]
    r2 = k['k2']*C[1]-k['k2r']*C[2]**2
    r3 = k['k3']*C[2]*C[3]-k['k3r']*C[4]
    r4 = k['k4']*C[2]*C[5]
    Qdot = -(r1*H[0]+r2*H[1]+r3*H[2]+r4*H[3])*V
    ToH2O = min(T-10,423) #313
    mH2O = (p*Cp*F*(Tin-T)+Qdot)/(CpH2O*(ToH2O-TinH2O))
    mH2O[mH2O<0] = 0
    return np.concatenate([r1, r2, r3, r4, mH2O])


# ECONOMICS
def econ1(c, F, C, C0, mH2O):
    Ct = c[0]*(C[:, 0]+C0[:, 0])*F[:, 0]+c[1]*C[:, 1]*F[:, 0]+c[2]*C[:, 2]*\
        F[:,0]+c[3]*mH2O
    return Ct


def econ2(c, F, C, C0, mH2O):
    Ct=c[0]*C[:, 0]*F[:, 0]+c[1]*C[:, 1]*F[:, 0]+c[2]*C[:, 2]*F[:, 0]+\
        c[3]*(C[:, 3]+C0[:, 3])*F[:, 0]+c[4]*C[:, 4]*F[:, 0]+c[5]*mH2O
    return Ct


def econ_recycle1(c, F, C, C0, mH2O):
    Ct=c[0]*(C[:, 0]+C0[:, 0])*F[:, 0]+c[1]*C[:,1]*F[:, 0]+c[2]*C[:, 2]*F[:,0]+\
        c[3]*(C[:, 3])*F[:, 0]+c[4]*(C[:, 4])*F[:, 0]+\
        c[5]*(C[:, 5]+C0[:, 5])*F[:, 0]+c[6]*mH2O
    return Ct


def econ_recycle2(c,F,C,C0,mH2O):
    Ct=c[0]*(C[:, 0])*F[:, 0]+c[1]*C[:, 1]*F[:, 0]+c[2]*C[:, 2]*F[:, 0]+\
        c[3]*(C[:, 3]+C0[:, 3])*F[:, 0]+c[4]*(C[:, 4])*F[:, 0]+\
        c[5]*(C[:, 5])*F[:, 0]+c[6]*mH2O
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
            self.Nin = np.array(list(self.Nin.values()), dtype=tuple)
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
def Rxtr_Recycle(FR, TR, R_Frac, Cdist = np.ones(3),  rxtr_dist = False):
    err = 1e6
    FinR3 = FR[-1]
    while err > 1e-6:
        ρR3 = (FinR3/(FinR3+FR[2]))*ρR[1]+(FR[2]/(FinR3+FR[2]))*ρR[2]
        Fr = R_Frac*(ρR[0]*FR[0]+ρR[1]*FR[1]+ρR[2]*FR[2])/((1-R_Frac)*ρR3)
        err = Fr - FinR3
        FinR3 = Fr*1
        
    ρR3 = ρR3*np.ones(TR.shape[0])
    C0R3 = C0F3*np.ones(TR[:, 0].reshape(-1, 1).shape)
    CpR3 = (FinR3/(FinR3+FR[2]))*CpR[1]+(FR[2]/(FinR3+FR[2]))*CpR[2]
    CpR3 = CpR3*np.ones(TR.shape[0])
    TinR3 = (FR[2]*CpR[2]*TinR[-1]+FinR3*CpR3*TR[:, -1])/(FinR3*CpR3+FR[2]*CpR[2])
    err = np.array([1e6])
    FinR3 = (FinR3+FR[2])*np.ones(TR.shape[0])
    j = 0
    
    while all(err > 1e-6):
        C0R1 = np.ones((TR.shape[0], Cinit3.shape[0]))
        TinR1 = np.ones(TR.shape[0])
        ρR1 = np.ones(TR.shape[0])
        CpR1 = np.ones(TR.shape[0])
        FinR1 = np.ones(TR.shape[0])
        for i in range(TR.shape[0]):
            CinR1 = np.array([C0R3[i], C0F1*Cdist[0]], dtype = tuple)
            Tmix1 = np.array([TinR3[i], TinR[0]]) # make this an array
            Fmix1 = np.array([FinR3[i], FR[0]])
            pmix1 = np.array([ρR3[i], ρR[0]])
            Cpmix1 = np.array([CpR3[i], CpR[0]])
            MixR1 = MIXER(CinR1, Fmix1, Tmix1, pmix1, Cpmix1, all_C0 = True)
            C0R1[i] = MixR1.mass_bal()
            TinR1[i] = MixR1.heat_bal()
            ρR1[i] = MixR1.pmix
            CpR1[i] = MixR1.Cpmix
            FinR1[i] = MixR1.Fout
            
        ReacR1 = CSTR(TR[:, 0], FinR1, C0R1, k0R, EaR, Cinit3, ρR1, CpR1, HR,
                      TinR1, c3, VR[0])
        ReacR1.MandE_bal(mass_bal_recycle, heat_bal_recycle)
        
        C0R2 = np.ones((TR.shape[0], Cinit3.shape[0]))
        TinR2 = np.ones(TR.shape[0])
        ρR2 = np.ones(TR.shape[0])
        CpR2 = np.ones(TR.shape[0])
        FinR2 = np.ones(TR.shape[0])
        for i in range(TR.shape[0]):
            CinR2 = np.array([ReacR1.Cf[i], C0F2*Cdist[1]], dtype = tuple)
            Tmix2 = np.array([ReacR1.T[i, 0], TinR[1]])
            Fmix2 = np.array([ReacR1.F[i,0], FR[1]])
            pmix2=np.array([ReacR1.p[i], ρR[1]])
            Cpmix2 = np.array([ReacR1.Cp[i], CpR[1]])
            MixR2 = MIXER(CinR2, Fmix2, Tmix2, pmix2, Cpmix2, all_C0 = True)
            C0R2[i] = MixR2.mass_bal()
            TinR2[i] = MixR2.heat_bal()
            ρR2[i] = MixR2.pmix
            CpR2[i] = MixR2.Cpmix
            FinR2[i] = MixR2.Fout
            
        ReacR2 = CSTR(TR[:, 1], FinR2, C0R2, k0R, EaR, Cinit3, ρR2, CpR2, HR,
                      TinR2, c3, VR[1])
        ReacR2.MandE_bal(mass_bal_recycle, heat_bal_recycle)
        
        Nsep = np.sum(ReacR2.Cf*ReacR2.F, axis = 1)
        xsep = ReacR2.Cf*ReacR2.F/Nsep.reshape(-1, 1)
        VtoF = xsep[:, -2].flatten()
        Fl1 = FLASH(Nsep, VtoF, xsep, alpha, Hvap)
        Fl1.mass_bal(K_init*np.ones(TR.shape[0]))
        Fl1.heat_bal()
        Fl2 = FLASH(Fl1.L, (1-Fl1.x_liq[:, 1]).flatten(), Fl1.x_liq, alpha, Hvap)
        Fl2.mass_bal(K_init*np.ones(TR.shape[0]))
        Fl2.heat_bal()
        Fpd1 = (Fl1.f.reshape(-1, 1)*ReacR2.F).flatten()
        Npd1 = Fl1.V.reshape(-1, 1)*Fl1.y_vap
        Fpd2 = ((1-Fl2.f).reshape(-1, 1)*(ReacR2.F-Fpd1.reshape(-1, 1))).flatten()
        Npd2 = Fl2.L.reshape(-1, 1)*Fl2.x_liq
        V = Fl2.f.reshape(-1, 1)*(ReacR2.F-Fpd1.reshape(-1, 1))
        CfFl2 = Fl2.y_vap*Fl2.V.reshape(-1, 1)/V
        TFl2 = 423*np.ones(TR.shape[0])
        
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
            pmix3 = np.array([ReacR2.p[i], ρR[-1]])
            Cpmix3 = np.array([ReacR2.Cp[i], CpR[-1]])
            MixR3 = MIXER(CinR3, Fmix3, Tmix3, pmix3, Cpmix3, all_C0 = True)
            C0R3[i] = MixR3.mass_bal()
            TinR3[i] = MixR3.heat_bal()
            ρR3[i] = MixR3.pmix
            CpR3[i] = MixR3.Cpmix
            FinR3[i] = MixR3.Fout
            
        err = ((Fpd1*ReacR2.p+Fpd2*ReacR2.p+Fprg*ReacR2.p-\
                (FR[0]*ρR[0]+FR[1]*ρR[1]+FR[2]*ρR[2]))**2)
        j += 1
    
    if rxtr_dist:
        rxtr1_prod = np.hstack([ReacR1.Cf[:, :3]*ReacR1.F, ReacR1.Cf[:, -1]*ReacR1.F.reshape(-1, 1)])
        rxtr1_util = ReacR1.mH2O.reshape(-1, 1)
        
        rxtr2_prod = ReacR2.Cf*ReacR2.F
        rxtr2_util = ReacR2.mH2O.reshape(-1, 1)
        
        flsh1_prod = 1e1*Fl1.V.reshape(-1, 1)
        flsh1_equl = 1e4*Fl1.K_ref.reshape(-1, 1)
        flsh1_util = 1e2*Fl1.m_steam.reshape(-1, 1)
        
        flsh2_prod = Fl2.L.reshape(-1, 1)
        flsh2_equl = 1e3*Fl2.K_ref.reshape(-1, 1)
        flsh2_util = 1e1*Fl2.m_steam.reshape(-1, 1)

        return rxtr1_prod, rxtr1_util,\
               rxtr2_prod, rxtr2_util,\
               flsh1_prod, flsh1_equl, flsh1_util,\
               flsh2_prod, flsh2_equl, flsh2_util
    
    else:
        CtR1 = ReacR1.Econ(econ_recycle1)
        
        CtR2 = ReacR2.Econ(econ_recycle2)
        
        CtR = Npd1@c3[:-1]+Npd2@c3[:-1]+c3[-1]*(ReacR1.mH2O+ReacR2.mH2O)+\
            100*c3[-1]*(Fl1.m_steam+Fl2.m_steam)+FR[0]*C0F1@c3[:-1]
        CtR = CtR/1e3
        
        return 8760*CtR1, 8760*CtR2, 8760*CtR


def SYST_RECYCLE(T, FR, R_Frac, Cdist, rxtr_dist = False):
    T = T.flatten()
    T = T.reshape(int(T.shape[0]/2), 2)
    
    if rxtr_dist:
        y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = Rxtr_Recycle(FR, T, R_Frac, Cdist, rxtr_dist = rxtr_dist)
        return y1, y2, y3, y4, y5, y6, y7, y8, y9, y10
    
    else:
        CtR1, CtR2, CtR = Rxtr_Recycle(FR, T, R_Frac, Cdist)
        return CtR


#%% BO functions
def cost_fun(Y_in, alpha, idx, p, feed_cost):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        
        if type(Y_in) == list:
            Y = []
            for i, y in enumerate(Y_in):
                if len(y.shape) == 1:
                    y = y.reshape(-1, 1).T
                Y.append(y[:, idx[i]])
            Y = np.hstack(Y)
        elif type(Y_in) == np.ndarray:
            if len(Y_in.shape) == 1:
                Y_in = Y_in.reshape(-1, 1).T
            Y = Y_in
        else:
            raise Exception("Incorrect data type for input Y_in")
        
        m_watRXT1 = Y[:, 0].reshape(-1, 1)
        m_watRXT2 = Y[:, 7].reshape(-1, 1)
        m_stmFl1 = 1e-2*Y[:, 10].reshape(-1, 1)
        m_stmFl2 = 1e-1*Y[:, 13].reshape(-1, 1)
        
        n_rxt2 = np.sum(Y[:, 1:7], axis = 1).reshape(-1, 1)
        x_rxt2 = Y[:, 1:7]/n_rxt2
        
        n_prd1 = 1e-1*Y[:, 8].reshape(-1, 1)
        K_sep1 = 1e-4*Y[:, 9].reshape(-1, 1)
        
        n_prd2 = Y[:, 11].reshape(-1, 1)
        K_sep2 = 1e-3*Y[:, 12].reshape(-1, 1)
        
        f_Fl1 = n_prd1/n_rxt2
        x_Fl1 = x_rxt2/(f_Fl1*(K_sep1*alpha-1)+1) 
        y_prd1 = K_sep1*alpha*x_Fl1
        
        n_Fl2 = n_rxt2-n_prd1
        f_Fl2 = 1-(n_prd2/n_Fl2)
        x_prd2 = x_Fl1/(f_Fl2*(K_sep2*alpha-1)+1)
        
        Y_out = np.hstack([m_watRXT1, m_watRXT2, n_prd1*y_prd1, m_stmFl1, n_prd2*x_prd2, m_stmFl2])
        
        fun = 8.76*(np.array([y.T@p for y in Y_out]).flatten()+feed_cost)
        
    return fun


def gp_sim(x, y_mod, mu, sigma, x_idx, y_idx):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1).T

    mu_r = {}
    sig_r = {}
    x_r = {}
    
    for i, key in enumerate(y_mod):
        index = x_idx[i]
        index1 = y_idx[0][i]
        index2 = y_idx[1][i]
        if index is not None and index1 is not None:
            yt = [((mu_r[f'{k+1}']-mu[k])/sigma[k])[:, l] for k, l in zip(index1, index2)]
            x_r[f'{i+1}'] = np.hstack([x[:, index], np.hstack(yt)])
        
        elif index1 is None:
            x_r[f'{i+1}'] = x[:, index]
            
        elif index is None:
            yt = [((mu_r[f'{k+1}']-mu[k])/sigma[k])[:, l] for k, l in zip(index1, index2)]
            x_r[f'{i+1}'] = np.hstack(yt)
        
        if len(x_r[f'{i+1}'].shape) == 1:
            x_r[f'{i+1}'] = x_r[f'{i+1}'].reshape(-1, 1).T
        
        mu_r[f'{i+1}'], sig_r[f'{i+1}'] = y_mod[key].predict(x_r[f'{i+1}'],
                                                           return_std = True)
        
        if len(mu_r[f'{i+1}'].shape) == 1:
            mu_r[f'{i+1}'] = mu_r[f'{i+1}'].reshape(-1, 1)
            sig_r[f'{i+1}'] = sig_r[f'{i+1}'].reshape(-1, 1)
        
    return list(mu_r.values()), list(sig_r.values())


#%% BO setup
ub = np.array([423, 423])
lb = np.array([303, 303])
dim = len(ub)
exp_w = [2.6]
kernel = gpr.kernels.Matern(np.ones(2), np.array([[1e-1, 1e3]]*dim), nu = 2.5)
bounds = Bounds(np.zeros(dim), np.ones(dim))
feed_cost = FR[0]*C0F1[0]*cA
Cdist = np.ones(3)
args = (FR, R_Frac, Cdist)
args_dist = (FR, R_Frac, Cdist, True)
p = np.array([cT, cT, cA, cB, cC, cD, cE, cF, 100*cT, cA, cB, cC, cD, cE,
              cF, 100*cT])


idx = [None,
       
       [np.array([], dtype = int),                                          # RXTR 1 PROD
        np.array([0]),                                                      # RXTR 1 UTIL
        np.array([0, 1, 2, 3, 4, 5]),                                       # RXTR 2 PROD
        np.array([0]),                                                      # RXTR 2 UTIL
        np.array([0]),                                                      # FLSH 1 PROD
        np.array([0]),                                                      # FLSH 1 EQUL
        np.array([0]),                                                      # FLSH 1 UTIL
        np.array([0]),                                                      # FLSH 2 PROD
        np.array([0]),                                                      # FLSH 2 EQUL
        np.array([0]),                                                      # FLSH 2 UTIL
        ]]

idx_opbo = [None,

            [4,                                                             # RXTR 1 UTIL
             5, 6, 7, 8, 9, 10,                                             # RXTR 2 PROD
             11,                                                            # RXTR 2 UTIL
             12,                                                            # FLSH 1 PROD
             13,                                                            # FLSH 1 EQUL
             14,                                                            # FLSH 1 UTIL
             15,                                                            # FLSH 2 PROD
             16,                                                            # FLSH 2 EQUL
             17,                                                            # FLSH 2 UTIL
             ]]

x_idx = [np.array([0]),                                                     # RXTR 1 PROD
         np.array([0]),                                                     # RXTR 1 UTIL
         np.array([1]),                                                     # RXTR 2 PROD
         np.array([0, 1]),                                                  # RXTR 2 UTIL
         np.array([0, 1]),                                                  # FLSH 1 PROD
         np.array([0, 1]),                                                  # FLSH 1 EQUL
         np.array([1]),                                                     # FLSH 1 UTIL
         np.array([0, 1]),                                                  # FLSH 2 PROD
         np.array([0, 1]),                                                  # FLSH 2 EQUL
         np.array([1]),                                                     # FLSH 2 UTIL
         ]

y_idx = [[None,                                                             # RXTR 1 PROD -- 0
          [0],                                                              # RXTR 1 UTIL -- 1
          [0],                                                              # RXTR 2 PROD -- 2
          [2],                                                              # RXTR 2 UTIL -- 3
          None,                                                             # FLSH 1 PROD -- 4
          None,                                                             # FLSH 1 EQUL -- 5
          [4],                                                              # FLSH 1 UTIL -- 6
          None,                                                             # FLSH 2 PROD -- 7
          None,                                                             # FLSH 2 EQUL -- 8
          [6, 7],                                                           # FLSH 2 UTIL -- 9
          ],

          [None,                                                            # RXTR 1 PROD
           [np.array([0, 1, 2, 3])],                                        # RXTR 1 UTIL
           [np.array([0, 1, 2, 3])],                                        # RXTR 2 PROD
           [np.array([0, 1, 2, 3, 4])],                                     # RXTR 2 UTIL
           [np.array([5])],                                                 # FLSH 1 PROD
           [np.array([0, 1, 2, 3, 4])],                                     # FLSH 1 EQUL
           [np.array([0])],                                                 # FLSH 1 UTIL
           [np.array([2])],                                                 # FLSH 2 PROD
           [np.array([0])],                                                 # FLSH 2 EQUL
           [np.array([0]), np.array([0])],                                  # FLSH 2 UTIL
           ]]


trials = 3
init_pts = 2
eps = 1e-3
n_samples = 100
f_args = (alpha, idx[1], p, feed_cost)
gp_args = (x_idx, y_idx)
restarts = 100
af_cores = 4
kernel_length_scale_bnds = np.array([[5e-2, 1e1], [1e-1, 1e3],
                                     [1e-1, 1e3], [1e-1, 1e3],
                                     [5e-2, 1e3], [5e-2, 1e3],
                                     [5e-2, 1e3], [5e-2, 1e3],
                                     [5e-2, 1e3], [5e-2, 1e3]])
nu1pt5 = np.array([1.5]*10)
nu2pt5 = np.array([2.5]*10)
feasible_lb = 1e-6*np.ones(14)
feasible_ub = np.inf*np.ones(14)
feasible_lb_opbo = 1e-6*np.ones(18)
feasible_ub_opbo = np.inf*np.ones(18)


x_init = np.linspace(bounds.lb, bounds.ub, 5)
x_init = np.meshgrid(*x_init.T)
x_init = np.reshape(x_init, (dim, -1)).T

x_BO = np.ones((trials*len(x_init), dim))
F_BO = np.ones((trials, len(x_init)))

x_bois = np.ones((trials*len(x_init), dim))
F_bois = np.ones((trials, len(x_init)))

x_mcbo = np.ones((trials*len(x_init), dim))
F_mcbo = np.ones((trials, len(x_init)))

x_opbo = np.ones((trials*len(x_init), dim))
F_opbo = np.ones((trials, len(x_init)))


RXT_DIST = BO_algos.BO(ub, lb, dim, exp_w[0], kernel, SYST_RECYCLE, bounds)

for i, x_0 in enumerate(x_init):
    RXT_DIST.exp_w = exp_w[0]
    RXT_DIST.args = args
    RXT_DIST.optimizer_sbo(trials = trials, x_init = x_0)
    
    x_BO[i*trials:(i+1)*trials] = RXT_DIST.x_sbo 
    F_BO[:, i] = RXT_DIST.y_sbo.flatten()
    np.savetxt('x_sbo_2D_rxtr.txt', x_BO)
    np.savetxt('f_sbo_2D_rxtr.txt', F_BO)

    RXT_DIST.exp_w = exp_w
    RXT_DIST.args = args_dist
    RXT_DIST.optimizer_bois(trials = trials, init_pts = init_pts, eps = eps,
                            idx = idx, x_idx = x_idx, y_idx = y_idx,
                            gp_sim = gp_sim, cost_fun = cost_fun,
                            restarts = restarts, af_cores = af_cores,
                            f_args = f_args, gp_args = gp_args,
                            x_init = RXT_DIST.scale(RXT_DIST.x_sbo[:2]),
                            kernel_length_scale_bnds = kernel_length_scale_bnds, nu = nu1pt5,
                            feasibility_check = True, clip_to_bounds = True,
                            feasible_lb = feasible_lb, feasible_ub = feasible_ub)
    
    x_bois[i*trials:(i+1)*trials] = RXT_DIST.x_bois 
    F_bois[:, i] = RXT_DIST.f_bois.flatten()
    np.savetxt('x_bois_2D_rxtr_equil_factors.txt', x_bois)
    np.savetxt('f_bois_2D_rxtr_equil_factors.txt', F_bois)
    
    RXT_DIST.optimizer_mcbo(trials = trials, init_pts = init_pts, n_samples = n_samples,
                            idx = idx, x_idx = x_idx, y_idx = y_idx,
                            gp_sim = gp_sim, cost_fun = cost_fun,
                            restarts = restarts, af_cores = af_cores,
                            f_args = f_args, gp_args = gp_args,
                            x_init = RXT_DIST.scale(RXT_DIST.x_sbo[:2]),
                            kernel_length_scale_bnds = kernel_length_scale_bnds, nu = nu1pt5,
                            feasibility_check = True, clip_to_bounds = True,
                            feasible_lb = feasible_lb, feasible_ub = feasible_ub)
    
    x_mcbo[i*trials:(i+1)*trials] = RXT_DIST.x_mcbo 
    F_mcbo[:, i] = RXT_DIST.f_mcbo.flatten()
    np.savetxt('x_mcbo_2D_rxtr_equil_factors.txt', x_bois)
    np.savetxt('f_mcbo_2D_rxtr_equil_factors.txt', F_bois)
    
    RXT_DIST.optimizer_optimism_bo(trials = trials, init_pts = init_pts,
                               idx = idx_opbo, x_idx = x_idx, y_idx = y_idx,
                               gp_sim = gp_sim, cost_fun = cost_fun,
                               feasible_lb = feasible_lb_opbo, feasible_ub = feasible_ub_opbo,
                               restarts = restarts, af_cores = af_cores,
                               f_args = f_args, gp_args = gp_args,
                               x_init = RXT_DIST.scale(RXT_DIST.x_sbo[:2]),
                               kernel_length_scale_bnds = kernel_length_scale_bnds, nu = nu1pt5,
                               norm_xdat = False, split_gps = False)
    
    x_opbo[i*trials:(i+1)*trials] = RXT_DIST.x_opbo
    F_opbo[:, i] = RXT_DIST.f_opbo.flatten()
    np.savetxt('x_opbo_2D_rxtr_equil_factors.txt', x_opbo)
    np.savetxt('f_opbo_2D_rxtr_equil_factors.txt', F_opbo)

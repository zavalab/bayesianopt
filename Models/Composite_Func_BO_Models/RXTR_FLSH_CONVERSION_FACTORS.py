import numpy as np
from scipy.optimize import Bounds, fsolve
import sklearn.gaussian_process as gpr
import biosteam as bst
from thermo import SRK, PR

import sys
sys.path.append('./../../BO_algos')
import Composite_Func_Algos as BO_algos

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', bst.exceptions.CostWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

n_feed = np.array([1000, 3000, 0])*1e3
M = np.array([28, 2, 17])
p = np.array([0.0060*1.0, 0.0014*1.0, 0.0085*1.0, 0.4*2.5])
p_elec = 0.0782*1.82
p_wat = 0.00041911551691021853

# BioSTEAM
Gamma = bst.IdealActivityCoefficients
Phi = bst.IdealFugacityCoefficients
chemicals = [bst.Chemical('N2', eos = PR),
             bst.Chemical('H2', eos = PR),
             bst.Chemical('NH3', eos = SRK)]
props = bst.Thermo(chemicals, Gamma = Gamma, Phi = Phi)
props.mixture.include_excess_energies = True
bst.settings.set_thermo(props)

bst.settings.cooling_agents = [bst.settings.cooling_agents[1]]


#%% UNIT FUNCTIONS
def COMPRESSOR_BIOSTEAM(P, P_in, T_in, n, eta):
    bst.settings.set_thermo(props)
    n = n/1e3
    comp_feed = bst.Stream(None, N2 = n[0], H2 = n[1], NH3 = n[2],
                           T = T_in, P = P_in, phase = 'g')
    COMP = bst.units.IsentropicCompressor(None, ins = comp_feed,
                                          driver = 'Electric motor',
                                          outs = None, P = P,
                                          eta = eta)
    COMP.simulate()
    T = COMP.outlet.T
    E = COMP.power_utility.consumption
    Cst = COMP.utility_cost*1.82
    return np.array([T, E, Cst])


def HEATER(n, Tin, Tout, p_ng):
    R = 8.314
    H_ng = 51.98 # MJ/kg natural gas
    
    # Heat Capacity Constants
    A = np.array([3.280, 3.249, 3.578])
    B = np.array([0.593, 0.422, 3.020])*1e-3
    C = np.array([0.000, 0.000, 0.000])*1e-6
    D = np.array([0.040, 0.083, -0.186])*1e5
    
    ICPH = A*(Tout-Tin)+B/2*(Tout**2-Tin**2)+C/3*(Tout**3-Tin**3)-\
           D*(Tout**-1-Tin**-1)
    Q = R*n*ICPH
    Q = np.sum(Q)
    m_ng = Q/1e6/H_ng
    Cst = m_ng*p_ng
    
    y = np.hstack([m_ng.reshape(-1, 1), Cst.reshape(-1, 1)])
    return y


def COOLER(n, Tin, Tout):
    R = 8.314
    H_wat = 4184*10 # J/kg water with ΔT fixed to 10
    p_wat = 2.12e-7 # USD/kJ
    
    # Heat Capacity Constants
    A = np.array([3.280, 3.249, 3.578])
    B = np.array([0.593, 0.422, 3.020])*1e-3
    C = np.array([0.000, 0.000, 0.000])*1e-6
    D = np.array([0.040, 0.083, -0.186])*1e5
    
    ICPH = A*(Tout-Tin)+B/2*(Tout**2-Tin**2)+C/3*(Tout**3-Tin**3)-\
           D*(Tout**-1-Tin**-1)
    Q = R*n*ICPH
    Q = np.sum(Q)
    m_wat = -Q/H_wat
    Cst = -Q/1000*p_wat
    
    y = np.hstack([m_wat.reshape(-1, 1), Cst.reshape(-1, 1)])
    return y


def CONDENSER(n, deltaH_lat):
    H_wat = 4184*10 # J/kg water with ΔT fixed to 10
    p_wat = 2.12e-7 # USD/kJ
    
    Q = n*deltaH_lat
    Q = np.sum(Q)
    m_wat = -Q/H_wat
    Cst = -Q/1000*p_wat
    
    y = np.hstack([m_wat.reshape(-1, 1), Cst.reshape(-1, 1)])
    return y


def FLASH_BIOSTEAM(n, T, P, T_in, P_in):
    bst.settings.set_thermo(props)
    n = n/1e3
    
    inlet = bst.Stream(None, N2 = n[0], H2 = n[1], NH3 = n[2],
                       T = T_in, P = P_in, phase = 'g')
    flash = bst.units.Flash(None, ins = inlet, outs = None, T = T, P = P)
    flash.simulate()
    
    n_gas = flash.outs[0].z_mol*flash.outs[0].F_mol*1e3
    n_liq = flash.outs[1].z_mol*flash.outs[1].F_mol*1e3
    Ut = flash.heat_utilities[0]
    m_wat = Ut.flow*18
    Cst = Ut.cost
    
    y = np.hstack([m_wat.reshape(-1, 1), Cst.reshape(-1, 1)])
    return y, n_gas, n_liq


def RXTR(x, n_0):
    T = x[:, 0]
    P = x[:, 1]
    
    v = np.array([-1/2, -3/2, 1])
    
    R = 8.314
    T_0 = 298
    P_0 = 1
    deltaH_rxn0 = 46110*0.85#1.05
    deltaG_rxn0 = 16450*2.0#1.70
    A = np.array([3.280, 3.249, 3.578])
    B = np.array([0.593, 0.422, 3.020])*1e-3
    C = np.array([0.000, 0.000, 0.000])*1e-6
    D = np.array([0.040, 0.083, -0.186])*1e5
    
    deltaA = v.T@A
    deltaB = v.T@B
    deltaC = v.T@C
    deltaD = v.T@D
    
    ICPH = deltaA*(T-T_0)+deltaB/2*(T**2-T_0**2)+deltaC/3*(T**3-T_0**3)-\
           deltaD*(T**-1-T_0**-1)
    ICPS = deltaA*(np.log(T)-np.log(T_0))+deltaB*(T-T_0)+deltaC/2*(T**2-T_0**2)\
           -deltaD/2*(T**-2-T_0**-2)
    deltaH_rxn = deltaH_rxn0+R*ICPH
    deltaG_rxn = deltaH_rxn0-T/T_0*(deltaH_rxn0-deltaG_rxn0)+R*ICPH-R*T*ICPS
    K = np.exp(-deltaG_rxn/(R*T))
    
    def equilibrium(eps, n_0 = n_0, v = v, K = K, P = P, P_0 = P_0):
        y_A = (n_0[0]+v[0]*eps)/(np.sum(n_0)+np.sum(v)*eps)
        y_B = (n_0[1]+v[1]*eps)/(np.sum(n_0)+np.sum(v)*eps)
        y_C = (n_0[2]+v[2]*eps)/(np.sum(n_0)+np.sum(v)*eps)
        
        fug = y_A**v[0]*y_B**v[1]*y_C**v[2]
        err = fug-(P/P_0)**(-np.sum(v))*K
        return err
    
    eps0 = 0.99
    eps = fsolve(equilibrium, eps0)
    
    y = np.hstack([eps.reshape(-1, 1), deltaH_rxn.reshape(-1, 1)])
    return y


## System build
def SYSTEM(x, p, n_feed, M):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1).T
        
    T = x[:, 0]
    P = x[:, 1]
    r = x[:, 2]
    T_flash = x[:, 3]
    P_flash = x[:, 4]
    
    # Economic parameters
    p_C = p[2]
    p_ng = p[3]
    
    # Thermodynamic parameters
    H_ng = 51.98                     # MJ/kg natural gas
    v = np.array([-1/2, -3/2, 1])    # reaction coefficients
    n_0 = np.array([1/2, 3/2, 0])    # mol feed basis   
    
    # Front End
    n_N2 = n_feed*np.array([1, 0, 0])
    CP_A = COMPRESSOR_BIOSTEAM(P*1e5, 1.5e7, 323, n_N2, eta = 0.75)
    p_CPA = CP_A[2]#*1.82                  # Cost of compressing feed of A
    HX_A = HEATER(n_N2, CP_A[0], T, p_ng)
    p_HXA = HX_A[:, 1]               # Cost of heating feed of A
    
    n_H2 = n_feed*np.array([0, 1, 0])
    CP_B = COMPRESSOR_BIOSTEAM(P*1e5, 1.5e7, 323, n_H2, eta = 0.75)
    p_CPB = CP_B[2]#*1.82                  # Cost of compressing feed of B
    HX_B = HEATER(n_H2, CP_B[0], T, p_ng)
    p_HXB = HX_B[:, 1]               # Cost of heating feed of B
    
    # Recycle Loop
    n_0tot = n_feed*1
    n_prd = np.zeros(3)
    n_prg = np.zeros(3)
    
    m_in = n_feed.T@M
    m_out = (n_prd+n_prg).T@M
    i = 0
    
    while abs(m_in-m_out) > 1e-6:
        
        # Reactor
        RX_1 = RXTR(x, n_0)
        eps = RX_1[0, 0]
        e_tot = (n_0tot[0]/n_0[0])*eps # scale up to correct production value
    
        y_A = (n_0[0]+v[0]*eps)/(np.sum(n_0)+np.sum(v)*eps)
        n_A = y_A*(np.sum(n_0tot)+np.sum(v)*e_tot)    
        y_B = (n_0[1]+v[1]*eps)/(np.sum(n_0)+np.sum(v)*eps)
        n_B = y_B*(np.sum(n_0tot)+np.sum(v)*e_tot)    
        y_C = (n_0[2]+v[2]*eps)/(np.sum(n_0)+np.sum(v)*eps)
        n_C = y_C*(np.sum(n_0tot)+np.sum(v)*e_tot)
        n_tot = np.array([n_A, n_B, n_C])
    
        # Flash
        FL_1 = FLASH_BIOSTEAM(n_tot, T_flash[0], P_flash[0]*1e5, T[0], P[0]*1e5)
        n_prd = FL_1[2].flatten()
        n_fl = FL_1[1].flatten()
        n_prg = (1-r)*n_fl
        n_rcyl = r*n_fl
        
        # Recycle compressor and heater
        CP_R = COMPRESSOR_BIOSTEAM(P*1e5, P_flash*1e5, T_flash, n_rcyl,
                                   eta = 0.75)
        HX_R = HEATER(n_rcyl, CP_R[0], T, p_ng)
        n_0tot = n_rcyl+n_feed
        n_0 = n_0tot/2e6
        
        m_in = n_feed.T@M
        m_out = (n_prd+n_prg).T@M
        i += 1
        
    p_RX1 = (RX_1[:, 1]*n_C/1e6/H_ng)*p_ng
    p_FL1 = FL_1[0][:, 1]
    p_CPR = CP_R[2]#*1.82
    p_HXR = HX_R[:, 1]
    
    
    C_system = p_CPA+p_HXA+p_CPB+p_HXB+p_RX1+p_FL1+p_CPR+p_HXR+p[:3].T@n_feed
    C_prod = -p_C*n_prd[2]+p_C*np.sqrt((max(0, n_prd[2]-1.9e6))**2)+(100*(1.9e6-n_prd[2])/1.9e6)**2*p_C*5e2
    
    Cst = C_system+C_prod
        
    return np.hstack([Cst.reshape(-1, 1), n_prd.reshape(1, 3)])


def SYSTEM_BO(x, p, n_feed, M):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1).T
    y = np.ones((len(x), 4))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', bst.exceptions.CostWarning)
        for i in range(len(x)):    
            y[i] = SYSTEM(x[i], p, n_feed, M)
    return y[:, 0].flatten()


def SYSTEM_DIST(x, n_feed, M):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1).T
        
    T = x[:, 0]
    P = x[:, 1]
    r = x[:, 2]
    T_flash = x[:, 3]
    P_flash = x[:, 4]
    
    # Thermodynamic parameters
    H_ng = 51.98    # MJ/kg natural gas
    v = np.array([-1/2, -3/2, 1])    # reaction coefficients
    n_0 = np.array([1/2, 3/2, 0])    # mol feed basis   
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', bst.exceptions.CostWarning)
        
        # Front End
        n_N2 = n_feed*np.array([1, 0, 0])
        CP_A = COMPRESSOR_BIOSTEAM(P*1e5, 1.5e7, 323, n_N2, eta = 0.75)
        T_oA = CP_A[0]
        
        HX_A = HEATER(n_N2, T_oA, T, p_ng = 0)
        
        
        n_H2 = n_feed*np.array([0, 1, 0])
        CP_B = COMPRESSOR_BIOSTEAM(P*1e5, 1.5e7, 323, n_H2, eta = 0.75)
        T_oB = CP_B[0]
        
        HX_B = HEATER(n_H2, T_oB, T, p_ng = 0)
        
        # Recycle Loop
        n_0tot = n_feed*1
        n_prd = np.zeros(3)
        n_prg = np.zeros(3)
        
        m_in = n_feed.T@M
        m_out = (n_prd+n_prg).T@M
        i = 0
        
        while abs(m_in-m_out) > 1e-6:
            
            # Reactor
            RX_1 = RXTR(x, n_0)
            eps = RX_1[0, 0]
            e_tot = (n_0tot[0]/n_0[0])*eps # scale up to correct production value
        
            y_A = (n_0[0]+v[0]*eps)/(np.sum(n_0)+np.sum(v)*eps)
            n_A = y_A*(np.sum(n_0tot)+np.sum(v)*e_tot)    
            y_B = (n_0[1]+v[1]*eps)/(np.sum(n_0)+np.sum(v)*eps)
            n_B = y_B*(np.sum(n_0tot)+np.sum(v)*e_tot)    
            y_C = (n_0[2]+v[2]*eps)/(np.sum(n_0)+np.sum(v)*eps)
            n_C = y_C*(np.sum(n_0tot)+np.sum(v)*e_tot)
            n_tot = np.array([n_A, n_B, n_C])
        
            # Flash
            FL_1 = FLASH_BIOSTEAM(n_tot, T_flash, P_flash*1e5, T, P*1e5)
            n_prd = FL_1[2].flatten()
            n_fl = FL_1[1].flatten()
            n_prg = (1-r)*n_fl
            n_rcyl = r*n_fl
            
            # Recycle compressor and heater
            CP_R = COMPRESSOR_BIOSTEAM(P*1e5, P_flash*1e5, T_flash, n_rcyl,
                                       eta = 0.75)
            T_oR = CP_R[0]
            
            HX_R = HEATER(n_rcyl, T_oR, T, p_ng = 0)
            n_0tot = n_rcyl + n_feed
            n_0 = n_0tot/2e6
            
            m_in = n_feed.T@M
            m_out = (n_prd+n_prg).T@M
            i += 1
            
        m_ngRX1 = (RX_1[:, 1]*n_C/1e6/H_ng)
        m_watFL1 = FL_1[0][:, 0]
        eta = 1000*np.array([n_prg[1]/n_feed[1], n_prd[0]/n_prg[0], n_prg[-1]/n_prd[-1]])
    
    return np.array([eta[0]]).reshape(-1, 1), np.array([eta[1]]).reshape(-1, 1),\
           np.array([eta[2]]).reshape(-1, 1), np.array([m_ngRX1]), np.array([m_watFL1/1e3])


#%% BOIS FUNCTIONS

def cost_fun(Y_in, X, n_feed, p):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', bst.exceptions.CostWarning)
    
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        if type(Y_in) == list:
            Y = []
            for i, y in enumerate(Y_in):
                if len(y.shape) == 1:
                    y = y.reshape(-1, 1)
                Y.append(y[:])
            Y = np.hstack(Y)
        elif type(Y_in) == np.ndarray:
            if len(Y_in.shape) == 1:
                Y_in = Y_in.reshape(1, -1)
            Y = Y_in
        else:
            raise Exception("Incorrect data type for input Y_in")
        
        
        # Economic parameters
        p_C = p['C']
        p_ng = p['nat_gas']
        p_feed = np.array([p['A'], p['B'], p['C']])
        p_wat = p['water']
        Cst = np.zeros(len(Y))
        
        
        # Thermodynamic parameters
        H_ng = 51.98    # MJ/kg natural gas
        v = np.array([-1/2, -3/2, 1])    # reaction coefficients
        n_0 = np.array([1/2, 3/2, 0])    # mol feed basis   
        
        
        for i, (x, y) in enumerate(zip(X, Y)):
            x = x.reshape(1, -1)        
            T = x[0, 0]
            P = x[0, 1]
            r = x[0, 2]
            T_flash = x[0, 3]
            P_flash = x[0, 4]
            
            y = y.reshape(1, -1)
            y[y<0] = 0
            # GP-modeled parameters
            eta_B = y[0, 0]/1000
            eta_A = y[0, 1]/1000
            eta_C = y[0, 2]/1000
            m_ngRX = max(0, y[0, 3])
            m_watFL = max(0, y[0, 4]*1e3)
            
            # Reactor and Flash Utilities (data-driven model)
            p_RX = m_ngRX*p_ng
            p_FL = p_wat*m_watFL
            
            # component B balance
            n_prgB = eta_B*n_feed[1]
            n_rxtB = -(n_feed[1]-n_prgB)
            n_prdB = 0
            
            # component C balance
            n_rxtC = v[2]/v[1]*n_rxtB
            n_prdC = n_rxtC/(1+eta_C)
            n_prgC = eta_C*n_prdC
            
            # component A balance
            n_rxtA = v[0]/v[1]*n_rxtB
            n_prgA = (n_feed[0]+n_rxtA)/(1+eta_A)
            n_prdA = eta_A*n_prgA
            
            # Stream component arrays
            n_rxt = np.array([n_rxtA, n_rxtB, n_rxtC])
            n_prd = np.array([n_prdA, n_prdB, n_prdC])
            n_prg = np.array([n_prgA, n_prgB, n_prgC])
            n_rcyl = n_prg*r/(1-r)
            
            
            # Front End
            ## reagent A compression and heating
            n_N2 = n_feed*np.array([1, 0, 0])
            CP_A = COMPRESSOR_BIOSTEAM(P*1e5, 1.5e7, 323, n_N2, eta = 0.75)
            T_oA = CP_A[0]
            p_CPA = CP_A[2] # Utlity cost of CP_A
            
            HX_A = HEATER(n_N2, T_oA, T, p_ng)
            p_HXA = HX_A[:, 1] # utility cost of HX_A
            
            ## reagent B compression and heating
            n_H2 = n_feed*np.array([0, 1, 0])
            CP_B = COMPRESSOR_BIOSTEAM(P*1e5, 1.5e7, 323, n_H2, eta = 0.75)
            T_oB = CP_B[0]
            p_CPB = CP_B[2] # Utlity cost of CP_B
            
            HX_B = HEATER(n_H2, T_oB, T, p_ng)
            p_HXB = HX_B[:, 1] # Utlity cost of HX_B
            
            # Recycle Compressor and Heater
            CP_R = COMPRESSOR_BIOSTEAM(P*1e5, P_flash*1e5, T_flash, n_rcyl, eta = 0.75)
            p_CPR = CP_R[2] # Utlity cost of CP_R
            
            HX_R = HEATER(n_rcyl, CP_R[0], T, p_ng)
            p_HXR = HX_R[:, 1] # Utlity cost of HX_R
            
            C_system = p_CPA+p_HXA+p_CPB+p_HXB+p_RX+p_FL+p_CPR+p_HXR+p_feed[:3].T@n_feed
            C_prod = -p_C*n_prd[2]+p_C*np.sqrt((max(0, n_prd[2]-1.9e6))**2)+(100*(1.9e6-n_prd[2])/1.9e6)**2*p_C*5e2
            
            # System Cost
            Cst[i] = C_system+C_prod
    
    return Cst


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
        if x_idx[i] is not None and y_idx[0][i] is not None:
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


#%% RUN SETUP

n_feed = np.array([1e6, 3e6, 0])
p_dict = {}
p_dict['A'] = p[0]
p_dict['B'] = p[1]
p_dict['C'] = p[2]
p_dict['nat_gas'] = p[3]
p_dict['elec'] = p_elec
p_dict['water'] = p_wat

ub = np.array([973, 450, 0.9, 338, 170])
lb = np.array([673, 250, 0.5, 288, 140])
dim = len(ub)
exp_w = [2.6, 1.5, 1.0, 0.5, 0]
shift_exp_w = [50, 65, 75, 85]
kernel = gpr.kernels.Matern(length_scale = np.ones(dim),
                            length_scale_bounds = np.array([[1e-1, 2e1]]*dim),
                            nu = 2.5)
bounds = Bounds(np.zeros(dim), np.ones(dim))
args = (p, n_feed, M)
args_dist = (n_feed, M)

idx = [[0, 1, 2, 3, 4],

       [np.array([0]),     # eta_B
        np.array([0]),     # eta_A
        np.array([0]),     # eta_C
        np.array([0]),     # m_ngRX
        np.array([0])      # m_watFL
        ]]

idx_opbo = [[0, 1, 2, 3, 4],
            
            [0,            # eta_B
             1,            # eta_A
             2,            # eta_C
             3,            # m_ngRX
             4]]           # m_watFL

x_idx = [np.array([0, 1, 2, 3]),            # eta_B
         np.array([3, 4]),                  # eta_A  
         np.array([3, 4]),                  # eta_C
         np.array([0, 1]),                  # m_ngRX
         np.array([0, 2, 3, 4]),            # m_watFL
         ]

y_idx = [[None,                             # eta_B
          [0],                              # eta_A
          [0],                              # eta_C
          [0, 1],                           # m_ngRX
          [2],                              # m_watFL
          ],

         [None,                             # eta_B
          [np.array([0])],                  # eta_A
          [np.array([0])],                  # eta_C
          [np.array([0]), np.array([0])],   # m_ngRX
          [np.array([0])],                  # m_watFL
          ]]

trials = 100
init_pts = 2
eps = 1e-3
n_samples = 20
nu = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
kernel_length_scale_bnds = np.array([[1e-1, 1e2]]*len(nu))
f_args = (n_feed, p_dict)
gp_args = (x_idx, y_idx)
f_prime_regularizer = np.array([0.25, 0.25, 0.25, 1, 1])
feasible_lb = 1e-6*np.ones(len(nu))
feasible_ub = np.inf*np.ones(len(nu))

x_init = np.random.uniform(bounds.lb, bounds.ub, (25, dim))

X_SBO = np.ones((trials*len(x_init), dim))
F_SBO = np.ones((trials, len(x_init)))

X_BOIS = np.ones((trials*len(x_init), dim))
F_BOIS = np.ones((trials, len(x_init)))

X_MCBO = np.ones((trials*len(x_init), dim))
F_MCBO = np.ones((trials, len(x_init)))

X_OPBO = np.ones((trials*len(x_init), dim))
F_OPBO = np.ones((trials, len(x_init)))

RXT_DIST = BO_algos.BO(ub, lb, dim, exp_w[0], kernel, SYSTEM_BO, bounds, args)

for i, x_0 in enumerate(x_init):
    RXT_DIST.system = SYSTEM_BO
    RXT_DIST.exp_w = exp_w[0]
    RXT_DIST.args = args
    
    RXT_DIST.optimizer_sbo(trials = trials, x_init = x_0, init_pts = 1)
    
    X_SBO[i*trials:(i+1)*trials] = RXT_DIST.x_sbo
    F_SBO[:, i] = RXT_DIST.y_sbo.flatten()
    np.savetxt('RXT_FLSH_RCYL_x_sbo.txt', X_SBO)
    np.savetxt('RXT_FLSH_RCYL_f_sbo.txt', F_SBO)
        
    RXT_DIST.system = SYSTEM_DIST
    RXT_DIST.exp_w = exp_w
    RXT_DIST.args = args_dist
    
    RXT_DIST.optimizer_bois(trials = trials, init_pts = init_pts, eps = eps,
                            idx = idx, x_idx = x_idx, y_idx = y_idx,
                            gp_sim = gp_sim, cost_fun = cost_fun,
                            restarts = 10, af_cores = 1,
                            f_args = f_args, gp_args = gp_args,
                            x_init = RXT_DIST.scale(RXT_DIST.x_sbo[:2]),
                            kernel_length_scale_bnds = kernel_length_scale_bnds, nu = nu,
                            split_gps = False, norm_xdat = False,
                            f_prime_regularizer = None,
                            feasibility_check = True,
                            feasible_lb = feasible_lb, feasible_ub = feasible_ub,
                            clip_to_bounds = True)
    
    X_BOIS[i*trials:(i+1)*trials] = RXT_DIST.x_bois
    F_BOIS[:, i] = RXT_DIST.f_bois.flatten()
    np.savetxt('RXT_FLSH_RCYL_x_bois.txt', X_BOIS)
    np.savetxt('RXT_FLSH_RCYL_f_bois.txt', F_BOIS)
    
    RXT_DIST.optimizer_mcbo(trials = trials, init_pts = init_pts, n_samples = n_samples,
                            idx = idx, x_idx = x_idx, y_idx = y_idx,
                            gp_sim = gp_sim, cost_fun = cost_fun,
                            restarts = 10, af_cores = 1,
                            f_args = f_args, gp_args = gp_args,
                            x_init = RXT_DIST.scale(RXT_DIST.x_sbo[:2]),
                            kernel_length_scale_bnds = kernel_length_scale_bnds, nu = nu,
                            split_gps = False, norm_xdat = False,
                            feasibility_check = True,
                            feasible_lb = feasible_lb, feasible_ub = feasible_ub,
                            clip_to_bounds = True)
    
    X_MCBO[i*trials:(i+1)*trials] = RXT_DIST.x_mcbo
    F_MCBO[:, i] = RXT_DIST.f_mcbo.flatten()
    np.savetxt('RXT_FLSH_RCYL_x_mcbo.txt', X_MCBO)
    np.savetxt('RXT_FLSH_RCYL_f_mcbo.txt', F_MCBO)
    
    RXT_DIST.optimizer_optimism_bo(trials = trials, init_pts = init_pts,
                                   idx = idx_opbo, x_idx = x_idx, y_idx = y_idx,
                                   gp_sim = gp_sim, cost_fun = cost_fun,
                                   feasible_lb = feasible_lb, feasible_ub = feasible_ub,
                                   restarts = 10, af_cores = 1,
                                   f_args = f_args, gp_args = gp_args,
                                   x_init = RXT_DIST.scale(RXT_DIST.x_sbo[:2]),
                                   kernel_length_scale_bnds = kernel_length_scale_bnds, nu = nu,
                                   norm_xdat = False, split_gps = False)
    
    X_OPBO[i*trials:(i+1)*trials] = RXT_DIST.x_opbo
    F_OPBO[:, i] = RXT_DIST.f_opbo.flatten()
    np.savetxt('RXT_FLSH_RCYL_x_opbo.txt', X_OPBO)
    np.savetxt('RXT_FLSH_RCYL_f_opbo.txt', F_OPBO)
import numpy as np
from scipy.optimize import fsolve, Bounds
import sklearn.gaussian_process as gpr

import sys
sys.path.append('./../../BO_algos')
import Composite_Func_Algos as BO_algos

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# CONSTANTS AND PARAMETERS
## Physical Properties
rho_CH4 = 0.7168         # methane density (kg/m^3)
eta_gasturbine = 0.3    # efficiency of gas turbine used for electricity generation
btu_h2o = 1500          # energy required to remove 1lb H2O (BTU/lb H2O)


## Conversions
t2lbs = 2205            # lbs to tonnes conversion factor (lb/tonne)
day2second = 86400      # seconds in a day (s/day)
acre2meter = 4046       # m^2 in 1 acre (m^2/acre)
joule2kwh = 3.6*1e6     # joules in 1 kWh (J/kWh)
ft3_m3 = 35.3           # cubic feet in a cubic meter (ft^3/m^3)
BTU_kWh = 3412          # BTU in a kWh (BTU/kWh)
SCF_BTU = 1000          # SCF in MMBTU (SCF/MMBTU)
lb_kg = 2.2             # lbs in kg (lb/kg)


## Prices
p = {}                  # price dictionary
p['bags'] = 5147/1e6    # b-PBR bag replacement cost (MMUSD/acres/yr)
p['inoc'] = 1590/1e6    # b-PBR inoculum and cultivation costs (MMUSD/acre/yr)
p['urea'] = 500/1e6     # urea cost (MMUSD/tonne)
p['mix'] = 3.3154e-5    # b-PBR mixing requirements (MMUSD/ha/day)
p['wat_del'] = 8.8e-8   # b-PBR power requirements for water delivery to units (MMUSD/ha/day)
p['cb'] = 7000/1e6      # CB price (MMUSD/tonne)
p['floc'] = 1e-4        # flocculation tank operating cost (MMUSD/tonne CB processed)
p['lam'] = 4.30e-7      # lamella clarifier operating cost (MMUSD/ tonne CB processed)
p['pf'] = 3.3e-8        # pressure filter operating energy demand (MMUSD/m^3 feed)
p['nat_gas'] = 5.84     # natural gas price (USD/MMBTU <--> USD/1000 SCF) 
p['elec'] = 0.11        # price of electricity (USD/kWh)
p['co2_rem'] = 40       # price of removing CO2 from biogas (USD/tonne CO2)
p['h2s_rem'] = 0.0667   # price of removing H2S from biogas (USD/tonne biogas)
p['labor'] = 9.34/2428  # labor costs for PBR based on a facility of size 2428 acres (MMUSD/acre/yr)
p['maint'] = 0.05       # maintenance cost fraction (MMUSD/MMUSD ISBL)
p['op'] = 0.025         # business costs cost fraction (MMUSD/MMUSD ISBL)
p['ovhd'] = 0.05        # overhead costs cost fraction (MMUSD/MMUSD ISBL)
p['rin'] = 0.839        # RIN credit value for methane production (USD/kg CH4)
p['lcfs'] = 0.548       # LCFS credit value for methne production (USD/kg CH4)
p['p_credit'] = 74.5    # P credit value for phosphorus capture (USD/kg P)


## Economic Constants
### general project values 
oprD = 365              # operational days a year
CEPCI_2020 = 596.2      # 2020 consumer price index (reference year)
project_life = 10       # project life (years)

### digester and solids-liquids separator (SLS)
AD_CEPCI = 539.1
SLS_CEPCI = 556.8

### bioreactors
PBR_scale_factor = 0.6      # scaling factor for purchase of PBRs
PBR_base_size = 5.08/4046   # size of reference PBR system (acres)
PBR_base_price = 0.000279   # installed cost of reference PBR system (MMUSD)
PBR_CEPCI = 556.8           # CEPCI for reference PBR system price

### separation train
FLOC_scale_factor = 0.6     # scaling factor for purchase of flocculation tank
FLOC_base_size = 266939     # size of reference flocculation tank (tonnes CB processed/yr)
FLOC_base_price = 0.1147    # installed cost of reference flocculation tank (MMUSD)
FLOC_CEPCI = 585.7          # CEPCI for reference flocculation tank

LAM_scale_factor = 0.6      # scaling factor for purchase of lamella clarifier
LAM_base_size = 266939      # size of reference lamella clarifier (tonnes CB processed/yr)
LAM_base_price = 2.5        # installed cost of reference lamella clarifier (MMUSD)
LAM_CEPCI = 585.7           # CEPCI for reference lamella clarifier

PF_scale_factor = 0.6       # scaling factor for purchase of pressure filter
PF_base_size = 17.76*365    # size of reference pressure filter (tonnes effluent/yr)
PF_base_price = 0.137       # installed cost of reference pressure filter (MMUSD)
PF_CEPCI = 381.8            # CEPCI for reference pressure filter

### thermal dryer
TD_scale_factor = 0.6       # scaling factor for purchase of thermal dryer
TD_base_size = 5.182*365    # size of reference thermal dryer
TD_base_price = 0.706427    # installed cost of reference thermal dryer
TD_CEPCI = 539.1            # CEPCI for reference thermal dryer

### cogeneration and amine
CGA_scale_factor = [0.8]    # scaling factor for cogeneration and amine unit
CGA_base_size = [620*365]   # size of reference cogeneration and amine unit
CGA_base_price = [13.1352]  # installed cost of reference cogeneration and amine unit
CGA_CEPCI = [539.1, 444.2]  # CEPCI for reference cogeneration and amine unit


## Manure Feed Parameters
M_in = 20832                        # Manure feed to process (tonnes/yr)
P_in = 1*1000*1.8e6/77245/2.2/1000  # Phosphorus (P) in feed to process (tonnes/yr)
NtoP = 1.1                          # Manure N to P (N:P) ratio
N_in = NtoP*P_in                    # Nitrogen (N) in feed to process (tonnes/yr)

Feed = np.array([M_in, P_in, NtoP, N_in])


## Digester, Solids Separator, Cogen and Amine Unit Parameters
y_MtoBG = 1/21                          # biogas production to manure feed ratio
x_CH4 = 0.6488                          # mass frac of CH4 in biogas
x_CO2 = 0.3488                          # mass frac of CO2 in biogas
x_H2S = 0.0024                          # mass frac of H2S in biogas
x_BG = np.array([x_CH4, x_CO2, x_H2S])  # composition of biogas stream
x_TS = 0.09                             # total solids fraction in digestate (assume N and P are uniformly distributed)
x_CH4mark = 1.0*(2/3)                   # fraction of generated methane sent to the market

front_end_args = np.array([y_MtoBG, x_CH4, x_CO2, x_H2S, x_TS, x_CH4mark]) 


## Bag Photobioreactor (b-PBR) Parameters
"""
Dimension of X will depend on units used for m_v and Y_xv, recommend using SI units
"""
Y_xv = 0.00202*1e-6                   # Biomass production per photon consumed (kg*umol^-1)
eta = 0.23538                         # CB photon use efficiency (unit-less)
m_v = 917.5/3.6                       # CB maintencance photon need (umol*kg^-1*s^-1)
pbr_NtoP = 7                          # CB N to P (N:P) ratio
X_0 = 0.03                            # Initial CB concentration (kg/m^3)
I_0 = 350                             # incident light intensity (umol*s^-1*m^-2)
t_batch =  30*86400                   # batch time (seconds)
SV_reactor = (4.48*1.22)/0.355        # Surface area to volume ration of bioreactors (m^-1)
sigma_a = 0.355/5.08                  # Bioreactor mass surface density of rack system (tonne/m^2)
P_dem = 0.023                         # P content of the CB (tonne P/tonne CB)
max_prod = 175                        # maximum demand for CB fertilizer (tonnes/yr)

x = np.array([I_0, SV_reactor, t_batch, P_dem])
pbr_args = np.array([Y_xv, eta, m_v, pbr_NtoP, X_0, sigma_a])


## Separation Train and Thermal Dryer Parameters
floc_req = 0.097   # flocculant required on CB mass basis (tonnes floc/tonnes CB)
x_cbF = 0.01       # mass frac of algae in flocculant output
x_cbL = 0.016      # mass frac of algae in lamella output
x_cbPF = 0.27      # mass frac of algae in pressure filter output
x_cbTD = 1.0       # mass frac of algae in dryer output

dewatering_train_args = np.array([floc_req, x_cbF, x_cbL, x_cbPF, x_cbTD])



#%% UNIT DEFINITIONS
## Digester and SLS
class DIGEST():
    def __init__(self, m_in, y_MtoBG, x_BG, CEPCI):
        self.m_in = m_in
        self.y_MtoBG = y_MtoBG
        self.x_BG = x_BG
        self.CEPCI = CEPCI

    # digester mass balance based on biogas yield (y_MtoBG) and composition (x_BG) values
    def mass_bal(self):
        self.m_BG = self.m_in*self.y_MtoBG
        self.m_D = self.m_in-self.m_BG
        self.m_CH4 = self.m_BG*self.x_BG[0]
        self.m_CO2 = self.m_BG*self.x_BG[1]
        self.m_H2S = self.m_BG*self.x_BG[2]
        self.error =  self.m_in-self.m_BG-self.m_D
        
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around digester is not closed')

    # digester costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = (937.1*self.m_in**0.6+75355)*(CEPCI_2020/self.CEPCI) # Capacity is in tonnes/yr
        return self.C/1e6

    
class SLD_SEP():
    def __init__(self, m_in, Pin, Nin, x_TS, CEPCI):
        self.m_in = m_in
        self.x_TS = x_TS
        self.Pin = Pin
        self.Nin = Nin
        self.CEPCI = CEPCI
    
    # SLS mass balance based on separation factor x_TS
    def mass_bal(self):
        self.m_SSL = (1-self.x_TS)*self.m_in
        self.m_SSS = self.x_TS*self.m_in
        self.m_PS = self.x_TS*self.Pin
        self.m_PL = (1-self.x_TS)*self.Pin
        self.m_NS = self.x_TS*self.Nin
        self.m_NL = (1-self.x_TS)*self.Nin
        self.error = self.m_in-self.m_SSL-self.m_SSS
        
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around solids-liquids separator is not closed')
    
    # SLS costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.m_lb = t2lbs*self.m_in/oprD/24 # flow in lbs/hr
        self.C = (3.75*self.m_in+1786.9*np.log(self.m_lb)-9506.6)*(CEPCI_2020/self.CEPCI)
        return self.C/1e6


## b-PBR
class PBR_DESIGN():
    def __init__(self, X_0, Y_xv, eta, m_v, m_P, m_N, NtoP, m_CO2, sigma_a, scale_factor, base_size, base_price, CEPCI):
        # Bioreactor constants
        self.X_0 = X_0
        self.Y_xv = Y_xv
        self.eta = eta
        self.m_v = m_v 
        self.rho_eff = 1                # density of effluent (tonne/m^3)
        
        # Reactor feed
        self.m_P = m_P
        self.m_N = m_N
        self.m_CO2 = m_CO2
        self.NtoP = NtoP
        
        # Economic parameters
        self.sf = scale_factor
        self.PBR_0 = base_size
        self.Ct_0 = base_price
        self.CEPCI = CEPCI
        
        # Conversions and constants
        self.sigma_a = sigma_a
        self.day2second = 86400         # seconds in a day
        self.oprD = 365                 # operational days a year
        self.acre2meter = 4046          # m^2 in 1 acre
        self.joule2kwh = 3.6*1e6        # joules in 1 kWh
        self.gPERL_tonnePERm3 = 1e3     # g/L in tonnes/m^3 ((g/L)/(tonne/m^3))        
    
    # simulate CB growth only 
    def CB_GRO(self, x, lam = 655*1e-9, c = 3.0*1e8, h = 6.63*1e-34, A = 6.023*1e23):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        I0 = x[0, 0]    # incident light intensity
        SV = x[0, 1]    # surface area to volume ratio
        t_end = x[0, 2] # batch time
        self.P_dem = x[0, 3]
        
        # Biomass and supplemental N required
        self.m_cb = self.m_P/self.P_dem    # mass of CB required (tonne/yr)
        self.N_dem = min(5e-2, self.NtoP*self.P_dem)    # (tonne N/tonne CB)
        self.m_NR = self.N_dem*self.m_cb                # Required N (tonne/yr)
        self.m_NS = max(0, self.m_NR-self.m_N)          # Supplemental N requirements (from urea) (tonnes/yr)
        
        # Supplemental CO2 required after biogas contribution (tonnes/yr)
        self.x_Ccb = 0.54                                  # mass frac of C in CB
        self.e_CO2 = 0.1                                   # excess CO2 provided (mass fraction)
        self.CO2_dem = 44*(1+self.e_CO2)*(self.x_Ccb/12)   # tonne CO2/tonne CB
        self.m_CO2R = self.CO2_dem*self.m_cb               # Required CO2 in tonne/yr
        self.m_CO2S = max(0, self.m_CO2R-self.m_CO2)       # Supplemental CO2 requirements (tonnes/yr)
        
        
        t = np.linspace(0, t_end, int(t_end)+1)
        kappa = self.Y_xv*self.m_v
        X_S = self.eta*(I0*SV)/self.m_v/3.11 # scale to reduce very high titer
        
        self.X = self.X_0*np.exp(-kappa*t)+X_S*(1-np.exp(-kappa*t))                                  # effluent titer (kg/m^3)
        self.m_out = self.m_cb/(self.X[-1]/self.gPERL_tonnePERm3)*self.rho_eff                       # reactor outflow (tonnes/yr)
        self.V = (t_end/self.day2second)*(self.m_cb/self.oprD)/(self.X[-1]/self.gPERL_tonnePERm3)    # reactor volume (m^3)
        self.SA = self.V*self.rho_eff/self.sigma_a/self.acre2meter                                   # reactor surface area (acres)
        
        E_hat = h*c/lam                                                 # Energy per photon (J/photon)
        photons = I0*1e-6*A*(self.V*SV)*self.day2second*self.oprD       # Photon flow (photons/yr)
        self.E_tot = E_hat*photons/self.joule2kwh                       # Photon energy flow (kWh/yr)
       
    # reactor costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = self.Ct_0*(CEPCI_2020/self.CEPCI)*(self.SA/self.PBR_0)**self.sf
        return self.C


## Sepration Train
### flocculation tank
class FLOCCULATION_TANK(): # There should not be any separation happeining inside the flocculator
    def __init__(self, m_in, m_cb, floc_req, scale_factor, base_size, base_price, CEPCI):
        self.m_in = m_in
        self.m_cb = m_cb
        self.f_req = floc_req
        self.sf = scale_factor
        self.F0 = base_size
        self.C0 = base_price
        self.CEPCI = CEPCI

    # flocculation tank mass balance based on addition of flocculant (still preliminary, might not do anything) 
    def mass_bal(self):
        self.m_floc = self.m_cb*self.f_req
        self.m_BM = self.m_in+self.m_floc
        self.error = self.m_BM-self.m_in-self.m_floc
    
        if abs(self.error) > 1e-6:
            print(self.m_in, self.m_floc, self.m_BM)
            raise Exception('Mass balance around flocculation tank is not closed')

    # flocculation tank costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = self.C0*(CEPCI_2020/self.CEPCI)*(self.m_cb/self.F0)**self.sf
        return self.C


### lamella clarifier
class LAMELLA(): # The lamella is the clarifier separator
    def __init__(self, m_in, m_cb, x_BM_effl, scale_factor, base_size, base_price, CEPCI):
        self.m_in = m_in
        self.m_cb = m_cb
        self.x_cb = x_BM_effl
        self.sf = scale_factor
        self.L0 = base_size
        self.C0 = base_price
        self.CEPCI = CEPCI

    # lamella clarifier mass balance based on effluent moisture content factor x_BM_effl
    def mass_bal(self):
        self.m_BM = self.m_cb/self.x_cb
        self.m_W = self.m_in - self.m_BM
        self.error = self.m_in-self.m_BM-self.m_W
        
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around lamella clarifier is not closed')

    # lamella clarifier costing 
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = self.C0*(CEPCI_2020/self.CEPCI)*(self.m_cb/self.L0)**self.sf
        return self.C


## pressure filter
class PRESSURE_FILTER():
    def __init__(self, m_in, m_cb, x_BM_effl, scale_factor, base_size, base_price, CEPCI):
        self.m_in = m_in
        self.m_cb = m_cb
        self.x_cb = x_BM_effl
        self.sf = scale_factor
        self.PF0 = base_size
        self.C0 = base_price
        self.CEPCI = CEPCI

    # pressure filter mass balance based on effluent moisture content x_BM_effl
    def mass_bal(self):
        self.m_BM = self.m_cb/self.x_cb
        self.m_W = self.m_in - self.m_BM
        self.error = self.m_in-self.m_BM-self.m_W
        
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around pressure filter is not closed')

    # pressure filter costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = self.C0*(CEPCI_2020/self.CEPCI)*(self.m_BM/self.PF0)**self.sf
        return self.C


## Thermal Dryer
class THERMAL_DRYER():
    def __init__(self, m_in, m_cb, x_BM_effl, scale_factor, base_size, base_price, CEPCI):
        self.m_in = m_in
        self.m_cb = m_cb
        self.x_cb = x_BM_effl
        self.sf = scale_factor
        self.TD0 = base_size
        self.C0 = base_price
        self.CEPCI = CEPCI

    # thermal dryer mass balance based on specified final moisture content x_BM_effl
    def mass_bal(self):
        self.m_BM = self.m_cb/self.x_cb  # mass flow of effluent (tonnes/yr)
        self.m_W = self.m_in - self.m_BM  # Recycled water (tonnes/yr)
        self.error = self.m_in-self.m_BM-self.m_W
        
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around thermal dryer is not closed')

    # thermal dryer costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = self.C0*(CEPCI_2020/self.CEPCI)*(self.m_in/self.TD0)**self.sf
        return self.C
    

## Cogeneration and Amine Scrubbing
class COGEN_AMINE():
    def __init__(self, m_in, x_CH4market, Ct_Dig, scale_factor, base_size, base_price, CEPCI):
        self.m_CH4 = m_in[0]
        self.m_CO2 = m_in[1]
        self.m_H2S = m_in[2]
        self.x_CH4 = x_CH4market        # Fraction of CH4 sold to market off-farm
        self.y_O2air = 0.21             # mol frac of O2 in air
        self.Ct_Dig = Ct_Dig
        self.A0 = base_size[0]          # size of reference A unit
        self.C0_A = base_price[0]       # installed cost of reference amine unit
        self.sf_A = scale_factor[0]     # scaling facotr for amine estimate
        self.CEPCI_CG = CEPCI[0]        # CEPCI of reference CG unit
        self.CEPCI_A = CEPCI[1]         # CEPCI of reference amine unit

    # cogen and amine unit mass balance based on complete combustion of not exported methane
    # CO2 and H2S in biogas stream are assumed to be completely removed
    def mass_bal(self):
        # mass flow of BG to amine scrubber
        self.m_Ain = self.m_CH4+self.m_CO2+self.m_H2S
        # mass flow of gases exiting amine unit (removed CO2 and H2S)
        self.m_Aout = self.m_CO2+self.m_H2S
        # mass flow CH4 to market
        self.m_CH4M = self.m_CH4*self.x_CH4
        # mass flow of CH4 into co-gen plant (tonnes/yr)
        self.m_CH4CG = (1-self.x_CH4)*self.m_CH4
        # mass of CO2 produced in co-gen (tonnes/yr)
        self.m_CO2CG = self.m_CH4CG/16*44
        # mass of H2O produced in co-gen (tonnes/yr)
        self.m_H2OCG = self.m_CH4CG/16*2*18
        self.m_O2 = self.m_CH4CG/16*2*32  # required O2 (tonnes/yr)
        # mass of N2 in co-gen (tonnes/yr)
        self.m_N2 = self.m_CH4CG/16*2*(1-self.y_O2air)/(self.y_O2air)*28
        self.m_flue = self.m_N2+self.m_H2OCG+self.m_CO2CG
        self.error = self.m_Ain+self.m_O2-(self.m_Aout+self.m_CH4M+self.m_CO2CG+self.m_H2OCG)
        
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around cogen and amine scrubbing unit is not closed')

    # cogen and amine unit costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C_CG = 0.67*self.Ct_Dig*(CEPCI_2020/self.CEPCI_CG)*(1-self.x_CH4)
        self.C_A = self.C0_A*(CEPCI_2020/self.CEPCI_A)*\
            (self.m_Ain/self.A0)**self.sf_A
        return self.C_CG+self.C_A
    

#%% PROCESS SETUP AND SIMULATION

Digester = DIGEST(M_in, y_MtoBG, x_BG, AD_CEPCI)
Digester.mass_bal()
Ct_Digester = Digester.econ()

SLS = SLD_SEP(Digester.m_D, P_in, N_in, x_TS, SLS_CEPCI)
SLS.mass_bal()
Ct_SLS = SLS.econ()

CGA = COGEN_AMINE([Digester.m_CH4, Digester.m_CO2, Digester.m_H2S], x_CH4mark, Ct_Digester,
                  CGA_scale_factor, CGA_base_size, CGA_base_price, CGA_CEPCI)
CGA.mass_bal()
Ct_CGA = CGA.econ()

PBR_mod = PBR_DESIGN(X_0, Y_xv, eta, m_v, SLS.m_PL, SLS.m_NL, pbr_NtoP, CGA.m_CO2CG, 
                     sigma_a, PBR_scale_factor, PBR_base_size, PBR_base_price, PBR_CEPCI)
PBR_mod.CB_GRO(x)
Ct_PBR = PBR_mod.econ()

Floc = FLOCCULATION_TANK(PBR_mod.m_out, PBR_mod.m_cb, floc_req,
                         FLOC_scale_factor, FLOC_base_size, FLOC_base_price, FLOC_CEPCI)
Floc.mass_bal()
Ct_FlocculationTank = Floc.econ()

Lam = LAMELLA(Floc.m_BM, Floc.m_cb, x_cbL,
              LAM_scale_factor, LAM_base_size, LAM_base_price, LAM_CEPCI)
Lam.mass_bal()
Ct_Lamella = Lam.econ()
m_innxt = Lam.m_BM

PF = PRESSURE_FILTER(m_innxt, PBR_mod.m_cb, x_cbPF,
                     PF_scale_factor, PF_base_size, PF_base_price, PF_CEPCI)
PF.mass_bal()
Ct_PressureFilter = PF.econ()

TD = THERMAL_DRYER(PF.m_BM, PBR_mod.m_cb, x_cbTD,
                   TD_scale_factor, TD_base_size, TD_base_price, TD_CEPCI)
TD.mass_bal()
Ct_ThermalDryer = TD.econ()


#%% ECONOMICS CALCULATIONS
def MSP(msp, p, units, parameters, constants, DROI = 0.15, tax = 0.21):
    CGA = units[0]
    
    P_in = parameters[0]
    m_cb = parameters[1]
    depreciation = parameters[2]
    TCI = parameters[3]
    TOC = parameters[4]
    project_life = int(parameters[5])
    
    rho_CH4 = constants[0]
    ft3_m3 = constants[1]
    SCF_BTU = constants[2]
    BTU_kWh = constants[3]
    eta_gasturbine = constants[4]
    
    
    cb_revenue = msp*1000*m_cb
    biogas_revenue = p['nat_gas']*((CGA.x_CH4*CGA.m_CH4*1000/rho_CH4)*ft3_m3/SCF_BTU)
    electricity_revenue = p['elec']*eta_gasturbine*((((1-CGA.x_CH4)*CGA.m_CH4*1000/rho_CH4)*ft3_m3/
                                                     SCF_BTU)*1e3/BTU_kWh)*1000
    
    p_credit_revenue = (74.5*0.0)*P_in*1000
    rin_credits_revenue = 1.0*p['rin']*CGA.m_CH4*CGA.x_CH4*1000
    lcfs_credits_revenue = 1.0*p['lcfs']*CGA.m_CH4*CGA.x_CH4*1000
    
    REV = (cb_revenue+biogas_revenue+electricity_revenue+\
           p_credit_revenue+rin_credits_revenue+lcfs_credits_revenue)/1e6
    AATP = (1-tax)*(REV-TOC)+depreciation
    
    NPV = -TCI*np.ones(project_life+1)
    PVCF = NPV.copy()  # present value cashflow
    for i in range(1, project_life+1):
        PVCF[i] = AATP*(1+DROI)**(-i)
        NPV[i] = NPV[i-1]+PVCF[i]
    return NPV[-1].flatten()


## Capital Costs
ISBL = Ct_Digester+Ct_SLS+Ct_CGA+Ct_PBR+Ct_FlocculationTank+Ct_Lamella+Ct_PressureFilter+Ct_ThermalDryer 
OSBL = 0.4*ISBL
ENG = (ISBL+OSBL)*0.3
CONT = (ISBL+OSBL)*0.2
LAND = 5890*PBR_mod.SA/1e6
TCI = ISBL+OSBL+ENG+CONT+LAND


## Operating Costs
### fixed operating costs
maintenance_costs = p['maint']*ISBL
operations_costs = p['op']*ISBL
overhead_costs = p['ovhd']*ISBL
depreciation = ISBL/project_life

### digester, SLS, cogen and amine units operating costs
digester_cost = 0.096*Ct_Digester
sls_cost = (0.488*SLS.m_in+0.1*(1786.9*np.log(SLS.m_lb)-9506.6))/1e6
amine_cost = p['co2_rem']/1e6*CGA.m_CO2+p['h2s_rem']/1e6*Digester.m_BG*1000

### b-PBR operating costs
pbr_replacement_cost = p['bags']*(PBR_mod.SA)*max(1, x[1]/SV_reactor) # increase cost for higher S/V's that will require more complex geometries
inoculum_cost = p['inoc']*(PBR_mod.SA)/1 ###
mixing_cost = p['mix']*(PBR_mod.SA*acre2meter/1e4)*oprD*max(1, 1+(x[1]/SV_reactor-1)/2) # increase cost for higher S/V's as these will likely increase turbulence and require more energy
water_delivery_cost = p['wat_del']*(PBR_mod.SA*acre2meter/1e4)*oprD
urea_cost = p['urea']*PBR_mod.m_NS/0.4666666666666667

### separation train operating costs
flocculation_tank_cost = p['floc']*Floc.m_cb
lamella_cost = p['lam']*Lam.m_cb
pressure_filter_cost = p['pf']*PF.m_in
dryer_cost = p['nat_gas']/1e6*(TD.m_W*1000*lb_kg*btu_h2o)/1e6

### labor costs
labor = p['labor']*PBR_mod.SA

### total operating costs
VOC = digester_cost+sls_cost+amine_cost+\
      pbr_replacement_cost+inoculum_cost+mixing_cost+water_delivery_cost+urea_cost+\
      flocculation_tank_cost+lamella_cost+pressure_filter_cost+dryer_cost+labor
FOC = maintenance_costs+operations_costs+overhead_costs+depreciation
TOC = VOC+FOC


## Revenues
cb_revenue = p['cb']*1000*PBR_mod.m_cb
biogas_revenue = p['nat_gas']*((CGA.x_CH4*CGA.m_CH4*1000/rho_CH4)*ft3_m3/SCF_BTU)
electricity_revenue = p['elec']*eta_gasturbine*((((1-CGA.x_CH4)*CGA.m_CH4*1000/rho_CH4)*ft3_m3/
                                                 SCF_BTU)*1e3/BTU_kWh)*1000
REV = (cb_revenue+biogas_revenue+electricity_revenue)/1e6  # Total revenue in MMUSD/yr
p_credit_revenue = 74.5*P_in*1000
rin_credits_revenue = p['rin']*CGA.m_CH4*CGA.x_CH4*1000
lcfs_credits_revenue = p['lcfs']*CGA.m_CH4*CGA.x_CH4*1000


## MSP Determination
msp_args = (p,
            [CGA],
            np.array([P_in, PBR_mod.m_cb, depreciation, TCI, TOC, project_life]),
            np.array([rho_CH4, ft3_m3, SCF_BTU, BTU_kWh, eta_gasturbine]))
msp = fsolve(MSP, 3, args = msp_args)


#%% SYSTEM  FUNCTION
def SYSTEM_TEA(x, p, Feed, front_end_args, pbr_args, dewatering_train_args,
               max_prod = None, dist = False, initial_guess = 3):
    
    warnings.simplefilter('ignore', DeprecationWarning)
    
    x = x.flatten()
    ## Physical Properties
    rho_CH4 = 0.7168         # methane density (kg/m^3)
    eta_gasturbine = 0.3    # efficiency of gas turbine used for electricity generation
    btu_h2o = 1500          # energy required to remove 1lb H2O (BTU/lb H2O)


    ## Conversions
    acre2meter = 4046       # m^2 in 1 acre (m^2/acre)
    ft3_m3 = 35.3           # cubic feet in a cubic meter (ft^3/m^3)
    BTU_kWh = 3412          # BTU in a kWh (BTU/kWh)
    SCF_BTU = 1000          # SCF in MMBTU (SCF/MMBTU)
    lb_kg = 2.2             # lbs in kg (lb/kg)
    SV_0 = 15.39605633802817
    
    
    ## Manure Feed Parameters
    M_in = Feed[0]          # Manure feed to process (tonnes/yr)
    P_in = Feed[1]          # Phosphorus (P) in feed to process (tonnes/yr)
    NtoP = Feed[2]          # Manure N to P (N:P) ratio
    N_in = NtoP*P_in        # Nitrogen (N) in feed to process (tonnes/yr)
    
    
    ## Digester, Solids Separator, Cogen and Amine Unit Parameters
    y_MtoBG = front_end_args[0]                 # biogas production to manure feed ratio
    x_CH4 = front_end_args[1]                   # mass frac of CH4 in biogas
    x_CO2 = front_end_args[2]                   # mass frac of CO2 in biogas
    x_H2S = front_end_args[3]                   # mass frac of H2S in biogas
    x_BG = np.array([x_CH4, x_CO2, x_H2S])      # composition of biogas stream
    x_TS = front_end_args[4]                    # total solids fraction in digestate (assume N and P are uniformly distributed)
    x_CH4mark = front_end_args[5]               # fraction of generated methane sent to the market
    
    
    ## Bag Photobioreactor (b-PBR) Parameters
    """
    Dimension of X will depend on units used for m_v and Y_xv, recommend using SI units
    """
    Y_xv = pbr_args[0]                    # Biomass production per photon consumed (kg*umol^-1)
    eta = pbr_args[1]                     # CB photon use efficiency (unit-less)
    m_v = pbr_args[2]                     # CB maintencance photon need (umol*kg^-1*s^-1)
    pbr_NtoP = pbr_args[3]                # CB N to P (N:P) ratio
    X_0 = pbr_args[4]                     # Initial CB concentration (kg/m^3)
    sigma_a = pbr_args[5]
    
    
    ## Separation Train and Thermal Dryer Parameters
    floc_req = dewatering_train_args[0]   # flocculant required on CB mass basis (tonnes floc/tonnes CB)
    x_cbF = dewatering_train_args[1]      # mass frac of algae in flocculant output
    x_cbL = dewatering_train_args[2]      # mass frac of algae in lamella output
    x_cbPF = dewatering_train_args[3]     # mass frac of algae in pressure filter output
    x_cbTD = dewatering_train_args[4]     # mass frac of algae in dryer output
    
    
    ## Economic Constants
    ### general project values 
    oprD = 365              # operational days a year
    project_life = 10       # project life (years)

    ### digester and solids-liquids separator (SLS)
    AD_CEPCI = 539.1
    SLS_CEPCI = 556.8

    ### bioreactors
    PBR_scale_factor = 0.6      # scaling factor for purchase of PBRs
    PBR_base_size = 5.08/4046   # size of reference PBR system (m^3)
    PBR_base_price = 0.000279   # installed cost of reference PBR system (MMUSD)
    PBR_CEPCI = 556.8           # CEPCI for reference PBR system price

    ### separation train
    FLOC_scale_factor = 0.6     # scaling factor for purchase of flocculation tank
    FLOC_base_size = 266939     # size of reference flocculation tank (tonnes CB processed/yr)
    FLOC_base_price = 0.1147    # installed cost of reference flocculation tank (MMUSD)
    FLOC_CEPCI = 585.7          # CEPCI for reference flocculation tank

    LAM_scale_factor = 0.6      # scaling factor for purchase of lamella clarifier
    LAM_base_size = 266939      # size of reference lamella clarifier (tonnes CB processed/yr)
    LAM_base_price = 2.5        # installed cost of reference lamella clarifier (MMUSD)
    LAM_CEPCI = 585.7           # CEPCI for reference lamella clarifier

    PF_scale_factor = 0.6       # scaling factor for purchase of pressure filter
    PF_base_size = 17.76*365    # size of reference pressure filter (tonnes effluent/yr)
    PF_base_price = 0.137       # installed cost of reference pressure filter (MMUSD)
    PF_CEPCI = 381.8            # CEPCI for reference pressure filter

    ### thermal dryer
    TD_scale_factor = 0.6       # scaling factor for purchase of thermal dryer
    TD_base_size = 5.182*365    # size of reference thermal dryer
    TD_base_price = 0.706427    # installed cost of reference thermal dryer
    TD_CEPCI = 539.1            # CEPCI for reference thermal dryer

    ### cogeneration and amine
    CGA_scale_factor = [0.8]    # scaling factor for cogeneration and amine unit
    CGA_base_size = [620*365]   # size of reference cogeneration and amine unit
    CGA_base_price = [13.1352]  # installed cost of reference cogeneration and amine unit
    CGA_CEPCI = [539.1, 444.2]  # CEPCI for reference cogeneration and amine unit
    
    
    ## System setup and simulation
    Digester = DIGEST(M_in, y_MtoBG, x_BG, AD_CEPCI)
    Digester.mass_bal()
    Ct_Digester = Digester.econ()

    SLS = SLD_SEP(Digester.m_D, P_in, N_in, x_TS, SLS_CEPCI)
    SLS.mass_bal()
    Ct_SLS = SLS.econ()

    CGA = COGEN_AMINE([Digester.m_CH4, Digester.m_CO2, Digester.m_H2S], x_CH4mark, Ct_Digester,
                      CGA_scale_factor, CGA_base_size, CGA_base_price, CGA_CEPCI)
    CGA.mass_bal()
    Ct_CGA = CGA.econ()

    PBR_mod = PBR_DESIGN(X_0, Y_xv, eta, m_v, SLS.m_PL, SLS.m_NL, pbr_NtoP, CGA.m_CO2CG, 
                         sigma_a, PBR_scale_factor, PBR_base_size, PBR_base_price, PBR_CEPCI)
    PBR_mod.CB_GRO(x)
    Ct_PBR = PBR_mod.econ()

    Floc = FLOCCULATION_TANK(PBR_mod.m_out, PBR_mod.m_cb, floc_req,
                             FLOC_scale_factor, FLOC_base_size, FLOC_base_price, FLOC_CEPCI)
    Floc.mass_bal()
    Ct_FlocculationTank = Floc.econ()

    Lam = LAMELLA(Floc.m_BM, Floc.m_cb, x_cbL,
                  LAM_scale_factor, LAM_base_size, LAM_base_price, LAM_CEPCI)
    Lam.mass_bal()
    Ct_Lamella = Lam.econ()
    m_innxt = Lam.m_BM

    PF = PRESSURE_FILTER(m_innxt, PBR_mod.m_cb, x_cbPF,
                         PF_scale_factor, PF_base_size, PF_base_price, PF_CEPCI)
    PF.mass_bal()
    Ct_PressureFilter = PF.econ()

    TD = THERMAL_DRYER(PF.m_BM, PBR_mod.m_cb, x_cbTD,
                       TD_scale_factor, TD_base_size, TD_base_price, TD_CEPCI)
    TD.mass_bal()
    Ct_ThermalDryer = TD.econ()
    
    #### return intermediate values of interest
    if dist:
        return PBR_mod.V.reshape(-1, 1), PBR_mod.X[-1].reshape(-1, 1)
    
    #### run economics calculations and return system MSP
    else:
        ## Capital Costs
        ISBL = Ct_Digester+Ct_SLS+Ct_CGA+Ct_PBR+Ct_FlocculationTank+Ct_Lamella+Ct_PressureFilter+Ct_ThermalDryer 
        OSBL = 0.4*ISBL
        ENG = (ISBL+OSBL)*0.3
        CONT = (ISBL+OSBL)*0.2
        LAND = 5890*PBR_mod.SA/1e6
        TCI = ISBL+OSBL+ENG+CONT+LAND
    
        ### fixed operating costs
        maintenance_costs = p['maint']*ISBL
        operations_costs = p['op']*ISBL
        overhead_costs = p['ovhd']*ISBL
        depreciation = ISBL/project_life
    
        ### digester, SLS, cogen and amine units operating costs
        digester_cost = 0.096*Ct_Digester
        sls_cost = (0.488*SLS.m_in+0.1*(1786.9*np.log(SLS.m_lb)-9506.6))/1e6
        amine_cost = p['co2_rem']/1e6*CGA.m_CO2+p['h2s_rem']/1e6*Digester.m_BG*1000
    
        ### b-PBR operating costs
        pbr_replacement_cost = p['bags']*(PBR_mod.SA)*max(1, 3*(x[1]/SV_0)-2)  # scale cost based on SV to reflect cost of making more complex geometries
        inoculum_cost = p['inoc']*(PBR_mod.SA)/1 ###
        mixing_cost = p['mix']*(PBR_mod.SA*acre2meter/1e4)*oprD*max(1, 3*(x[1]/SV_0)-2)  # scale cost based on SV to reflect cost of turbulence induced by more complex geometries
        water_delivery_cost = p['wat_del']*(PBR_mod.SA*acre2meter/1e4)*oprD
        urea_cost = p['urea']*PBR_mod.m_NS/0.4666666666666667
    
        ### separation train operating costs
        flocculation_tank_cost = p['floc']*Floc.m_cb
        lamella_cost = p['lam']*Lam.m_cb
        pressure_filter_cost = p['pf']*PF.m_in
        dryer_cost = p['nat_gas']/1e6*(TD.m_W*1000*lb_kg*btu_h2o)/1e6
    
        ### labor costs
        labor_penalty = 5e4*(max(0, 30-x[2]/day2second))/1e6
        labor = p['labor']*PBR_mod.SA+labor_penalty
    
        ### total operating costs
        VOC = digester_cost+sls_cost+amine_cost+\
              pbr_replacement_cost+inoculum_cost+mixing_cost+water_delivery_cost+urea_cost+\
              flocculation_tank_cost+lamella_cost+pressure_filter_cost+dryer_cost+labor
        FOC = maintenance_costs+operations_costs+overhead_costs+depreciation
        TOC = VOC+FOC
    
    
        ## MSP Calculation
        if max_prod is not None:
            m_cb = min(max_prod, PBR_mod.m_cb)
        else:
            m_cb = PBR_mod.m_cb
        
        msp_args = (p,
                    [CGA],
                    np.array([P_in, m_cb, depreciation, TCI, TOC, project_life]),
                    np.array([rho_CH4, ft3_m3, SCF_BTU, BTU_kWh, eta_gasturbine]))
        msp = fsolve(MSP, initial_guess, args = msp_args)
        
        return msp


#%% S-BO & BOIS SETUP
def SYSTEM_BO(x, *args):
    x = x.reshape(-1, dim)
    msp = np.ones(len(x))
    for i, x_0 in enumerate(x):    
        msp[i] = SYSTEM_TEA(x_0, *args)
    return msp#1e3*msp/cb_value


def cost_fun(Y_in, X, p, Feed, front_end_args, pbr_args, dewatering_train_args,
             max_prod = None, initial_guess = 3):   
    
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', RuntimeWarning)
    
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
    
    
    msps = np.ones(len(Y))
    
    
    ## Physical Properties
    rho_CH4 = 0.7168        # methane density (kg/m^3)
    rho_eff = 1             # density reactor of effluent (tonne/m^3)
    eta_gasturbine = 0.3    # efficiency of gas turbine used for electricity generation
    btu_h2o = 1500          # energy required to remove 1lb H2O (BTU/lb H2O)


    ## Conversions
    acre2meter = 4046       # m^2 in 1 acre (m^2/acre)
    day2second = 86400      # seconds in a day (s/day)
    ft3_m3 = 35.3           # cubic feet in a cubic meter (ft^3/m^3)
    BTU_kWh = 3412          # BTU in a kWh (BTU/kWh)
    SCF_BTU = 1000          # SCF in MMBTU (SCF/MMBTU)
    lb_kg = 2.2             # lbs in kg (lb/kg)
    gPERL_tonnePERm3 = 1e3  # g/L in tonnes/m^3 ((g/L)/(tonne/m^3))
    SV_0 = 15.39605633802817
    
    
    ## Manure Feed Parameters
    M_in = Feed[0]          # Manure feed to process (tonnes/yr)
    P_in = Feed[1]          # Phosphorus (P) in feed to process (tonnes/yr)
    NtoP = Feed[2]          # Manure N to P (N:P) ratio
    N_in = NtoP*P_in        # Nitrogen (N) in feed to process (tonnes/yr)
    
    
    ## PBR parameters
    pbr_NtoP = pbr_args[0]
    sigma_a = pbr_args[1]
    
    
    ## Digester, Solids Separator, Cogen and Amine Unit Parameters
    y_MtoBG = front_end_args[0]                 # biogas production to manure feed ratio
    x_CH4 = front_end_args[1]                   # mass frac of CH4 in biogas
    x_CO2 = front_end_args[2]                   # mass frac of CO2 in biogas
    x_H2S = front_end_args[3]                   # mass frac of H2S in biogas
    x_BG = np.array([x_CH4, x_CO2, x_H2S])      # composition of biogas stream
    x_TS = front_end_args[4]                    # total solids fraction in digestate (assume N and P are uniformly distributed)
    x_CH4mark = front_end_args[5]               # fraction of generated methane sent to the market
    
    
    ## Separation Train and Thermal Dryer Parameters
    floc_req = dewatering_train_args[0]   # flocculant required on CB mass basis (tonnes floc/tonnes CB)
    x_cbF = dewatering_train_args[1]      # mass frac of algae in flocculant output
    x_cbL = dewatering_train_args[2]      # mass frac of algae in lamella output
    x_cbPF = dewatering_train_args[3]     # mass frac of algae in pressure filter output
    x_cbTD = dewatering_train_args[4]     # mass frac of algae in dryer output
    
    
    ## Economic Constants
    ### general project values 
    oprD = 365              # operational days a year
    project_life = 10       # project life (years)

    ### digester and solids-liquids separator (SLS)
    AD_CEPCI = 539.1
    SLS_CEPCI = 556.8

    ### bioreactors
    PBR_scale_factor = 0.6      # scaling factor for purchase of PBRs
    PBR_base_size = 5.08/4046   # size of reference PBR system (m^3)
    PBR_base_price = 0.000279   # installed cost of reference PBR system (MMUSD)
    PBR_CEPCI = 556.8           # CEPCI for reference PBR system price

    ### separation train
    FLOC_scale_factor = 0.6     # scaling factor for purchase of flocculation tank
    FLOC_base_size = 266939     # size of reference flocculation tank (tonnes CB processed/yr)
    FLOC_base_price = 0.1147    # installed cost of reference flocculation tank (MMUSD)
    FLOC_CEPCI = 585.7          # CEPCI for reference flocculation tank

    LAM_scale_factor = 0.6      # scaling factor for purchase of lamella clarifier
    LAM_base_size = 266939      # size of reference lamella clarifier (tonnes CB processed/yr)
    LAM_base_price = 2.5        # installed cost of reference lamella clarifier (MMUSD)
    LAM_CEPCI = 585.7           # CEPCI for reference lamella clarifier

    PF_scale_factor = 0.6       # scaling factor for purchase of pressure filter
    PF_base_size = 17.76*365    # size of reference pressure filter (tonnes effluent/yr)
    PF_base_price = 0.137       # installed cost of reference pressure filter (MMUSD)
    PF_CEPCI = 381.8            # CEPCI for reference pressure filter

    ### thermal dryer
    TD_scale_factor = 0.6       # scaling factor for purchase of thermal dryer
    TD_base_size = 5.182*365    # size of reference thermal dryer
    TD_base_price = 0.706427    # installed cost of reference thermal dryer
    TD_CEPCI = 539.1            # CEPCI for reference thermal dryer

    ### cogeneration and amine
    CGA_scale_factor = [0.8]    # scaling factor for cogeneration and amine unit
    CGA_base_size = [620*365]   # size of reference cogeneration and amine unit
    CGA_base_price = [13.1352]  # installed cost of reference cogeneration and amine unit
    CGA_CEPCI = [539.1, 444.2]  # CEPCI for reference cogeneration and amine unit
    
    
    for i, (x, y) in enumerate(zip(X, Y)):
        y[y<0] = 0
        
        SV = x[1]
        t_batch = x[2]
        P_dem = x[3]
        
        PBR_volume = y[0]
        PBR_titer = y[1]
        
        ## System setup and simulation
        Digester = DIGEST(M_in, y_MtoBG, x_BG, AD_CEPCI)
        Digester.mass_bal()
        Ct_Digester = Digester.econ()

        SLS = SLD_SEP(Digester.m_D, P_in, N_in, x_TS, SLS_CEPCI)
        SLS.mass_bal()
        Ct_SLS = SLS.econ()

        CGA = COGEN_AMINE([Digester.m_CH4, Digester.m_CO2, Digester.m_H2S], x_CH4mark, Ct_Digester,
                          CGA_scale_factor, CGA_base_size, CGA_base_price, CGA_CEPCI)
        CGA.mass_bal()
        Ct_CGA = CGA.econ()
        
        if y[0] < 0:
            print(y[0])
            print(x)
            raise Exception("Volume cannot be negative")
            
        if y[1] < 0:
            print(y[1])
            print(x)
            raise Exception("Titer cannot be negative")
        
        PBR_surface_area = PBR_volume*rho_eff/sigma_a/acre2meter
        PBR_volume_flow = PBR_volume/(t_batch/day2second)*oprD
        PBR_mass_flow = PBR_volume_flow*rho_eff
        PBR_cb_flow = PBR_volume_flow*PBR_titer/gPERL_tonnePERm3
        
        # PBR_surface_area = PBR_volume*rho_eff/sigma_a/acre2meter
        # PBR_cb_flow = SLS.m_PL/P_dem
        # PBR_volume_flow = PBR_cb_flow/(PBR_titer/gPERL_tonnePERm3)
        # PBR_mass_flow = PBR_volume_flow*rho_eff
        
        PBR_NS = max(0, min(0.05, P_dem*pbr_NtoP)*PBR_cb_flow-SLS.m_NL)
        Ct_PBR = PBR_base_price*(CEPCI_2020/PBR_CEPCI)*(PBR_surface_area/PBR_base_size)**PBR_scale_factor
        
        if PBR_cb_flow < 0:
            print(PBR_cb_flow)
            print(x)
            raise Exception("Mass flow cannot be negative")
            
        if PBR_surface_area < 0:
            print(PBR_surface_area)
            print(x)
            raise Exception("Area cannot be negative")

        Floc = FLOCCULATION_TANK(PBR_mass_flow, PBR_cb_flow, floc_req,
                                 FLOC_scale_factor, FLOC_base_size, FLOC_base_price, FLOC_CEPCI)
        Floc.mass_bal()
        Ct_FlocculationTank = Floc.econ()

        Lam = LAMELLA(Floc.m_BM, Floc.m_cb, x_cbL,
                      LAM_scale_factor, LAM_base_size, LAM_base_price, LAM_CEPCI)
        Lam.mass_bal()
        Ct_Lamella = Lam.econ()
        m_innxt = Lam.m_BM

        PF = PRESSURE_FILTER(m_innxt, PBR_cb_flow, x_cbPF,
                             PF_scale_factor, PF_base_size, PF_base_price, PF_CEPCI)
        PF.mass_bal()
        Ct_PressureFilter = PF.econ()

        TD = THERMAL_DRYER(PF.m_BM, PBR_cb_flow, x_cbTD,
                           TD_scale_factor, TD_base_size, TD_base_price, TD_CEPCI)
        TD.mass_bal()
        Ct_ThermalDryer = TD.econ()
        
        
        ## Capital Costs
        ISBL = Ct_Digester+Ct_SLS+Ct_CGA+Ct_PBR+Ct_FlocculationTank+Ct_Lamella+Ct_PressureFilter+Ct_ThermalDryer 
        OSBL = 0.4*ISBL
        ENG = (ISBL+OSBL)*0.3
        CONT = (ISBL+OSBL)*0.2
        LAND = 5890*PBR_surface_area/1e6
        TCI = ISBL+OSBL+ENG+CONT+LAND
    
        ### fixed operating costs
        maintenance_costs = p['maint']*ISBL
        operations_costs = p['op']*ISBL
        overhead_costs = p['ovhd']*ISBL
        depreciation = ISBL/project_life
    
        ### digester, SLS, cogen and amine units operating costs
        digester_cost = 0.096*Ct_Digester
        sls_cost = (0.488*SLS.m_in+0.1*(1786.9*np.log(SLS.m_lb)-9506.6))/1e6
        amine_cost = p['co2_rem']/1e6*CGA.m_CO2+p['h2s_rem']/1e6*Digester.m_BG*1000
    
        ### b-PBR operating costs
        pbr_replacement_cost = p['bags']*(PBR_surface_area)*max(1, 3*(x[1]/SV_0)-2)  # scale cost based on SV to reflect cost of making more complex geometries
        inoculum_cost = p['inoc']*(PBR_surface_area)/1 ###
        mixing_cost = p['mix']*(PBR_surface_area*acre2meter/1e4)*oprD*max(1, 3*(x[1]/SV_0)-2)  # scale cost based on SV to reflect cost of turbulence induced by more complex
        water_delivery_cost = p['wat_del']*(PBR_surface_area*acre2meter/1e4)*oprD
        urea_cost = p['urea']*PBR_NS/0.4666666666666667
    
        ### separation train operating costs
        flocculation_tank_cost = p['floc']*Floc.m_cb
        lamella_cost = p['lam']*Lam.m_cb
        pressure_filter_cost = p['pf']*PF.m_in
        dryer_cost = p['nat_gas']/1e6*(TD.m_W*1000*lb_kg*btu_h2o)/1e6
    
        ### labor costs
        labor_penalty = 5e4*(max(0, 30-x[2]/day2second))/1e6
        labor = p['labor']*PBR_surface_area+labor_penalty
    
        ### total operating costs
        VOC = digester_cost+sls_cost+amine_cost+\
              pbr_replacement_cost+inoculum_cost+mixing_cost+water_delivery_cost+urea_cost+\
              flocculation_tank_cost+lamella_cost+pressure_filter_cost+dryer_cost+labor
        FOC = maintenance_costs+operations_costs+overhead_costs+depreciation
        TOC = VOC+FOC
     
        
        ## MSP Calculation
        if max_prod is not None:
            m_cb = min(max_prod, PBR_cb_flow)
        else:
            m_cb = PBR_cb_flow
        
        msp_args = (p,
                    [CGA],
                    np.array([P_in, m_cb, depreciation, TCI, TOC, project_life]),
                    np.array([rho_CH4, ft3_m3, SCF_BTU, BTU_kWh, eta_gasturbine]))
        msps[i] = fsolve(MSP, initial_guess, args = msp_args)
        #cb_value = (1486*SLS.m_PL+1356*min(0.05, P_dem*pbr_NtoP)*PBR_cb_flow)/PBR_cb_flow
        msps[i] = 1e0*msps[i]#/cb_value
        
    return msps


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


def gp_sim_ind(x, y_mod, mu, sigma, x_idx, y_idx):
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

        mu_r[f'{i+1}'] = np.ones((len(x), len(y_mod[key])))
        sig_r[f'{i+1}'] = mu_r[f'{i+1}'].copy()
        for j, model in enumerate(y_mod[key]):
            mu_r[f'{i+1}'][:, j], sig_r[f'{i+1}'][:, j] =\
                model.predict(x_r[f'{i+1}'], return_std = True)


    return list(mu_r.values()), list(sig_r.values())


#%% BO RUNS SETUP
trials = 50
ub = np.array([I_0+1e-6, SV_reactor*1.25, t_batch*1.25, 0.154])
lb = np.array([I_0, SV_reactor*0.75, t_batch*0.75, 0.0128])
dim = len(ub)
bounds = Bounds(np.zeros(dim), np.ones(dim))
kernel = gpr.kernels.Matern(np.ones(dim), np.array([[1e-1, 1e3]]*dim), nu = 1.5)
exp_w = [2.6]
eps = 1e-3
init_pts = 2
nu = np.array([1.5, 1.5])
kernel_length_scale_bnds = np.array([[1e-1, 1e3], [1e-1, 1e3]])

feasible_lb = 1e-6*np.ones(2)
feasible_ub = np.inf*np.ones(2)
feasible_ub_opbo = np.array([1e6, 10])


idx = [[0, 1, 2],

       [np.array([0]),          # PBR volume
        np.array([0]),          # PBR titer
        ]]

idx_opbo = [[0, 1, 2],

            [0,                 # PBR volume
             1]]                # PBR titer

x_idx = [np.array([1, 2, 3]),   # PBR volume
         np.array([1, 2, 3]),   # PBR titer
         ]

y_idx = [[None,                 # PBR volume
          None,                 # PBR titer
          ],
         
         [None,                 # PBR volume
          None,                 # PBR titer
          ]]


args = (p, Feed, front_end_args, pbr_args, dewatering_train_args, max_prod)
args_dist = (p, Feed, front_end_args, pbr_args, dewatering_train_args, max_prod, True)
f_args = (p, Feed, front_end_args, [pbr_args[3], sigma_a], dewatering_train_args, max_prod)
gp_args = (x_idx, y_idx)

X = np.linspace(np.zeros(3), np.ones(3), 5)
X = np.meshgrid(*X.T)
X = np.reshape(X, (3, -1)).T
X = np.hstack([np.zeros((len(X), 1)), X])

F_SBO = np.ones((trials, len(X)))
X_SBO = np.ones((trials*len(X), dim))
F_BOIS = F_SBO.copy()
X_BOIS = X_SBO.copy()
F_MCBO = F_SBO.copy()
X_MCBO = X_SBO.copy()
F_OPBO = F_SBO.copy()
X_OPBO = X_SBO.copy()


TEA_OPT = BO_algos.BO(ub, lb, dim, exp_w[0], kernel, SYSTEM_BO, bounds, args = args)

for i, x0 in enumerate(X):
    TEA_OPT.system = SYSTEM_BO
    TEA_OPT.args = args
    TEA_OPT.exp_w = exp_w[0]
    TEA_OPT.optimizer_sbo(trials, x_init = x0, init_pts = 1)
    
    X_SBO[i*trials:(i+1)*trials, :] = TEA_OPT.x_sbo
    F_SBO[:, i] = TEA_OPT.y_sbo.flatten()
    np.savetxt('EFRI_SYSTEM_5x5x5_grid_series_x_sbo.txt', X_SBO)
    np.savetxt('EFRI_SYSTEM_5x5x5_grid_series_f_sbo.txt', F_SBO)


    TEA_OPT.system = SYSTEM_TEA
    TEA_OPT.exp_w = exp_w
    TEA_OPT.args = args_dist
    TEA_OPT.optimizer_bois(trials = trials, init_pts = init_pts, eps = eps,
                           idx = idx, x_idx = x_idx, y_idx = y_idx,
                           gp_sim = gp_sim, cost_fun = cost_fun,
                           restarts = 10, af_cores = 1,
                           f_args = f_args, gp_args = gp_args, 
                           x_init = TEA_OPT.scale(TEA_OPT.x_sbo[:2]),
                           norm_xdat = False, split_gps = False,
                           kernel_length_scale_bnds = kernel_length_scale_bnds, nu = nu,
                           feasibility_check = True, clip_to_bounds = True, 
                           feasible_lb = feasible_lb, feasible_ub = feasible_ub)
    
    X_BOIS[i*trials:(i+1)*trials, :] = TEA_OPT.x_bois
    F_BOIS[:, i] = TEA_OPT.f_bois.flatten()
    np.savetxt('EFRI_SYSTEM_5x5x5_grid_series_x_bois.txt', X_BOIS)
    np.savetxt('EFRI_SYSTEM_5x5x5_grid_series_f_bois.txt', F_BOIS)
    
    
    TEA_OPT.optimizer_mcbo(trials = trials, init_pts = init_pts, n_samples = 100,
                           idx = idx, x_idx = x_idx, y_idx = y_idx,
                           gp_sim = gp_sim, cost_fun = cost_fun,
                           restarts = 10, af_cores = 1,
                           f_args = f_args, gp_args = gp_args, 
                           x_init = TEA_OPT.scale(TEA_OPT.x_sbo[:2]),
                           norm_xdat = False, split_gps = False,
                           kernel_length_scale_bnds = kernel_length_scale_bnds, nu = nu,
                           feasibility_check = True, clip_to_bounds = True, 
                           feasible_lb = feasible_lb, feasible_ub = feasible_ub)
    
    X_MCBO[i*trials:(i+1)*trials, :] = TEA_OPT.x_mcbo
    F_MCBO[:, i] = TEA_OPT.f_mcbo.flatten()
    np.savetxt('EFRI_SYSTEM_5x5x5_grid_series_x_mcbo.txt', X_MCBO)
    np.savetxt('EFRI_SYSTEM_5x5x5_grid_series_f_mcbo.txt', F_MCBO)


    TEA_OPT.optimizer_optimism_bo(trials = trials, init_pts = init_pts,
                                  idx = idx_opbo, x_idx = x_idx, y_idx = y_idx,
                                  gp_sim = gp_sim, cost_fun = cost_fun,
                                  feasible_lb = feasible_lb, feasible_ub = feasible_ub_opbo,
                                  restarts = 10, af_cores = 1,
                                  f_args = f_args, gp_args = gp_args, 
                                  x_init = TEA_OPT.scale(TEA_OPT.x_sbo[:2]),
                                  norm_xdat = False, split_gps = False,
                                  kernel_length_scale_bnds = kernel_length_scale_bnds, nu = nu)
    
    X_OPBO[i*trials:(i+1)*trials, :] = TEA_OPT.x_opbo
    F_OPBO[:, i] = TEA_OPT.f_opbo.flatten()
    np.savetxt('EFRI_SYSTEM_5x5x5_grid_series_x_opbo.txt', X_OPBO)
    np.savetxt('EFRI_SYSTEM_5x5x5_grid_series_f_opbo.txt', F_OPBO)

    
    print('RESTART IS AT 'f'{i+1}, BEST S-BO, BOIS, MC-BO, and OP-BO VALUES ARE 'f'{np.min(TEA_OPT.y_sbo):.3f},'\
          f'{np.min(TEA_OPT.f_bois):.3f},'\
          f'{np.min(TEA_OPT.f_mcbo):.3f},'\
          '& ' f'{np.min(TEA_OPT.f_opbo):.3f},')
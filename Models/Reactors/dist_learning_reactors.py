from numpy import arange, random, array, argmin, vstack, atleast_1d, round
from numpy import hstack, exp, matmul, log, ones, linalg, transpose as trans
from numpy import meshgrid, concatenate, linspace, mean, std, max as nmax, min as nmin
from matplotlib import pyplot as pyp,cm
from scipy.optimize import minimize, Bounds, fsolve
from joblib import Parallel, delayed
import sklearn.gaussian_process as gpr
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
import time

exp_w=2.6; t=35; bnds=Bounds(-1,1); bnds2=Bounds((-1,-1),(1,1)); noise=1e-6
F=90; T=423; lp=1;
CBEST=0*ones((t+1,1)); CBESTGP=0*ones((t+1,1)); idx=ones((t,1),dtype=int);
m=2/450; b=-1-313*m; mf=5; bf=95;

##################Initial Conditions and Physical Parameters##################
## FIRST REACTOR
# Physicanl, Kinetic, and Thermodynamic parameters 
k01=array([1000,1250]); Ea1=array([32000,35000]); H1=array([-210000,-1700000])
ρ1=850; Cp1=3000; CpH2O=4184; R=8.314;
# Inlet Specs and Reactor Size
C01=array([5000,0,0]); Tin1=298; V1=1000; TinH2O=298

## SECOND REACTOR
# Physicanl, Kinetic, and Thermodynamic parameters
k02=array([1000,1250,950]); Ea2=array([32000,35000,40000]); H2=array([-210000,-1700000,-1500000])
ρ2=950; Cp2=4000;
# Inlet Specs and Reactor Size
Tin2f=298; V2=1500; CR02=1000 

## RECYCLE REACTOR NETWORK
# Physicanl, Kinetic, and Thermodynamic parameters
k0R=array([1000,1250,950,1100]); EaR=array([32000,35000,40000,33000]);
HR=array([-210000,-1700000,-1500000,-350000])
ρR=array([850,950,875]); CpR=array([3000,4000,2900]);
# Inlet Specs and Reactor Size
TinR=array([298,298,398]); VR=array([1000,1500]);
C0F1=array([5000,0,0,0,0,0])
C0F2=array([0,0,0,1000,0,0])
C0F3=array([0,0,0,0,0,1000])

## Economic Parameters
cB=-0.35; cE=-0.3; cA=0.12; cC=0.075; cD=0.15; cF=0.05; cT=0.0035;
c1=array([cA,cB,cC,cT])
c2=array([cA,cB,cC,cD,cE,cT])
c3=array([cA,cB,cC,cD,cE,cF,cT])

## Initial Guesses
Cinit1=array([5,2260,475])
Cinit2=array([5,1500,400,5,400])
Cinit3=array([80,1300,200,10,130,120])
##############################################################################
#T range is 313 to 763
#F range is 90 to 100
#Units are all in MKH (Hours) 

## REACTOR CLASS 
class CSTR():
    def __init__(self,T,F,C0,k0,Ea,Cinit,ρ,Cp,H,Tin,c,V,bnds,rev_ratio=100):
        self.T=T.reshape(-1,1)
        self.F=F.reshape(-1,1)
        self.F,self.T=meshgrid(self.F,self.T)
        self.T=self.T.flatten().reshape(-1,1)
        self.F=self.F.flatten().reshape(-1,1)
        self.Cin=C0
        self.k0=k0
        self.Ea=Ea
        self.Cinit=Cinit.reshape(Cinit.shape[0])
        self.p=ρ
        self.Cp=Cp
        self.H=H
        self.Tin=Tin
        self.c=c
        self.V=V
        self.bounds=bnds
        self.rev_ratio=rev_ratio
        self.rxn_num=self.k0.shape[0]
        self.Cf=ones((self.T.shape[0],self.Cinit.shape[0]))
        self.rxn=ones((self.T.shape[0],self.rxn_num))
        self.mH2O=ones((self.T.shape[0]))
    def MandE_bal(self,m_bal,q_bal):
        for i in range(self.T.shape[0]):
            self.k={}
            for j in range(self.rxn_num):
                self.k['k'+str(j+1)]=self.k0[j]*exp(-self.Ea[j]/(R*(self.T[i])))
                self.k['k'+str(j+1)+'r']=self.k['k'+str(j+1)]/self.rev_ratio;
            def C(C0):
                self.C=m_bal(self.F[i],C0,self.Cin,self.k,self.V).reshape(self.Cinit.shape[0])
                return self.C-C0
            self.soln=fsolve(C,self.Cinit)
            self.Cf[i,:]=self.soln
            self.Cinit=self.soln.copy()
            self.soln=q_bal(float(self.T[i]),float(self.F[i]),self.Cinit,self.k,self.p,self.Cp,self.H,self.Tin,self.V)
            self.rxn[i,:]=self.soln[0:self.rxn_num]
            self.mH2O[i]=self.soln[-1]
    def Econ(self,econ):
        self.Ct=econ(self.c,self.F,self.Cf,self.Cin,self.mH2O)/1e4
        return self.Ct

# MASS BALANCES
def mass_bal1(F,C,C0,k,V):
    #Cr=((1/(2*V1*k['k1']))*(F*(C0[0]-C[0])+2*k['k1r']*C[1]*V1))**(1/2) # If the quadratic isn't allowing convergence, use this seems to be fairly robust for determining Cr
    Cr=(-F/(2*k['k1']*V)+((F/(2*k['k1']*V))**2+4*(F*C0[0]/(2*k['k1']*V)+\
        C[1]*k['k1r']/k['k1']))**0.5)/2
    Ci=(k['k1']*Cr**2*V+k['k2r']*C[2]**2*V)/(F+k['k1r']*V+k['k2']*V)
    Cpd=(2*k['k2']*Ci*V-2*k['k2r']*V*C[2]**2)/(F)
    return concatenate([Cr,Ci,Cpd])

def mass_bal2(F,C,C0,k,V):
    #Cr=((1/(2*V2*k['k1']))*(F*(C0[0]-C[0])+2*k['k1r']*C[1]*V2))**(1/2) # If the quadratic isn't allowing convergence, use this seems to be fairly robust for determining Cr
    Cr=(-F/(2*k['k1']*V)+((F/(2*k['k1']*V))**2+4*(F*C0[0]/(2*k['k1']*V)+\
        C[1]*k['k1r']/k['k1']))**0.5)/2
    Ci=(F*C0[1]+k['k1']*Cr**2*V+k['k2r']*C[2]**2*V)/(F+k['k1r']*V+k['k2']*V)
    Cpd=(F*C0[2]+2*k['k2']*Ci*V-2*k['k2r']*V*C[2]**2-k['k3']*C[2]*C[3]*V+\
          k['k3r']*C[4]*V)/(F)
    Cr2=(F*C0[3]+k['k3r']*C[4]*V)/(F+k['k3']*C[2]*V)
    Ci2=(k['k3']*C[2]*C[3]*V)/(F+k['k3r']*V2)
    return concatenate([Cr,Ci,Cpd,Cr2,Ci2])

def mass_bal_recycle(F,C,C0,k,V):
    Ca = C0[0]-2*(k['k1']*C[0]**2-k['k1r']*C[1]-k['k4']*C[2]*C[5])*V/F
    Cb = C0[1]+(k['k1']*C[0]**2-k['k1r']*C[1]-k['k2']*C[1]+k['k2r']*C[2]**2)*V/F
    Cc = C0[2]+(2*(k['k2']*C[1]-k['k2r']*C[2]**2)-k['k3']*C[2]*C[3]+\
            k['k3r']*C[4]-k['k4']*C[2]*C[5])*V/F
    Cd = C0[3]-(k['k3']*C[2]*C[3]-k['k3r']*C[4])*V/F
    Ce = C0[4]+(k['k3']*C[2]*C[3]-k['k3r']*C[4])*V/F
    Cf = C0[5]-k['k4']*C[2]*C[5]*V/F
    return concatenate([Ca, Cb, Cc, Cd, Ce, Cf])

# ENERGY BALANCES
def heat_bal1(T,F,C,k,ρ,Cp,H,Tin,V):
    r1=k['k1']*C[0]**2-k['k1r']*C[1]
    r2=k['k2']*C[1]-k['k2r']*C[2]**2
    Qdot=-r1*H[0]*V-r2*H[1]*V
    ToH2O=min(T-10,323)
    mH2O=(ρ*Cp*F*(Tin-T)+Qdot)/(CpH2O*(ToH2O-TinH2O))
    if mH2O<0:
        mH2O=array([0])
    return concatenate([r1,r2,mH2O])

def heat_bal2(T,F,C,k,ρ,Cp,H,Tin,V):
    r1=k['k1']*C[0]**2-k['k1r']*C[1]
    r2=k['k2']*C[1]-k['k2r']*C[2]**2
    r3=k['k3']*C[2]*C[3]-k['k3r']*C[4]
    Qdot=-r1*H[0]*V-r2*H[1]*V2-r3*H[2]*V
    ToH2O=min(T-10,323)
    mH2O=(ρ*Cp*F*(Tin-T)+Qdot)/(CpH2O*(ToH2O-TinH2O))
    if mH2O<0:
        mH2O=array([0])
    return concatenate([r1,r2,r3,mH2O])

def heat_bal_recycle(T,F,C,k,ρ,Cp,H,Tin,V):
    r1=k['k1']*C[0]**2-k['k1r']*C[1]
    r2=k['k2']*C[1]-k['k2r']*C[2]**2
    r3=k['k3']*C[2]*C[3]-k['k3r']*C[4]
    r4=k['k4']*C[2]*C[5]
    Qdot=-(r1*H[0]+r2*H[1]+r3*H[2]+r4*H[3])*V
    ToH2O=min(T-10,323)
    mH2O=(ρ*Cp*F*(Tin-T)+Qdot)/(CpH2O*(ToH2O-TinH2O))
    if mH2O<0:
        mH2O=array([0])
    return concatenate([r1,r2,r3,r4,mH2O])

# ECONOMICS
def econ1(c,F,C,C0,mH2O):
    Ct=c[0]*(C[:,0]+C0[0])*F[:,0]+c[1]*C[:,1]*F[:,0]+c[2]*C[:,2]*F[:,0]+c[3]*mH2O
    return Ct

def econ2(c,F,C,C0,mH2O):
    Ct=c[0]*C[:,0]*F[:,0]+c[1]*C[:,1]*F[:,0]+c[2]*C[:,2]*F[:,0]+\
        c[3]*(C[:,3]+C0[3])*F[:,0]+c[4]*C[:,4]*F[:,0]+c[5]*mH2O
    return Ct

def econ_recycle1(c,F,C,C0,mH2O):
    Ct=c[0]*(C[:,0]+C0[0])*F[:,0]+c[1]*C[:,1]*F[:,0]+c[2]*C[:,2]*F[:,0]+\
        c[3]*(C[:,3])*F[:,0]+c[4]*(C[:,4])*F[:,0]+\
        c[5]*(C[:,5]+C0[5])*F[:,0]+c[6]*mH2O
    return Ct

def econ_recycle2(c,F,C,C0,mH2O):
    Ct=c[0]*(C[:,0])*F[:,0]+c[1]*C[:,1]*F[:,0]+c[2]*C[:,2]*F[:,0]+\
        c[3]*(C[:,3]+C0[3])*F[:,0]+c[4]*(C[:,4])*F[:,0]+\
        c[5]*(C[:,5])*F[:,0]+c[6]*mH2O
    return Ct

## MIXER CLASS
class MIXER():
    def __init__(self,Cin,Fin,Tin,ρ,Cp,Tref=298,all_C0=False):
        self.Cin=Cin
        self.Fin=Fin
        self.Tin=Tin
        self.p=ρ
        self.Cp=Cp
        self.Tref=Tref
        self.m_in=ρ*Fin
        self.xin=self.m_in/sum(self.m_in)
        self.Cpmix=sum(self.Cp*self.xin)
        self.pmix=sum(self.p*self.xin)
        self.all_C0=all_C0
    def mass_bal(self):
        self.Nin={}
        for i in range(self.Fin.shape[0]):
            self.Nin['N'+str(i+1)]=(self.Fin[i]*self.Cin[i])
        if not self.all_C0:
            self.Nin=array(list(self.Nin.values()),dtype=tuple)
            self.Nin=hstack(self.Nin[:]).astype(float)
        else:
            self.Nin=sum(array(list(self.Nin.values())))
        self.mout=sum(self.p*self.Fin)
        self.Fout=self.mout/self.pmix
        self.Cout=self.Nin/self.Fout
        return self.Cout
    def heat_bal(self):
        self.Tout=self.Tref+sum(self.xin*self.Cp*(self.Tin-self.Tref))/self.Cpmix
        return self.Tout
    
## SPLITTER CLASS
class SPLITTER():
    def __init__(self, Fin, split, split_frac = True):
        self.Fin=Fin
        if split_frac:
            self.split_frac=split
        else:
            self.split_frac=split/self.Fin
        self.split_orig = 1-sum(self.split_frac)
    def split(self):
        self.Fout=concatenate([self.split_frac*self.Fin, self.split_orig*self.Fin])
        return self.Fout

T=arange(313,764,1).reshape(-1,1)
F=arange(90,100.02,0.02).reshape(-1,1)
F2=50

T1,T2=meshgrid(T,T)
d1=T1.shape[0]; d2=T1.shape[1]
T1=T1.flatten().reshape(-1,1)
T2=T2.flatten().reshape(-1,1)
TT=hstack([T1,T2])

#%% TRIAL RUN
# Reactors in Series
Reac1=CSTR(T,array([100]),C01,k01,Ea1,Cinit1,ρ1,Cp1,H1,Tin1,c1,V1,
           Bounds((0,0,0),(2500,2500,2500)))
Reac1.MandE_bal(mass_bal1,heat_bal1)
Ct1=Reac1.Econ(econ1)

bst=argmin(Ct1)
C02mix=array([Reac1.Cf[bst],array([CR02])],dtype=tuple)
Tmix=array([Reac1.T[bst,0],Tin2f])
Fmix=array([Reac1.F[bst,0],F2])
pmix=array([ρ1,ρ2])
Cpmix=array([Cp1,Cp2])
Mix=MIXER(C02mix,Fmix,Tmix,pmix,Cpmix)
C02=Mix.mass_bal()
Tin2=Mix.heat_bal()
ρ3=Mix.pmix
Cp3=Mix.Cpmix
F3=Mix.Fout

Reac2=CSTR(T,array([F3]),C02,k02,Ea2,Cinit2,ρ3,Cp3,H2,Tin2,c2,V2,
           Bounds((0,0,0,0,0),(2500,2500,2500,2500,2500)))
Reac2.MandE_bal(mass_bal2,heat_bal2)
Ct2=Reac2.Econ(econ2)

fig,ax1=pyp.subplots(1,1,figsize=(11,8.5));
ax1.grid(color='gray',axis='both',alpha=0.25); ax1.set_axisbelow(True);
ax1.set_xlim((T[0],T[-1])); #ax1.set_ylim((-1.8,-0.6));
ax1.set_xlabel(r'$Temperature (K)$',fontsize=24); pyp.xticks(fontsize=24);
ax1.set_ylabel(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24); pyp.yticks(fontsize=24);
ax1.plot(T,Ct1+Ct2,color='blue',linewidth=3); 
pyp.savefig('Rxtr_covergResults.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

# Reactors with Recycle
FR = array([100, 50, 50])
TR = array([637, 757]).reshape(-1,1)
R_Frac = 0.1

def Rxtr_Recycle(FR, TR, R_Frac, rxn_reg = False):
    err = 1e6
    FinR3 = FR[-1]
    while err > 1e-6:
        ρR3 = (FinR3/(FinR3+FR[2]))*ρR[1]+(FR[2]/(FinR3+FR[2]))*ρR[2]
        Fr = R_Frac*(ρR[0]*FR[0]+ρR[1]*FR[1]+ρR[2]*FR[2])/((1-R_Frac)*ρR3)
        err = Fr - FinR3
        FinR3 = Fr*1
    CinR = vstack([C0F1, C0F3])
    CpR3 = (FinR3/(FinR3+FR[2]))*CpR[1]+(FR[2]/(FinR3+FR[2]))*CpR[2]
    TinR3 = (FR[2]*CpR[2]*TinR[-1]+FinR3*CpR3*TR[1,0])/(FinR3*CpR3+FR[2]*CpR[2])
    err = 1e6
    FinR3 = FinR3+FR[2]
    i = 0
    while err > 1e-6:
        MixR1 = MIXER(CinR, array([FR[0], FinR3]), array([TinR[0], TinR3]), array([ρR[0], ρR3]), array([CpR[0], CpR3]), all_C0=True)
        C0R1 = MixR1.mass_bal()
        TinR1 = MixR1.heat_bal()
        FinR1 = MixR1.Fout
        CpR1 = MixR1.Cpmix
        ρR1 = MixR1.pmix
        ReacR1 = CSTR(TR[0], FinR1, C0R1, k0R, EaR, Cinit3, ρR1, CpR1, HR, TinR1, c3, V1,
                      Bounds((0,0,0,0,0,0),(2500,2500,2500,2500,2500,1000)))
        ReacR1.MandE_bal(mass_bal_recycle, heat_bal_recycle)
        CinR2 = vstack([ReacR1.Cf, C0F2])
        MixR2 = MIXER(CinR2, array([ReacR1.F[0,0], FR[1]]), array([ReacR1.T[0,0], TinR[1]]), array([ρR1, ρR[1]]), array([CpR1, CpR[1]]), all_C0=True)
        C0R2 = MixR2.mass_bal()
        TinR2 = MixR2.heat_bal()
        FinR2 = MixR2.Fout
        CpR2 = MixR2.Cpmix
        ρR2 = MixR2.pmix
        ReacR2 = CSTR(TR[1], FinR2, C0R2, k0R, EaR, Cinit3, ρR2, CpR2, HR, TinR2, c3, V2,
                      Bounds((0,0,0,0,0,0),(2500,2500,2500,2500,2500,1000)))
        ReacR2.MandE_bal(mass_bal_recycle, heat_bal_recycle)
        SplitR1 = SPLITTER(ReacR2.F, array([0.1]))
        Fr, Fpd = SplitR1.split()
        CinR3 = vstack([ReacR2.Cf, C0F3])
        MixR3 = MIXER(CinR3, array([Fr[0], FR[2]]), array([ReacR2.T[0,0], TinR[-1]]), array([ρR2, ρR[-1]]), array([CpR2, CpR[-1]]), all_C0 = True)
        C0R3 = MixR3.mass_bal()
        TinR3 = MixR3.heat_bal()
        FinR3 = MixR3.Fout
        CpR3 = MixR3.Cpmix
        ρR3 = MixR3.pmix
        CinR = vstack([C0F1, C0R3])
        err = (Fpd*ReacR2.p-(FR[0]*ρR[0]+FR[1]*ρR[1]+FR[2]*ρR[2]))**2
        i += 1
    if rxn_reg:
        R1_rxn = ReacR1.soln[:-1]
        R2_rxn = ReacR2.soln[:-1]
        return R1_rxn, R2_rxn
    else:
        CtR1 = ReacR1.Econ(econ_recycle1)
        CtR2 = ReacR2.Econ(econ_recycle2)
        return CtR1, CtR2

CtR1, CtR2 = Rxtr_Recycle(FR, TR, R_Frac, CSTR)

#%% T1 and T2 system formulation and visualization
def SYST(T,rxn_reg=False):
    T=T.reshape(-1,1)
    T1=T[0]
    T2=T[1]
    F1=array([100]);
    F2=array([50]);
    Reac1=CSTR(T1,F1,C01,k01,Ea1,Cinit1,ρ1,Cp1,H1,Tin1,c1,V1,
               Bounds((0,0,0),(2500,2500,2500)))
    Reac1.MandE_bal(mass_bal1,heat_bal1)
    Ct1=Reac1.Econ(econ1)
    C02mix=array([Reac1.Cf[0],array([CR02])],dtype=tuple)
    Tmix=array([Reac1.T[0,0],Tin2f])
    Fmix=array([Reac1.F[0,0],F2[0]])
    pmix=array([ρ1,ρ2])
    Cpmix=array([Cp1,Cp2])
    Mix=MIXER(C02mix,Fmix,Tmix,pmix,Cpmix)
    C02=Mix.mass_bal()
    Tin2=Mix.heat_bal()
    ρ3=Mix.pmix
    Cp3=Mix.Cpmix
    F3=Mix.Fout
    Reac2=CSTR(T2,array([F3]),C02,k02,Ea2,Cinit2,ρ3,Cp3,H2,Tin2,c2,V2,
               Bounds((0,0,0,0,0),(2500,2500,2500,2500,2500)))
    Reac2.MandE_bal(mass_bal2,heat_bal2)
    Ct2=Reac2.Econ(econ2)
    Ct=Ct1+Ct2
    if rxn_reg:
        r11,r12=Reac1.soln[0:-1]
        r21,r22,r23=Reac2.soln[0:-1]
        return r11, r12, r21, r22, r23
    else:
        return Ct1,Ct2,Ct

# print('System calculation...')
# start = time.time()
# CT = Parallel(n_jobs = -1)(delayed(SYST)(start_point) for start_point in TT)
# end = time.time()
# print(end-start)
# CT = hstack(CT[:]).T

# fig3D=pyp.figure(figsize=[10.5,8.5])
# ax3D1=fig3D.add_subplot(111);
# ax3D1.grid(color='gray',axis='both',alpha=0.25); ax3D1.set_axisbelow(True);
# fig1=ax3D1.contourf(T1.reshape(d1,d2),T2.reshape(d1,d2),CT[:,2].reshape(d1,d2),cmap=cm.jet);
# ax3D1.scatter(T1.flatten()[argmin(CT[:,2])],T2.flatten()[argmin(CT[:,2])],color='white',edgecolor='k',marker='o',s=100);
# cbar=pyp.colorbar(fig1);
# cbar.set_label(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,labelpad=15);
# cbar.ax.tick_params(labelsize=20)
# pyp.xlabel(r'$T_1\ (K)$',fontsize=24);
# pyp.xticks([325,425,525,625,725]);
# pyp.ylabel(r'$T_2\ (K)$',fontsize=24);
# pyp.yticks([325,425,525,625,725]);
# pyp.xlim((313,763)); pyp.ylim((313,763));
# ax3D1.tick_params(axis='both',which='major',labelsize=20);
# pyp.title(r'$Reactor\ Cost\ Function\ f({\bf x})$',fontsize=24,pad=15);

# fig3Dp=pyp.figure(figsize=[10.5,8.5])
# ax3D1p=fig3Dp.add_subplot(111,projection='3d')
# ax3D1p.grid(color='gray',axis='both',alpha=0.25); ax3D1p.set_axisbelow(True);
# fig1p=ax3D1p.plot_surface(T1.reshape(d1,d2),T2.reshape(d1,d2),CT[:,2].reshape(d1,d2),rstride=1,
#                           cstride=1,linewidth=0,antialiased=False,cmap=cm.jet);
# #ax3D1p.scatter(T1.flatten()[argmin(CT)],T2.flatten()[argmin(CT)],min(CT),color='white',edgecolor='black',s=75);
# ax3D1p.view_init(41,69);
# ax3D1p.xaxis.set_rotate_label(False)
# pyp.xlabel(r'$T_1\ (K)$',fontsize=24,rotation=0,labelpad=20);
# ax3D1p.yaxis.set_rotate_label(False)
# pyp.ylabel(r'$T_2\ (K)$',fontsize=24,rotation=0,labelpad=30);
# ax3D1p.zaxis.set_rotate_label(False)
# ax3D1p.set_zlabel(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,rotation=90,
#                   labelpad=38);
# pyp.xlim((313,763)); pyp.ylim((313,763)); #ax3D1p.set_zlim((0,85)); 
# pyp.xticks([325,425,525,625,725]);
# pyp.yticks([325,425,525,625,725]);
# #ax3D1p.set_zticks([-1.6,-1.45,-1.3,-1.15,-1.0,-0.85]);
# ax3D1p.tick_params(axis='both',which='major',labelsize=20);
# ax3D1p.tick_params(axis='z',pad=20)
# pyp.title(r'$Reactor\ Cost\ Function\ f({\bf x})$',fontsize=24,pad=35)

#%% REFERENCE MODEL FORMULATION
# Multivariate Reaction Regressions
TTr=random.uniform(313,763,(200,2))
CTR=lambda T:SYST(T,rxn_reg=True)
CTR=array(list(map(CTR,TTr))).reshape(TTr.shape[0],5)
# Inverse T relation
Am=ones(TTr.shape[0]).reshape(-1,1)
Am=hstack([1/TTr,Am])
logCTR=log(CTR)
pseudoAinvm=matmul(linalg.inv(matmul(trans(Am),Am)),trans(Am))
thetam=ones((CTR.shape[1],Am.shape[1]))
for i in range(thetam.shape[0]):
    thetam[i]=matmul(pseudoAinvm,logCTR[:,i])
# Reference Model Reactor
class CSTR_REF():
    def __init__(self,T1,T2,F,C0,ρ,Cp,H,Tin,c,V,thetam,rev_ratio=100,rxt_num=1):
        self.T1=T1.reshape(-1,1)
        self.F=F.reshape(-1,1)
        self.F,self.T1=meshgrid(self.F,self.T1)
        self.T1=self.T1.flatten().reshape(-1,1)
        self.F=self.F.flatten().reshape(-1,1)
        self.T2=T2*ones(self.T1.shape)
        self.Cin=C0
        self.p=ρ
        self.Cp=Cp
        self.H=H
        self.Tin=Tin
        self.c=c
        self.V=V
        self.thetam=thetam
        self.rev_ratio=rev_ratio
        self.rxt_num=rxt_num
        self.rxn_num=self.thetam.shape[0]
    def MandE_bal(self,m_bal,q_bal):
        self.Cf=m_bal(hstack([self.T1,self.T2]),self.F[:,0],self.Cin,self.V,self.thetam,self.rxt_num)
        self.Cf=trans(self.Cf)
        self.soln=q_bal(hstack([self.T1,self.T2]),self.F[:,0],self.p,self.Cp,self.H,self.Tin,self.V,self.thetam,self.rxt_num)
        self.rxn=hstack(self.soln[0:self.rxn_num])
        self.mH2O=self.soln[-1]
    def Econ(self,econ):
        self.Ct=econ(self.c,self.F,self.Cf,self.Cin,self.mH2O)/1e4
        return self.Ct
# Regressed Mass Balances
def mass_bal1_reg(T,F,C0,V,thetam,rxt_num):
    T1=T[:,0]
    T2=T[:,1]
    r1=exp(thetam[0,0]/T1+thetam[0,1]/T2+thetam[0,2])
    r2=exp(thetam[1,0]/T1+thetam[1,1]/T2+thetam[1,2])
    Cr=C0[0]-2*r1*V/F
    Ci=C0[1]+(r1-r2)*V/F
    Cpd=C0[2]+2*r2*V/F
    return array([Cr,Ci,Cpd])

def mass_bal2_reg(T,F,C0,V,thetam,rxt_num):
    T1=T[:,1]
    T2=T[:,0]
    r1=exp(thetam[0,0]/T1+thetam[0,1]/T2+thetam[0,2])
    r2=exp(thetam[1,0]/T1+thetam[1,1]/T2+thetam[1,2])
    r3=exp(thetam[2,0]/T1+thetam[2,1]/T2+thetam[2,2])
    Cr=C0[0]-2*r1*V/F
    Ci=C0[1]+(r1-r2)*V/F
    Cpd=C0[2]+(2*r2-r3)*V/F
    Cr2=C0[3]-r3*V/F
    Ci2=r3*V/F
    return array([Cr,Ci,Cpd,Cr2,Ci2])
# Regressed Energy Balances
def heat_bal1_reg(T,F,ρ,Cp,H,Tin,V,thetam,rxt_num):
    T1=T[:,0]
    T2=T[:,1]
    r1=exp(thetam[0,0]/T1+thetam[0,1]/T2+thetam[0,2])
    r2=exp(thetam[1,0]/T1+thetam[1,1]/T2+thetam[1,2])
    Qdot=-r1*H[0]*V-r2*H[1]*V
    ToH2O=ones(T1.shape)
    for i in range(T1.shape[0]):
        ToH2O[i]=min(T1[i]-10,323)
    mH2O=(ρ*Cp*F*(Tin-T1)+Qdot)/(CpH2O*(ToH2O-TinH2O))
    for i in range(mH2O.shape[0]):
        if mH2O[i]<0:
            mH2O[i]=0
    return r1,r2,mH2O

def heat_bal2_reg(T,F,ρ,Cp,H,Tin,V,thetam,rxt_num):
    T1=T[:,1]
    T2=T[:,0]
    r1=exp(thetam[0,0]/T1+thetam[0,1]/T2+thetam[0,2])
    r2=exp(thetam[1,0]/T1+thetam[1,1]/T2+thetam[1,2])
    r3=exp(thetam[2,0]/T1+thetam[2,1]/T2+thetam[2,2])
    Qdot=-r1*H[0]*V-r2*H[1]*V-r3*H[2]*V
    ToH2O=ones(T2.shape)
    for i in range(T2.shape[0]):
        ToH2O[i]=min(T2[i]-10,323)
    mH2O=(ρ*Cp*F*(Tin-T2)+Qdot)/(CpH2O*(ToH2O-TinH2O))
    for i in range(mH2O.shape[0]):
        if mH2O[i]<0:
            mH2O[i]=0
    return r1,r2,r3,mH2O
# Reference System Connect
def SYST_REF(T,rxn_reg=False):
    T=T.reshape(-1,1)
    T1=T[0]
    T2=T[1]
    F1=array([100]);
    F2=array([50]);
    Reac1_reg=CSTR_REF(T1,T2,F1,C01,ρ1,Cp1,H1,Tin1,c1,V1,thetam[0:2])
    Reac1_reg.MandE_bal(mass_bal1_reg,heat_bal1_reg)
    Ct1=Reac1_reg.Econ(econ1)
    C02mix=array([Reac1.Cf[0],array([CR02])],dtype=tuple)
    Tmix=array([Reac1.T[0,0],Tin2f])
    Fmix=array([Reac1.F[0,0],F2[0]])
    pmix=array([ρ1,ρ2])
    Cpmix=array([Cp1,Cp2])
    Mix=MIXER(C02mix,Fmix,Tmix,pmix,Cpmix)
    C02=Mix.mass_bal()
    Tin2=Mix.heat_bal()
    p3=Mix.pmix
    Cp3=Mix.Cpmix
    F3=Mix.Fout
    Reac2_reg=CSTR_REF(T2,T1,array([F3]),C02,p3,Cp3,H2,Tin2,c2,V2,thetam[2:thetam.shape[0]])
    Reac2_reg.MandE_bal(mass_bal2_reg,heat_bal2_reg)
    Ct2=Reac2_reg.Econ(econ2)
    Ct=Ct1+Ct2
    if rxn_reg:
        r11,r12=Reac1_reg.soln[0:-1]
        r21,r22,r23=Reac2_reg.soln[0:-1]
        return r11, r12, r21, r22, r23
    else:
        return Ct1,Ct2,Ct

# CT_REF=lambda T:SYST_REF(T)
# CT_REF=array(list(map(CT_REF,TT))).reshape(TT.shape[0],3)
# print('Reference model calculation...')
# start = time.time()
# CT_REF = Parallel(n_jobs = -1)(delayed(SYST_REF)(start_point) for start_point in TT)
# end = time.time()
# print(end-start)
# CT_REF = hstack(CT_REF[:]).T

# fig3D=pyp.figure(figsize=[10.5,8.5])
# ax3D1=fig3D.add_subplot(111);
# ax3D1.grid(color='gray',axis='both',alpha=0.25); ax3D1.set_axisbelow(True);
# fig1=ax3D1.contourf(T1.reshape(d1,d2),T2.reshape(d1,d2),CT_REF[:,2].reshape(d1,d2),cmap=cm.jet);
# ax3D1.scatter(T1.flatten()[argmin(CT)],T2.flatten()[argmin(CT)],color='white',edgecolor='k',marker='o',s=100);
# cbar=pyp.colorbar(fig1);
# cbar.set_label(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,labelpad=15);
# cbar.ax.tick_params(labelsize=20)
# pyp.xlabel(r'$T_1\ (K)$',fontsize=24);
# pyp.xticks([325,425,525,625,725]);
# pyp.ylabel(r'$T_2\ (K)$',fontsize=24);
# pyp.yticks([325,425,525,625,725]);
# pyp.xlim((313,763)); pyp.ylim((313,763));
# ax3D1.tick_params(axis='both',which='major',labelsize=20);
# pyp.title(r'$Approximated\ Reactor\ Cost\ Function\ f({\bf x})$',fontsize=24,pad=15);

# fig3Dp=pyp.figure(figsize=[10.5,8.5])
# ax3D1p=fig3Dp.add_subplot(111,projection='3d')
# ax3D1p.grid(color='gray',axis='both',alpha=0.25); ax3D1p.set_axisbelow(True);
# fig1p=ax3D1p.plot_surface(T1.reshape(d1,d2),T2.reshape(d1,d2),CT_REF[:,2].reshape(d1,d2),rstride=1,
#                           cstride=1,linewidth=0,antialiased=False,cmap=cm.jet);
# #ax3D1p.scatter(T1.flatten()[argmin(CT)],T2.flatten()[argmin(CT)],min(CT),color='white',edgecolor='black',s=75);
# ax3D1p.view_init(41,69);
# ax3D1p.xaxis.set_rotate_label(False)
# pyp.xlabel(r'$T_1\ (K)$',fontsize=24,rotation=0,labelpad=20);
# ax3D1p.yaxis.set_rotate_label(False)
# pyp.ylabel(r'$T_2\ (K)$',fontsize=24,rotation=0,labelpad=30);
# ax3D1p.zaxis.set_rotate_label(False)
# ax3D1p.set_zlabel(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,rotation=90,
#                   labelpad=38);
# pyp.xlim((313,763)); pyp.ylim((313,763)); #ax3D1p.set_zlim((0,85)); 
# pyp.xticks([325,425,525,625,725]);
# pyp.yticks([325,425,525,625,725]);
# #ax3D1p.set_zticks([-1.6,-1.45,-1.3,-1.15,-1.0,-0.85]);
# ax3D1p.tick_params(axis='both',which='major',labelsize=20);
# ax3D1p.tick_params(axis='z',pad=20)
# pyp.title(r'$Approximated\ Reactor\ Cost\ Function\ f({\bf x})$',fontsize=24,pad=35)

#%% RECYCLE SYSTEM MODEL as f(T1, T2) plots and definition
def SYST_RECYCLE(T, rxn_reg=False):
    T=T.reshape(-1,1)
    FR = array([100, 50, 50])
    R_Frac = 0.1
    if rxn_reg:
        R1_rxn, R2_rxn = Rxtr_Recycle(FR, T, R_Frac, rxn_reg = rxn_reg)
        return hstack([R1_rxn, R2_rxn]).reshape(-1,1).T
    else:
        CtR1, CtR2 = Rxtr_Recycle(FR, T, R_Frac)
        CtR = CtR1+CtR2
        return CtR1, CtR2, CtR

# print('Recycle system calculation...')
# start = time.time()
# CT_RECYL = Parallel(n_jobs = -1)(delayed(SYST_RECYCLE)(start_point) for start_point in TT)
# end = time.time()
# print(end-start)
# CT_RECYL = hstack(CT_RECYL[:]).T

# fig3D=pyp.figure(figsize=[10.5,8.5])
# ax3D1=fig3D.add_subplot(111);
# ax3D1.grid(color='gray',axis='both',alpha=0.25);
# ax3D1.set_axisbelow(True);
# fig1=ax3D1.contourf(T1.reshape(d1,d2),T2.reshape(d1,d2),
#                     CT_RECYL[:,-1].reshape(d1,d2),cmap=cm.jet);
# ax3D1.scatter(T1.flatten()[argmin(CT_RECYL[:,-1])],
#               T2.flatten()[argmin(CT_RECYL[:,-1])],
#               color='white',edgecolor='k',marker='o',s=100);
# cbar=pyp.colorbar(fig1);
# cbar.set_label(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,labelpad=15);
# cbar.ax.tick_params(labelsize=20)
# pyp.xlabel(r'$T_1\ (K)$',fontsize=24);
# pyp.xticks([325,425,525,625,725]);
# pyp.ylabel(r'$T_2\ (K)$',fontsize=24);
# pyp.yticks([325,425,525,625,725]);
# pyp.xlim((313,763)); pyp.ylim((313,763));
# ax3D1.tick_params(axis='both',which='major',labelsize=20);
# pyp.title(r'$Reactor\ Cost\ Function\ f({\bf x})$',fontsize=24,pad=15);

# fig3Dp=pyp.figure(figsize=[10.5,8.5])
# ax3D1p=fig3Dp.add_subplot(111,projection='3d')
# ax3D1p.grid(color='gray',axis='both',alpha=0.25);
# ax3D1p.set_axisbelow(True);
# fig1p=ax3D1p.plot_surface(T1.reshape(d1,d2),T2.reshape(d1,d2),
#                           CT_RECYL[:,-1].reshape(d1,d2),rstride=1,cstride=1,
#                           linewidth=0,antialiased=False,cmap=cm.jet);
# #ax3D1p.scatter(T1.flatten()[argmin(CT)],T2.flatten()[argmin(CT)],min(CT),color='white',edgecolor='black',s=75);
# ax3D1p.view_init(41,69);
# ax3D1p.xaxis.set_rotate_label(False)
# pyp.xlabel(r'$T_1\ (K)$',fontsize=24,rotation=0,labelpad=20);
# ax3D1p.yaxis.set_rotate_label(False)
# pyp.ylabel(r'$T_2\ (K)$',fontsize=24,rotation=0,labelpad=30);
# ax3D1p.zaxis.set_rotate_label(False)
# ax3D1p.set_zlabel(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,rotation=90,
#                   labelpad=38);
# pyp.xlim((313,763)); pyp.ylim((313,763)); #ax3D1p.set_zlim((0,85)); 
# pyp.xticks([325,425,525,625,725]);
# pyp.yticks([325,425,525,625,725]);
# #ax3D1p.set_zticks([-1.6,-1.45,-1.3,-1.15,-1.0,-0.85]);
# ax3D1p.tick_params(axis='both',which='major',labelsize=20);
# ax3D1p.tick_params(axis='z',pad=20)
# pyp.title(r'$Reactor\ Cost\ Function\ f({\bf x})$',fontsize=24,pad=35)

#%% RECYCLE LOOP REFERENCE MODEL FORMULATION
# Reaction Regressions
def scale(x, ub, lb, sf = 1):
    # m = sf/(ub-lb)
    # b = -lb*sf/(ub-lb)
    m = 2*sf/(ub-lb)
    b = sf-m*ub
    return m*x+b

def descale(x, ub, lb, sf = 1):
    # m = (ub-lb)/sf
    # b = lb
    b = (ub+lb)/2
    m = (b-lb)/sf
    return m*x+b

TTr = random.uniform(313, 763, (200, 2))
# TTr = linspace(313, 763, 14).reshape(-1, 1)
# TTr = meshgrid(TTr, TTr)
# TTr = hstack([TTr[0].flatten().reshape(-1, 1), TTr[1].flatten().reshape(-1, 1)])
RXNS = Parallel(n_jobs = -1)(delayed(SYST_RECYCLE)(start_point, rxn_reg = True)
                             for start_point in TTr)
#TTr = TTr+random.normal(0, 5, (200, 2))
TTr = scale(1/TTr, ub = 1/313, lb = 1/763)
TTr = hstack([TTr, ones((TTr.shape[0], 1))])
RXNS = vstack(RXNS[:][:])
lnRXNS = log(abs(RXNS))
lo = nmin(lnRXNS, axis = 0)
hi = nmax(lnRXNS, axis = 0)
sf = std(lnRXNS, axis = 0)
n_params = 3
n_eqns = RXNS.shape[1]
idx = random.uniform(-10, 10, size = 100)
idx1 = [1, 4, 7, 10, 15]
lb = -100*ones((n_params*n_eqns))
lb = lb.reshape(n_eqns, n_params)
ub = 100*ones((n_params*n_eqns))
ub = ub.reshape(n_eqns, n_params)
thetaR = ones(shape = (n_eqns, n_params))
theta = ones(n_params)
print('Calculation of reaction model weights...')
start = time.time()
for j in range(n_eqns):
    sf[j] = max(sf[j], 1)
    lnRXNSn = scale(lnRXNS[:, j], hi[j], lo[j], sf = sf[j])
    def fit(theta):
        λ = 0.05
        lnrn = TTr@theta
        return sum((lnRXNSn-lnrn)**2)+λ*theta.T@theta
    bndsm = Bounds(lb[j], ub[j])
    soln = Parallel(n_jobs = -1)(delayed(minimize)(fit, i*theta, method = 'L-BFGS-B', 
                                        bounds = bndsm) for i in idx)
    loss = array([atleast_1d(res.fun)[0] for res in soln])
    theta = array([res.x for res in soln],dtype='float')
    theta = theta[argmin(loss)]
    thetaR[j] = theta
end = time.time()
print(end-start)
r = ones(RXNS.shape)
for i in range(8):
    r[:, i] = exp(descale(TTr@thetaR[i], hi[i], lo[i], sf = sf[i]))
hi = hstack([hi[:4].reshape(-1,1), hi[4:].reshape(-1,1)])
lo = hstack([lo[:4].reshape(-1,1), lo[4:].reshape(-1,1)])
sf = hstack([sf[:4].reshape(-1,1), sf[4:].reshape(-1,1)])    

# Reference mass and energy balances
def mass_bal_reg_recycle(T,F,C0,V,theta, rxt_num):
    rn = rxt_num-1
    invT = scale(1/T, ub = 1/313, lb = 1/763)
    invT = hstack([invT, ones((T.shape[0], 1))])
    r = exp(descale(invT@theta.T, hi[:,rn], lo[:,rn], sf[:,rn])).T
    Ca=C0[0]-2*(r[0]-r[3])*V/F#,array([0]))
    Cb=C0[1]+(r[0]-r[1])*V/F#,array([0]))
    Cc=C0[2]+(2*r[1]-r[2]-r[3])*V/F#,array([0]))
    Cd=C0[3]-r[2]*V/F#,array([0]))
    Ce=C0[4]+r[2]*V/F#,array([0]))
    Cf=C0[4]-r[3]*V/F#,array([0]))
    return array([Ca,Cb,Cc,Cd,Ce,Cf])
    
def heat_bal_reg_recycle(T,F,p,Cp,H,Tin,V,theta,rxt_num):
    rn = rxt_num-1
    T1=T[:, rn]
    invT = scale(1/T, ub = 1/313, lb = 1/763)
    invT = hstack([invT, ones((T.shape[0], 1))])
    r = exp(descale(invT@theta.T, hi[:,rn], lo[:,rn], sf[:,rn])).T
    Qdot=-(r[0]*H[0]+r[1]*H[1]+r[2]*H[2]+r[3]*H[3])*V
    ToH2O=ones(T1.shape)
    for i in range(T1.shape[0]):
        ToH2O[i]=min(T1[i]-10,323)
    mH2O=(p*Cp*F*(Tin-T1)+Qdot)/(CpH2O*(ToH2O-TinH2O))
    for i in range(mH2O.shape[0]):
        if mH2O[i]<0:
            mH2O[i]=0
    return r[0],r[1],r[2],r[3],mH2O

def Rxtr_Recycle_Ref(FR, TR, R_Frac, theta):
    err = 1e6
    FinR3 = FR[-1]
    while err > 1e-6:
        ρR3 = (FinR3/(FinR3+FR[2]))*ρR[1]+(FR[2]/(FinR3+FR[2]))*ρR[2]
        Fr = R_Frac*(ρR[0]*FR[0]+ρR[1]*FR[1]+ρR[2]*FR[2])/((1-R_Frac)*ρR3)
        err = Fr - FinR3
        FinR3 = Fr*1
    CinR = vstack([C0F1, C0F3])
    CpR3 = (FinR3/(FinR3+FR[2]))*CpR[1]+(FR[2]/(FinR3+FR[2]))*CpR[2]
    TinR3 = (FR[2]*CpR[2]*TinR[-1]+FinR3*CpR3*TR[1,0])/(FinR3*CpR3+FR[2]*CpR[2])
    err = 1e6
    FinR3 = FinR3+FR[2]
    i = 0
    while err > 1e-6:
        MixR1 = MIXER(CinR, array([FR[0], FinR3]), array([TinR[0], TinR3]), array([ρR[0], ρR3]), array([CpR[0], CpR3]), all_C0=True)
        C0R1 = MixR1.mass_bal()
        TinR1 = MixR1.heat_bal()
        FinR1 = MixR1.Fout
        CpR1 = MixR1.Cpmix
        ρR1 = MixR1.pmix
        ReacR1_REF = CSTR_REF(TR[0], TR[1], FinR1, C0R1, ρR1, CpR1, HR, TinR1, c3, V1,
                      theta[:4])
        ReacR1_REF.MandE_bal(mass_bal_reg_recycle, heat_bal_reg_recycle)
        CinR2 = vstack([ReacR1_REF.Cf, C0F2])
        MixR2 = MIXER(CinR2, array([ReacR1_REF.F[0,0], FR[1]]), array([ReacR1_REF.T1[0,0], TinR[1]]), array([ρR1, ρR[1]]), array([CpR1, CpR[1]]), all_C0=True)
        C0R2 = MixR2.mass_bal()
        TinR2 = MixR2.heat_bal()
        FinR2 = MixR2.Fout
        CpR2 = MixR2.Cpmix
        ρR2 = MixR2.pmix
        ReacR2_REF = CSTR_REF(TR[0], TR[1], FinR2, C0R2, ρR2, CpR2, HR, TinR2, c3, V2,
                      theta[4:], rxt_num=2)
        ReacR2_REF.MandE_bal(mass_bal_reg_recycle, heat_bal_reg_recycle)
        SplitR1 = SPLITTER(ReacR2_REF.F, array([0.1]))
        Fr, Fpd = SplitR1.split()
        CinR3 = vstack([ReacR2_REF.Cf, C0F3])
        MixR3 = MIXER(CinR3, array([Fr[0], FR[2]]), array([ReacR2_REF.T2[0,0], TinR[-1]]), array([ρR2, ρR[-1]]), array([CpR2, CpR[-1]]), all_C0 = True)
        C0R3 = MixR3.mass_bal()
        TinR3 = MixR3.heat_bal()
        FinR3 = MixR3.Fout
        CpR3 = MixR3.Cpmix
        ρR3 = MixR3.pmix
        CinR = vstack([C0F1, C0R3])
        err = (Fpd*ReacR2_REF.p-(FR[0]*ρR[0]+FR[1]*ρR[1]+FR[2]*ρR[2]))**2
        i += 1
        CtR1 = ReacR1_REF.Econ(econ_recycle1)
        CtR2 = ReacR2_REF.Econ(econ_recycle2)
    return CtR1, CtR2

def SYST_RECYCLE_REF(T):
    T=T.reshape(-1,1)
    FR = array([100, 50, 50])
    R_Frac = 0.1
    CtR1, CtR2 = Rxtr_Recycle_Ref(FR, T, R_Frac, thetaR)
    CtR = CtR1+CtR2
    return CtR1, CtR2, CtR

# print('Reference recycle system calculation...')
# start = time.time()
# CT_RECYL_REF = Parallel(n_jobs = -1)(delayed(SYST_RECYCLE_REF)(start_point) for start_point in TT)
# end = time.time()
# print(end-start)
# CT_RECYL_REF = hstack(CT_RECYL_REF[:]).T

# fig3D=pyp.figure(figsize=[10.5,8.5])
# ax3D1=fig3D.add_subplot(111);
# ax3D1.grid(color='gray',axis='both',alpha=0.25);
# ax3D1.set_axisbelow(True);
# fig1=ax3D1.contourf(T1.reshape(d1,d2),T2.reshape(d1,d2),
#                     CT_RECYL_REF[:,-1].reshape(d1,d2),cmap=cm.jet);
# ax3D1.scatter(T1.flatten()[argmin(CT_RECYL_REF[:,-1])],
#               T2.flatten()[argmin(CT_RECYL_REF[:,-1])],
#               color='white',edgecolor='k',marker='o',s=100);
# cbar=pyp.colorbar(fig1);
# cbar.set_label(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,labelpad=15);
# cbar.ax.tick_params(labelsize=20)
# pyp.xlabel(r'$T_1\ (K)$',fontsize=24);
# pyp.xticks([325,425,525,625,725]);
# pyp.ylabel(r'$T_2\ (K)$',fontsize=24);
# pyp.yticks([325,425,525,625,725]);
# pyp.xlim((313,763)); pyp.ylim((313,763));
# ax3D1.tick_params(axis='both',which='major',labelsize=20);
# pyp.title(r'$Reactor\ Cost\ Function\ f({\bf x})$',fontsize=24,pad=15);

# fig3Dp=pyp.figure(figsize=[10.5,8.5])
# ax3D1p=fig3Dp.add_subplot(111,projection='3d')
# ax3D1p.grid(color='gray',axis='both',alpha=0.25);
# ax3D1p.set_axisbelow(True);
# fig1p=ax3D1p.plot_surface(T1.reshape(d1,d2),T2.reshape(d1,d2),
#                           CT_RECYL_REF[:,-1].reshape(d1,d2),rstride=1,cstride=1,
#                           linewidth=0,antialiased=False,cmap=cm.jet);
# #ax3D1p.scatter(T1.flatten()[argmin(CT)],T2.flatten()[argmin(CT)],min(CT),color='white',edgecolor='black',s=75);
# ax3D1p.view_init(41,69);
# ax3D1p.xaxis.set_rotate_label(False)
# pyp.xlabel(r'$T_1\ (K)$',fontsize=24,rotation=0,labelpad=20);
# ax3D1p.yaxis.set_rotate_label(False)
# pyp.ylabel(r'$T_2\ (K)$',fontsize=24,rotation=0,labelpad=30);
# ax3D1p.zaxis.set_rotate_label(False)
# ax3D1p.set_zlabel(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,rotation=90,
#                   labelpad=38);
# pyp.xlim((313,763)); pyp.ylim((313,763)); #ax3D1p.set_zlim((0,85)); 
# pyp.xticks([325,425,525,625,725]);
# pyp.yticks([325,425,525,625,725]);
# #ax3D1p.set_zticks([-1.6,-1.45,-1.3,-1.15,-1.0,-0.85]);
# ax3D1p.tick_params(axis='both',which='major',labelsize=20);
# ax3D1p.tick_params(axis='z',pad=20)
# pyp.title(r'$Reactor\ Cost\ Function\ f({\bf x})$',fontsize=24,pad=35)

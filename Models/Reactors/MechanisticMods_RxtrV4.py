# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:11:31 2020

@author: leonardo
"""

from numpy import arange, random, array, argmin, vstack, atleast_1d, round
from numpy import hstack, exp, matmul, log, ones, linalg, transpose as trans
from numpy import meshgrid, min as nmin
from matplotlib import pyplot as pyp,cm
from scipy.optimize import minimize, Bounds, fsolve
from joblib import Parallel, delayed
import sklearn.gaussian_process as gpr
# from pyGPGO.covfunc import matern52
# from pyGPGO.acquisition import Acquisition
# from pyGPGO.surrogates.GaussianProcess import GaussianProcess
# from pyGPGO.GPGO import GPGO
# from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D

exp_w=2.6; t=35; bnds=Bounds(-1,1); bnds2=Bounds((0,0),(1,1)); noise=1e-6
F=90; T=423; lp=1;
CBEST=0*ones((t+1,1)); CBESTGP=0*ones((t+1,1)); idx=ones((t,1),dtype=int);
cons = ({'type':'ineq','fun':lambda x:x[0]},
        {'type':'ineq','fun':lambda x:1-x[0]})
m=2/450; b=-1-313*m; mf=5; bf=95;

##################Initial Conditions and Physical Parameters##################
Cr0=5000; Ci0=0; Cp0=0;
k01=1000; k02=1250; E1=32000; E2=35000;

p=850; Cp=3000; CpH2O=4184; R=8.314; H1=-210000; H2=-1700000;
Tin=298; TinH2O=298; V=1000;
cP=-0.35; cR=0.12; cT=0.0035; cI=0.075
##############################################################################
#T range is 313 to 763
#F range is 90 to 100
#Units are all in MKS 

def LCB(x):
    x=array([x]).reshape(-1,1); x=x.reshape(1,x.shape[0])
    mu,std=modelgp.predict(x,return_std=True); std=std.reshape(-1,1);
    return (mu-exp_w*std)

def LCB_ref(x):
    x=array([x]).reshape(-1,1); x=x.reshape(1,x.shape[0])
    mu,std=model.predict(x,return_std=True); std=std.reshape(-1,1)
    Cmod=Ctfxn_mod((x-b)/m,F)[0]
    return (Cmod+mu-exp_w*std)

def LCB_refM(x):
    x=array([x]).reshape(-1,1); x=x.reshape(1,x.shape[0])
    mu,std=model.predict(x,return_std=True); std=std.reshape(-1,1)
    Cmod=Ctfxn_modM((x[0,0]-b)/m,mf*x[0,1]+bf)[0]
    return Cmod+mu-exp_w*std
    
def Ctfxn(T,F):
    T=array([T],dtype='float64').reshape(-1,1); T=round(T,3);
    F=array([F],dtype='float64').reshape(-1,1); F=round(F,3);
    k1=k01*exp(-E1/(R*(T))); k2=k02*exp(-E2/(R*(T))); ToH2O=T-10;
    k1r=k1/100; k2r=k2/100;

    Crinit=5; Ciinit=2260; Cpdinit=475;
    Crinit=((F*(Cr0-Crinit)+2*k1r*Ciinit*V)/(2*k1*V))**(1/2);
    Ciinit=(k1*Crinit**2*V+k2r*Cpdinit**2*V)/(F+k1r*V+k2*V);
    Cpdinit=(2*k2*Ciinit*V-2*k2r*V*Cpdinit**2)/(F)
    C0=array([Crinit,Ciinit,Cpdinit]); C0=C0.reshape(C0.shape[0])
    k1f=k1*1; k2f=k2*1; k1rf=k1r*1; k2rf=k2r*1; Ff=F*1    
    def C(C):
        Crg=C[0]; Cig=C[1]; Cpdg=C[2]
        Cr=((Ff*(Cr0-Crg)+2*k1rf*Cig*V)/(2*k1f*V))**(1/2);
        Ci=(k1f*Crg**2*V+k2rf*Cpdg**2*V)/(Ff+k1rf*V+k2f*V);
        Cpd=(2*k2f*Cig*V-2*k2rf*V*Cpdg**2)/(Ff)
        return array([Cr-Crg,Ci-Cig,Cpd-Cpdg]).reshape(C0.shape[0])        
    soln=fsolve(C,C0);    
    C0=array(soln); Cr=C0[0]; Ci=C0[1]; Cpd=C0[2];

    r1=k1*Cr**2-k1r*Ci; r2=k2*Ci-k2r*Cpd**2;
    mH2O=(p*Cp*F*(Tin-T)-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O))    
    g=max([0,Cpd-600])
    Ct=cR*(Cr0+Cr)*F+cP*F*Ci+cI*Cpd*F+cT*mH2O#+100*g
    return Ct/1e4,r1,r2,mH2O,Cr,Ci,Cpd

def Ctfxn_mod(T,F):
    T=array([T],dtype='float64').reshape(-1,1); T=round(T,2);
    F=array([F],dtype='float64').reshape(-1,1); F=round(F,2);
    ## Inverse log
    r1=exp(theta[0,0]*1/T+theta[0,1])
    r2=exp(theta[1,0]*1/T+theta[1,1])    
    ToH2O=T-10; 
    Cr=Cr0-2*r1*V/F
    Ci=(r1-r2)*V/F
    Cpd=2*r2*V/F
    mH2O=(p*Cp*F*(Tin-T)-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O))    
    g=ones(Cpd.shape)
    for i in range(Cpd.shape[0]):
        g[i]=max([0,Cpd[i]-600])    
    Ct=cR*(Cr0+Cr)*F+cP*F*Ci+cI*Cpd*F+cT*mH2O#+100*g
    return Ct/1e4,r1,r2,mH2O,Cr,Ci,Cpd

def Ctfxn_modM(T,F):
    T=array([T],dtype='float64').reshape(-1,1); T=round(T,2);
    F=array([F],dtype='float64').reshape(-1,1); F=round(F,2);
    ## Log-linear log
    r1=exp(thetam[0,0]*1/T+thetam[0,1]*log(F)+thetam[0,2]);
    r2=exp(thetam[1,0]*1/T+thetam[1,1]*log(F)+thetam[1,2]);    
    ToH2O=T-10; 
    Cr=Cr0-2*r1*V/F
    Ci=(r1-r2)*V/F
    Cpd=2*r2*V/F
    mH2O=(p*Cp*F*(Tin-T)-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O))    
    g=ones(Cpd.shape)
    for i in range(Cpd.shape[0]):
        g[i]=max([0,Cpd[i]-600])    
    Ct=cR*(Cr0+Cr)*F+cP*F*Ci+cI*Cpd*F+cT*mH2O#+100*g
    return Ct/1e4,r1,r2,mH2O,Cr,Ci,Cpd

####################### Reaction Regression ##################################
## Univariate Regression
Tr=random.uniform(313,763,(200,1)); Fr=90*ones(Tr.shape)
Ctr=lambda T,F:Ctfxn(T,F)[1:3];
Ctr=array(list(map(Ctr,Tr,Fr))).reshape(Tr.shape[0],2);
# Inverse log
A=ones(Tr.shape[0]).reshape(-1,1); Ctr=log(Ctr); A=hstack([1/Tr,A]);
psuedoAinv=matmul(linalg.inv(matmul(trans(A),A)),trans(A));
w1=matmul(psuedoAinv,Ctr[:,0]); w2=matmul(psuedoAinv,Ctr[:,1]);
theta=vstack([w1,w2]);

## Multivariate Regression
Tr=random.uniform(313,763,(200,1)); Fr=random.uniform(90,100,(200,1));
Ctrm=lambda T,F:Ctfxn(T,F)[1:3];
Ctrm=array(list(map(Ctrm,Tr,Fr))).reshape(Tr.shape[0],2);
# Log-linear inverse T
Am=ones(Tr.shape[0]).reshape(-1,1); Ctrm=log(Ctrm); Am=hstack([1/Tr,log(Fr),Am]);
psuedoAinvm=matmul(linalg.inv(matmul(trans(Am),Am)),trans(Am));
w1=matmul(psuedoAinvm,Ctrm[:,0]); w2=matmul(psuedoAinvm,Ctrm[:,1]);
thetam=vstack([w1,w2]);

#################################SETUP########################################
## Univariate Model
Tm=arange(-1,1.001,0.002,dtype='float32').reshape(-1,1); Fm=F*ones(Tm.shape);
Ck=lambda T,F:Ctfxn((T-b)/m,F)[0];
Ck=array(list(map(Ck,Tm,Fm))).reshape(-1,1);
Cmk=Ctfxn_mod((Tm-b)/m,Fm)[0].reshape(-1,1);
CPRED=0*ones(Ck.shape); CM=CPRED.copy(); STD=CPRED.copy(); STDGP=CPRED.copy();

## Multivariate Model
Tp=arange(313,764,1); Fp=arange(90,100.01,0.1);
Fp,Tp=meshgrid(Fp,Tp); Ckp=ones(Tp.shape); Cmkp=ones(Tp.shape);
TFm=hstack([Tp.flatten().reshape(-1,1),Fp.flatten().reshape(-1,1)]);
d1=Tp.shape[0]; d2=Tp.shape[1];
Ckp=lambda T,F:Ctfxn(T,F)[0]; Cmkp=lambda T,F:Ctfxn_modM(T,F)[0];
Ckp=array(list(map(Ckp,Tp.reshape(-1,1),Fp.reshape(-1,1)))).reshape(Tp.shape);
Cmkp=array(list(map(Cmkp,Tp.reshape(-1,1),Fp.reshape(-1,1)))).reshape(Tp.shape);

CBEST=ones((lp)); CINIT=ones((t+1,lp)); TINIT=ones((t+1,lp)); FINIT=ones((t+1,lp));
CBESTGP=ones((lp)); CINITGP=CINIT.copy(); TINITGP=TINIT.copy(); FINITGP=FINIT.copy();
FUNS=arange(0.0,t,1); FUNSGP=1*FUNS;

## Figures
fig3Dp=pyp.figure(figsize=[10.5,8.5])
ax3D1p=fig3Dp.add_subplot(111,projection='3d')
ax3D1p.grid(color='gray',axis='both',alpha=0.25); ax3D1p.set_axisbelow(True);
fig1p=ax3D1p.plot_surface(Tp,Fp,Ckp,rstride=1,
                          cstride=1,linewidth=0,antialiased=False,cmap=cm.jet);
#ax3D1p.scatter(Tp.flatten()[argmin(Ckp)],Fp.flatten()[argmin(Ckp)],min(Ckp),color='white',edgecolor='black',s=75);
ax3D1p.view_init(41,69);
ax3D1p.xaxis.set_rotate_label(False)
pyp.xlabel(r'$T\ (K)$',fontsize=24,rotation=0,labelpad=20);
ax3D1p.yaxis.set_rotate_label(False)
pyp.ylabel(r'$F\ (\frac{m^3}{hr})$',fontsize=24,rotation=0,labelpad=30);
ax3D1p.zaxis.set_rotate_label(False)
ax3D1p.set_zlabel(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,rotation=90,
                  labelpad=38);
pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1p.set_zlim((0,85)); 
pyp.xticks([325,425,525,625,725]);
pyp.yticks([90,92,94,96,98,100]);
ax3D1p.set_zticks([-1.6,-1.45,-1.3,-1.15,-1.0,-0.85]);
ax3D1p.tick_params(axis='both',which='major',labelsize=20);
ax3D1p.tick_params(axis='z',pad=20)
pyp.title(r'$Reactor\ Cost\ Function\ f({\bf x})$',fontsize=24,pad=35)
pyp.savefig('Rxtr_mod.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

fig3D=pyp.figure(figsize=[10.5,8.5])
ax3D1=fig3D.add_subplot(111);
ax3D1.grid(color='gray',axis='both',alpha=0.25); ax3D1.set_axisbelow(True);
fig1=ax3D1.contourf(Tp,Fp,Ckp.reshape(d1,d2),cmap=cm.jet);
ax3D1.scatter(Tp.flatten()[argmin(Ckp)],Fp.flatten()[argmin(Ckp)],color='white',edgecolor='k',marker='o',s=100);
cbar=pyp.colorbar(fig1);
cbar.set_label(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,labelpad=15);
cbar.ax.tick_params(labelsize=20)
pyp.xlabel(r'$T\ (K)$',fontsize=24);
pyp.xticks([325,425,525,625,725]);
pyp.ylabel(r'$F\ (\frac{m^3}{hr})$',fontsize=24);
pyp.yticks([90,92,94,96,98,100]);
pyp.xlim((313,763)); pyp.ylim((90,100));
ax3D1.tick_params(axis='both',which='major',labelsize=20);
pyp.title(r'$Reactor\ Cost\ Function\ f({\bf x})$',fontsize=24,pad=15);
pyp.savefig('Rxtr_mod_cont.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

fig3Dp=pyp.figure(figsize=[10.5,8.5])
ax3D1p=fig3Dp.add_subplot(111,projection='3d')
ax3D1p.grid(color='gray',axis='both',alpha=0.25); ax3D1p.set_axisbelow(True);
fig1p=ax3D1p.plot_surface(Tp,Fp,Cmkp,rstride=1,
                          cstride=1,linewidth=0,antialiased=False,cmap=cm.jet);
#ax3D1p.scatter(Tp.flatten()[argmin(Cmkp)],Fp.flatten()[argmin(Cmkp)],min(Cmkp),color='white',edgecolor='black',s=75);
ax3D1p.view_init(41,69);
ax3D1p.xaxis.set_rotate_label(False)
pyp.xlabel(r'$T\ (K)$',fontsize=24,rotation=0,labelpad=20);
ax3D1p.yaxis.set_rotate_label(False)
pyp.ylabel(r'$F\ (\frac{m^3}{hr})$',fontsize=24,rotation=0,labelpad=30);
ax3D1p.zaxis.set_rotate_label(False)
ax3D1p.set_zlabel(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,rotation=90,
                  labelpad=35);
pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1p.set_zlim((0,85));
pyp.xticks([325,425,525,625,725]);
pyp.yticks([90,92,94,96,98,100]);
ax3D1p.set_zticks([-1.6,-1.4,-1.2,-1.0,-0.8,-0.6]);
ax3D1p.tick_params(axis='both',which='major',labelsize=20);
ax3D1p.tick_params(axis='z',pad=20)
pyp.title(r'$Reference\ Model\ g({\bf x})$',fontsize=24,pad=35)
pyp.savefig('RefRxtr_mod.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

fig3D=pyp.figure(figsize=[10.5,8.5])
ax3D1=fig3D.add_subplot(111);
ax3D1.grid(color='gray',axis='both',alpha=0.25); ax3D1.set_axisbelow(True);
fig1=ax3D1.contourf(Tp,Fp,Cmkp.reshape(d1,d2),
                    levels=[-1.8,-1.65,-1.5,-1.35,-1.2,-1.05,-0.9,-0.75,-0.6],cmap=cm.jet);
ax3D1.scatter(Tp.flatten()[argmin(Cmkp)],Fp.flatten()[argmin(Cmkp)],color='white',edgecolor='k',marker='o',s=100);
cbar=pyp.colorbar(fig1);
cbar.set_label(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24,labelpad=15);
cbar.ax.tick_params(labelsize=20)
pyp.xlabel(r'$T\ (K)$',fontsize=24);
pyp.xticks([325,425,525,625,725]);
pyp.ylabel(r'$F\ (\frac{m^3}{hr})$',fontsize=24);
pyp.yticks([90,92,94,96,98,100]);
pyp.xlim((313,763)); pyp.ylim((90,100));
ax3D1.tick_params(axis='both',which='major',labelsize=20);
pyp.title(r'$Reference\ Model\ g({\bf x})$',fontsize=24,pad=15)
pyp.savefig('RefRxtr_mod_cont.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

fig3Dp=pyp.figure(figsize=[10.5,8.5])
ax3D1p=fig3Dp.add_subplot(111,projection='3d')
ax3D1p.grid(color='gray',axis='both',alpha=0.25); ax3D1p.set_axisbelow(True);
fig1p=ax3D1p.plot_surface(Tp,Fp,Ckp-Cmkp,rstride=1,
                          cstride=1,linewidth=0,antialiased=False,cmap=cm.jet);
ax3D1p.view_init(41,69);
ax3D1p.xaxis.set_rotate_label(False)
pyp.xlabel(r'$T\ (K)$',fontsize=24,rotation=0,labelpad=20);
ax3D1p.yaxis.set_rotate_label(False)
pyp.ylabel(r'$F\ (\frac{m^3}{hr})$',fontsize=24,rotation=0,labelpad=30);
ax3D1p.zaxis.set_rotate_label(False)
ax3D1p.set_zlabel(r'$Model\ Error\ (10k\ USD/hr)$',fontsize=24,rotation=90,
                  labelpad=35);
pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1p.set_zlim((0,85)); 
pyp.xticks([325,425,525,625,725]);
pyp.yticks([90,92,94,96,98,100]);
ax3D1p.set_zticks([-0.3,-0.2,-0.1,0.0,0.1]);
ax3D1p.tick_params(axis='both',which='major',labelsize=20);
ax3D1p.tick_params(axis='z',pad=20)
pyp.title(r'$Residual\ Model\ \epsilon({\bf x})$',fontsize=24,pad=35)
pyp.savefig('ResRxtr_mod.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

##################################RUNS########################################
#%% Univariate
## Setup
Tm=arange(-1,1.001,0.002,dtype='float32').reshape(-1,1);
Tinit=random.uniform(-1,1,1).reshape(-1,1); F=F*ones(Tinit.shape); Tinitgp=1*Tinit;
Cinit=Ctfxn((Tinit-b)/m,F)[0].reshape(-1,1); Cinitgp=1*Cinit;
Cbest=nmin(Cinit).reshape(-1,1); Cbestgp=1*Cbest;
kernel=gpr.kernels.Matern(1,(0.1,100),nu=2.5);
model=gpr.GaussianProcessRegressor(kernel,alpha=1e-6,normalize_y=True,
                                     n_restarts_optimizer=10);
modelgp=gpr.GaussianProcessRegressor(kernel,alpha=1e-6,normalize_y=True,
                                     n_restarts_optimizer=10);
T0=random.uniform(-1,1,100).reshape(-1,1); T0=T0.astype('float64')

# Reference model
dinit=(Cinit-Ctfxn_mod((Tinit-b)/m,F)[0]).reshape(-1,1);
model.fit(Tinit,dinit); #model.kernel_.get_params();
opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB_ref,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds) for start_point in T0);
Tnxtm=array([res.x for res in opt],dtype='float64');
funs=array([atleast_1d(res.fun)[0] for res in opt]);

# BO
modelgp.fit(Tinit,Cinit); #modelgp.kernel_.get_params();
#Cm,std=model.predict(Tm,return_std=True); std=std.reshape(-1,1);
opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds) for start_point in T0);#'SLSQP'
Tnxtmgp=array([res.x for res in opt],dtype='float64');
funsgp=array([atleast_1d(res.fun)[0] for res in opt]); 

## Loop
for i in range(t):
    T0=random.uniform(-1,1,100).reshape(-1,1); T0=T0.astype('float64')
    
    ## Reference model
    Tnxt=Tnxtm[argmin(funs)].reshape(-1,1); Tinit=vstack([Tinit,Tnxt]);
    FUNS[i]=funs[argmin(funs)]; Cnxt=Ctfxn((Tnxt-b)/m,F)[0].reshape(-1,1);
    Cinit=vstack([Cinit,Cnxt]); Cbest=vstack([Cbest,nmin(Cinit)]);
    dnxt=(Cnxt-Ctfxn_mod((Tnxt-b)/m,F)[0]).reshape(-1,1); dinit=vstack([dinit,dnxt]);
    model.fit(Tinit,dinit); #model.kerne;_.get_params();
    opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB_ref,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds) for start_point in T0);
    Tnxtm=array([res.x for res in opt],dtype='float64');
    funs=array([atleast_1d(res.fun)[0] for res in opt]);
    dpred,stdpred=model.predict(Tm,return_std=True); stdpred=stdpred.reshape(-1,1);
    Cpred=Ctfxn_mod((Tm-b)/m,F)[0]+dpred;
    
    ## BO
    Tnxt=Tnxtmgp[argmin(funsgp)].reshape(-1,1); Tinitgp=vstack([Tinitgp,Tnxt])
    FUNSGP[i]=funsgp[argmin(funsgp)]; Cnxt=Ctfxn((Tnxt-b)/m,F)[0].reshape(-1,1);
    Cinitgp=vstack([Cinitgp,Cnxt]); Cbestgp=vstack([Cbestgp,nmin(Cinitgp)]);
    modelgp.fit(Tinitgp,Cinitgp); #modelgp.kernel_.get_params();
    opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds) for start_point in T0);#'SLSQP'
    Tnxtmgp=array([res.x for res in opt],dtype='float64');
    funsgp=array([atleast_1d(res.fun)[0] for res in opt]);
    Cm,std=modelgp.predict(Tm,return_std=True); std=std.reshape(-1,1);
    af=Cm-exp_w*std; #pyp.figure("AF Runs"); pyp.plot(Tm,af)

Tm=(Tm-b)/m; Tinit=(Tinit-b)/m; Tinitgp=(Tinitgp-b)/m;
itr=array(arange(t+1)).reshape(-1,1);

pyp.figure(); pyp.plot(Tm,Ck,color='blue',linewidth=1);
pyp.plot(Tm,Cpred,color='red',linewidth=1);
pyp.fill_between(Tm.reshape(-1),(Cpred-2*stdpred).reshape(-1),
                   (Cpred+2*stdpred).reshape(-1),alpha=0.1);
pyp.scatter(Tinit,Cinit,marker='o',color='white',edgecolor='k');  
pyp.xlim((300,800)); pyp.ylim((-1.5,-0.86));
pyp.legend(['True Model','BO+Ref model']);
pyp.figure();
pyp.plot(itr,Cinit,color='blue',linewidth=1);
pyp.scatter(itr,Cinit,marker='o',color='white',edgecolor='blue');
pyp.plot(itr,Cbest,color='red',linewidth=1);
pyp.scatter(itr,Cbest,marker='o',color='white',edgecolor='red');
pyp.legend(['Current Solution','Optimal Solution']);

pyp.figure(); pyp.plot(Tm,Ck,color='blue',linewidth=1);
pyp.plot(Tm,Cm,color='red',linewidth=1);
pyp.fill_between(Tm.reshape(-1),(Cm-2*std).reshape(-1),
                   (Cm+2*std).reshape(-1),alpha=0.1);
pyp.scatter(Tinitgp,Cinitgp,marker='o',color='white',edgecolor='k');  
pyp.xlim((300,800)); pyp.ylim((-1.5,-0.86));
pyp.legend(['True Model','BO model'])
pyp.figure();
pyp.plot(itr,Cinitgp,color='blue',linewidth=1);
pyp.scatter(itr,Cinitgp,marker='o',color='white',edgecolor='blue');
pyp.plot(itr,Cbestgp,color='red',linewidth=1);
pyp.scatter(itr,Cbestgp,marker='o',color='white',edgecolor='red');
pyp.legend(['Current Solution','Optimal Solution']);

#%% Multivariate##
for j in range(lp):
    ## Setup
    TFinit=random.uniform(-1,1,(1,2)); TFinitgp=1*TFinit;
    Cinit=Ctfxn((TFinit[0,0]-b)/m,mf*TFinit[0,1]+bf)[0].reshape(-1,1); Cinitgp=1*Cinit;
    Cbest=nmin(Cinit).reshape(-1,1); Cbestgp=1*Cbest;
    kernel=gpr.kernels.Matern(1,(0.1,100),nu=2.5);
    model=gpr.GaussianProcessRegressor(kernel,alpha=1e-6,normalize_y=True,
                                       n_restarts_optimizer=10);
    modelgp=gpr.GaussianProcessRegressor(kernel,alpha=1e-6,normalize_y=True,
                                         n_restarts_optimizer=10);
    TF0=random.uniform(-1,1,(100,2)); TF0=TF0.astype('float64')

    # Reference model
    dinit=(Cinit-Ctfxn_modM((TFinit[0,0]-b)/m,mf*TFinit[0,1]+bf)[0]).reshape(-1,1);
    model.fit(TFinit,dinit); #model.kernel_.get_params();
    opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB_refM,x0=start_point,method='L-BFGS-B',
                                              bounds=bnds2) for start_point in TF0);
    TFnxtm=array([res.x for res in opt],dtype='float64');
    funs=array([atleast_1d(res.fun)[0] for res in opt]);

    # BO
    modelgp.fit(TFinitgp,Cinitgp); #modelgp.kernel_.get_params();
    #Cm,std=model.predict(Tm,return_std=True); std=std.reshape(-1,1);
    opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB,x0=start_point,method='L-BFGS-B',
                                              bounds=bnds2) for start_point in TF0);#'SLSQP'
    TFnxtmgp=array([res.x for res in opt],dtype='float64');
    funsgp=array([atleast_1d(res.fun)[0] for res in opt]);

    ## Loop
    for i in range(t):
        TF0=random.uniform(-1,1,(100,2)); TF0=TF0.astype('float64')
    
        ## Reference model
        TFnxt=TFnxtm[argmin(funs)].reshape(1,2); TFinit=vstack([TFinit,TFnxt]);
        FUNS[i]=funs[argmin(funs)];
        Cnxt=Ctfxn((TFnxt[0,0]-b)/m,mf*TFnxt[0,1]+bf)[0].reshape(-1,1);
        Cinit=vstack([Cinit,Cnxt]); Cbest=vstack([Cbest,nmin(Cinit)]);
        dnxt=(Cnxt-Ctfxn_modM((TFnxt[0,0]-b)/m,mf*TFnxt[0,1]+bf)[0]).reshape(-1,1);
        dinit=vstack([dinit,dnxt]); model.fit(TFinit,dinit); #model.kerne;_.get_params();
        opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB_refM,x0=start_point,method='L-BFGS-B',
                                                  bounds=bnds2) for start_point in TF0);
        TFnxtm=array([res.x for res in opt],dtype='float64');
        funs=array([atleast_1d(res.fun)[0] for res in opt]);
    
        ## BO
        TFnxt=TFnxtmgp[argmin(funsgp)].reshape(1,2); TFinitgp=vstack([TFinitgp,TFnxt]);
        FUNSGP[i]=funsgp[argmin(funsgp)];
        Cnxt=Ctfxn((TFnxt[0,0]-b)/m,mf*TFnxt[0,1]+bf)[0].reshape(-1,1);
        Cinitgp=vstack([Cinitgp,Cnxt]); Cbestgp=vstack([Cbestgp,nmin(Cinitgp)]);
        modelgp.fit(TFinitgp,Cinitgp); #modelgp.kernel_.get_params();
        opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB,x0=start_point,method='L-BFGS-B',
                                                  bounds=bnds2) for start_point in TF0);#'SLSQP'
        TFnxtmgp=array([res.x for res in opt],dtype='float64');
        funsgp=array([atleast_1d(res.fun)[0] for res in opt]);   

    TINIT[:,j]=TFinit[:,0]; FINIT[:,j]=TFinit[:,1]; CINIT[:,j]=Cinit[:,0];
    TINITGP[:,j]=TFinitgp[:,0]; FINITGP[:,j]=TFinitgp[:,1]; CINITGP[:,j]=Cinitgp[:,0];
    CBEST[j]=Cbest[-1]; CBESTGP[j]=Cbestgp[-1];

TFinit[:,0]=(TINIT[:,argmin(CBEST)]-b)/m;
TFinit[:,1]=mf*FINIT[:,argmin(CBEST)]+bf;
Cinit=CINIT[:,argmin(CBEST)].reshape(-1,1)
TFinitgp[:,0]=(TINITGP[:,argmin(CBEST)]-b)/m;
TFinitgp[:,1]=mf*FINITGP[:,argmin(CBEST)]+bf;
Cinitgp=CINITGP[:,argmin(CBEST)].reshape(-1,1)
TFm[:,0]=m*TFm[:,0]+b; TFm[:,1]=(TFm[:,1]-bf)/mf;
dpred,stdpred=model.predict(TFm,return_std=True); stdpred=stdpred.reshape(-1,1);
Cpred=Ctfxn_modM((TFm[:,0]-b)/m,mf*TFm[:,1]+bf)[0]+dpred;
afgp=Cpred-exp_w*stdpred;
Cm,std=modelgp.predict(TFm,return_std=True); std=std.reshape(-1,1);
afgp=Cm-exp_w*std;
TFm[:,0]=(TFm[:,0]-b)/m; TFm[:,1]=mf*TFm[:,1]+bf;

itr=arange(1,t+2,1);
fig,ax1=pyp.subplots(1,1,figsize=(11,8.5));
ax1.grid(color='gray',axis='both',alpha=0.25); ax1.set_axisbelow(True);
ax1.set_xlim((1,t+1)); ax1.set_ylim((-1.8,-0.6));
ax1.scatter(itr,Cinit,marker='o',color='white',edgecolor='blue',zorder=3,s=200);
ax1.scatter(itr,Cinitgp,marker='o',color='white',edgecolor='red',zorder=3,s=200);
ax1.set_xlabel(r'$Sample\ Number$',fontsize=24); pyp.xticks(fontsize=24);
ax1.set_ylabel(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=24); pyp.yticks(fontsize=24);
ax1.plot(itr,Cinit,color='blue',linewidth=3);
ax1.plot(itr,Cinitgp,color='red',linewidth=3); 
ax1.legend([r'$BO+Ref$', r'$BO$'],fontsize=24);
pyp.savefig('Rxtr_covergResults.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

# fig3Dp=pyp.figure(figsize=[10,8.5])
# ax3D1p=fig3Dp.add_subplot(111,projection='3d')
# fig1p=ax3D1p.plot_surface(Tp,Fp,Cpred.reshape(d1,d2),rstride=1,
#                           cstride=1,linewidth=0,antialiased=False,cmap=cm.jet);
# #ax3D1p.scatter(Tp.flatten()[argmin(Ckp)],Fp.flatten()[argmin(Ckp)],min(Ckp),color='white',edgecolor='black',s=75);
# ax3D1p.view_init(41,69);
# ax3D1p.xaxis.set_rotate_label(False)
# pyp.xlabel(r'$T\ (K)$',fontsize=20,rotation=0,labelpad=15);
# ax3D1p.yaxis.set_rotate_label(False)
# pyp.ylabel(r'$F\ (\frac{m^3}{hr})$',fontsize=20,rotation=0,labelpad=25);
# ax3D1p.zaxis.set_rotate_label(False)
# ax3D1p.set_zlabel(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=20,rotation=90,
#                   labelpad=30);
# pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1p.set_zlim((0,85)); 
# ax3D1p.tick_params(axis='both',which='major',labelsize=16);
# ax3D1p.tick_params(axis='z',pad=15)
# pyp.savefig('Rxtr_ref_mod.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

# fig3Dp=pyp.figure(figsize=[10,8.5])
# ax3D1p=fig3Dp.add_subplot(111,projection='3d')
# fig1p=ax3D1p.plot_surface(Tp,Fp,Cm.reshape(d1,d2),rstride=1,
#                           cstride=1,linewidth=0,antialiased=False,cmap=cm.jet);
# #ax3D1p.scatter(Tp.flatten()[argmin(Ckp)],Fp.flatten()[argmin(Ckp)],min(Ckp),color='white',edgecolor='black',s=75);
# ax3D1p.view_init(41,69);
# ax3D1p.xaxis.set_rotate_label(False)
# pyp.xlabel(r'$T\ (K)$',fontsize=20,rotation=0,labelpad=15);
# ax3D1p.yaxis.set_rotate_label(False)
# pyp.ylabel(r'$F\ (\frac{m^3}{hr})$',fontsize=20,rotation=0,labelpad=25);
# ax3D1p.zaxis.set_rotate_label(False)
# ax3D1p.set_zlabel(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=20,rotation=90,
#                   labelpad=30);
# pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1p.set_zlim((0,85)); 
# ax3D1p.tick_params(axis='both',which='major',labelsize=16);
# ax3D1p.tick_params(axis='z',pad=15)
# pyp.savefig('Rxtr_gen_mod.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

fig3D=pyp.figure(figsize=[10,8.5])
ax3D1=fig3D.add_subplot(111);
fig1=ax3D1.contourf(Tp,Fp,Ckp.reshape(d1,d2),cmap=cm.jet);
ax3D1.scatter(TFinit[:,0],TFinit[:,1],color='red',marker='x',s=100);
ax3D1.scatter(Tp.flatten()[argmin(Ckp)],Fp.flatten()[argmin(Ckp)],color='white',edgecolor='k',marker='o',s=100);
cbar=pyp.colorbar(fig1);
cbar.set_label(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=20,labelpad=15);
cbar.ax.tick_params(labelsize=16)
pyp.xlabel(r'$T\ (K)$',fontsize=20); pyp.xticks(fontsize=16);
pyp.ylabel(r'$F\ (\frac{m^3}{hr})$',fontsize=20); pyp.yticks(fontsize=16);
pyp.xlim((313,763)); pyp.ylim((90,100));
ax3D1.tick_params(axis='both',which='major',labelsize=16);
pyp.savefig('Rxtr_mod_samps.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

fig3D=pyp.figure(figsize=[10,8.5])
ax3D1=fig3D.add_subplot(111);
fig1=ax3D1.contourf(Tp,Fp,Ckp.reshape(d1,d2),cmap=cm.jet);
ax3D1.scatter(TFinitgp[:,0],TFinitgp[:,1],color='red',marker='x',s=100);
ax3D1.scatter(Tp.flatten()[argmin(Ckp)],Fp.flatten()[argmin(Ckp)],color='white',edgecolor='k',marker='o',s=100);
cbar=pyp.colorbar(fig1);
cbar.set_label(r'$Operating\ cost\ (10k\ USD/hr)$',fontsize=20,labelpad=15);
cbar.ax.tick_params(labelsize=16)
pyp.xlabel(r'$T\ (K)$',fontsize=20); pyp.xticks(fontsize=16);
pyp.ylabel(r'$F\ (\frac{m^3}{hr})$',fontsize=20); pyp.yticks(fontsize=16);
pyp.xlim((313,763)); pyp.ylim((90,100));
ax3D1.tick_params(axis='both',which='major',labelsize=16);
pyp.savefig('Rxtr_gen_samps.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

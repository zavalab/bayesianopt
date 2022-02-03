# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:55:13 2020

@author: leonardo
"""

import sys;
sys.path.insert(1,r'C:\Users\leonardo\OneDrive - UW-Madison\Research\bayesianopt\Scripts')
from numpy import exp, arange, random, array, vstack, argmax, delete, hstack
from numpy import argmin, loadtxt, apply_along_axis, ones, round, intersect1d
from numpy import log, matmul, linalg, transpose as trans, meshgrid
from matplotlib import pyplot as pyp, cm
import time
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import GPy
from Acquisition_Funcs import LCB, EI, PI
from scipy.optimize import fsolve, minimize
import imageio
from mpl_toolkits.mplot3d import Axes3D

##################Initial Conditions and Physical Parameters##################
Cr0=5000; Ci0=0; Cp0=0;
k01=8.0e9; k02=7.5e10; E1=78400; E2=86200; #k01=8.0e9; E2=86200; k02=7.5e10;
#k01=5.07e10; k02=3.25e11; E1=78400; E2=85700
#k01=1000; k02=1250; E1=32000; E2=35000;

p=850; Cp=3000; CpH2O=4184; R=8.314; H1=-210000; H2=-1700000;
Tin=298; TinH2O=288; V=1000;
cP=-34; cR=12; cT=0.32; cI=6# cP=-34 cR=12; cT=0.32; cI=6
#cP=-35; cR=12; cT=0.35; cI=3.5
##############################################################################
#T range is 323 to 423
#F range is 40 to 100
#Units are all in MKS 

def Ctfxn(T,F):
    T=array([T],dtype='float64').reshape(-1,1);
    F=array([F],dtype='float64').reshape(-1,1);
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
    return Ct/1e6,r1,r2,mH2O,Cr,Ci,Cpd

def Ctfxn_mod(T,F):
    T=array([T],dtype='float64').reshape(-1,1);
    F=array([F],dtype='float64').reshape(-1,1);
    ## Linear
    #r1=theta[0,0]*T+theta[0,1]
    #r2=theta[1,0]*T+theta[1,1]
    ## Cubic
    #r1=theta[0,0]*T**3+theta[0,1]*T**2+theta[0,2]*T+theta[0,3]
    #r2=theta[1,0]*T**3+theta[1,1]*T**2+theta[1,2]*T+theta[1,3]
    ## Inverse log
    r1=exp(theta[0,0]*1/T+theta[0,1])
    r2=exp(theta[1,0]*1/T+theta[1,1])
    ## Inverse cubic of log
    #r1=exp(theta[0,1]*1/T**2+theta[0,2]*1/T+theta[0,3])
    #r2=exp(theta[1,0]*1/T**3+theta[1,1]*1/T**2+theta[1,2]*1/T+theta[1,3])
    
    ToH2O=T-10; 
    Cr=Cr0-2*r1*V/F
    Ci=(r1-r2)*V/F
    Cpd=2*r2*V/F
    mH2O=(p*Cp*F*(Tin-T)-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O))    
    g=ones(Cpd.shape)
    for i in range(Cpd.shape[0]):
        g[i]=max([0,Cpd[i]-600])    
    Ct=cR*(Cr0+Cr)*F+cP*F*Ci+cI*Cpd*F+cT*mH2O#+100*g
    return Ct/1e6,r1,r2,mH2O,Cr,Ci,Cpd

def Ctfxn_modM(T,F):
    T=array([T],dtype='float64').reshape(-1,1);
    F=array([F],dtype='float64').reshape(-1,1);
    ## Quadratic
    #r1=thetam[0,0]*T**2+thetam[0,1]*T+thetam[0,2]*F+thetam[0,3]
    #r2=thetam[1,0]*T**2+thetam[1,1]*T+thetam[1,2]*F+thetam[1,3]
    ## Log-linear iverse T
    r1=exp(thetam[0,0]*1/T+thetam[0,1]*log(F)+thetam[0,2])
    r2=exp(thetam[1,0]*1/T+thetam[1,1]*log(F)+thetam[1,2])
    
    ToH2O=T-10; 
    Cr=Cr0-2*r1*V/F
    Ci=(r1-r2)*V/F
    Cpd=2*r2*V/F
    mH2O=(p*Cp*F*(Tin-T)-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O))    
    g=ones(Cpd.shape)
    for i in range(Cpd.shape[0]):
        g[i]=max([0,Cpd[i]-600])    
    Ct=cR*(Cr0+Cr)*F+cP*F*Ci+cI*Cpd*F+cT*mH2O#+100*g
    return Ct/1e6,r1,r2,mH2O,Cr,Ci,Cpd


####################### Reaction Regression ##################################
# Univariate Regression
Tr=random.uniform(303,503,(200,1)); Fr=90*ones(Tr.shape)
#Tr=random.uniform(323,773,(200,1)); Fr=90*ones(Tr.shape)
Ctr=lambda T,F:Ctfxn(T,F)[1:3];
Ctr=array(list(map(Ctr,Tr,Fr))).reshape(Tr.shape[0],2);
## Linear
#A=ones(Tr.shape[0]).reshape(-1,1); A=hstack([Tr,A]);
#psuedoAinv=matmul(linalg.inv(matmul(trans(A),A)),trans(A));
#w1=matmul(psuedoAinv,Ctr[:,0]); w2=matmul(psuedoAinv,Ctr[:,1]);
#theta=vstack([w1,w2]);
## Cubic
#A=ones(Tr.shape[0]).reshape(-1,1); A=hstack([Tr**3,Tr**2,Tr,A]);
#psuedoAinv=matmul(linalg.inv(matmul(trans(A),A)),trans(A));
#w1=matmul(psuedoAinv,Ctr[:,0]); w2=matmul(psuedoAinv,Ctr[:,1]);
#theta=vstack([w1,w2]);
## Inverse log
A=ones(Tr.shape[0]).reshape(-1,1); Ctr=log(Ctr); A=hstack([1/Tr,A]);
psuedoAinv=matmul(linalg.inv(matmul(trans(A),A)),trans(A));
w1=matmul(psuedoAinv,Ctr[:,0]); w2=matmul(psuedoAinv,Ctr[:,1]);
theta=vstack([w1,w2]);
## Inverse log cubic
#A=ones(Tr.shape[0]).reshape(-1,1); Ctr=log(Ctr)
#A1=hstack([(1/Tr)**2,1/Tr,A]); A2=hstack([(1/Tr)**3,(1/Tr)**2,1/Tr,A]);
#psuedoAinv1=matmul(linalg.inv(matmul(trans(A1),A1)),trans(A1));
#psuedoAinv2=matmul(linalg.inv(matmul(trans(A2),A2)),trans(A2));
#w1=matmul(psuedoAinv1,Ctr[:,0]); w2=matmul(psuedoAinv2,Ctr[:,1]);
#theta=vstack([hstack([[0],w1]),w2]);

#Multivariate regression
Trm=random.uniform(303,503,(200,1));
Frm=random.uniform(90,100,(200,1));
#Trm=random.uniform(773,423,(200,1));
#Frm=random.uniform(40,100,(200,1));
Ctrm=lambda T,F:Ctfxn(T,F)[1:3]
Ctrm=array(list(map(Ctrm,Trm,Frm))).reshape(Trm.shape[0],2);
## Quadratic
#Am=ones(Tr.shape[0]).reshape(-1,1); Am=hstack([Tr**2,Tr,Frm,Am]);
#psuedoAinvm=matmul(linalg.inv(matmul(trans(Am),Am)),trans(Am));
#w1=matmul(psuedoAinvm,Ctrm[:,0]); w2=matmul(psuedoAinvm,Ctrm[:,1]);
#thetam=vstack([w1,w2]);
## Log-linear inverse T
Am=ones(Trm.shape[0]).reshape(-1,1); Am=hstack([1/Trm,log(Frm),Am]); Ctrm=log(Ctrm)
psuedoAinvm=matmul(linalg.inv(matmul(trans(Am),Am)),trans(Am));
w1=matmul(psuedoAinvm,Ctrm[:,0]); w2=matmul(psuedoAinvm,Ctrm[:,1]);
thetam=vstack([w1,w2]);

#################################SETUP########################################
noise=1e-6; exp_w=2.6; t=100; itr=array(arange(t+1)).reshape(-1,1); F=90; T=423;
lp=100; CBEST=0*ones((t+1,1)); CBESTGP=0*ones((t+1,1)); idx=ones((t,1),dtype=int);

Tm=arange(0,1.001,0.001).reshape(-1,1); Fm=F*ones(Tm.shape);
Ck=lambda T,F:Ctfxn(200*T+303,F)[0];
#Ck=lambda T,F:Ctfxn(450*T+323,F)[0];
Ck=array(list(map(Ck,Tm,Fm))).reshape(-1,1);
Cmk=Ctfxn_mod(200*Tm+303,Fm)[0].reshape(-1,1);
#Cmk=Ctfxn_mod(450*Tm+323,Fm)[0].reshape(-1,1);
# Fm=arange(0,1.001,0.001).reshape(-1,1); Tm=T*ones(Fm.shape)
# Ck=Ctfxn(Tm,60*Fm+40)[0].reshape(-1,1)
# Cmk=Ctfxn_mod(Tm,50*Fm+50)[0].reshape(-1,1);

## MULTIVARIATE CASE:
Tp=arange(303,504,1); Fp=arange(90,100.1,0.1)
#Tp=arange(323,773,1); Fp=arange(40,100.1,0.1)
Fp,Tp=meshgrid(Fp,Tp); Ckp=ones(Tp.shape); Cmkp=ones(Tp.shape);
d1=Tp.shape[0]; d2=Tp.shape[1]
Ckp=lambda T,F:Ctfxn(T,F)[0]; Cmkp=lambda T,F:Ctfxn_modM(T,F)[0];
Ckp=array(list(map(Ckp,Tp.reshape(-1,1),Fp.reshape(-1,1)))).reshape(Tp.shape);
Cmkp=array(list(map(Cmkp,Tp.reshape(-1,1),Fp.reshape(-1,1)))).reshape(Tp.shape);
Tm=(Tp.reshape(-1,1)-303)/200; Fm=(Fp.reshape(-1,1)-90)/10;
Ck=Ckp.reshape(-1,1); Cmk=Cmkp.reshape(-1,1); TFm=hstack([Tm,Fm])

CPRED=0*ones(Ck.shape); CM=CPRED.copy(); STD=CPRED.copy(); STDGP=CPRED.copy();
CIDX=ones((lp,2)); CIDXGP=ones((lp,2));

#%%###############################RUNS#####################################%%#            
for j in range(lp):
    # Tinit=random.uniform(0,1,1).reshape(-1,1); F=F*ones(Tinit.shape); #array([0.8])
    # Cinit=lambda T,F:Ctfxn(450*T+323,F)[0];
    # #Cinit=lambda T,F:Ctfxn(100*T+323,F)[0];
    # Cinit=array(list(map(Cinit,Tinit,F))).reshape(-1,1);
    # Cbest=min(Cinit).reshape(-1,1);
    # #dinit=Cinit-Ctfxn_mod(100*Tinit+323,F)[0]
    # dinit=Cinit-Ctfxn_mod(450*Tinit+323,F)[0]

    # kernel=GPy.kern.Matern52(1,variance=15,lengthscale=1)
    # model=GPy.models.GPRegression(Tinit,dinit,kernel);
    # model.Gaussian_noise.variance=noise;
    # model.Gaussian_noise.variance.fix();
    # dm,std=model.predict(Tm); std=std**0.5; af=LCB(Cmk+dm,std,exp_w);
    # Cpred=Cmk+dm;
    # if j==lp-1:
    #     pyp.figure()
    #     pyp.plot(100*Tm+323,Cpred)
    #     pyp.fill_between(100*Tm.reshape(-1)+323,(Cpred-2*std).reshape(-1),
    #                       (Cpred+2*std).reshape(-1),alpha=0.1);
    #     pyp.scatter(100*Tinit[0]+323,Cinit[0],marker='x',color='k');
    #     pyp.plot(100*Tm+323,Ck,'g--');
    #     pyp.plot(100*Tm+323,Cmk,'r:');
    #     pyp.xlim((323,423)); #pyp.ylim((-23.2,-22.15));
    #     pyp.legend(['Mean','f(x)',r'$f_{mod}(x)$','Confidence','Data'],loc='upper right');
    #     pyp.savefig('Rxtr_1D_Progression1_Rest.png',dpi=300);
    #     pyp.close();
    #     pyp.figure();
    #     pyp.scatter(100*Tinit[0]+323,dinit[0],marker='x',color='k');
    #     pyp.plot(100*Tm+323,dm,'c-.');
    #     pyp.xlim((323,423));
    #     pyp.savefig('Rxtr_1D_Error1_Rest.png',dpi=300);
    #     pyp.close();

    # kernelgp=GPy.kern.Matern52(1,variance=15,lengthscale=1)
    # modelgp=GPy.models.GPRegression(Tinit,Cinit,kernelgp);
    # modelgp.Gaussian_noise.variance=noise;
    # modelgp.Gaussian_noise.variance.fix();
    # Cm,stdgp=modelgp.predict(Tm); stdgp=stdgp**0.5; afgp=LCB(Cm,stdgp,exp_w);
    # if j==lp-1:
    #     pyp.figure()
    #     pyp.plot(100*Tm+323,Cm)
    #     pyp.fill_between(100*Tm.reshape(-1)+323,(Cm-2*stdgp).reshape(-1),
    #                       (Cm+2*stdgp).reshape(-1),alpha=0.1);
    #     pyp.scatter(100*Tinit[0]+323,Cinit[0],marker='x',color='k')
    #     pyp.plot(100*Tm+323,Ck,'g--');
    #     pyp.xlim((323,423)); #pyp.ylim((-23.2,-22.15))
    #     pyp.legend(['Mean','f(x)','Confidence','Data'],loc='upper right')
    #     pyp.savefig('Rxtr_1Dgp_Progression1_Rest.png',dpi=300)
    #     pyp.close()

    
# MULTIVARIATE CASE:    
    #Ck=apply_along_axis(CtfxnMD,1,TFm)[:,0].reshape(-1,1);
    #Cmk=apply_along_axis(Ctfxn_modMD,1,TFm)[:,0].reshape(-1,1);
    #TFm=loadtxt('2D_mech_rxtr.txt');
    
    TFinit=random.uniform(0,1,2).reshape(1,2)#array([[1,1]])
    Cinit=Ctfxn(200*TFinit[0,0]+303,10*TFinit[0,1]+90)[0].reshape(-1,1); Cbest=min(Cinit);
    d2dinit=Cinit-Ctfxn_modM(200*TFinit[0,0]+303,10*TFinit[0,1]+90)[0].reshape(-1,1);

    kernel2d=GPy.kern.Matern52(2,variance=30,lengthscale=1)
    model2d=GPy.models.GPRegression(TFinit,d2dinit,kernel2d)
    model2d.Gaussian_noise.variance=noise;
    model2d.Gaussian_noise.variance.fix();
    dm2d,std2d=model2d.predict(TFm); std2d=std2d**0.5; af2d=LCB((Cmk+dm2d),std2d,exp_w);
    Cpred=Cmk+dm2d;
    
    kernel2dgp=GPy.kern.Matern52(2,variance=30,lengthscale=1)
    model2dgp=GPy.models.GPRegression(TFinit,Cinit,kernel2dgp)
    model2dgp.Gaussian_noise.variance=noise;
    model2dgp.Gaussian_noise.variance.fix();
    Cm,std2dgp=model2dgp.predict(TFm); std2dgp=std2dgp**0.5; af2dgp=LCB(Cm,std2dgp,exp_w);
    TFinitgp=TFinit[0:1]; Cinitgp=Cinit[0:1]; Cbestgp=Cbest[0];
    
    if j==lp-1:
        # fig3D=pyp.figure(figsize=[20,8])
        # ax3D1=fig3D.add_subplot(121)
        # fig1=ax3D1.contourf(Tp,Fp,Ckp)
        # ax3D1.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,color='r',
        #               marker='x',s=40)
        # pyp.colorbar(fig1)
        # pyp.xlim((303,503)); pyp.ylim((90,100))
        # pyp.title('True Model')
        # ax3D2=fig3D.add_subplot(122)
        # fig2=ax3D2.contourf(Tp,Fp,Cpred.reshape((d1,d2)))
        # ax3D2.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,color='r',
        #               marker='x',s=40)
        # pyp.colorbar(fig2)
        # pyp.xlim((303,503)); pyp.ylim((90,100))
        # pyp.title('Estimated Model')
        # pyp.savefig('Rxtr_2D_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
        # pyp.close()
        
        fig3Dp=pyp.figure(figsize=[20,8])
        ax3D1p=fig3Dp.add_subplot(121,projection='3d')
        fig1p=ax3D1p.plot_surface(Tp,Fp,Ckp,cmap=cm.coolwarm);
        ax3D1p.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost')
        pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D1p.set_zlim((-23.1,-22.3));
        ax3D2p=fig3Dp.add_subplot(122,projection='3d')
        fig2p=ax3D2p.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),cmap=cm.coolwarm);
        ax3D2p.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2p.set_zlabel('Cost')
        pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D2p.set_zlim((-23.1,-22.3));
        pyp.savefig('Rxtr_2D_Surf_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
        pyp.close()
        
        # fig3Dgp=pyp.figure(figsize=[20,8])
        # ax3D1gp=fig3Dgp.add_subplot(121)
        # fig1gp=ax3D1gp.contourf(Tp,Fp,Ckp)
        # ax3D1gp.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,color='r',
        #               marker='x',s=40)
        # pyp.colorbar(fig1gp)
        # pyp.xlim((303,503)); pyp.ylim((90,100))
        # pyp.title('True Model')
        # ax3D2gp=fig3Dgp.add_subplot(122)
        # fig2gp=ax3D2gp.contourf(Tp,Fp,Cm.reshape((d1,d2)))
        # ax3D2gp.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,color='r',
        #               marker='x',s=40)
        # pyp.colorbar(fig2gp)
        # pyp.xlim((303,503)); pyp.ylim((90,100));
        # pyp.title('Estimated Model');
        # pyp.savefig('Rxtr_2Dgp_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
        # pyp.close()
        
        fig3Dpgp=pyp.figure(figsize=[20,8])
        ax3D1pgp=fig3Dpgp.add_subplot(121,projection='3d')
        fig1pgp=ax3D1pgp.plot_surface(Tp,Fp,Ckp,cmap=cm.coolwarm);
        ax3D1pgp.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1pgp.set_zlabel('Cost')
        pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D1pgp.set_zlim((-23.1,-22.3));
        ax3D2pgp=fig3Dpgp.add_subplot(122,projection='3d')
        fig2pgp=ax3D2pgp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),cmap=cm.coolwarm);
        ax3D2pgp.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2pgp.set_zlabel('Cost')
        pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D2pgp.set_zlim((-23.1,-22.3));
        pyp.savefig('Rxtr_2Dgp_Surf_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
        pyp.close()
        
        fig3Dcomp=pyp.figure(figsize=[20,8])
        ax3D1comp=fig3Dcomp.add_subplot(121,projection='3d')
        fig1comp=ax3D1comp.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),cmap=cm.coolwarm);
        ax3D1comp.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1comp.set_zlabel('Cost')
        pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D1comp.set_zlim((-23.1,-22.3));
        ax3D2comp=fig3Dcomp.add_subplot(122,projection='3d')
        fig2comp=ax3D2comp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),cmap=cm.coolwarm);
        ax3D2comp.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2comp.set_zlabel('Cost')
        pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D2comp.set_zlim((-23.1,-22.3));
        pyp.savefig('Rxtr_2D_Surf_Comp_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
        pyp.close()
        
#        plt3d=pyp.figure().gca(projection='3d')
#        plt3d.plot_surface(Tp,Fp,Ckp)
#        plt3d.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)))
#        plt3d.scatter(100*TFinit[:,0]+323,10*TFinit[:,1]+90,Cinit[0],color='r',
#                      marker='x',size=20)
#        pyp.show()
        
        
#######################Using 'mechanistic' model##############################
    # for i in range(t):
    #     Tnxt=Tm[argmax(af)];
    #     #Cnxt=Ctfxn(100*Tnxt+323,F)[0];
    #     Cnxt=Ctfxn(450*Tnxt+323,F)[0];
    #     Cmnxt=Cmk[argmax(af)]; dnxt=Cnxt-Cmnxt; idx[i]=argmax(af)
    #     Tinit=vstack([Tinit,Tnxt]); Cinit=vstack([Cinit,Cnxt]);
    #     dinit=vstack([dinit,dnxt]); Cbest=vstack([Cbest,min(Cinit)])
    #     model=GPy.models.GPRegression(Tinit,dinit,kernel);
    #     model.Gaussian_noise.variance=noise;
    #     model.Gaussian_noise.variance.fix();
    #     dm,std=model.predict(Tm); std=std**0.5; af=LCB(Cmk+dm,std,exp_w);
    #     Cpred=Cmk+dm;
    #     if j==lp-1:
    #         pyp.figure()
    #         pyp.plot(100*Tm+323,Cpred)
    #         pyp.fill_between(100*Tm.reshape(-1)+323,(Cpred-2*std).reshape(-1),
    #                           (Cpred+2*std).reshape(-1),alpha=0.1);
    #         pyp.scatter(100*Tinit[0:i+2]+323,Cinit[0:i+2],marker='x',color='k');
    #         pyp.plot(100*Tm+323,Ck,'g--');
    #         pyp.plot(100*Tm+323,Cmk,'r:');
    #         pyp.xlim((323,423)); #pyp.ylim((-23.2,-22.15));
    #         pyp.legend(['Mean','f(x)',r'$f_{mod}(x)$','Confidence','Data'],loc='upper right');
    #         str1=str(i+2);
    #         pyp.savefig('Rxtr_1D_Progression'+str1+'_Rest.png',dpi=300);
    #         pyp.close();  
    #         pyp.figure();
    #         pyp.scatter(100*Tinit[0:i+2]+323,dinit[0:i+2],marker='x',color='k');
    #         pyp.plot(100*Tm+323,dm,'c-.');
    #         pyp.xlim((323,423));
    #         pyp.savefig('Rxtr_1D_Error'+str1+'_Rest.png',dpi=300);
    #         pyp.close();
    # CBEST=CBEST+Cbest; CPRED=CPRED+Cpred; STD=STD+std; 
    # CIDX[j,0]=argmin(Cpred); CIDX[j,1]=min(Cpred)
    
## MULTIVARIATE CASE:
    for i in range(t):
        TFnxt=TFm[argmax(af2d)].reshape(1,2);
        Cnxt=Ctfxn(200*TFnxt[0,0]+303,10*TFnxt[0,1]+90)[0];
        Cmnxt=Ctfxn_modM(200*TFnxt[0,0]+303,10*TFnxt[0,1]+90)[0];
        d2dnxt=Cnxt-Cmnxt;
        TFinit=vstack([TFinit,TFnxt]); Cinit=vstack([Cinit,Cnxt]);
        d2dinit=vstack([d2dinit,d2dnxt]); Cbest=vstack([Cbest,min(Cinit)])
        model2d=GPy.models.GPRegression(TFinit,d2dinit,kernel2d);
        model2d.Gaussian_noise.variance=noise;
        model2d.Gaussian_noise.variance.fix();
        dm2d,std2d=model2d.predict(TFm); std2d=std2d**0.5; af2d=LCB((Cmk+dm2d),std2d,exp_w);
        Cpred=Cmk+dm2d;
        
        TFnxt=TFm[argmax(af2dgp)].reshape(1,2);
        Cnxt=Ctfxn(200*TFnxt[0,0]+303,10*TFnxt[0,1]+90)[0];
        TFinitgp=vstack([TFinitgp,TFnxt]); Cinitgp=vstack([Cinitgp,Cnxt]);
        Cbestgp=vstack([Cbestgp,min(Cinitgp)])
        model2dgp=GPy.models.GPRegression(TFinitgp,Cinitgp,kernel2dgp)
        model2dgp.Gaussian_noise.variance=noise;
        model2dgp.Gaussian_noise.variance.fix();
        Cm,std2dgp=model2dgp.predict(TFm); std2dgp=std2dgp**0.5; af2dgp=LCB(Cm,std2dgp,exp_w);
        
        if j==lp-1:
            str1=str(i+2);
            # fig3D=pyp.figure(figsize=[20,8])
            # ax3D1=fig3D.add_subplot(121)
            # fig1=ax3D1.contourf(Tp,Fp,Ckp)
            # ax3D1.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,color='r',
            #               marker='x',s=40)
            # pyp.colorbar(fig1)
            # pyp.xlim((303,503)); pyp.ylim((90,100))
            # pyp.title('True Model')
            # ax3D2=fig3D.add_subplot(122)
            # fig2=ax3D2.contourf(Tp,Fp,Cpred.reshape((d1,d2)))
            # ax3D2.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,color='r',
            #               marker='x',s=40)
            # pyp.colorbar(fig2)
            # pyp.xlim((303,503)); pyp.ylim((90,100))
            # pyp.title('Estimated Model')
            # pyp.savefig('Rxtr_2D_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
            # pyp.close()
            
            fig3Dp=pyp.figure(figsize=[20,8])
            ax3D1p=fig3Dp.add_subplot(121,projection='3d')
            fig1p=ax3D1p.plot_surface(Tp,Fp,Ckp,cmap=cm.coolwarm);
            ax3D1p.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,Cinit,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost')
            pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D1p.set_zlim((-23.1,-22.3));
            ax3D2p=fig3Dp.add_subplot(122,projection='3d')
            fig2p=ax3D2p.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),cmap=cm.coolwarm);
            ax3D2p.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,Cinit,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2p.set_zlabel('Cost')
            pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D2p.set_zlim((-23.1,-22.3));
            pyp.savefig('Rxtr_2D_Surf_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
            pyp.close()
            
            # fig3Dgp=pyp.figure(figsize=[20,8])
            # ax3D1gp=fig3Dgp.add_subplot(121)
            # fig1gp=ax3D1gp.contourf(Tp,Fp,Ckp)
            # ax3D1gp.scatter(200*TFinitgp[:,0]+303,10*TFinitgp[:,1]+90,color='r',
            #               marker='x',s=40)
            # pyp.colorbar(fig1gp)
            # pyp.xlim((303,503)); pyp.ylim((40,100))
            # pyp.title('True Model')
            # ax3D2gp=fig3Dgp.add_subplot(122)
            # fig2gp=ax3D2gp.contourf(Tp,Fp,Cm.reshape((d1,d2)))
            # ax3D2gp.scatter(200*TFinitgp[:,0]+303,10*TFinitgp[:,1]+90,color='r',
            #               marker='x',s=40)
            # pyp.colorbar(fig2gp)
            # pyp.xlim((303,503)); pyp.ylim((90,100))
            # pyp.title('Estimated Model')
            # pyp.savefig('Rxtr_2Dgp_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
            # pyp.close()
            
            fig3Dpgp=pyp.figure(figsize=[20,8])
            ax3D1pgp=fig3Dpgp.add_subplot(121,projection='3d')
            fig1pgp=ax3D1pgp.plot_surface(Tp,Fp,Ckp,cmap=cm.coolwarm);
            ax3D1pgp.scatter(200*TFinitgp[:,0]+303,10*TFinitgp[:,1]+90,Cinitgp,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1pgp.set_zlabel('Cost')
            pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D1pgp.set_zlim((-23.1,-22.3));
            ax3D2pgp=fig3Dpgp.add_subplot(122,projection='3d')
            fig2pgp=ax3D2pgp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),cmap=cm.coolwarm);
            ax3D2pgp.scatter(200*TFinitgp[:,0]+303,10*TFinitgp[:,1]+90,Cinitgp,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2pgp.set_zlabel('Cost')
            pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D2pgp.set_zlim((-23.1,-22.3));
            pyp.savefig('Rxtr_2Dgp_Surf_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
            pyp.close()            
            
            fig3Dcomp=pyp.figure(figsize=[20,8])
            ax3D1comp=fig3Dcomp.add_subplot(121,projection='3d')
            fig1comp=ax3D1comp.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),cmap=cm.coolwarm);
            ax3D1comp.scatter(200*TFinit[:,0]+303,10*TFinit[:,1]+90,Cinit,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1comp.set_zlabel('Cost')
            pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D1comp.set_zlim((-23.1,-22.3));
            ax3D2comp=fig3Dcomp.add_subplot(122,projection='3d')
            fig2comp=ax3D2comp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),cmap=cm.coolwarm);
            ax3D2comp.scatter(200*TFinitgp[:,0]+303,10*TFinitgp[:,1]+90,Cinit,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); #ax3D2comp.set_zlabel('Cost')
            pyp.xlim((303,503)); pyp.ylim((90,100)); #ax3D2comp.set_zlim((-23.1,-22.3));
            pyp.savefig('Rxtr_2D_Surf_Comp_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
            pyp.close()
            
            # plt3d=pyp.figure().gca(projection='3d')
            # plt3d.plot_surface(Tp,Fp,Ckp)
            # plt3d.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)))
            # plt3d.scatter(450*TFinit[:,0]+323,50*TFinit[:,1]+50,Cinit[0],color='r',
            #               marker='x',size=20)
            # pyp.show()
    CBEST=CBEST+Cbest; CBESTGP=CBESTGP+Cbestgp;
    CPRED=CPRED+Cpred; STD=STD+std2d; CM=CM+Cm; STDGP=STDGP+std2dgp; 
    CIDX[j,0]=argmin(Cpred); CIDX[j,1]=min(Cpred); CIDXGP[j,0]=argmin(Cm); CIDXGP[j,1]=min(Cm);
    
        
#########################Using 'pure' GP model################################
    # Tinitgp=Tinit[0:1]; Cinitgp=Cinit[0:1]; Cbestgp=Cbest[0]; 

    # for i in range(t):
    #     Tnxt=Tm[argmax(afgp)]; Tinitgp=vstack([Tinitgp,Tnxt]);
    #     #Cnxt=Ctfxn(100*Tnxt+323,F)[0];
    #     Cnxt=Ctfxn(450*Tnxt+323,F)[0];
    #     Cinitgp=vstack([Cinitgp,Cnxt]); Cbestgp=vstack([Cbestgp,min(Cinitgp)])
    #     modelgp=GPy.models.GPRegression(Tinitgp,Cinitgp,kernelgp)
    #     modelgp.Gaussian_noise.variance=noise;
    #     modelgp.Gaussian_noise.variance.fix();
    #     Cm,stdgp=modelgp.predict(Tm); stdgp=stdgp**0.5; afgp=LCB(Cm,stdgp,exp_w);
    #     if j==lp-1:
    #         pyp.figure()
    #         pyp.plot(100*Tm+323,Cm)
    #         pyp.fill_between(100*Tm.reshape(-1)+323,(Cm-2*stdgp).reshape(-1),
    #                           (Cm+2*stdgp).reshape(-1),alpha=0.1);
    #         pyp.scatter(100*Tinitgp[0:i+2]+323,Cinitgp[0:i+2],marker='x',color='k')
    #         pyp.plot(100*Tm+323,Ck,'g--');
    #         pyp.xlim((323,423)); #pyp.ylim((-23.2,-22.15))
    #         pyp.legend(['Mean','f(x)','Confidence','Data'],loc='upper right')
    #         str1gp=str(i+2)
    #         pyp.savefig('Rxtr_1Dgp_Progression'+str1gp+'_Rest.png',dpi=300)
    #         pyp.close()
    # CBESTGP=CBESTGP+Cbestgp; CM=CM+Cm; STDGP=STDGP+stdgp;
    # CIDXGP[j,0]=argmin(Cm); CIDXGP[j,1]=min(Cm);

#Tm=100*Tm+323; Tinit=100*Tinit+323; Tinitgp=100*Tinitgp+323;
# Tm=450*Tm+323; Tinit=450*Tinit+323; Tinitgp=450*Tinitgp+323;
CBEST=CBEST/lp; CBESTGP=CBESTGP/lp; CPRED=CPRED/lp; CM=CM/lp;
STD=STD/lp; STDGP=STDGP/lp;
    
#%%################################Plots###################################%%#
# pyp.figure()
# pyp.plot(Tm,CPRED)
# pyp.fill_between(Tm.reshape(-1),(CPRED-2*STD).reshape(-1),
#                   (CPRED+2*STD).reshape(-1),alpha=0.1);
# pyp.plot(Tm,Ck,'g--')
# pyp.plot(Tm,Cmk,'r:')
# #pyp.xlim((323,423));
# pyp.xlim((323,773));
# pyp.legend(['Mean','f(x)',r'$f_{mod}(x)$','Confidence','Data'],loc='upper right')
# pyp.savefig('Avg_Mod_1D.png',dpi=300,bbox_inches='tight',pad_inches=0)

# pyp.figure()
# pyp.plot(Tm,dm,'c-.')
# pyp.scatter(Tinit,dinit,marker='x',color='k')
# pyp.xlim((323,773));

# # pyp.figure()
# # pyp.plot(Tm,af)

# #pyp.figure()
# #pyp.plot(Cbest)
# #pyp.scatter(itr,Cbest)

# pyp.figure()
# pyp.plot(Tm,CM)
# pyp.fill_between(Tm.reshape(-1),(CM-2*STDGP).reshape(-1),
#                   (CM+2*STDGP).reshape(-1),alpha=0.1);
# pyp.plot(Tm,Ck,'g--')
# #pyp.xlim((323,423));
# pyp.xlim((323,773));
# pyp.legend(['Mean','f(x)','Confidence','Data'],loc='upper right')
# pyp.savefig('Avg_Mod_1Dgp.png',dpi=300,bbox_inches='tight',pad_inches=0)

# pyp.figure()
# pyp.plot(Tm,afgp)

# pyp.figure()
# pyp.plot(Cbestgp)
# pyp.scatter(itr,Cbestgp)

pyp.figure()
pyp.plot(CBEST); pyp.scatter(itr,CBEST);
pyp.plot(CBESTGP); pyp.scatter(itr,CBESTGP);
#pyp.ylim((min(Ck)-0.01,-22.9))#max(Ck)+0.01));
pyp.xlabel('Iteration Number'); pyp.ylabel('Optimal Cost');
pyp.savefig('Avg_Comp_2D_Rest2.png',dpi=300,bbox_inches='tight',pad_inches=0)

pyp.figure()
pyp.hlines(min(Ck),0,lp+1,colors='k')
pyp.scatter(arange(1,lp+1,1),CIDX[:,1]); #pyp.plot(arange(1,101,1),CIDX[:,1]);
pyp.xlim((0,lp+1)); #pyp.ylim((-23.10,-22.90))#max(Ck)+0.01));
pyp.xlabel('Simulation Number'); pyp.ylabel('Optimal Cost');
pyp.legend(['True Solution','Hybrid BO Solution'])
pyp.savefig('Avg_Comp_2D_Sol2.png',dpi=300,bbox_inches='tight',pad_inches=0)

pyp.figure()
pyp.hlines(min(Ck),0,lp+1,colors='k')
pyp.scatter(arange(1,lp+1,1),CIDXGP[:,1],color='red'); #pyp.plot(arange(1,101,1),CIDXGP[:,1]);
pyp.xlim((0,lp+1)); #pyp.ylim((min(Ck)-0.01,-22.90))#max(Ck)+0.01));
pyp.xlabel('Simulation Number'); pyp.ylabel('Optimal Cost');
pyp.legend(['True Solution','Pure BO Solution'])
pyp.savefig('Avg_Comp_2D_SolGP2.png',dpi=300,bbox_inches='tight',pad_inches=0)

# images=[];
# for j in range(Tinit.shape[0]):
#     str2=str(j+1);
#     images.append(imageio.imread('Rxtr_1D_Progression'+str2+'_Rest.png'));
# imageio.mimsave('Rxtr_1D.gif',images,duration=1)

# imageserr=[];
# for j in range(Tinit.shape[0]):
#     str2=str(j+1);
#     imageserr.append(imageio.imread('Rxtr_1D_Error'+str2+'_Rest.png'));
# imageio.mimsave('Rxtr_1D_Error.gif',imageserr,duration=1)

# imagesgp=[];
# for j in range(Tinitgp.shape[0]):
#     str2gp=str(j+1);
#     imagesgp.append(imageio.imread('Rxtr_1Dgp_Progression'+str2gp+'_Rest.png'));
# imageio.mimsave('Rxtr_1Dgp.gif',imagesgp,duration=1)

# TFr=TFm[0]; Ckr=Ck[0]; Cpredr=Cpred[0]; Cmr=Cm[0]
# for i in range(1,TFm.shape[0]):
#     if i%50==0:
#         TFr=vstack([TFr,TFm[i]])
#         Ckr=vstack([Ckr,Ck[i]])
#         Cpredr=vstack([Cpredr,Cpred[i]])
#         Cmr=vstack([Cmr,Cm[i]])

# zr=0*ones(Cmr.shape)
# from mpl_toolkits.mplot3d import Axes3D
# fig3D=pyp.figure()
# ax3D=fig3D.add_subplot(111,projection='3d')
# #ax3D.scatter(TFr[:,0],TFr[:,1],Ckr,c='c',marker='o')
# ax3D.scatter(TFr[:,0],TFr[:,1],abs(Cpredr-Ckr),c='r',marker='x')
# ax3D.scatter(TFr[:,0],TFr[:,1],abs(Cmr-Ckr),c='g',marker='*')
# ax3D.scatter(TFr[:,0],TFr[:,1],zr,c='k',marker='.')

# fig3D=pyp.figure()
# ax3D=fig3D.add_subplot(111,projection='3d')
# #ax3D.scatter(TFr[:,0],TFr[:,1],Ckr,c='c',marker='o')
# ax3D.scatter(TFr[:,0],TFr[:,1],(Ckr),c='r',marker='x')
# ax3D.scatter(TFr[:,0],TFr[:,1],(Cmr),c='g',marker='*')

# fig3D=pyp.figure()
# ax3D=fig3D.add_subplot(111,projection='3d')
# #ax3D.scatter(TFr[:,0],TFr[:,1],Ckr,c='c',marker='o')
# ax3D.scatter(TFr[:,0],TFr[:,1],(Ckr),c='r',marker='x')
# ax3D.scatter(TFr[:,0],TFr[:,1],(Cpredr),c='g',marker='*')

# fig3Dp=pyp.figure(figsize=[20,8])
# ax3D1p=fig3Dp.add_subplot(121,projection='3d')
# fig1p=ax3D1p.plot_surface(Tp,Fp,Ckp,color='g',alpha=0.5);
# ax3D1p.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
# pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost')
# pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D1p.set_zlim((-23.1,-22.3));
# ax3D2p=fig3Dp.add_subplot(122,projection='3d')
# fig2p=ax3D2p.plot_surface(Tp,Fp,CPRED.reshape((d1,d2)),color='g',alpha=0.5);
# ax3D2p.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
# pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2p.set_zlabel('Cost')
# pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D2p.set_zlim((-23.1,-22.3));
# pyp.savefig('Avg_Mod_2D.png',dpi=300,bbox_inches='tight',pad_inches=0)

# fig3Dp=pyp.figure(figsize=[20,8])
# ax3D1p=fig3Dp.add_subplot(121,projection='3d')
# fig1p=ax3D1p.plot_surface(Tp,Fp,Ckp,color='g',alpha=0.5);
# ax3D1p.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
# pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost')
# pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D1p.set_zlim((-23.1,-22.3));
# ax3D2p=fig3Dp.add_subplot(122,projection='3d')
# fig2p=ax3D2p.plot_surface(Tp,Fp,CM.reshape((d1,d2)),color='g',alpha=0.5);
# ax3D2p.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
# pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2p.set_zlabel('Cost')
# pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D2p.set_zlim((-23.1,-22.3));
# pyp.savefig('Avg_Mod_2Dgp.png',dpi=300,bbox_inches='tight',pad_inches=0)

# images2d=[];
# for j in range(TFinit.shape[0]):
#     str2=str(j+1);
#     images2d.append(imageio.imread('Rxtr_2D_Progression'+str2+'_Rest.png'));
# imageio.mimsave('Rxtr_2D_Rest2.gif',images2d,duration=0.5)

images2ds=[];
for j in range(TFinit.shape[0]):
    str2=str(j+1);
    images2ds.append(imageio.imread('Rxtr_2D_Surf_Progression'+str2+'_Rest.png'));
imageio.mimsave('Rxtr_2D_Surf_Rest2.gif',images2ds,duration=0.5)

# images2dgp=[];
# for j in range(TFinitgp.shape[0]):
#     str2=str(j+1);
#     images2dgp.append(imageio.imread('Rxtr_2Dgp_Progression'+str2+'_Rest.png'));
# imageio.mimsave('Rxtr_2Dgp_Rest2.gif',images2dgp,duration=0.5)

images2dsgp=[];
for j in range(TFinitgp.shape[0]):
    str2=str(j+1);
    images2dsgp.append(imageio.imread('Rxtr_2Dgp_Surf_Progression'+str2+'_Rest.png'));
imageio.mimsave('Rxtr_2Dgp_Surf_Rest2.gif',images2dsgp,duration=0.5)

images2dscomp=[];
for j in range(TFinitgp.shape[0]):
    str2=str(j+1);
    images2dscomp.append(imageio.imread('Rxtr_2D_Surf_Comp_Progression'+str2+'_Rest.png'));
imageio.mimsave('Rxtr_2D_Surf_Comp_Rest2.gif',images2dscomp,duration=0.5)
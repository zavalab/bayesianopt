# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:55:13 2020

@author: leonardo
"""

import sys;
sys.path.insert(1,r'C:\Users\leonardo\OneDrive - UW-Madison\Research\bayesianopt\Scripts')
from numpy import exp, arange, random, array, vstack, argmax, delete, hstack
from numpy import argmin, loadtxt, apply_along_axis, ones, round, intersect1d
from numpy import cos, sin, log, matmul, linalg, transpose as trans, meshgrid
from matplotlib import pyplot as pyp
import time
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import GPy
from Acquisition_Funcs import LCB, EI, PI
from scipy.optimize import fsolve
import imageio
from mpl_toolkits.mplot3d import Axes3D

##################Initial Conditions and Physical Parameters##################
k01=550000; k02=500000; p=850; Cp=3000; CpH2O=4184; R=8.314;
E1=30500; E2=33500; H1=-210000; H2=-1700000; Cr0=10000; Ci0=0; Cp0=0;
Tin=323; TinH2O=298; V=1; cP=-510; cR=13; cT=0.070; cO=5; W=10000; cI=29#14.5
##############################################################################
#T range is 323 to 423
#F rabge is 50 to 100
#Units are all in MKS 

def Ctfxn(T,F):
    T=array([T],dtype='float64').reshape(-1,1);
    F=array([F],dtype='float64').reshape(-1,1);
    k1=k01*exp(-E1/(R*T)); k2=k02*exp(-E2/(R*T)); ToH2O=T-10;
    k1r=k1/100; k2r=k2/100;
    Crinit=200; Ciinit=5000; Cpdinit=170;
    Crinit=((F[0,0]*(Cr0-Crinit)+2*k1r[0,0]*Ciinit*V)/(2*k1[0,0]*V))**(1/2);
    Ciinit=(k1[0,0]*Crinit**2*V+k2r[0,0]*Cpdinit**2*V)/(F[0,0]+k1r[0,0]*V+k2[0,0]*V);
    Cpdinit=Cpdinit+(((2*k2[0,0]*Ciinit*V-F[0,0]*Cpdinit)/(2*k2r[0,0]*V))**(1/2)
                     -Cpdinit)/5
    
    C0=[Crinit,Ciinit,Cpdinit];
    Csoln=ones([1,3])    
    def C(C):
        Crg=C[0]; Cig=C[1]; Cpdg=C[2]
#        Cr=(F*Cr0+k1r*Cig**2*V)/(F+k1*V);
#        Ci=((2*(k1*Crg+k2r*Cpdg)*V-F*Cig)/(2*(k1r+k2)*V))**(0.5);
#        Cpd=(k2*Cig**2*V)/(F+k2r*V);
        Cr=((Ff*(Cr0-Crg)+2*k1rf*Cig*V)/(2*k1f*V))**(1/2);
        Ci=(k1f*Crg**2*V+k2rf*Cpdg**2*V)/(Ff+k1rf*V+k2f*V);
        Cpd=((2*k2f*Cig*V-Ff*Cpdg)/(2*k2rf*V))**(1/2)
        return [Cr-Crg,Ci-Cig,Cpd-Cpdg]
    
    for j in range(T.shape[0]):
        k1f=k1[j,0]; k2f=k2[j,0]; k1rf=k1r[j,0]; k2rf=k2r[j,0]; Ff=F[j,0]
        soln=fsolve(C,C0); C0=array(soln); Cr=C0[0]; Ci=C0[1]; Cpd=C0[2];
        Csoln=vstack([Csoln,array([C0[0],C0[1],C0[2]]).reshape(1,3)]);
    Csoln=Csoln[1:Csoln.shape[0],:]
        
#    a=2*(k1-(k1*k1r*V)/(F+(k1r+k2)*V))*V
#    Cr=(-F/a+((F/a)**2+4*F*Cr0/a)**0.5)/2
#    Cr=(-F/(2*k1*V)+((F/(2*k1*V))**2+4*F*Cr0/(2*k1*V))**0.5)/2
#    for k in range(100):
#        Cr=((F*(Cr0-Crg)+2*k1r*Cig*V)/(2*k1*V))**(1/2);
#        Ci=(k1*Crg**2*V+k2r*Cpdg**2*V)/(F+k1r*V+k2*V);
#        Cpd=((2*k2*Cig*V-F*Cpdg)/(2*k2r*V))**(1/2);
#        if (2*k2*Cig*V-F*Cpdg)<0:
#            Cpd=2*k2*Cig*V/F#random.uniform(100,200)
#        res=(Cr-Crg)**2+(Ci-Cig)**2+(Cpd-Cpdg)**2
#        if res<=1e-6:
#            break
#        Crg=Cr*1; Cig=Ci*1; Cpdg=Cpdg+(Cpd-Cpdg)/5#Cpdg=Cpd*1
#        if T>353:
#            Cpdg=Cpd*1
#    Ci=(k1*Cr**2*V)/(F+(k1r+k2)*V)
#    Ci=(F*Ci0+k1*Cr**3*V)/(F+k2*V)
#    Cpd=k2*Ci*V/F
#    for k in range(100):    
#        Cr=(F*Cr0+2*k1r*Ci*V)/(F+2*k1*Cr*V);
#        Ci=(k1*Cr**2*V+3*k2r*Cpd*V)/(F+k1r*V+3*k2*Ci**2*V)
#        Cpd=(k2*Ci**3*V)/(F+k2r*V)
    Cr=Csoln[:,0].reshape(-1,1); Ci=Csoln[:,1].reshape(-1,1);
    Cpd=Csoln[:,2].reshape(-1,1); r1=k1*Cr**2-k1r*Ci; r2=k2*Ci-k2r*Cpd**2;
    mH2O=(p*Cp*F*(Tin-T)-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O))
    
    g=ones(Cpd.shape)
    for i in range(Cpd.shape[0]):
        g[i]=max([0,Cpd[i]-600])

    Ct=cP*V*Ci+cO*W+cT*mH2O+cI*Cpd*V+cR*Cr*V+250*g
    return Ct/1e5,r1,r2,mH2O,Cr,Ci,Cpd

def Ctfxn_mod(T,F):
    r1=theta[0,0]*T**2+theta[0,1]*T+theta[0,2]
    r2=theta[1,0]*T**2+theta[1,1]*T+theta[1,2]
    
    ToH2O=T-10; 
    Cr=Cr0-2*r1*V/F
    Ci=(r1-r2)*V/F
    Cpd=2*r2*V/F
    mH2O=(p*Cp*F*(Tin-T)-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O))
    
    g=ones(Cpd.shape)
    for i in range(Cpd.shape[0]):
        g[i]=max([0,Cpd[i]-600])
    
    Ct=cP*V*Ci+cO*W+cT*mH2O+cI*Cpd*V+cR*Cr*V+250*g
    return Ct/1e5,r1,r2,mH2O,Cr,Ci,Cpd

def Ctfxn_modM(T,F):
    T=array([T],dtype='float64').reshape(-1,1);
    F=array([F],dtype='float64').reshape(-1,1);
    r1=thetam[0,0]*T**2+thetam[0,1]*T+thetam[0,2]*F+thetam[0,3]
    r2=thetam[1,0]*T**2+thetam[1,1]*T+thetam[1,2]*F+thetam[1,3]
    
    ToH2O=T-10; 
    Cr=Cr0-2*r1*V/F
    Ci=(r1-r2)*V/F
    Cpd=2*r2*V/F
    mH2O=(p*Cp*F*(Tin-T)-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O))
    
    g=ones(Cpd.shape)
    for i in range(Cpd.shape[0]):
        g[i]=max([0,Cpd[i]-600])
    
    Ct=cP*V*Ci+cO*W+cT*mH2O+cI*Cpd*V+cR*Cr*V+250*g
    return Ct/1e5,r1,r2,mH2O,Cr,Ci,Cpd

# def CtfxnMD(x):
#     x=x.reshape(1,2)
#     T=200*x[0,0]+323; F=50*x[0,1]+50; #Cr0=9000*x[0,2]+1000
#     k1=k01*exp(-E1/(R*T)); k2=k02*exp(-E2/(R*T)); ToH2O=T-10; k1r=k1/100
    
#     a=2*(k1-(k1*k1r*V)/(F+(k1r+k2)*V))*V
#     Cr=(-F/a+((F/a)**2+4*F*Cr0/a)**0.5)/2
# #    Cr=(-F/(2*k1*V)+((F/(2*k1*V))**2+4*F*Cr0/(2*k1*V))**0.5)/2
#     Ci=(k1*Cr**2*V)/(F+(k1r+k2)*V)
# #    Ci=(F*Ci0+k1*Cr**2*V)/(F+k2*V)
#     Cpd=k2*Ci*V/F
#     r1=k1*Cr**2-k1r*Ci; r2=k2*Ci;
#     mH2O=(p*Cp*F*(Tin-T)-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O))

#     C=cP*V*Ci+cO*W+cT*mH2O+cI*Cpd*V
#     return C/1e4,r1/1e13,r2/1e13,mH2O,Cr,Ci,Cpd

# def Ctfxn_modMD(x):
#     x=x.reshape(1,2)
#     T=200*x[0,0]+323; F=50*x[0,1]+50; #Cr0=9000*x[0,2]+1000
#     r1=theta[0,0]*T+theta[0,1]*F+theta[0,2]
#     r2=theta[1,0]*T+theta[1,1]*F+theta[1,2]
    
#     ToH2O=T-10; #r1=r1/2; r2=r2/2; 
#     Cr=(Cr0*F-2*r1*V)/F
#     Ci=(r1-r2)*V/F
#     Cpd=r2*V/F
#     mH2O=(p*Cp*F*(Tin-T)-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O))
    
#     C=cP*V*Ci+cO*W+cT*mH2O+cI*Cpd*V
#     return C/1e4,r1,r2,mH2O,Cr,Ci,Cpd

#T=arange(373,974,1); 
#Ct=array(list(map(Ctfxn, T)));
#Cr=Ct[:,1]; Ci=Ct[:,2]; Cp=Ct[:,3]; Ca=Ct[:,4]; Ct=Ct[:,0]

####################### Reaction Regression ##################################
# Univariate Regression
Tr=random.uniform(323,774,(200,1)); Fr=90*ones(Tr.shape)
IR=hstack([Tr,Fr]); Ctr=ones((Tr.shape[0],2))
for i in range(IR.shape[0]):
    Ctr[i,:]=Ctfxn(IR[i,0],IR[i,1])[1:3];
#Ctr=Ctfxn(IR[:,0],IR[:,1])[1:3]
Ctr=array(Ctr).reshape(IR.shape[0],2)
A=ones(IR.shape[0]).reshape(-1,1); A=hstack([Tr**2,Tr,A]);
psuedoAinv=matmul(linalg.inv(matmul(trans(A),A)),trans(A));
w1=matmul(psuedoAinv,Ctr[:,0]); w2=matmul(psuedoAinv,Ctr[:,1]);
theta=vstack([w1,w2]);

#Multivariate regression
Frm=random.uniform(40,100,(200,1));
IRm=hstack([Tr,Frm]); Ctrm=ones((Tr.shape[0],2))
for i in range(IRm.shape[0]):
     Ctrm[i,:]=Ctfxn(IRm[i,0],IRm[i,1])[1:3];
#Ctr=Ctfxn(IR[:,0],IR[:,1])[1:3]
Ctrm=array(Ctrm).reshape(IRm.shape[0],2)
Am=ones(IRm.shape[0]).reshape(-1,1); Am=hstack([IRm[:,0].reshape(-1,1)**2,IRm,Am]);
psuedoAinvm=matmul(linalg.inv(matmul(trans(Am),Am)),trans(Am));
w1=matmul(psuedoAinvm,Ctrm[:,0]); w2=matmul(psuedoAinvm,Ctrm[:,1]);
thetam=vstack([w1,w2]);

#theta=array([[14.74793401,4881.686336,-3935.661266],
#               [290.6382819,39.96915156,-93949.59079]])

#theta=array([[ 3.04726624e+01,4.94513631e+03,-1.18770315e+04],
#       [ 1.51278679e+03,8.50418845e+02,-5.75150104e+05]])
#################################SETUP########################################
noise=1e-4; exp_w=2; t=20; itr=array(arange(t+1)).reshape(-1,1); F=90; T=423;
lp=100; CBEST=0*ones((t+1,1)); CBESTGP=0*ones((t+1,1)); idx=ones((t,1),dtype=int);
#TINIT=0*ones((t+1,1)); TINITGP=0*ones((t+1,1));
#CINIT=0*ones((t+1,1)); CINITGP=0*ones((t+1,1));

Tm=arange(0,1.001,0.001).reshape(-1,1); Fm=F*ones(Tm.shape);
Ck=Ctfxn(450*Tm+323,Fm)[0].reshape(-1,1);
Cmk=Ctfxn_mod(450*Tm+323,Fm)[0].reshape(-1,1);
#Fm=arange(0,1.001,0.001).reshape(-1,1); Tm=T*ones(Fm.shape)
#Ck=Ctfxn(Tm,50*Fm+50)[0].reshape(-1,1)
#Cmk=Ctfxn_mod(Tm,50*Fm+50)[0].reshape(-1,1);

## MULTIVARIATE CASE:
# Tp=arange(323,774,1); Fp=arange(40,100.1,0.1)
# Fp,Tp=meshgrid(Fp,Tp); Ckp=ones(Tp.shape); Cmkp=ones(Tp.shape);
# d1=Tp.shape[0]; d2=Tp.shape[1]
# for g in range(Fp.shape[1]):
#     for h in range(Fp.shape[0]):
#         Ckp[h,g]=Ctfxn(Tp[h,g],Fp[h,g])[0]
#         Cmkp[h,g]=Ctfxn_modM(Tp[h,g],Fp[h,g])[0]
# Tm=(Tp.reshape(-1,1)-323)/450; Fm=(Fp.reshape(-1,1)-40)/60;
# Ck=Ckp.reshape(-1,1); Cmk=Cmkp.reshape(-1,1); TFm=hstack([Tm,Fm])

#%%###############################RUNS#####################################%%#            
for j in range(lp):
    Tinit=random.uniform(0,1,1).reshape(-1,1);#array([[0.8]])
    Cinit=Ctfxn(450*Tinit+323,F)[0].reshape(-1,1); Cbest=min(Cinit).reshape(-1,1)
    dinit=Ctfxn(450*Tinit+323,F)[0]-Ctfxn_mod(450*Tinit+323,F)[0]

    kernel=GPy.kern.Matern52(1,variance=30,lengthscale=1)
    model=GPy.models.GPRegression(Tinit,dinit,kernel);
    model.Gaussian_noise.variance=noise;
    model.Gaussian_noise.variance.fix();
    dm,std=model.predict(Tm); std=std**0.5; af=LCB(Cmk+dm,std,exp_w);
    Cpred=Cmk+dm;
    if j==lp-1:
        pyp.figure()
        pyp.plot(450*Tm+323,Cpred)
        pyp.fill_between(450*Tm.reshape(-1)+323,(Cpred-2*std).reshape(-1),
                          (Cpred+2*std).reshape(-1),alpha=0.1);
        pyp.scatter(450*Tinit[0]+323,Cinit[0],marker='x',color='k');
        pyp.plot(450*Tm+323,Ck,'g--');
        pyp.plot(450*Tm+323,Cmk,'r:');
        pyp.xlim((323,773)); pyp.ylim((-23.2,-22.15));
#        pyp.legend(['Mean','f(x)',r'$f_{mod}(x)$','Confidence','Data'],loc='upper right');
#        pyp.savefig('Rxtr_1D_Progression1_Rest.png',dpi=300);
        pyp.close();
        pyp.figure();
        pyp.scatter(450*Tinit[0]+323,dinit[0],marker='x',color='k');
        pyp.plot(450*Tm+323,dm,'c-.');
        pyp.xlim((323,773));
#        pyp.savefig('Rxtr_1D_Error1_Rest.png',dpi=300);
#        pyp.close();

    kernelgp=GPy.kern.Matern52(1,variance=30,lengthscale=1)
    modelgp=GPy.models.GPRegression(Tinit,Cinit,kernelgp);
    modelgp.Gaussian_noise.variance=noise;
    modelgp.Gaussian_noise.variance.fix();
    Cm,stdgp=modelgp.predict(Tm); stdgp=stdgp**0.5; afgp=LCB(Cm,stdgp,exp_w);
    if j==lp-1:
        pyp.figure()
        pyp.plot(450*Tm+323,Cm)
        pyp.fill_between(450*Tm.reshape(-1)+323,(Cm-2*stdgp).reshape(-1),
                          (Cm+2*stdgp).reshape(-1),alpha=0.1);
        pyp.scatter(450*Tinit[0]+323,Cinit[0],marker='x',color='k')
        pyp.plot(450*Tm+323,Ck,'g--');
        pyp.xlim((323,773)); pyp.ylim((-23.2,-22.15))
        pyp.legend(['Mean','f(x)','Confidence','Data'],loc='upper right')
#        pyp.savefig('Rxtr_1Dgp_Progression1_Rest.png',dpi=300)
#        pyp.close()

    
## MULTIVARIATE CASE:    
    ## Ck=apply_along_axis(CtfxnMD,1,TFm)[:,0].reshape(-1,1);
    ## Cmk=apply_along_axis(Ctfxn_modMD,1,TFm)[:,0].reshape(-1,1);
    ## TFm=loadtxt('2D_mech_rxtr.txt');
    
    # TFinit=random.uniform(0,1,2).reshape(1,2)#array([[1,1]])
    # Cinit=Ctfxn(450*TFinit[0,0]+323,60*TFinit[0,1]+40)[0].reshape(-1,1); Cbest=min(Cinit);
    # d2dinit=Cinit-Ctfxn_modM(450*TFinit[0,0]+323,60*TFinit[0,1]+40)[0].reshape(-1,1);

    # kernel2d=GPy.kern.Matern52(2,variance=30,lengthscale=1)
    # model2d=GPy.models.GPRegression(TFinit,d2dinit,kernel2d)
    # model2d.Gaussian_noise.variance=noise;
    # model2d.Gaussian_noise.variance.fix();
    # dm2d,std2d=model2d.predict(TFm); std2d=std2d**0.5; af2d=LCB((Cmk+dm2d),std2d,exp_w);
    # Cpred=Cmk+dm2d;
    
    # kernel2dgp=GPy.kern.Matern52(2,variance=30,lengthscale=1)
    # model2dgp=GPy.models.GPRegression(TFinit,Cinit,kernel2dgp)
    # model2dgp.Gaussian_noise.variance=noise;
    # model2dgp.Gaussian_noise.variance.fix();
    # Cm,std2dgp=model2dgp.predict(TFm); std2dgp=std2dgp**0.5; af2dgp=LCB(Cm,std2dgp,exp_w);
    # TFinitgp=TFinit[0:1]; Cinitgp=Cinit[0:1]; Cbestgp=Cbest[0];
    
    # if j==lp-1:
    #     fig3D=pyp.figure(figsize=[20,8])
    #     ax3D1=fig3D.add_subplot(121)
    #     fig1=ax3D1.contourf(Tp,Fp,Ckp)
    #     ax3D1.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,color='r',
    #                   marker='x',s=40)
    #     pyp.colorbar(fig1)
    #     pyp.xlim((323,773)); pyp.ylim((40,100))
    #     pyp.title('True Model')
    #     ax3D2=fig3D.add_subplot(122)
    #     fig2=ax3D2.contourf(Tp,Fp,Cpred.reshape((d1,d2)))
    #     ax3D2.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,color='r',
    #                   marker='x',s=40)
    #     pyp.colorbar(fig2)
    #     pyp.xlim((323,773)); pyp.ylim((40,100))
    #     pyp.title('Estimated Model')
    #     pyp.savefig('Rxtr_2D_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
    #     pyp.close()
        
    #     fig3Dp=pyp.figure(figsize=[20,8])
    #     ax3D1p=fig3Dp.add_subplot(121,projection='3d')
    #     fig1p=ax3D1p.plot_surface(Tp,Fp,Ckp,color='g',alpha=0.5);
    #     ax3D1p.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
    #     pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost')
    #     pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D1p.set_zlim((-23.1,-22.3));
    #     ax3D2p=fig3Dp.add_subplot(122,projection='3d')
    #     fig2p=ax3D2p.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),color='g',alpha=0.5);
    #     ax3D2p.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
    #     pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2p.set_zlabel('Cost')
    #     pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D2p.set_zlim((-23.1,-22.3));
    #     pyp.savefig('Rxtr_2D_Surf_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
    #     pyp.close()
        
    #     fig3Dgp=pyp.figure(figsize=[20,8])
    #     ax3D1gp=fig3Dgp.add_subplot(121)
    #     fig1gp=ax3D1gp.contourf(Tp,Fp,Ckp)
    #     ax3D1gp.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,color='r',
    #                   marker='x',s=40)
    #     pyp.colorbar(fig1gp)
    #     pyp.xlim((323,773)); pyp.ylim((40,100))
    #     pyp.title('True Model')
    #     ax3D2gp=fig3Dgp.add_subplot(122)
    #     fig2gp=ax3D2gp.contourf(Tp,Fp,Cm.reshape((d1,d2)))
    #     ax3D2gp.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,color='r',
    #                   marker='x',s=40)
    #     pyp.colorbar(fig2gp)
    #     pyp.xlim((323,773)); pyp.ylim((40,100));
    #     pyp.title('Estimated Model');
    #     pyp.savefig('Rxtr_2Dgp_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
    #     pyp.close()
        
    #     fig3Dpgp=pyp.figure(figsize=[20,8])
    #     ax3D1pgp=fig3Dpgp.add_subplot(121,projection='3d')
    #     fig1pgp=ax3D1pgp.plot_surface(Tp,Fp,Ckp,color='g',alpha=0.5);
    #     ax3D1pgp.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
    #     pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1pgp.set_zlabel('Cost')
    #     pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D1pgp.set_zlim((-23.1,-22.3));
    #     ax3D2pgp=fig3Dpgp.add_subplot(122,projection='3d')
    #     fig2pgp=ax3D2pgp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),color='g',alpha=0.5);
    #     ax3D2pgp.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
    #     pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2pgp.set_zlabel('Cost')
    #     pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D2pgp.set_zlim((-23.1,-22.3));
    #     pyp.savefig('Rxtr_2Dgp_Surf_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
    #     pyp.close()
        
    #     fig3Dcomp=pyp.figure(figsize=[20,8])
    #     ax3D1comp=fig3Dcomp.add_subplot(121,projection='3d')
    #     fig1comp=ax3D1comp.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),color='g',alpha=0.5);
    #     ax3D1comp.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
    #     pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1comp.set_zlabel('Cost')
    #     pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D1comp.set_zlim((-23.1,-22.3));
    #     ax3D2comp=fig3Dcomp.add_subplot(122,projection='3d')
    #     fig2comp=ax3D2comp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),color='g',alpha=0.5);
    #     ax3D2comp.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
    #     pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2comp.set_zlabel('Cost')
    #     pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D2comp.set_zlim((-23.1,-22.3));
    #     pyp.savefig('Rxtr_2D_Surf_Comp_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
    #     pyp.close()
        
#        plt3d=pyp.figure().gca(projection='3d')
#        plt3d.plot_surface(Tp,Fp,Ckp)
#        plt3d.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)))
#        plt3d.scatter(450*TFinit[:,0]+323,50*TFinit[:,1]+50,Cinit[0],color='r',
#                      marker='x',size=20)
#        pyp.show()
        
        
#######################Using 'mechanistic' model##############################
    for i in range(t):
        Tnxt=Tm[argmax(af)]; Cnxt=Ctfxn(450*Tnxt+323,F)[0];
#    mnxt=m(xnxt); dnxt=ynxt-mnxt; 
        Cmnxt=Cmk[argmax(af)]; dnxt=Cnxt-Cmnxt; idx[i]=argmax(af)
        Tinit=vstack([Tinit,Tnxt]); Cinit=vstack([Cinit,Cnxt]);
        dinit=vstack([dinit,dnxt]); Cbest=vstack([Cbest,min(Cinit)])
        model=GPy.models.GPRegression(Tinit,dinit,kernel);
        model.Gaussian_noise.variance=noise;
        model.Gaussian_noise.variance.fix();
        dm,std=model.predict(Tm); std=std**0.5; af=LCB(Cmk+dm,std,exp_w);
        Cpred=Cmk+dm;
        if j==lp-1:
            pyp.figure()
            pyp.plot(450*Tm+323,Cpred)
            pyp.fill_between(450*Tm.reshape(-1)+323,(Cpred-2*std).reshape(-1),
                              (Cpred+2*std).reshape(-1),alpha=0.1);
            pyp.scatter(450*Tinit[0:i+2]+323,Cinit[0:i+2],marker='x',color='k');
            pyp.plot(450*Tm+323,Ck,'g--');
            pyp.plot(450*Tm+323,Cmk,'r:');
            pyp.xlim((323,773)); pyp.ylim((-23.2,-22.15));
            pyp.legend(['Mean','f(x)',r'$f_{mod}(x)$','Confidence','Data'],loc='upper right');
            str1=str(i+2);
#            pyp.savefig('Rxtr_1D_Progression'+str1+'_Rest.png',dpi=300);
#            pyp.close();  
            pyp.figure();
            pyp.scatter(450*Tinit[0:i+2]+323,dinit[0:i+2],marker='x',color='k');
            pyp.plot(450*Tm+323,dm,'c-.');
            pyp.xlim((323,773));
#            pyp.savefig('Rxtr_1D_Error'+str1+'_Rest.png',dpi=300);
#            pyp.close();
    CBEST=CBEST+Cbest; #TINIT=TINIT+Tinit; CINIT=CINIT+Cinit
    
## MULTIVARIATE CASE:
    # for i in range(t):
    #     TFnxt=TFm[argmax(af2d)].reshape(1,2);
    #     Cnxt=Ctfxn(450*TFnxt[0,0]+323,60*TFnxt[0,1]+40)[0];
    #     Cmnxt=Ctfxn_modM(450*TFnxt[0,0]+323,60*TFnxt[0,1]+40)[0];
    #     d2dnxt=Cnxt-Cmnxt;
    #     TFinit=vstack([TFinit,TFnxt]); Cinit=vstack([Cinit,Cnxt]);
    #     d2dinit=vstack([d2dinit,d2dnxt]); Cbest=vstack([Cbest,min(Cinit)])
    #     model2d=GPy.models.GPRegression(TFinit,d2dinit,kernel2d);
    #     model2d.Gaussian_noise.variance=noise;
    #     model2d.Gaussian_noise.variance.fix();
    #     dm2d,std2d=model2d.predict(TFm); std2d=std2d**0.5; af2d=LCB((Cmk+dm2d),std2d,exp_w);
    #     Cpred=Cmk+dm2d;
        
    #     TFnxt=TFm[argmax(af2dgp)].reshape(1,2);
    #     Cnxt=Ctfxn(450*TFnxt[0,0]+323,60*TFnxt[0,1]+40)[0];
    #     TFinitgp=vstack([TFinitgp,TFnxt]); Cinitgp=vstack([Cinitgp,Cnxt]);
    #     Cbestgp=vstack([Cbestgp,min(Cinitgp)])
    #     model2dgp=GPy.models.GPRegression(TFinitgp,Cinitgp,kernel2dgp)
    #     model2dgp.Gaussian_noise.variance=noise;
    #     model2dgp.Gaussian_noise.variance.fix();
    #     Cm,std2dgp=model2dgp.predict(TFm); std2dgp=std2dgp**0.5; af2dgp=LCB(Cm,std2dgp,exp_w);
        
    #     if j==lp-1:
    #         str1=str(i+2);
    #         fig3D=pyp.figure(figsize=[20,8])
    #         ax3D1=fig3D.add_subplot(121)
    #         fig1=ax3D1.contourf(Tp,Fp,Ckp)
    #         ax3D1.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,color='r',
    #                       marker='x',s=40)
    #         pyp.colorbar(fig1)
    #         pyp.xlim((323,773)); pyp.ylim((40,100))
    #         pyp.title('True Model')
    #         ax3D2=fig3D.add_subplot(122)
    #         fig2=ax3D2.contourf(Tp,Fp,Cpred.reshape((d1,d2)))
    #         ax3D2.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,color='r',
    #                       marker='x',s=40)
    #         pyp.colorbar(fig2)
    #         pyp.xlim((323,773)); pyp.ylim((40,100))
    #         pyp.title('Estimated Model')
    #         pyp.savefig('Rxtr_2D_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
    #         pyp.close()
            
    #         fig3Dp=pyp.figure(figsize=[20,8])
    #         ax3D1p=fig3Dp.add_subplot(121,projection='3d')
    #         fig1p=ax3D1p.plot_surface(Tp,Fp,Ckp,color='g',alpha=0.5);
    #         ax3D1p.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
    #         pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost')
    #         pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D1p.set_zlim((-23.1,-22.3));
    #         ax3D2p=fig3Dp.add_subplot(122,projection='3d')
    #         fig2p=ax3D2p.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),color='g',alpha=0.5);
    #         ax3D2p.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
    #         pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2p.set_zlabel('Cost')
    #         pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D2p.set_zlim((-23.1,-22.3));
    #         pyp.savefig('Rxtr_2D_Surf_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
    #         pyp.close()
            
    #         fig3Dgp=pyp.figure(figsize=[20,8])
    #         ax3D1gp=fig3Dgp.add_subplot(121)
    #         fig1gp=ax3D1gp.contourf(Tp,Fp,Ckp)
    #         ax3D1gp.scatter(450*TFinitgp[:,0]+323,60*TFinitgp[:,1]+40,color='r',
    #                       marker='x',s=40)
    #         pyp.colorbar(fig1gp)
    #         pyp.xlim((323,773)); pyp.ylim((40,100))
    #         pyp.title('True Model')
    #         ax3D2gp=fig3Dgp.add_subplot(122)
    #         fig2gp=ax3D2gp.contourf(Tp,Fp,Cm.reshape((d1,d2)))
    #         ax3D2gp.scatter(450*TFinitgp[:,0]+323,60*TFinitgp[:,1]+40,color='r',
    #                       marker='x',s=40)
    #         pyp.colorbar(fig2gp)
    #         pyp.xlim((323,773)); pyp.ylim((40,100))
    #         pyp.title('Estimated Model')
    #         pyp.savefig('Rxtr_2Dgp_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
    #         pyp.close()
            
    #         fig3Dpgp=pyp.figure(figsize=[20,8])
    #         ax3D1pgp=fig3Dpgp.add_subplot(121,projection='3d')
    #         fig1pgp=ax3D1pgp.plot_surface(Tp,Fp,Ckp,color='g',alpha=0.5);
    #         ax3D1pgp.scatter(450*TFinitgp[:,0]+323,60*TFinitgp[:,1]+40,Cinitgp,color='k')
    #         pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1pgp.set_zlabel('Cost')
    #         pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D1pgp.set_zlim((-23.1,-22.3));
    #         ax3D2pgp=fig3Dpgp.add_subplot(122,projection='3d')
    #         fig2pgp=ax3D2pgp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),color='g',alpha=0.5);
    #         ax3D2pgp.scatter(450*TFinitgp[:,0]+323,60*TFinitgp[:,1]+40,Cinitgp,color='k')
    #         pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2pgp.set_zlabel('Cost')
    #         pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D2pgp.set_zlim((-23.1,-22.3));
    #         pyp.savefig('Rxtr_2Dgp_Surf_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
    #         pyp.close()            
            
    #         fig3Dcomp=pyp.figure(figsize=[20,8])
    #         ax3D1comp=fig3Dcomp.add_subplot(121,projection='3d')
    #         fig1comp=ax3D1comp.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),color='g',alpha=0.5);
    #         ax3D1comp.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
    #         pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1comp.set_zlabel('Cost')
    #         pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D1comp.set_zlim((-23.1,-22.3));
    #         ax3D2comp=fig3Dcomp.add_subplot(122,projection='3d')
    #         fig2comp=ax3D2comp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),color='g',alpha=0.5);
    #         ax3D2comp.scatter(450*TFinit[:,0]+323,60*TFinit[:,1]+40,Cinit,color='k')
    #         pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2comp.set_zlabel('Cost')
    #         pyp.xlim((323,773)); pyp.ylim((40,100)); ax3D2comp.set_zlim((-23.1,-22.3));
    #         pyp.savefig('Rxtr_2D_Surf_Comp_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
    #         pyp.close()
            
            # plt3d=pyp.figure().gca(projection='3d')
            # plt3d.plot_surface(Tp,Fp,Ckp)
            # plt3d.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)))
            # plt3d.scatter(450*TFinit[:,0]+323,50*TFinit[:,1]+50,Cinit[0],color='r',
            #               marker='x',size=20)
            # pyp.show()
#    CBEST=CBEST+Cbest; CBESTGP=CBESTGP+Cbestgp;
        
#########################Using 'pure' GP model################################
    Tinitgp=Tinit[0:1]; Cinitgp=Cinit[0:1]; Cbestgp=Cbest[0]; 

    for i in range(t):
        Tnxt=Tm[argmax(afgp)]; Cnxt=Ctfxn(450*Tnxt+323,F)[0]; Tinitgp=vstack([Tinitgp,Tnxt]);
        Cinitgp=vstack([Cinitgp,Cnxt]); Cbestgp=vstack([Cbestgp,min(Cinitgp)])
        modelgp=GPy.models.GPRegression(Tinitgp,Cinitgp,kernelgp)
        modelgp.Gaussian_noise.variance=noise;
        modelgp.Gaussian_noise.variance.fix();
        Cm,stdgp=modelgp.predict(Tm); stdgp=stdgp**0.5; afgp=LCB(Cm,stdgp,exp_w);
        if j==lp-1:
            pyp.figure()
            pyp.plot(450*Tm+323,Cm)
            pyp.fill_between(450*Tm.reshape(-1)+323,(Cm-2*stdgp).reshape(-1),
                              (Cm+2*stdgp).reshape(-1),alpha=0.1);
            pyp.scatter(450*Tinitgp[0:i+2]+323,Cinitgp[0:i+2],marker='x',color='k')
            pyp.plot(450*Tm+323,Ck,'g--');
            pyp.xlim((323,773)); pyp.ylim((-23.2,-22.15))
            pyp.legend(['Mean','f(x)','Confidence','Data'],loc='upper right')
            str1gp=str(i+2)
#            pyp.savefig('Rxtr_1Dgp_Progression'+str1gp+'_Rest.png',dpi=300)
#            pyp.close()
    CBESTGP=CBESTGP+Cbestgp; #TINITGP=TINITGP+Tinitgp; CINITGP=CINITGP+Cinitgp;
    
###################################Plots######################################
# Tm=450*Tm+323; Tinit=450*Tinit+323; Tinitgp=450*Tinitgp+323;
CBEST=CBEST/lp; CBESTGP=CBESTGP/lp; #CINIT=CINIT/lp; CINITGP=CINITGP/lp;
# #TINIT=450*(TINIT/lp)+323; TINITGP=450*(TINITGP/lp)+323;
# pyp.figure()
# pyp.plot(Tm,Cpred)
# pyp.fill_between(Tm.reshape(-1),(Cpred-2*std).reshape(-1),
#                   (Cpred+2*std).reshape(-1),alpha=0.1);
# pyp.scatter(Tinit,Cinit,marker='x',color='black')
# pyp.plot(Tm,Ck,'g--')
# pyp.plot(Tm,Cmk,'r:')
# #pyp.plot(Tm,dm,'c-.')
# pyp.xlim((323,773)); #pyp.ylim((-64,-44));
# pyp.legend(['Mean','f(x)',r'$f_{mod}(x)$','Confidence','Data'],loc='upper right')

# pyp.figure()
# pyp.plot(Tm,dm,'c-.')
# pyp.scatter(Tinit,dinit,marker='x',color='k')
# pyp.xlim((323,773));

# pyp.figure()
# pyp.plot(Tm,af)

#pyp.figure()
#pyp.plot(Cbest)
#pyp.scatter(itr,Cbest)

# pyp.figure()
# pyp.plot(Tm,Cm)
# pyp.fill_between(Tm.reshape(-1),(Cm-2*stdgp).reshape(-1),
#                   (Cm+2*stdgp).reshape(-1),alpha=0.1);
# pyp.scatter(Tinitgp,Cinitgp,marker='x',color='black')
# pyp.plot(Tm,Ck,'g--')
# pyp.xlim((323,773)); #pyp.ylim((-64,-44));
# pyp.legend(['Mean','f(x)','Confidence','Data'],loc='upper right')

# pyp.figure()
# pyp.plot(Tm,afgp)

# pyp.figure()
# pyp.plot(Cbestgp)
# pyp.scatter(itr,Cbestgp)

pyp.figure()
pyp.plot(CBEST)
pyp.scatter(itr,CBEST)
pyp.plot(CBESTGP)
pyp.scatter(itr,CBESTGP)
pyp.xlabel('Iteration Number'); pyp.ylabel('Optimal Cost')
pyp.savefig('Avg_Comp_2D_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)


# for i in range(Tinit.shape[0]):
#     pyp.figure()
#     pyp.plot(Tm,Cpred)
#     pyp.fill_between(Tm.reshape(-1),(Cpred-2*std).reshape(-1),
#                      (Cpred+2*std).reshape(-1),alpha=0.1);
#     pyp.scatter(Tinit[0:i+1],Cinit[0:i+1],marker='x',color='k')
#     pyp.plot(Tm,Ck,'g--');
#     pyp.plot(Tm,Cmk,'r:')
#     pyp.xlim((323,773));
#     pyp.legend(['Mean','f(x)','f_mod(x)','Confidence','Data'],loc='upper right')
#     str1=str(i+1)
#     pyp.savefig('Rxtr_1D_Progression'+str1+'.png',dpi=300)
#     pyp.close()

images=[];
for j in range(Tinit.shape[0]):
    str2=str(j+1);
    images.append(imageio.imread('Rxtr_1D_Progression'+str2+'_Rest.png'));
imageio.mimsave('Rxtr_1D_Rest.gif',images,duration=1)

imageserr=[];
for j in range(Tinit.shape[0]):
    str2=str(j+1);
    imageserr.append(imageio.imread('Rxtr_1D_Error'+str2+'_Rest.png'));
imageio.mimsave('Rxtr_1D_ErrorRest.gif',imageserr,duration=1)

imagesgp=[];
for j in range(Tinitgp.shape[0]):
    str2gp=str(j+1);
    imagesgp.append(imageio.imread('Rxtr_1Dgp_Progression'+str2gp+'_Rest.png'));
imageio.mimsave('Rxtr_1Dgp_Rest.gif',imagesgp,duration=1)

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

# plt3d=pyp.figure().gca(projection='3d')
# plt3d.plot_surface(Tp,Fp,Ckp)
# pyp.show()

# plt3d=pyp.figure().gca(projection='3d')
# plt3d.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)))
# pyp.show()

# images2d=[];
# for j in range(TFinit.shape[0]):
#     str2=str(j+1);
#     images2d.append(imageio.imread('Rxtr_2D_Progression'+str2+'_Rest.png'));
# imageio.mimsave('Rxtr_2D_Rest.gif',images2d,duration=0.5)

# images2ds=[];
# for j in range(TFinit.shape[0]):
#     str2=str(j+1);
#     images2ds.append(imageio.imread('Rxtr_2D_Surf_Progression'+str2+'_Rest.png'));
# imageio.mimsave('Rxtr_2D_Surf_Rest.gif',images2ds,duration=0.5)

# images2dgp=[];
# for j in range(TFinitgp.shape[0]):
#     str2=str(j+1);
#     images2dgp.append(imageio.imread('Rxtr_2Dgp_Progression'+str2+'_Rest.png'));
# imageio.mimsave('Rxtr_2Dgp_Rest.gif',images2dgp,duration=0.5)

# images2dsgp=[];
# for j in range(TFinitgp.shape[0]):
#     str2=str(j+1);
#     images2dsgp.append(imageio.imread('Rxtr_2Dgp_Surf_Progression'+str2+'_Rest.png'));
# imageio.mimsave('Rxtr_2Dgp_Surf_Rest.gif',images2dsgp,duration=0.5)

# images2dscomp=[];
# for j in range(TFinitgp.shape[0]):
#     str2=str(j+1);
#     images2dscomp.append(imageio.imread('Rxtr_2D_Surf_Comp_Progression'+str2+'_Rest.png'));
# imageio.mimsave('Rxtr_2D_Surf_Comp_Rest.gif',images2dscomp,duration=0.5)
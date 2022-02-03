# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:02:21 2020

@author: leonardo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:55:13 2020

@author: leonardo
"""

import sys;
sys.path.insert(1,r'C:\Users\leonardo\OneDrive - UW-Madison\Research\bayesianopt\Scripts')
from numpy import exp, arange, random, array, vstack, argmax, hstack, argmin
from numpy import ones, log, matmul, linalg, transpose as trans, meshgrid
from matplotlib import pyplot as pyp, cm
import GPy
from Acquisition_Funcs import LCB
from scipy.optimize import fsolve, minimize, Bounds
import imageio
from mpl_toolkits.mplot3d import Axes3D

##################Initial Conditions and Physical Parameters##################
Cr0=5000; Ci0=0; Cp0=0;
k01=1000; k02=1250; E1=32000; E2=35000;

p=850; Cp=3000; CpH2O=4184; R=8.314; H1=-210000; H2=-1700000;
Tin=298; TinH2O=298; V=1000;
cP=-35; cR=12; cT=0.35; cI=7.5
##############################################################################
#T range is 313 to 763
#F range is 90 to 100
#Units are all in MKS 

def Ctfxn(T,F):
    T=array([T],dtype='float64').reshape(-1,1);
    F=array([F],dtype='float64').reshape(-1,1);
    k1=k01*exp(-E1/(R*(T))); k2=k02*exp(-E2/(R*(T))); ToH2O=T-10;
    k1r=k1/100; k2r=k2/100;

    Crinit=100; Ciinit=2500; Cpdinit=300;
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
    g=max([0,Cpd-300])
    Ct=cR*(Cr0+Cr)*F+cP*F*Ci+cI*Cpd*F+cT*mH2O+500*g;
    return Ct/1e6,r1,r2,mH2O,Cr,Ci,Cpd

def Ctfxn_mod(T,F):
    T=array([T],dtype='float64').reshape(-1,1);
    F=array([F],dtype='float64').reshape(-1,1);
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
        g[i]=max([0,Cpd[i]-300])    
    Ct=cR*(Cr0+Cr)*F+cP*F*Ci+cI*Cpd*F+cT*mH2O+500*g
    return Ct/1e6,r1,r2,mH2O,Cr,Ci,Cpd

def Ctfxn_modM(T,F):
    T=array([T],dtype='float64').reshape(-1,1);
    F=array([F],dtype='float64').reshape(-1,1);
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
        g[i]=max([0,Cpd[i]-300])    
    Ct=cR*(Cr0+Cr)*F+cP*F*Ci+cI*Cpd*F+cT*mH2O+500*g
    return Ct/1e6,r1,r2,mH2O,Cr,Ci,Cpd


####################### Reaction Regression ##################################
# Univariate Regression
Tr=random.uniform(313,763,(200,1)); Fr=90*ones(Tr.shape)
Ctr=lambda T,F:Ctfxn(T,F)[1:3];
Ctr=array(list(map(Ctr,Tr,Fr))).reshape(Tr.shape[0],2);
## Inverse log
A=ones(Tr.shape[0]).reshape(-1,1); Ctr=log(Ctr); A=hstack([1/Tr,A]);
psuedoAinv=matmul(linalg.inv(matmul(trans(A),A)),trans(A));
w1=matmul(psuedoAinv,Ctr[:,0]); w2=matmul(psuedoAinv,Ctr[:,1]);
theta=vstack([w1,w2]);

#Multivariate regression
Trm=random.uniform(313,763,(200,1));
Frm=random.uniform(90,100,(200,1));
Ctrm=lambda T,F:Ctfxn(T,F)[1:3]
Ctrm=array(list(map(Ctrm,Trm,Frm))).reshape(Trm.shape[0],2);
## Log-linear inverse T
Am=ones(Trm.shape[0]).reshape(-1,1); Am=hstack([1/Trm,log(Frm),Am]); Ctrm=log(Ctrm)
psuedoAinvm=matmul(linalg.inv(matmul(trans(Am),Am)),trans(Am));
w1=matmul(psuedoAinvm,Ctrm[:,0]); w2=matmul(psuedoAinvm,Ctrm[:,1]);
thetam=vstack([w1,w2]);

#################################SETUP########################################
noise=1e-6; exp_w=2.0; t=50; itr=array(arange(t+1)).reshape(-1,1); F=90; T=423;
lp=100; CBEST=0*ones((t+1,1)); CBESTGP=0*ones((t+1,1)); idx=ones((t,1),dtype=int);

Tm=arange(0,1.001,0.001).reshape(-1,1); Fm=F*ones(Tm.shape);
Ck=lambda T,F:Ctfxn(450*T+313,F)[0];
Ck=array(list(map(Ck,Tm,Fm))).reshape(-1,1);
Cmk=Ctfxn_mod(450*Tm+313,Fm)[0].reshape(-1,1);

## MULTIVARIATE CASE:
Tp=arange(313,764,1); Fp=arange(90,100.1,0.1)
Fp,Tp=meshgrid(Fp,Tp); Ckp=ones(Tp.shape); Cmkp=ones(Tp.shape);
d1=Tp.shape[0]; d2=Tp.shape[1]
Ckp=lambda T,F:Ctfxn(T,F)[0]; Cmkp=lambda T,F:Ctfxn_modM(T,F)[0];
Ckp=array(list(map(Ckp,Tp.reshape(-1,1),Fp.reshape(-1,1)))).reshape(Tp.shape);
Cmkp=array(list(map(Cmkp,Tp.reshape(-1,1),Fp.reshape(-1,1)))).reshape(Tp.shape);
Tm=(Tp.reshape(-1,1)-313)/450; Fm=(Fp.reshape(-1,1)-90)/10;
Ck=Ckp.reshape(-1,1); Cmk=Cmkp.reshape(-1,1); TFm=hstack([Tm,Fm])

CPRED=0*ones(Ck.shape); CM=CPRED.copy(); STD=CPRED.copy(); STDGP=CPRED.copy();
CIDX=ones((lp,2)); CIDXGP=ones((lp,2));

#%%###############################RUNS#####################################%%#            
for j in range(lp):
    
# MULTIVARIATE CASE:    
    #Ck=apply_along_axis(CtfxnMD,1,TFm)[:,0].reshape(-1,1);
    #Cmk=apply_along_axis(Ctfxn_modMD,1,TFm)[:,0].reshape(-1,1);
    #TFm=loadtxt('2D_mech_rxtr.txt');
    
    TFinit=random.uniform(0,1,2).reshape(1,2)#array([[1,1]])
    Cinit=Ctfxn(450*TFinit[0,0]+313,10*TFinit[0,1]+90)[0].reshape(-1,1); Cbest=min(Cinit);
    d2dinit=Cinit-Ctfxn_modM(450*TFinit[0,0]+313,10*TFinit[0,1]+90)[0].reshape(-1,1);

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
        # ax3D1.scatter(450*TFinit[:,0]+323,10*TFinit[:,1]+90,color='r',
        #               marker='x',s=40)
        # pyp.colorbar(fig1)
        # pyp.xlim((323,773)); pyp.ylim((90,100))
        # pyp.title('True Model')
        # ax3D2=fig3D.add_subplot(122)
        # fig2=ax3D2.contourf(Tp,Fp,Cpred.reshape((d1,d2)))
        # ax3D2.scatter(450*TFinit[:,0]+323,10*TFinit[:,1]+90,color='r',
        #               marker='x',s=40)
        # pyp.colorbar(fig2)
        # pyp.xlim((323,773)); pyp.ylim((90,100))
        # pyp.title('Estimated Model')
        # pyp.savefig('Rxtr_2D_Progression1_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
        # pyp.close()
        
        fig3Dp=pyp.figure(figsize=[20,8])
        ax3D1p=fig3Dp.add_subplot(121,projection='3d')
        fig1p=ax3D1p.plot_surface(Tp,Fp,Ckp,cmap=cm.coolwarm);
        ax3D1p.scatter(450*TFinit[:,0]+313,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost')
        pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1p.set_zlim((-23.1,-22.3));
        ax3D2p=fig3Dp.add_subplot(122,projection='3d')
        fig2p=ax3D2p.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),cmap=cm.coolwarm);
        ax3D2p.scatter(450*TFinit[:,0]+313,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2p.set_zlabel('Cost')
        pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D2p.set_zlim((-23.1,-22.3));
        pyp.savefig('Rxtr_2D_Surf_Progression1_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)
        pyp.close()
        
        # fig3Dgp=pyp.figure(figsize=[20,8])
        # ax3D1gp=fig3Dgp.add_subplot(121)
        # fig1gp=ax3D1gp.contourf(Tp,Fp,Ckp)
        # ax3D1gp.scatter(450*TFinit[:,0]+323,10*TFinit[:,1]+90,color='r',
        #               marker='x',s=40)
        # pyp.colorbar(fig1gp)
        # pyp.xlim((323,773)); pyp.ylim((90,100))
        # pyp.title('True Model')
        # ax3D2gp=fig3Dgp.add_subplot(122)
        # fig2gp=ax3D2gp.contourf(Tp,Fp,Cm.reshape((d1,d2)))
        # ax3D2gp.scatter(450*TFinit[:,0]+323,10*TFinit[:,1]+90,color='r',
        #               marker='x',s=40)
        # pyp.colorbar(fig2gp)
        # pyp.xlim((323,773)); pyp.ylim((90,100));
        # pyp.title('Estimated Model');
        # pyp.savefig('Rxtr_2Dgp_Progression1_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)
        # pyp.close()
        
        fig3Dpgp=pyp.figure(figsize=[20,8])
        ax3D1pgp=fig3Dpgp.add_subplot(121,projection='3d')
        fig1pgp=ax3D1pgp.plot_surface(Tp,Fp,Ckp,cmap=cm.coolwarm);
        ax3D1pgp.scatter(450*TFinit[:,0]+313,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1pgp.set_zlabel('Cost')
        pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1pgp.set_zlim((-23.1,-22.3));
        ax3D2pgp=fig3Dpgp.add_subplot(122,projection='3d')
        fig2pgp=ax3D2pgp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),cmap=cm.coolwarm);
        ax3D2pgp.scatter(450*TFinit[:,0]+313,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2pgp.set_zlabel('Cost')
        pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D2pgp.set_zlim((-23.1,-22.3));
        pyp.savefig('Rxtr_2Dgp_Surf_Progression1_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)
        pyp.close()
        
        fig3D=pyp.figure(figsize=[20,8])
        ax3D1=fig3D.add_subplot(121)
        fig1=ax3D1.contourf(Tp,Fp,10*Cpred.reshape((d1,d2)),
                            levels=[-18,-16.5,-15,-13.5,-12,-10.5,-9,-7.5],
                            cmap=cm.coolwarm)
        pyp.colorbar(fig1)
        pyp.xlim((313,763)); pyp.ylim((90,100))
        pyp.title('True Model')
        ax3D2=fig3D.add_subplot(122)
        fig2=ax3D2.contourf(Tp,Fp,10*Cm.reshape((d1,d2)),
                            levels=[-18,-16.5,-15,-13.5,-12,-10.5,-9,-7.5],
                            cmap=cm.coolwarm)
        pyp.colorbar(fig2)
        pyp.xlim((313,763)); pyp.ylim((90,100))
        pyp.title('Estimated Model')
        pyp.savefig('Rxtr_2D_Progression_Comp1_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)
        pyp.close()
        
        fig3Dcomp=pyp.figure(figsize=[20,8])
        ax3D1comp=fig3Dcomp.add_subplot(121,projection='3d')
        fig1comp=ax3D1comp.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),cmap=cm.coolwarm);
        ax3D1comp.scatter(450*TFinit[:,0]+313,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1comp.set_zlabel('Cost')
        pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1comp.set_zlim((-23.1,-22.3));
        ax3D2comp=fig3Dcomp.add_subplot(122,projection='3d')
        fig2comp=ax3D2comp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),cmap=cm.coolwarm);
        ax3D2comp.scatter(450*TFinit[:,0]+313,10*TFinit[:,1]+90,Cinit,color='k')
        pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2comp.set_zlabel('Cost')
        pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D2comp.set_zlim((-23.1,-22.3));
        pyp.savefig('Rxtr_2D_Surf_Comp_Progression1_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)
        pyp.close()        
        
#######################Using 'mechanistic' model##############################    
## MULTIVARIATE CASE:
    for i in range(t):
        TFnxt=TFm[argmax(af2d)].reshape(1,2);
        Cnxt=Ctfxn(450*TFnxt[0,0]+313,10*TFnxt[0,1]+90)[0];
        Cmnxt=Ctfxn_modM(450*TFnxt[0,0]+313,10*TFnxt[0,1]+90)[0];
        d2dnxt=Cnxt-Cmnxt;
        TFinit=vstack([TFinit,TFnxt]); Cinit=vstack([Cinit,Cnxt]);
        d2dinit=vstack([d2dinit,d2dnxt]); Cbest=vstack([Cbest,min(Cinit)])
        model2d=GPy.models.GPRegression(TFinit,d2dinit,kernel2d);
        model2d.Gaussian_noise.variance=noise;
        model2d.Gaussian_noise.variance.fix();
        dm2d,std2d=model2d.predict(TFm); std2d=std2d**0.5; af2d=LCB((Cmk+dm2d),std2d,exp_w);
        Cpred=Cmk+dm2d;
        
        TFnxt=TFm[argmax(af2dgp)].reshape(1,2);
        Cnxt=Ctfxn(450*TFnxt[0,0]+313,10*TFnxt[0,1]+90)[0];
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
            # fig1=ax3D1.contourf(Tp,Fp,10*Ckp,
            #                     levels=[-18,-16.5,-15,-13.5,-12,-10.5,-9,-7.5])
            # ax3D1.scatter(450*TFinit[:,0]+323,10*TFinit[:,1]+90,color='r',
            #               marker='x',s=40)
            # pyp.colorbar(fig1)
            # pyp.xlim((323,773)); pyp.ylim((90,100))
            # pyp.title('True Model')
            # ax3D2=fig3D.add_subplot(122)
            # fig2=ax3D2.contourf(Tp,Fp,Cpred.reshape((d1,d2)))
            # ax3D2.scatter(450*TFinit[:,0]+323,10*TFinit[:,1]+90,color='r',
            #               marker='x',s=40)
            # pyp.colorbar(fig2)
            # pyp.xlim((323,773)); pyp.ylim((90,100))
            # pyp.title('Estimated Model')
            # pyp.savefig('Rxtr_2D_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
            # pyp.close()
            
            fig3Dp=pyp.figure(figsize=[20,8])
            ax3D1p=fig3Dp.add_subplot(121,projection='3d')
            fig1p=ax3D1p.plot_surface(Tp,Fp,Ckp,cmap=cm.coolwarm);
            ax3D1p.scatter(450*TFinit[:,0]+313,10*TFinit[:,1]+90,Cinit,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost')
            pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1p.set_zlim((-23.1,-22.3));
            ax3D2p=fig3Dp.add_subplot(122,projection='3d')
            fig2p=ax3D2p.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),cmap=cm.coolwarm);
            ax3D2p.scatter(450*TFinit[:,0]+313,10*TFinit[:,1]+90,Cinit,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2p.set_zlabel('Cost')
            pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D2p.set_zlim((-23.1,-22.3));
            pyp.savefig('Rxtr_2D_Surf_Progression'+str1+'_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)
            pyp.close()
            
            # fig3Dgp=pyp.figure(figsize=[20,8])
            # ax3D1gp=fig3Dgp.add_subplot(121)
            # fig1gp=ax3D1gp.contourf(Tp,Fp,Ckp)
            # ax3D1gp.scatter(450*TFinitgp[:,0]+323,10*TFinitgp[:,1]+90,color='r',
            #               marker='x',s=40)
            # pyp.colorbar(fig1gp)
            # pyp.xlim((323,773)); pyp.ylim((40,100))
            # pyp.title('True Model')
            # ax3D2gp=fig3Dgp.add_subplot(122)
            # fig2gp=ax3D2gp.contourf(Tp,Fp,Cm.reshape((d1,d2)))
            # ax3D2gp.scatter(450*TFinitgp[:,0]+323,10*TFinitgp[:,1]+90,color='r',
            #               marker='x',s=40)
            # pyp.colorbar(fig2gp)
            # pyp.xlim((323,773)); pyp.ylim((90,100))
            # pyp.title('Estimated Model')
            # pyp.savefig('Rxtr_2Dgp_Progression'+str1+'_Rest.png',dpi=300,bbox_inches='tight',pad_inches=0)
            # pyp.close()
            
            fig3Dpgp=pyp.figure(figsize=[20,8])
            ax3D1pgp=fig3Dpgp.add_subplot(121,projection='3d')
            fig1pgp=ax3D1pgp.plot_surface(Tp,Fp,Ckp,cmap=cm.coolwarm);
            ax3D1pgp.scatter(450*TFinitgp[:,0]+313,10*TFinitgp[:,1]+90,Cinitgp,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1pgp.set_zlabel('Cost')
            pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1pgp.set_zlim((-23.1,-22.3));
            ax3D2pgp=fig3Dpgp.add_subplot(122,projection='3d')
            fig2pgp=ax3D2pgp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),cmap=cm.coolwarm);
            ax3D2pgp.scatter(450*TFinitgp[:,0]+313,10*TFinitgp[:,1]+90,Cinitgp,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2pgp.set_zlabel('Cost')
            pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D2pgp.set_zlim((-23.1,-22.3));
            pyp.savefig('Rxtr_2Dgp_Surf_Progression'+str1+'_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)
            pyp.close()            
            
            fig3D=pyp.figure(figsize=[20,8])
            ax3D1=fig3D.add_subplot(121)
            fig1=ax3D1.contourf(Tp,Fp,10*Cpred.reshape((d1,d2)),
                                levels=[-18,-16.5,-15,-13.5,-12,-10.5,-9,-7.5],
                                cmap=cm.coolwarm)
            #ax3D1.scatter(450*TFinit[:,0]+323,10*TFinit[:,1]+90,color='r',
            #              marker='x',s=40)
            pyp.colorbar(fig1)
            pyp.xlim((313,763)); pyp.ylim((90,100))
            pyp.title('True Model')
            ax3D2=fig3D.add_subplot(122)
            fig2=ax3D2.contourf(Tp,Fp,10*Cm.reshape((d1,d2)),
                                levels=[-18,-16.5,-15,-13.5,-12,-10.5,-9,-7.5],
                                cmap=cm.coolwarm)
            #ax3D2.scatter(450*TFinit[:,0]+323,10*TFinit[:,1]+90,color='r',
            #              marker='x',s=40)
            pyp.colorbar(fig2)
            pyp.xlim((313,763)); pyp.ylim((90,100))
            pyp.title('Estimated Model')
            pyp.savefig('Rxtr_2D_Progression_Comp'+str1+'_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)
            pyp.close()
            
            fig3Dcomp=pyp.figure(figsize=[20,8])
            ax3D1comp=fig3Dcomp.add_subplot(121,projection='3d')
            fig1comp=ax3D1comp.plot_surface(Tp,Fp,Cpred.reshape((d1,d2)),cmap=cm.coolwarm);
            ax3D1comp.scatter(450*TFinit[:,0]+313,10*TFinit[:,1]+90,Cinit,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1comp.set_zlabel('Cost')
            pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D1comp.set_zlim((-23.1,-22.3));
            ax3D2comp=fig3Dcomp.add_subplot(122,projection='3d')
            fig2comp=ax3D2comp.plot_surface(Tp,Fp,Cm.reshape((d1,d2)),cmap=cm.coolwarm);
            ax3D2comp.scatter(450*TFinitgp[:,0]+313,10*TFinitgp[:,1]+90,Cinit,color='k')
            pyp.xlabel('Temperature'); pyp.ylabel('Flow'); #ax3D2comp.set_zlabel('Cost')
            pyp.xlim((313,763)); pyp.ylim((90,100)); #ax3D2comp.set_zlim((-23.1,-22.3));
            pyp.savefig('Rxtr_2D_Surf_Comp_Progression'+str1+'_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)
            pyp.close()
            
    CBEST=CBEST+Cbest; CBESTGP=CBESTGP+Cbestgp;
    CPRED=CPRED+Cpred; STD=STD+std2d; CM=CM+Cm; STDGP=STDGP+std2dgp; 
    CIDX[j,0]=argmin(Cpred); CIDX[j,1]=Ck[argmin(Cpred)];
    CIDXGP[j,0]=argmin(Cm); CIDXGP[j,1]=Ck[argmin(Cm)];
    
CBEST=CBEST/lp; CBESTGP=CBESTGP/lp; CPRED=CPRED/lp; CM=CM/lp;
STD=STD/lp; STDGP=STDGP/lp;
    
#%%################################Plots###################################%%#

pyp.figure(figsize=[12,8])
pyp.plot(10*CBEST); pyp.scatter(itr,10*CBEST);
pyp.plot(10*CBESTGP); pyp.scatter(itr,10*CBESTGP);
#pyp.ylim((min(Ck)-0.01,-22.9))#max(Ck)+0.01));
pyp.xlabel('Iteration Number'); pyp.ylabel('Optimal Cost (1e5)');
pyp.legend(['Hybrid BO','Pure BO'])
pyp.savefig('Avg_Comp_2D_Rest2_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)

pyp.figure()
pyp.hlines(min(10*Ck),0,lp+1,colors='k')
pyp.scatter(arange(1,lp+1,1),10*CIDX[:,1]); #pyp.plot(arange(1,101,1),CIDX[:,1]);
pyp.xlim((0,lp+1)); pyp.ylim((-17,-14))#max(Ck)+0.01));
pyp.xlabel('Simulation Number'); pyp.ylabel('Optimal Cost (1e5)');
pyp.legend(['True Solution','Hybrid BO Solution'])
pyp.savefig('Avg_Comp_2D_Sol2_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)

pyp.figure()
pyp.hlines(min(10*Ck),0,lp+1,colors='k')
pyp.scatter(arange(1,lp+1,1),10*CIDXGP[:,1],color='red'); #pyp.plot(arange(1,101,1),CIDXGP[:,1]);
pyp.xlim((0,lp+1)); pyp.ylim((-17,-14))#max(Ck)+0.01));
pyp.xlabel('Simulation Number'); pyp.ylabel('Optimal Cost(1e5)');
pyp.legend(['True Solution','Pure BO Solution'])
pyp.savefig('Avg_Comp_2D_SolGP2_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)

figC=pyp.figure(figsize=[12,8])
ax3D1=figC.add_subplot(211)
ax3D1.hlines(min(10*Ck),0,lp+1,colors='k')
ax3D1.scatter(arange(1,lp+1,1),10*CIDX[:,1])
pyp.xlim((0,lp+1)); pyp.ylim((-17,-14))#max(Ck)+0.01));
pyp.xlabel('Simulation Number'); pyp.ylabel('Optimal Cost');
pyp.legend(['True Solution','Hybrid BO Solution'])
ax3D2=figC.add_subplot(212)
ax3D2.hlines(min(10*Ck),0,lp+1,colors='k')
ax3D2.scatter(arange(1,lp+1,1),10*CIDXGP[:,1],color='red');
pyp.xlim((0,lp+1)); pyp.ylim((-17,-14))
pyp.xlabel('Simulation Number'); pyp.ylabel('Optimal Cost');
pyp.legend(['True Solution','Pure BO Solution'])
pyp.savefig('Avg_2D_Sol2_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)

fig3Dp=pyp.figure(figsize=[20,8])
ax3D1p=fig3Dp.add_subplot(111,projection='3d')
fig1p=ax3D1p.plot_surface(Tp,Fp,10*Ck.reshape((d1,d2)),cmap=cm.coolwarm);
pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost (1e5)')
pyp.xlim((313,763)); pyp.ylim((90,100)); ax3D1p.set_zlim((-17.5,-6.5));
ax3D1p.view_init(10,80);
pyp.savefig('True_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)

fig3Dp=pyp.figure(figsize=[20,8])
ax3D1p=fig3Dp.add_subplot(111,projection='3d')
fig1p=ax3D1p.plot_surface(Tp,Fp,10*Cmk.reshape((d1,d2)),cmap=cm.coolwarm);
pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost (1e5)')
pyp.xlim((313,763)); pyp.ylim((90,100)); ax3D1p.set_zlim((-17.5,-6.5));
ax3D1p.view_init(10,80);
pyp.savefig('Mod_approx_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)

fig3Dp=pyp.figure(figsize=[20,8])
ax3D1p=fig3Dp.add_subplot(111,projection='3d')
fig1p=ax3D1p.plot_surface(Tp,Fp,10*CPRED.reshape((d1,d2)),cmap=cm.coolwarm);
pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost (1e5)')
pyp.xlim((313,763)); pyp.ylim((90,100)); ax3D1p.set_zlim((-17.5,-6.5));
ax3D1p.view_init(10,80);
pyp.savefig('Mod_Avg_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)

fig3Dp=pyp.figure(figsize=[20,12])
ax3D1p=fig3Dp.add_subplot(211,projection='3d')
fig1p=ax3D1p.plot_surface(Tp,Fp,10*CPRED.reshape((d1,d2)),cmap=cm.coolwarm);
pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost (1e5)')
pyp.xlim((313,763)); pyp.ylim((90,100)); ax3D1p.set_zlim((-17.5,-6.5));
ax3D1p.view_init(10,80);
ax3D2p=fig3Dp.add_subplot(212,projection='3d')
fig1p=ax3D2p.plot_surface(Tp,Fp,10*(Ck-CPRED).reshape((d1,d2)),cmap=cm.coolwarm);
pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2p.set_zlabel('Model Error (1e5)')
pyp.xlim((313,763)); pyp.ylim((90,100)); ax3D2p.set_zlim((-5,2));
ax3D2p.view_init(10,80)
pyp.savefig('Mod_Avg_err_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)

fig3Dp=pyp.figure(figsize=[20,8])
ax3D1p=fig3Dp.add_subplot(111,projection='3d')
fig1p=ax3D1p.plot_surface(Tp,Fp,10*CM.reshape((d1,d2)),cmap=cm.coolwarm);
pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost (1e5)')
pyp.xlim((313,763)); pyp.ylim((90,100)); ax3D1p.set_zlim((-17.6,-6.5));
ax3D1p.view_init(10,80);
pyp.savefig('GP_Avg_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)

fig3Dp=pyp.figure(figsize=[20,12])
ax3D1p=fig3Dp.add_subplot(211,projection='3d')
fig1p=ax3D1p.plot_surface(Tp,Fp,10*CM.reshape((d1,d2)),cmap=cm.coolwarm);
pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D1p.set_zlabel('Cost (1e5)')
pyp.xlim((313,763)); pyp.ylim((90,100)); ax3D1p.set_zlim((-17.6,-6.5));
ax3D1p.view_init(10,80);
ax3D2p=fig3Dp.add_subplot(212,projection='3d')
fig1p=ax3D2p.plot_surface(Tp,Fp,10*(Ck-CM).reshape((d1,d2)),cmap=cm.coolwarm);
pyp.xlabel('Temperature'); pyp.ylabel('Flow'); ax3D2p.set_zlabel('Model Error (1e5)')
pyp.xlim((313,763)); pyp.ylim((90,100)); ax3D2p.set_zlim((-5,2));
ax3D2p.view_init(10,80)
pyp.savefig('GP_Avg_err_Mod2.png',dpi=300,bbox_inches='tight',pad_inches=0)

images2ds=[];
for j in range(TFinit.shape[0]):
    str2=str(j+1);
    images2ds.append(imageio.imread('Rxtr_2D_Surf_Progression'+str2+'_Mod2.png'));
imageio.mimsave('Rxtr_2D_Surf_Mod2.gif',images2ds,duration=0.5)

# images2dgp=[];
# for j in range(TFinitgp.shape[0]):
#     str2=str(j+1);
#     images2dgp.append(imageio.imread('Rxtr_2Dgp_Progression'+str2+'_Rest.png'));
# imageio.mimsave('Rxtr_2Dgp_Rest2.gif',images2dgp,duration=0.5)

images2dsgp=[];
for j in range(TFinitgp.shape[0]):
    str2=str(j+1);
    images2dsgp.append(imageio.imread('Rxtr_2Dgp_Surf_Progression'+str2+'_Mod2.png'));
imageio.mimsave('Rxtr_2Dgp_Surf_Mod2.gif',images2dsgp,duration=0.5)

images2dscomp=[];
for j in range(TFinitgp.shape[0]):
    str2=str(j+1);
    images2dscomp.append(imageio.imread('Rxtr_2D_Progression_Comp'+str2+'_Mod2.png'));
imageio.mimsave('Rxtr_2D_Comp_Mod2.gif',images2dscomp,duration=0.5)

images2dscomp=[];
for j in range(TFinitgp.shape[0]):
    str2=str(j+1);
    images2dscomp.append(imageio.imread('Rxtr_2D_Surf_Comp_Progression'+str2+'_Mod2.png'));
imageio.mimsave('Rxtr_2D_Surf_Comp_Mod2.gif',images2dscomp,duration=0.5)
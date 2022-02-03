# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:15:53 2020

@author: leonardo
"""

from numpy import exp, arange, vstack, array, zeros, argmin
from scipy.optimize import fsolve
from matplotlib import pyplot as pyp

Cr0=1e4; F=90; k01=5.5e5; k02=500000; E1=30500; E2=33500; T=323; R=8.314; V=1;
k1=k01*exp(-E1/(R*T)); k2=k02*exp(-E2/(R*T)); k1r=k1/100; k2r=k2/100;
Tin=323; TinH2O=298; p=850; Cp=3000; CpH2O=4184; H1=-210000; H2=-1700000;
cP=-510; cR=13; cT=0.070; cO=5; W=10000; cI=29#43.5

#for i in range(100):
#    Cr=(F*(Cr0-Cr)/(k1*V))**(1/3)
#    if abs(Cr-F*Cr0/(F+k1*Cr**2*V))<=1e-5:
#        break

def C(C):
#    C=C.reshape(1,3)
    Crg=C[0]; Cig=C[1]; Cpdg=C[2];
#    Cr=(F*Cr0+k1r*Cig**2*V)/(F+k1*V);
#    Ci=((2*(k1*Crg+k2r*Cpdg)*V-F*Cig)/(2*(k1r+k2)*V))**(0.5);
#    Cpd=(k2*Cig**2*V)/(F+k2r*V);
    Cr=((F*(Cr0-Crg)+2*k1r*Cig*V)/(2*k1*V))**(1/2);
    Ci=(k1*Crg**2*V+k2r*Cpdg**2*V)/(F+k1r*V+k2*V);
    Cpd=((2*k2*Cig*V-F*Cpdg)/(2*k2r*V))**(1/2);
    
    return [Cr-Crg,Ci-Cig,Cpd-Cpdg]

T=arange(323,774,1).reshape(-1,1); F=90;
C0=[Cr0/50,Cr0/2,Cr0/50];#[9400,160,566];
Csoln=zeros([1,5])
for i in range(T.shape[0]):
    k1=k01*exp(-E1/(R*(T[i,0]+0))); k2=k02*exp(-E2/(R*(T[i,0]+0)));
    k1r=k1/100; k2r=k2/100; ToH2O=T[i,0]-10
    root=fsolve(C,C0); C0=array(root); Cr=C0[0]; Ci=C0[1]; Cpd=C0[2];
    r1=k1*Cr**2-k1r*Ci; r2=k2*Ci-k2r*Cpd**2;#k1*Cr-k1r*Ci**2; r2=k2*Ci**2-k2r*Cpd;
    mH2O=(p*Cp*F*int(Tin-T[i,0])-r1*V*H1-r2*V*H2)/(CpH2O*(ToH2O-TinH2O));
    Ct=(cP*V*Ci+cO*W+cT*mH2O+cI*Cpd*V+cR*Cr*V)/1e5;
    Csoln=vstack([Csoln,array([Cr,Ci,Cpd,mH2O,Ct]).reshape(1,5)]);
Csoln=Csoln[1:Csoln.shape[0],:]

pyp.figure()
pyp.scatter(T,Csoln[:,0],marker='x',color='k')
pyp.plot(T,Csoln[:,0])

pyp.figure()
pyp.scatter(T,Csoln[:,1],marker='x',color='k')
pyp.plot(T,Csoln[:,1])

pyp.figure()
pyp.scatter(T,Csoln[:,2],marker='x',color='k')
pyp.plot(T,Csoln[:,2])

pyp.figure()
pyp.scatter(T,Csoln[:,3],marker='x',color='k')
pyp.plot(T,Csoln[:,3])

pyp.figure()
pyp.scatter(T,Csoln[:,4],marker='x',color='k')
pyp.plot(T,Csoln[:,4])
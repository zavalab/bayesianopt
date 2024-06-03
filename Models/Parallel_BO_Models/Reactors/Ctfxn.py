from numpy import exp

##################Initial Conditions and Physical Parameters##################
t0=0; dt=0.005;k01=0.1; k02=1; R=8.314; E1=7000; E2=20000;
V=1000; cP=-100; cR=5; cT=5; cO=1400;
##############################################################################

def Ctfxn(Cr0,Ci0,Cp0,T,tf):
    k1=k01*exp(-E1/(R*T)); k2=k02*exp(-E2/(R*T)); k1r=k1/10; n=int((tf-t0)/dt);
    Cr=Cr0*1; Ci=Ci0*1; Cp=Cp0*1; t=t0*1;

    
    def dCrdt(t,Cr,Ci,k1,k1r):
        return -k1*Cr+k1r*Ci
    def dCidt(t,Cr,Ci,k1,k1r,k2):
        return k1*Cr-k1r*Ci-k2*Ci
    def dCpdt(t,Ci,k2):
        return k2*Ci
    
    for i in range(n):
        k1R=dt*dCrdt(t,Cr,Ci,k1,k1r)
        k1I=dt*dCidt(t,Cr,Ci,k1,k1r,k2)
        k1P=dt*dCpdt(t,Ci,k2)
        k2R=dt*dCrdt(t+dt,Cr+k1R/2,Ci+k1I/2,k1,k1r)
        k2I=dt*dCidt(t+dt,Cr+k1R/2,Ci+k1I/2,k1,k1r,k2)
        k2P=dt*dCpdt(t+dt,Ci+k1I/2,k2)
        k3R=dt*dCrdt(t+dt,Cr+k2R/2,Ci+k2I/2,k1,k1r)
        k3I=dt*dCidt(t+dt,Cr+k2R/2,Ci+k2I/2,k1,k1r,k2)
        k3P=dt*dCpdt(t+dt,Ci+k2I/2,k2)
        k4R=dt*dCrdt(t+dt,Cr+k3R,Ci+k3I,k1,k1r)
        k4I=dt*dCidt(t+dt,Cr+k3R,Ci+k3I,k1,k1r,k2)
        k4P=dt*dCpdt(t+dt,Ci+k3I,k2)
        Cr=Cr+(k1R+2*(k2R+k3R)+k4R)/6
        Ci=Ci+(k1I+2*(k2I+k3I)+k4I)/6
        Cp=Cp+(k1P+2*(k2P+k3P)+k4P)/6
        t=t+dt;

    C=cP*V*Ci+cR*V*Cr0+cO*tf+cT*tf*(T-298)
    return C#,Ci,Cr


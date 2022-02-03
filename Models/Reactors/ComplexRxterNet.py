# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:28:52 2020

@author: leonardo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:46:57 2020

@author: leonardo
"""

from numpy import exp, arange, random, array, vstack, argmax, savetxt, hstack
from numpy import argmin, loadtxt, apply_along_axis, zeros
from matplotlib import pyplot as pyp
import time
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import GPy
from Acquisition_Funcs import LCB, EI, PI

#pyp.close('all')
##################Initial Conditions and Physical Parameters##################
Cr0=5; Ci0=0; Cp0=0; Ca0=0; Cb0=0; tf=50; t0=0; dt=0.05; k01=0.11; k02=2.2;
k03=5; k04=8.5; R=8.314; E1=7000; E2=23500; E3=2000; E4=26000; V=1000;
cP=-145; cR=0; cT=1; cO=5500; 
##############################################################################

def Ctfxn(T):
    k1=k01*exp(-E1/(R*T)); k2=k02*exp(-E2/(R*T)); k3=k03*exp(-E3/(R*T));
    k4=k04*exp(-E4/(R*T)); k1r=k1/5; k3r=k3/5; k4r=0; 
    n=int((tf-t0)/dt); Cr=Cr0*1; Ci=Ci0*1; Cp=Cp0*1; Ca=Ca0*1; Cb=Cb0*1; t=t0*1;

    
    def dCrdt(t,Cr,Ci,k1,k1r):
        return -k1*Cr+k1r*Ci
    def dCidt(t,Cr,Ci,Cp,Ca,k1,k1r,k2,k4,k4r,k3,k3r):
        return k1*Cr-k1r*Ci-k2*Ci-k3*Ci*Cp+k3r*Ca+2*(k4*Ca-k4r*Ci**2)
    def dCpdt(t,Ci,Cp,Ca,k2,k3,k3r):
        return k2*Ci-k3*Ci*Cp+k3r*Ca
    def dCadt(t,Ci,Cp,Ca,k3,k3r,k4,k4r):
        return k3*Ci*Cp-k3r*Ca-k4*Ca+k4r*Ci**2
    
    for i in range(n):
        k1R=dt*dCrdt(t,Cr,Ci,k1,k1r)
        k1I=dt*dCidt(t,Cr,Ci,Cp,Ca,k1,k1r,k2,k4,k4r,k3,k3r)
        k1P=dt*dCpdt(t,Ci,Cp,Ca,k2,k3,k3r)
        k1A=dt*dCadt(t,Ci,Cp,Ca,k3,k3r,k4,k4r)
        k2R=dt*dCrdt(t+dt,Cr+k1R/2,Ci+k1I/2,k1,k1r)
        k2I=dt*dCidt(t+dt,Cr+k1R/2,Ci+k1I/2,Cp+k1P/2,Ca+k1A/2,k1,k1r,k2,k4,k4r,k3,k3r)
        k2P=dt*dCpdt(t+dt,Ci+k1I/2,Cp+k1P/2,Ca+k1A/2,k2,k3,k3r)
        k2A=dt*dCadt(t,Ci+k1I/2,Cp+k1P/2,Ca+k1A/2,k3,k3r,k4,k4r)
        k3R=dt*dCrdt(t+dt,Cr+k2R/2,Ci+k2I/2,k1,k1r)
        k3I=dt*dCidt(t+dt,Cr+k2R/2,Ci+k2I/2,Cp+k2P/2,Ca+k2A/2,k1,k1r,k2,k4,k4r,k3,k3r)
        k3P=dt*dCpdt(t+dt,Ci+k2I/2,Cp+k2P/2,Ca+k2A/2,k2,k3,k3r)
        k3A=dt*dCadt(t,Ci+k2I/2,Cp+k2P/2,Ca+k2A/2,k3,k3r,k4,k4r)
        k4R=dt*dCrdt(t+dt,Cr+k3R,Ci+k3I,k1,k1r)
        k4I=dt*dCidt(t+dt,Cr+k3R,Ci+k3I,Cp+k3P,Ca+k3A,k1,k1r,k2,k4,k4r,k3,k3r)
        k4P=dt*dCpdt(t+dt,Ci+k3I,Cp+k3P,Ca+k3A,k2,k3,k3r)
        k4A=dt*dCadt(t,Ci+k3I,Cp+k3P,Ca+k3A,k3,k3r,k4,k4r)
        Cr=Cr+(k1R+2*(k2R+k3R)+k4R)/6
        Ci=Ci+(k1I+2*(k2I+k3I)+k4I)/6
        Cp=Cp+(k1P+2*(k2P+k3P)+k4P)/6
        Ca=Ca+(k1A+2*(k2A+k3A)+k4A)/6
        t=t+dt;

    C=cP*V*Ci+cR*V*Cr0+cO*tf+cT*tf*(T-298)
    return C#,Cr,Ci,Cp,Ca

#T=arange(373,974,1); 
#Ct=array(list(map(Ctfxn, T)));
#Cr=Ct[:,1]; Ci=Ct[:,2]; Cp=Ct[:,3]; Ca=Ct[:,4]; Ct=Ct[:,0]

# pyp.figure()
# pyp.plot(T,Ct)
# pyp.figure()
# pyp.plot(T,Ci)
# pyp.figure()
# pyp.plot(T,Cr)
# pyp.plot(T,Cp)
# pyp.plot(T,Ca)

noise=0.001#*random.normal(loc=0,scale=1)
def func(T):
    T=T*1000;
    return (Ctfxn(T))/1e4
#%%############################## GPyOpt #####################################
Tinit=random.rand(); Tinit=0.600*Tinit+0.373; Tinit=array(Tinit).reshape(-1,1)
Ctinit=func(Tinit); Ctbest=[]; trials=12; ar=arange(0,trials,1)
domain1=[{'name':'t','type':'continuous','domain':(0.373,1.000)}]
t1=time.perf_counter()
for i in range(trials):
    GPyOpt.acquisitions.LCB.AcquisitionLCB.exploration_weight=2**(-i)
#    GPyOpt.acquisitions.EI.AcquisitionEI.jitter=0.0001/(i+1)
#    print(GPyOpt.acquisitions.EI.AcquisitionEI.jitter)
    results3=BayesianOptimization(f=None,domain=domain1,
                              X=Tinit,Y=Ctinit,acqusition_type='LCB')
    Tnxt=results3.suggest_next_locations()
    Ctnxt=func(Tnxt)
    Tinit=vstack([Tinit,Tnxt])
    Ctinit=vstack([Ctinit,Ctnxt])
    Ctbest.append(min(Ctinit))
t2=time.perf_counter(); print(t2-t1)
results3.plot_acquisition()
pyp.figure(), pyp.plot(Ctbest); pyp.scatter(ar,Ctbest); pyp.show();
#print("The minimum value obtained was %f at T=%f"
#      %(results3.fx_opt,results3.x_opt))

#%%####################### GPReg and Made AFs ################################
Tinit=random.uniform(0.373,0.973,1).reshape(-1,1);
Ctinit=func(Tinit); Ctbest=min(Ctinit); trials=10;
#Tseed=Tinit.copy(); Ctseed=Ctinit.copy();
Tk=arange(0.373,0.973,0.001).reshape(-1,1); Ctk=func(Tk);
exp_w=4; Tm=arange(0.373,0.9731,0.0001).reshape(-1,1);
#kernel=GPy.kern.RBF(1,variance=0.1,lengthscale=1)
kernel=GPy.kern.Matern52(1,variance=49,lengthscale=0.2)
#kernel=GPy.kern.MLP(1,variance=6.3,weight_variance=18,bias_variance=3)
model=GPy.models.GPRegression(Tinit,Ctinit,kernel);
model.Gaussian_noise.variance=noise;
model.Gaussian_noise.variance.fix();
#model.optimize();
Ctm,std=model.predict(Tm); std=std**0.5; af=LCB(Ctm,std,exp_w);
#model.plot() #figsize=[1.7,1.05]
pyp.figure(); pyp.plot(Tm,Ctm);
pyp.scatter(Tinit,Ctinit,color='k',marker='x')
pyp.fill_between(Tm.reshape(-1),(Ctm-2*std).reshape(-1),(Ctm+2*std).reshape(-1)
                 ,alpha=0.1);
pyp.plot(Tk,Ctk,'g--'); #pyp.legend().remove();
pyp.xlim((0.373,0.973)); pyp.ylim((-4,-0.5));
#pyp.xticks([]); pyp.yticks([]);
pyp.legend(['Mean','f(x)','Data','Confidence'],loc='upper right')
#pyp.savefig('LCB_OBJ1.png',dpi=300,bbox_inches='tight',pad_inches=0)
#pyp.figure(figsize=[1.748,1.102])
pyp.figure()#figsize=[1.7,1.05]
pyp.plot(Tm,af,'k');
pyp.scatter(Tm[argmax(af)],max(af),color='r',marker='o')
pyp.xticks([]); pyp.yticks([]);
#pyp.savefig('LCB_AF1.png',dpi=300,bbox_inches='tight',pad_inches=0)

t1=time.perf_counter()
for i in range(trials):
    n=argmax(af); Tnxt=Tm[n]; Ctnxt=func(Tnxt)
    Tinit=vstack([Tinit,Tnxt]); Ctinit=vstack([Ctinit,Ctnxt]);
    Ctbest=vstack([Ctbest,min(Ctinit)]);
    model=GPy.models.GPRegression(Tinit,Ctinit,kernel)
    model.Gaussian_noise.variance=noise;
    model.Gaussian_noise.variance.fix();
    if i>=6:
        exp_w=exp_w/2
    Ctm,std=model.predict(Tm); std=std**0.5; af=LCB(Ctm,std,exp_w);
#    if i>9 and abs(Tinit[i+3]-Tinit[i+1])<=1e-6:
#        break
t2=time.perf_counter(); print(t2-t1)
#model.plot() #figsize=[1.7,1.05]
fig,(ax1,ax2)=pyp.subplots(2,1); fig.tight_layout(pad=3.0)
#ax1=fig.add_subplot(
ax1.plot(1e3*Tm,Ctm);
ax1.scatter(1e3*Tinit,Ctinit,color='k',marker='x')
ax1.fill_between(1e3*Tm.reshape(-1),(Ctm-2*std).reshape(-1),
                 (Ctm+2*std).reshape(-1),alpha=0.1);
ax1.plot(1e3*Tk,Ctk,'g--'); #pyp.legend().remove();
#pyp.scatter(xseed,yseed,color='r',marker='x')
ax1.set_xlim((373,973)); ax1.set_ylim((-4,-0.5));
ax1.set_xlabel('Temperature'); ax1.set_ylabel('Cost(10,000)')#pyp.xticks([]); pyp.yticks([]);
ax1.legend(['Mean','f(x)','Data','Confidence'],loc='upper right')
ax1.scatter(1e3*Tinit[-1],Ctbest[-1],color='r',marker='x')
#pyp.savefig('LCB_Rxtr.png',dpi=300,bbox_inches='tight',pad_inches=0)
#pyp.figure(figsize=[1.748,1.102])
#pyp.figure()#figsize=[1.7,1.05]
#ax2=fig.add_subplot(212)
ax2.plot(Tm,af,'k');
#ax2.scatter(Tm[argmax(af)],max(af),color='r',marker='o')
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_ylabel('PI')
fig.tight_layout(pad=1)
pyp.savefig('EI_Rxtr_AF.png',dpi=300)#,bbox_inches='tight',pad_inches=0)

it=list(range(Ctbest.shape[0]))
pyp.figure()
pyp.plot(Ctbest)
pyp.scatter(it,Ctbest)
pyp.xlabel('Iteration number'); pyp.ylabel('Best Observed Cost(10,000)')
pyp.savefig('EI_Rxtr_Prog.png',dpi=300)

#%% Nomad
import sys; sys.path.insert(1,r'C:\Users\leonardo\Desktop\nomad.3.9.1_Personal\bin')
import PyNomad; from numpy import zeros; from matplotlib import pyplot as pyp
T00=zeros(25).reshape(-1,1); TF=zeros(25).reshape(-1,1)
for j in range(1):
    l=25; T0=random.uniform(0.373,0.973,1).reshape(-1,1);
    fsol=zeros(l); xsol=zeros(l); itn=list(range(1,l+1)); nfevals=zeros(l)
    FSOL=itn.copy()
    for i in range(l):
        str1=str(i+1)
        def bb3(x):
            T=x.get_coord(0);
            f=func(T); #print(Cr0,T,tf,f)
            sigma=0.001; f=f+sigma*random.choice([-1,1]); #f=f*1e4
            x.set_bb_output(0,f)
            return 1

        x0=T0; lb=[0.373]; ub=[0.973];
#        params = ['BB_OUTPUT_TYPE OBJ PB','INITIAL_MESH_SIZE 5e-2','INITIAL_POLL_SIZE 5e-2','MAX_BB_EVAL '+str1,'UPPER_BOUND * 1']
        params = ['BB_OUTPUT_TYPE OBJ PB PB','MIN_MESH_SIZE 0.001','MAX_BB_EVAL '+str1,'UPPER_BOUND * 1']  
        [ x_return , f_return , h_return, nb_evals , nb_iters ,  stopflag ] = PyNomad.optimize(bb3,x0,lb,ub,params);
        fsol=f_return; xsol=x_return[0]; FSOL[i]=fsol
        nfevals[i]=nb_evals
#    T00[j]=T0*1000; TF[j]=xsol*1000;
#    if j==l-1:
    pyp.figure(), pyp.scatter(itn,FSOL,color='black',marker='x',s=20); pyp.plot(itn,FSOL,color='black');
    pyp.xlabel('Iteration number'); pyp.ylabel('Ct (10,000)'); pyp.show();
    pyp.savefig('Nomad_Results_Itr.png',dpi=300)
    pyp.figure(), pyp.scatter(itn,nfevals,color='black',s=20); pyp.show()

# TT=hstack([T00, TF]); TT=TT[TT[:,0].argsort()];
# #pyp.figure(), pyp.scatter(itn,fsol,color='black',s=20); pyp.plot(itn,fsol,color='black');
# #pyp.xlabel('Iteration number'); pyp.ylabel('Ct'); pyp.show();
# #pyp.figure(), pyp.scatter(itn,nfevals,color='black',s=20); pyp.show()
# pyp.figure(); pyp.scatter(T00,TF,marker='x',color='k')
# pyp.plot(TT[:,0],TT[:,1],color='k')
# pyp.xlabel(r'Initial Seed Point $(T_{0})$'); pyp.ylabel(r'Optimal Point $(T_{f})$')
# pyp.savefig('Nomad_Results.png',dpi=300)

#%% Generate Test Data
Traw=random.uniform(0.373,0.973,200).reshape(-1,1);
savetxt('Rxtr_Temp_Data.txt',Traw)
Ctraw=func(Tinit)
savetxt('Rxtr_Cost_Data.txt',Ctraw)

#%%#################### Multivaraite Case Study ###########################%%#
'row array: x=array([[x1,x2,x3]])'
##################Initial Conditions and Physical Parameters##################
t0=0; dt=0.05; k01=0.11; k02=2.2; k03=5; k04=8.5; R=8.314; E1=7000; E2=23500;
E3=2000; E4=26000; V=1000; cP=-145; cR=0; cT=1; cO=5500; 
##############################################################################

def CtfxnM(x):
    Cr0=x[0]; T=x[1]; tf=x[2]; Ci0=0; Cp0=0; Ca0=0; Cb0=0;
    k1=k01*exp(-E1/(R*T)); k2=k02*exp(-E2/(R*T)); k3=k03*exp(-E3/(R*T));
    k4=k04*exp(-E4/(R*T)); k1r=k1/5; k3r=k3/5; k4r=0; 
    n=int((tf-t0)/dt); Cr=Cr0*1; Ci=Ci0*1; Cp=Cp0*1; Ca=Ca0*1; Cb=Cb0*1; t=t0*1;

    
    def dCrdt(t,Cr,Ci,k1,k1r):
        return -k1*Cr+k1r*Ci
    def dCidt(t,Cr,Ci,Cp,Ca,k1,k1r,k2,k4,k4r,k3,k3r):
        return k1*Cr-k1r*Ci-k2*Ci-k3*Ci*Cp+k3r*Ca+2*(k4*Ca-k4r*Ci**2)
    def dCpdt(t,Ci,Cp,Ca,k2,k3,k3r):
        return k2*Ci-k3*Ci*Cp+k3r*Ca
    def dCadt(t,Ci,Cp,Ca,k3,k3r,k4,k4r):
        return k3*Ci*Cp-k3r*Ca-k4*Ca+k4r*Ci**2
    
    for i in range(n):
        k1R=dt*dCrdt(t,Cr,Ci,k1,k1r)
        k1I=dt*dCidt(t,Cr,Ci,Cp,Ca,k1,k1r,k2,k4,k4r,k3,k3r)
        k1P=dt*dCpdt(t,Ci,Cp,Ca,k2,k3,k3r)
        k1A=dt*dCadt(t,Ci,Cp,Ca,k3,k3r,k4,k4r)
        k2R=dt*dCrdt(t+dt,Cr+k1R/2,Ci+k1I/2,k1,k1r)
        k2I=dt*dCidt(t+dt,Cr+k1R/2,Ci+k1I/2,Cp+k1P/2,Ca+k1A/2,k1,k1r,k2,k4,k4r,k3,k3r)
        k2P=dt*dCpdt(t+dt,Ci+k1I/2,Cp+k1P/2,Ca+k1A/2,k2,k3,k3r)
        k2A=dt*dCadt(t,Ci+k1I/2,Cp+k1P/2,Ca+k1A/2,k3,k3r,k4,k4r)
        k3R=dt*dCrdt(t+dt,Cr+k2R/2,Ci+k2I/2,k1,k1r)
        k3I=dt*dCidt(t+dt,Cr+k2R/2,Ci+k2I/2,Cp+k2P/2,Ca+k2A/2,k1,k1r,k2,k4,k4r,k3,k3r)
        k3P=dt*dCpdt(t+dt,Ci+k2I/2,Cp+k2P/2,Ca+k2A/2,k2,k3,k3r)
        k3A=dt*dCadt(t,Ci+k2I/2,Cp+k2P/2,Ca+k2A/2,k3,k3r,k4,k4r)
        k4R=dt*dCrdt(t+dt,Cr+k3R,Ci+k3I,k1,k1r)
        k4I=dt*dCidt(t+dt,Cr+k3R,Ci+k3I,Cp+k3P,Ca+k3A,k1,k1r,k2,k4,k4r,k3,k3r)
        k4P=dt*dCpdt(t+dt,Ci+k3I,Cp+k3P,Ca+k3A,k2,k3,k3r)
        k4A=dt*dCadt(t,Ci+k3I,Cp+k3P,Ca+k3A,k3,k3r,k4,k4r)
        Cr=Cr+(k1R+2*(k2R+k3R)+k4R)/6
        Ci=Ci+(k1I+2*(k2I+k3I)+k4I)/6
        Cp=Cp+(k1P+2*(k2P+k3P)+k4P)/6
        Ca=Ca+(k1A+2*(k2A+k3A)+k4A)/6
        t=t+dt;
    C=cP*V*Ci+cR*V*Cr0+cO*tf+cT*tf*(T-298)+1e5*max(0,(2-Ci))
    return C#,Cr,Ci,Cp,Ca

def funcM(x):
    x=x.reshape(1,3)
    Ca0=x[0,0]*5;
    T=x[0,1]*600+373;
    tf=x[0,2]*90+10;
    return (CtfxnM([Ca0,T,tf]))/1e4
l=1; trials=50; exp_w=4; Ctavg=zeros(trials+1).reshape(-1,1)
for j in range(l):
    Ca0init=random.uniform(0,1,3).reshape(-1,1);
    Tinit=random.uniform(0,1,3).reshape(-1,1);
    tfinit=random.uniform(0,1,3).reshape(-1,1)
    xinit=hstack([Ca0init,Tinit,tfinit])
    Ctinit=apply_along_axis(funcM,1,xinit).reshape(-1,1); Ctbest=min(Ctinit);
#Ca0m=arange(0,1.0001,0.0001).reshape(-1,1);
#Tm=arange(0,1.0001,0.0001).reshape(-1,1);
#tfm=arange(0,1.0001,0.0001).reshape(-1,1);
    xm=loadtxt('Rxt_Param_Space_Multi.txt')
#kernel=GPy.kern.RBF(1,variance=0.1,lengthscale=1)
    kernel=GPy.kern.Matern52(3,variance=49,lengthscale=0.2)
#kernel=GPy.kern.MLP(1,variance=6.3,weight_variance=18,bias_variance=3)
    model=GPy.models.GPRegression(xinit,Ctinit,kernel);
    model.Gaussian_noise.variance=0#noise;
    model.Gaussian_noise.variance.fix();
#model.optimize();
    Ctm,std=model.predict(xm); std=std**0.5; af=PI(Ctm,std,exp_w);

    t1=time.perf_counter()
    for i in range(trials):
        n=argmax(af); xnxt=xm[n]; Ctnxt=funcM(xnxt)
        xinit=vstack([xinit,xnxt]); Ctinit=vstack([Ctinit,Ctnxt]);
        Ctbest=vstack([Ctbest,min(Ctinit)]);
        model=GPy.models.GPRegression(xinit,Ctinit,kernel)
        model.Gaussian_noise.variance=0#noise;
        model.Gaussian_noise.variance.fix();
        if i>=5:
            exp_w=exp_w/2
        Ctm,std=model.predict(xm); std=std**0.5; af=PI(Ctm,std,Ctbest[-1],0.15);
#    if i>9 and abs(Tinit[i+3]-Tinit[i+1])<=1e-6:
#        break
    t2=time.perf_counter(); print(t2-t1)
    Ctavg=(Ctavg+Ctbest)

Ctavg=Ctavg/(j+1)
it=list(range(Ctbest.shape[0]))
pyp.figure()
pyp.plot(Ctavg)
pyp.scatter(it,Ctavg)
pyp.xlabel('Iteration number'); pyp.ylabel('Best Observed Cost(10,000)')
pyp.savefig('PI_Rxtr_Prog_Multi.png',dpi=300)

idxb=argmin(Ctinit); xbest=xinit[idxb]*1; 
xbest[0]=xbest[0]*5; xbest[1]=xbest[1]*600+373; xbest[2]=xbest[2]*90+10
print(xbest,Ctinit[idxb])

#%%############################## GPyOptM ####################################
Ca0init=random.uniform(0,1,3).reshape(-1,1);
Tinit=random.uniform(0,1,3).reshape(-1,1);
tfinit=random.uniform(0,1,3).reshape(-1,1)
Ctinit=func(Tinit); Ctbest=[]; trials=15; ar=arange(0,trials,1)
domainM=[{'name':'Ca0','type':'continuous','domain':(0,1.000)},
        {'name':'T','type':'continuous','domain':(0,1.000)},
        {'name':'tf','type':'continuous','domain':(0,1.000)}]
t1=time.perf_counter()
resultsM=BayesianOptimization(f=funcM,domain=domainM, initial_design_numdata=3,
                              acqusition_type='LCB')
resultsM.run_optimization(max_iter=50,max_time=1800,eps=1e-8,verbosity=False)
t2=time.perf_counter(); print(t2-t1)
insM=resultsM.get_evaluations()[0]
outsM=resultsM.get_evaluations()[1]
evalsM=hstack([insM,outsM])
resultsM.plot_acquisition()
resultsM.plot_convergence()
#print("The minimum value obtained was %f at T=%f"
#      %(results3.fx_opt,results3.x_opt))
#%% NomadM
import sys; sys.path.insert(1,r'C:\Users\leonardo\Desktop\nomad.3.9.1_Personal\bin')
import PyNomad; from numpy import zeros; from matplotlib import pyplot as pyp
T00=zeros(25).reshape(-1,1); TF=zeros(25).reshape(-1,1)
FSOL=zeros(l); l=50;
for j in range(20):
    Ca0=random.uniform(0,1,1).reshape(-1,1);
    T0=random.uniform(0,1,1).reshape(-1,1);
    tf0=random.uniform(0,1,1).reshape(-1,1);
    fsol=zeros(l); xsol=zeros(l); itn=list(range(1,l+1)); nfevals=zeros(l)
    for i in range(l):
        str1=str(i+1)
        def bb3(x):
            Ca0=x.get_coord(0)
            T=x.get_coord(1);
            tf=x.get_coord(2);
            f=funcM(array([Ca0,T,tf])); #print(Cr0,T,tf,f)
            sigma=0.001; f=f+sigma*random.choice([-1,1]); #f=f*1e4
            x.set_bb_output(0,f)
            return 1

        x0=array([Ca0,T0,tf0]); lb=[0,0,0]; ub=[1,1,1];
#        params = ['BB_OUTPUT_TYPE OBJ PB','INITIAL_MESH_SIZE 5e-2','INITIAL_POLL_SIZE 5e-2','MAX_BB_EVAL '+str1,'UPPER_BOUND * 1']
        params = ['BB_OUTPUT_TYPE OBJ PB PB','MIN_MESH_SIZE 1e-6','MAX_BB_EVAL '+str1,'UPPER_BOUND * 1']  
        [ x_return , f_return , h_return, nb_evals , nb_iters ,  stopflag ] = PyNomad.optimize(bb3,x0,lb,ub,params);
        fsol=f_return; xsol=x_return[0]; FSOL[i]=fsol+FSOL[i]
        nfevals[i]=nb_evals
#    T00[j]=T0*1000; TF[j]=xsol*1000;
#    if j==l-1:
#    pyp.figure(), pyp.scatter(itn,FSOL,color='black',marker='x',s=20); pyp.plot(itn,FSOL,color='black');
#    pyp.xlabel('Iteration number'); pyp.ylabel('Ct (10,000)'); pyp.show();
#    pyp.savefig('Nomad_Results_ItrM.png',dpi=300)
#    pyp.figure(), pyp.scatter(itn,nfevals,color='black',s=20); pyp.show()

FSOL=FSOL/(j+1); it=list(range(FSOL.shape[0]))
pyp.figure(); pyp.plot(FSOL,color='k'); pyp.scatter(it,FSOL,color='k',marker='x')
# TT=hstack([T00, TF]); TT=TT[TT[:,0].argsort()];
# #pyp.figure(), pyp.scatter(itn,fsol,color='black',s=20); pyp.plot(itn,fsol,color='black');
pyp.xlabel('Iteration number'); pyp.ylabel('Ct (10,000)'); pyp.show();
# #pyp.figure(), pyp.scatter(itn,nfevals,color='black',s=20); pyp.show()
# pyp.figure(); pyp.scatter(T00,TF,marker='x',color='k')
# pyp.plot(TT[:,0],TT[:,1],color='k')
# pyp.xlabel(r'Initial Seed Point $(T_{0})$'); pyp.ylabel(r'Optimal Point $(T_{f})$')
pyp.savefig('Nomad_Results_ItrM.png',dpi=300)
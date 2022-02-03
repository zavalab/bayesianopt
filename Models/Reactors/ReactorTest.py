# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:57:55 2020

@author: leonardo
"""

import sys
sys.path.insert(1, r'C:\Users\leonardo\OneDrive - UW-Madison\Research')
from Ctfxn import Ctfxn
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from numpy import arange, array, random, argmin, asarray, append, shape, ones
from numpy import zeros, apply_along_axis, vstack
import warnings
from scipy.stats import norm
from matplotlib import pyplot as pyp
from mpl_toolkits.mplot3d import Axes3D
pyp.close('all')
#%%########################## UNIVARIATE CASE ################################
def Ctnoise(Cr0,Ci0,Cp0,T,tf,noise=1):
    Cr0=Cr0*5; Ci0=Ci0*5; Cp0=Cp0*5; T=T*1000; tf=tf*100;
    noise=0.01*random.normal(loc=0,scale=noise)
    Ct,Ci,Cr=Ctfxn(Cr0,Ci0,Cp0,T,tf)
    Ct=Ct/1e4+noise
    return Ct, Ci, Cr

def surrogate(model,x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(x, return_std=True)

def plot(x,y,model):
    pyp.scatter(x,y)
    xsamples=asarray(arange(0.3,1,0.001)).reshape(-1,1)
    ysamples,std=surrogate(model,xsamples)
    pyp.plot(xsamples, ysamples); pyp.show()

def acquisition(x, xsamples, model):
    yhat,_=surrogate(model,x)
    best=min(yhat)
    mu, std=surrogate(model,xsamples)
    mu=mu[:,0]
    probs=norm.cdf((mu-best)/(std+1e-9))
    return probs

def acquisitionlcb(x, xsamples, model):
    mu, std=surrogate(model,xsamples)
    mu=mu[:,0]; k=2#(2**(-i/10))
    G=((mu-k*std))
    return G,min(mu)

def opt_acquisition(x,y,model):
    xsamples=0.3+(1-0.3)*random.rand(100).reshape(-1,1)
    scores,minmu=acquisitionlcb(x,xsamples,model)
    idx=argmin(scores)
    return xsamples[idx],minmu

Cr0=1; Ci0=0; Cp0=0; tf=0.42; model=GP();
T=0.3+(1-0.3)*random.rand(1).reshape(-1,1)
C=array([Ctnoise(Cr0,Ci0,Cp0,x,tf) for x in T])
Ct=C[:,0]; Ci=C[:,1]; Cr=C[:,2]
model.fit(T,Ct);
fig1=pyp.figure(); plot(T,Ct,model);
test=10; l=len(Ct); est=Ct[:,0]*ones(test+l); estmin=min(est)*ones(test+l)
obsmin=estmin.copy();
for i in range(test):
    Tnxt,est[i+l]=opt_acquisition(T,Ct,model)
    Chat=array(Ctnoise(Cr0,Ci0,Cp0,Tnxt,tf))
    Cthat=Chat[0]; Cihat=Chat[1]; Crhat=Chat[2]
    T=append(T,Tnxt).reshape(-1,1)
    Ct=append(Ct,Cthat).reshape(-1,1)
    Ci=append(Ci,Cihat).reshape(-1,1)
    Cr=append(Cr,Crhat).reshape(-1,1)
    estmin[i+l]=min(est); obsmin[i+l]=min(Ct)
    model.fit(T,Ct)
fig2=pyp.figure(); plot(T,Ct,model);
idx=argmin(Ct); fig3=pyp.figure(); pyp.plot(obsmin); pyp.show()
print('Optimum is T=%.3f, Ct=%.3f' %(1e3*T[idx],1e4*Ct[idx]));

#%%######################### MULTIVARIATE CASE ###############################
def CtnoiseMVC(x,noise=1):
    Cr0=x[0]*5; Ci0=x[1]*5; Cp0=x[2]*5; T=x[3]*1000; tf=x[4]*100;
    noise=0.01*random.normal(loc=0,scale=noise)
    Ct,Ci,Cr=Ctfxn(Cr0,Ci0,Cp0,T,tf)
    Ct=Ct/1e4+noise
    return Ct, Ci, Cr

def surrogateMVC(model,x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(x, return_std=True)

def plot3D(x,y,model):
    fig=pyp.figure()
    ax=fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0],x[:,3],x[:,4],c='r',marker='o');
    ax.set_xlabel('Initial Concentration of A'); ax.set_ylabel('Temperature');
    ax.set_zlabel('Reaction Time'); pyp.show()
    #xsamples=asarray(arange(0.3,1,0.001)).reshape(-1,1)
    #ysamples,std=surrogateMVC(model,xsamples)
    #pyp.plot(xsamples, ysamples); pyp.show()

def acquisitionPI(x, xsamples,model):
    yhat,_=surrogateMVC(model,x)
    best=min(yhat)
    mu, std=surrogateMVC(model,xsamples)
    mu=mu[:,0]
    probs=norm.cdf((mu-best)/(std+1e-9))
    return probs

def acquisitionLCB(x,xsamples,model,i):
    mu, std=surrogateMVC(model,xsamples)
    mu=mu[:,0]; k=(2**(-i/10)); #print(k)
    G=((mu-k*std))
    return G,min(mu)

def opt_acquisitionMVC(x,xl,xu,y,model,i):
    c=shape(x)[1]; xsamples=ones((100,c));
    for j in range(c):
        xsamples[:,j]=xl[j]+(xu[j]-xl[j])*random.rand(100)
    scores,minmu=acquisitionLCB(x,xsamples,model,i)
    idx=argmin(scores)
    return xsamples[idx],minmu

model=GP(); xl=[0,0,0,0.3,0.01]; xu=[1,0,0,1,1];
Cr0=xl[0]+(xu[0]-xl[0])*random.rand(1).reshape(-1,1)
T=xl[3]+(xu[3]-xl[3])*random.rand(1).reshape(-1,1)
tf=xl[2]+(xu[4]-xl[4])*random.rand(1).reshape(-1,1)
l=len(T); w=len(xl); xmv=zeros((l,w));
xmv[:,0]=Cr0[:,0]; xmv[:,3]=T[:,0]; xmv[:,4]=tf[:,0]
C=apply_along_axis(CtnoiseMVC,1,xmv)
Ct=C[:,0].reshape(-1,1); Ci=C[:,1].reshape(-1,1); Cr=C[:,2].reshape(-1,1);
model.fit(xmv,Ct);
test=50; l2=len(Ct); est=Ct[:,0]*ones(test+l); estmin=min(est)*ones(test+l)
obsmin=estmin.copy();
for i in range(test):
    xnxt,est[i+l2]=opt_acquisitionMVC(xmv,xl,xu,Ct,model,i)
    estmin[i+l2]=min(estmin)
    Chat=CtnoiseMVC(xnxt)
    Cthat=Chat[0]; Cihat=Chat[1]; Crhat=Chat[2]
    xmv=vstack([xmv,xnxt])
    Ct=append(Ct,Cthat).reshape(-1,1)
    Ci=append(Ci,Cihat).reshape(-1,1)
    Cr=append(Cr,Crhat).reshape(-1,1)
    estmin[i+l2]=min(est); obsmin[i+l]=min(Ct);
    model.fit(xmv,Ct)
idx=argmin(Ct)
print('Optimum is Cr0=%.3f, T=%.3f, tf=%.3f, Ct=%.3f' 
      %(5*xmv[idx,0],1e3*xmv[idx,3],1e2*xmv[idx,4],1e4*Ct[idx]));
plot3D(xmv,Ct,model);
fig5=pyp.figure(); pyp.plot(obsmin); pyp.show()
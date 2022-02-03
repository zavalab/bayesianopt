# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:56:42 2020

@author: leonardo
"""

import sys;
sys.path.insert(1,r'C:\Users\leonardo\OneDrive - UW-Madison\Research\bayesianopt\Scripts')
from numpy import exp, arange, random, array, vstack, argmax, delete, hstack
from numpy import argmin, loadtxt, apply_along_axis, ones, round, intersect1d
from numpy import cos, sin, pi, matmul, linalg, transpose as trans
from matplotlib import pyplot as pyp
import time
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import GPy
from Acquisition_Funcs import LCB, EI, PI

def f(x):
    return (1/x)*sin(7*x)-2*cos(2*x)+sin(5*x)

def f2d(X):
    X=X.reshape(1,2)
    x=X[0,0]; y=X[0,1]
    return -(x*cos(x)+exp(-y))  
    

# f(x)
# x**6-3*x**5+8*x**2                                        [-1,2.65]
# x**9-x**8+x**7-1.5*x**6-2*x**5-0.5*x**4-2*x**3+5*x**2+x-1 [-1.15,1.5]
# x**9-x**8+x**7-1.5*x**6-x**5-0.5*x**4-2*x**3+5*x**2+x-1   [-1.05,1.4]
# (1/x)*sin(7*x)-2*cos(2*x)+sin(5*x)                        [-3,4]
# -(10*35*23*0.05*x)/((1+35*0.05+23*x)**2)                  [0.0,0.6]
# (1/(x**3-x+1))+x-5                                        [-1.3,10.0]

# f(x,y)
# x*cos(x)+exp(-y)                                          x:[0,10], y:[0:10]

def m(x):
    return 0.007298625*x**6+0.018417475*x**5-0.236156061*x**4-0.141672538*x**3+1.410502477*x**2+0*x-0.52803398

def m2d(X):
    X=X.reshape(1,2)
    x=X[0,0]; y=X[0,1]
    return -(0.082204508*(x)**2-0.1423999*x*y+0.011189003*(y)**2) 

# First f(x)
# -2.95471*x**3+7.845235*x**2-0.82337*x+0.329596
# -5.06579*x**3+9.256109*x**2+1.962406*x-0.3172
# 0.5*x**4-4.5*x**3+7.9375*x**2+0.9375*x-6.21725e-15
# 1.506100868*x**2-4.165265727*x+4.319414317
# Second f(x)
# 0.923549107*x**2-0.925502232*x-0.253348214
# -1.119791667*x**3+1.763392857*x**2+0.27827381*x-0.589285714
# 0.296875*x**4-1.416666667*x**3+1.37109375*x**2+0.511532738*x-0.525669643
# -0.56734262*x**2+0.510597944*x-0.390792622
# 1.047158147*x**3-1.038964859*x**2-0.946380318*x-0.206278869
# 1.159603179*x**4+0.225615084*x**3-2.968861467*x**2-0.135151953*x+0.10636666
# 0.809582148*x**4-0.069259329*x**3-2.282219695*x**2+0.525534775*x+0.143448915
# 6.03424375*x**5-6.234136875*x**4-10.33656994*x**3+7.520075094*x**2+3.199491437*x-1
# Third f(x)
# 1.983232771*x**2-0.511444573*x-0.461240931
# -2.137991564*x**3+4.002437137*x**2+1.059751239*x-0.893168693
# -1.104270938*x**4-1.134599688*x**3+5.041692734*x**2+0.740681172*x-1
# Fourth f(x)
# 0.007298625*x**6+0.018417475*x**5-0.236156061*x**4-0.141672538*x**3+1.410502477*x**2+0*x-0.52803398
# (1/x)*sin(7*x)-2*exp(-2.5*x**2)
# Fifth f(x)
# -(61.47141237*0.41539213*122.627843*0.05*x)/(1+0.41539213*0.05+122.627843*x)
# Sixth f(x)
# x-5
# (1/(x+1)**2)+x-5
#(-1*x**4+3.6*x**3+6.7*x**2+7.5*x+10)/(0*x**4-1.1*x**3-1.6*x**2-
#                          2.3*x-2.5)


# First f(x,y)
# 0.082204508*(x)**2-0.1423999*x*y+0.011189003*(y)**2


noise=0.001; exp_w=1; t=50; itr=array(arange(t+1)).reshape(-1,1)

xm=arange(-3,4.01,0.01).reshape(-1,1); yk=f(xm);
xinit=0.23487553*ones((1,1))#10*ones((1,1))#random.uniform(-3,4,1).reshape(-1,1)
yinit=f(xinit).reshape(-1,1); ybest=min(yinit).reshape(-1,1)
dinit=f(xinit)-m(xinit)
mk=m(xm).reshape(-1,1)

kernel=GPy.kern.Matern52(1,variance=30,lengthscale=1)
model=GPy.models.GPRegression(xinit,dinit,kernel);
model.Gaussian_noise.variance=noise;
model.Gaussian_noise.variance.fix();
dm,std=model.predict(xm); std=std**0.5; af=LCB(mk+dm,std,exp_w);
idx=ones((t,1),dtype=int); 

kernelgp=GPy.kern.Matern52(1,variance=30,lengthscale=1)
modelgp=GPy.models.GPRegression(xinit,yinit,kernelgp);
modelgp.Gaussian_noise.variance=noise;
modelgp.Gaussian_noise.variance.fix();
ym,stdgp=modelgp.predict(xm); stdgp=stdgp**0.5; afgp=LCB(ym,stdgp,exp_w); 

# XYm=loadtxt('2D_mech_funx1_param_space.txt');
# Zk=apply_along_axis(f2d,1,XYm).reshape(-1,1);
# Zmk=apply_along_axis(m2d,1,XYm).reshape(-1,1);
# XYinit=random.uniform(0,10,2).reshape(1,2)
# Zinit=f2d(XYinit).reshape(-1,1); Zbest=min(Zinit);
# d2dinit=f2d(XYinit)-m2d(XYinit).reshape(-1,1)

# kernel2d=GPy.kern.Matern52(2,variance=30,lengthscale=1)
# model2d=GPy.models.GPRegression(XYinit,d2dinit,kernel2d)
# model2d.Gaussian_noise.variance=noise;
# model2d.Gaussian_noise.variance.fix();
# dm2d,std2d=model2d.predict(XYm); std2d=std2d**0.5; af2d=LCB(Zmk+dm2d,std2d,exp_w);

# kernel2dgp=GPy.kern.Matern52(2,variance=30,lengthscale=1)
# model2dgp=GPy.models.GPRegression(XYinit,Zinit,kernel2dgp)
# model2dgp.Gaussian_noise.variance=noise;
# model2dgp.Gaussian_noise.variance.fix();
# Zm,std2dgp=model2dgp.predict(XYm); std2dgp=std2dgp**0.5; af2dgp=LCB(Zm,std2dgp,exp_w);

#######################Using 'mechanistic' model##############################
for i in range(t):
    xnxt=xm[argmax(af)]; ynxt=f(xnxt);
#    mnxt=m(xnxt); dnxt=ynxt-mnxt; 
    mnxt=mk[argmax(af)]; dnxt=ynxt-mnxt; idx[i]=argmax(af)
    xinit=vstack([xinit,xnxt]); yinit=vstack([yinit,ynxt]);
    dinit=vstack([dinit,dnxt]); ybest=vstack([ybest,min(yinit)])
    model=GPy.models.GPRegression(xinit,dinit,kernel);
    model.Gaussian_noise.variance=noise;
    model.Gaussian_noise.variance.fix();
    dm,std=model.predict(xm); std=std**0.5; af=LCB(mk+dm,std,exp_w);
    ypred=mk+dm; trms=10
    ypred2=ypred[0].reshape(-1,1);
    xmred=xm[0].reshape(-1,1)
    for j in range(1,ypred.shape[0]):
        if j%10==0:
            ypred2=vstack([ypred2,ypred[j]])
            xmred=vstack([xmred,xm[j]])
    xmred=(xmred-xmred[0])/(xmred[-1]-xmred[0])
    XM=ones((xmred.shape[0],1))
    for j in range(trms):
        XM=hstack([XM,cos((j+1)*pi*xmred),sin((j+1)*pi*xmred)])
    theta=matmul(linalg.inv(matmul(trans(XM),XM)),matmul(trans(XM),ypred2))
#    fpred=matmul(XM,theta)
    fpred=theta[0]*ones((xm.shape[0],1)); xmred=(xm-xm[0])/(xm[-1]-xm[0])
    for j in range(trms):
        fpred=fpred+theta[2*j+1]*cos((j+1)*pi*xmred)+theta[2*j+2]*sin((j+1)*pi*xmred)
    mk=fpred; dinit[1:i+2]=yinit[1:i+2]-vstack(mk[idx[0:i+1],:])

# for i in range(t):
#     XYnxt=XYm[argmax(af2d)]; Znxt=f2d(XYnxt); Zmnxt=m2d(XYnxt); d2dnxt=Znxt-Zmnxt;
#     XYinit=vstack([XYinit,XYnxt]); Zinit=vstack([Zinit,Znxt]);
#     d2dinit=vstack([d2dinit,d2dnxt]); Zbest=vstack([Zbest,min(Zinit)])
#     model2d=GPy.models.GPRegression(XYinit,d2dinit,kernel2d);
#     model2d.Gaussian_noise.variance=noise;
#     model2d.Gaussian_noise.variance.fix();
#     dm2d,std2d=model2d.predict(XYm); std2d=std2d**0.5; af2d=LCB(Zmk+dm2d,std2d,exp_w);
    
#########################Using 'pure' GP model################################
xinitgp=xinit[0:1]; yinitgp=yinit[0:1]; ybestgp=ybest[0];
# XYinitgp=XYinit[0:1]; Zinitgp=Zinit[0:1]; Zbestgp=Zbest[0]; 

for i in range(t):
    xnxt=xm[argmax(afgp)]; ynxt=f(xnxt); xinitgp=vstack([xinitgp,xnxt]);
    yinitgp=vstack([yinitgp,ynxt]); ybestgp=vstack([ybestgp,min(yinitgp)])
    modelgp=GPy.models.GPRegression(xinitgp,yinitgp,kernelgp)
    modelgp.Gaussian_noise.variance=noise;
    modelgp.Gaussian_noise.variance.fix();
    ym,stdgp=modelgp.predict(xm); stdgp=stdgp**0.5; afgp=LCB(ym,stdgp,exp_w);
    
# for i in range(t):
#     XYnxt=XYm[argmax(af2dgp)]; Znxt=f2d(XYnxt);
#     XYinitgp=vstack([XYinitgp,XYnxt]); Zinitgp=vstack([Zinitgp,Znxt]);
#     Zbestgp=vstack([Zbestgp,min(Zinitgp)])
#     model2dgp=GPy.models.GPRegression(XYinitgp,Zinitgp,kernel2dgp)
#     model2dgp.Gaussian_noise.variance=noise;
#     model2dgp.Gaussian_noise.variance.fix();
#     Zm,std2dgp=model2dgp.predict(XYm); std2dgp=std2dgp**0.5; af2dgp=LCB(Zm,std2dgp,exp_w);


pyp.figure()
pyp.plot(xm,mk+1*dm)
pyp.fill_between(xm.reshape(-1),(mk+1*dm-2*std).reshape(-1),
                  (mk+1*dm+2*std).reshape(-1),alpha=0.1);
pyp.scatter(xinit,yinit,marker='x',color='black')
pyp.plot(xm,yk,'g--')
pyp.plot(xm,mk,'r:')
pyp.plot(xm,dm,'c-.')
pyp.xlim((-3,4.00)); pyp.ylim((-4,6));

pyp.figure()
pyp.plot(xm,af)

pyp.figure()
pyp.plot(ybest)
pyp.scatter(itr,ybest)

# pyp.figure()
# pyp.plot(itr,Zbest)
# pyp.scatter(itr,Zbest)

pyp.figure()
pyp.plot(xm,ym)
pyp.fill_between(xm.reshape(-1),(ym-2*stdgp).reshape(-1),
                  (ym+2*stdgp).reshape(-1),alpha=0.1);
pyp.scatter(xinitgp,yinitgp,marker='x',color='black')
pyp.plot(xm,yk,'g--')
pyp.xlim((-3,4.00)); pyp.ylim((-4,6));

pyp.figure()
pyp.plot(xm,afgp)

pyp.figure()
pyp.plot(ybestgp)
pyp.scatter(itr,ybestgp)

#################### FOURIER APPROXIMATION OF FUNCTION #######################
# ypred=mk+dm; trms=10
# ypred2=ypred[0].reshape(-1,1);
# xmred=xm[0].reshape(-1,1)
# for i in range(1,ypred.shape[0]):
#     if i%10==0:
#         ypred2=vstack([ypred2,ypred[i]])
#         xmred=vstack([xmred,xm[i]])
# xmred=(xmred-xmred[0])/(xmred[-1]-xmred[0])
# XM=ones((xmred.shape[0],1))
# for i in range(trms):
#     XM=hstack([XM,cos((i+1)*pi*xmred),sin((i+1)*pi*xmred)])
# theta=matmul(linalg.inv(matmul(trans(XM),XM)),matmul(trans(XM),ypred2))
# fpred=matmul(XM,theta)
#fpred=theta[0]*ones((xm.shape[0],1)); xmred=(xm-xm[0])/(xm[-1]-xm[0])
#for i in range(trms):
#    fpred=fpred+theta[2*i+1]*cos((i+1)*pi*xmred)+theta[2*i+2]*
#    sin((i+1)*pi*xmred)
##############################################################################

# pyp.figure()
# pyp.plot(itr,Zbestgp)
# pyp.scatter(itr,Zbestgp)

# XYm2=XYm[0].reshape(1,2)
# for i in range(1,XYm.shape[0]):
#     if (i+1)%10==0:
#         XYm2=vstack([XYm2,XYm[i]])
  
# Zpred2=Zpred[0].reshape(-1,1)  
# for i in range(1,Zpred.shape[0]):
#     if (i+1)%10==0:
#         Zpred2=vstack([Zpred2,Zpred[i]])

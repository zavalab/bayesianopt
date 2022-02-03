# -*- coding: utf-8 -*-
"""
Created on Wed June 10 22:15:08 2020

@author: leonardo
"""

from numpy import arange, random, array, argmin, vstack, atleast_1d, meshgrid, hstack
from matplotlib import pyplot as pyp
import GPy
from scipy.optimize import minimize, Bounds
from joblib import Parallel, delayed

exp_w=2.6; t=15; bnds=Bounds(0,1); bnds2=Bounds((0,0),(1,1)); noise=1e-6
cons = ({'type':'ineq','fun':lambda x:x[0]},
        {'type':'ineq','fun':lambda x:1-x[0]})

def f(x):
    return x**6-3*x**5+8*x**2;

def f1(buffer_2, buffer_1):
    y = (5.066 - 33.31 * buffer_2 - 24.66 * buffer_1 + 119.9 * buffer_2**2 - 51.36 * buffer_2 * buffer_1 + 197.1 * buffer_1**2 - 130.5 * buffer_2**3
        + 309.3 * buffer_2**2 * buffer_1 - 126.4 * buffer_2 * buffer_1**2 - 497.9 * buffer_1**3 - 390.7 * buffer_2**3 * buffer_1 + 102.1 * buffer_2**2 * buffer_1**2 + 77.1 * buffer_2 * buffer_1**3 + 450.9 * buffer_1**4)
    return y

def LCB(x):
    x=array([x]).reshape(-1,1); x=x.reshape(1,x.shape[0])
    mu,std=model.predict(x); std=std**0.5
    return mu-exp_w*std

#%% Single Variable
x=arange(-1,2.66,0.01).reshape(-1,1)
y=f(x)

x=((x+1)/3.65); xinit=random.uniform(0,1,1).reshape(-1,1);#array([(2.4667+1)/3.65]).reshape(-1,1)
yinit=f(3.65*xinit[0]-1).reshape(-1,1); ybest=min(yinit);
kernel=GPy.kern.Matern52(1,variance=1,lengthscale=0.1)
model=GPy.models.GPRegression(xinit,yinit,kernel);
model.optimize(optimizer='lbfgsb');
model.Gaussian_noise.variance.fix(noise);
ym,std=model.predict(x); std=std**0.5; afs=ym-exp_w*std;
#xinits=xinit.copy(); yinits=yinit.copy()

x0=random.uniform(0,1,2).reshape(-1,1); funs=0*x0; xnxtm=0*x0
# Run Serially
# for i in range(x0.shape[0]):
#     af=minimize(LCB,x0[i],method='L-BFGS-B',bounds=bnds);#constraints=cons);
#     funs[i]=af.fun; xnxtm[i]=af.x
# Run in Parallel
opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds) for start_point in x0)
xnxtm=array([res.x for res in opt]);
funs=array([atleast_1d(res.fun)[0] for res in opt]); 
FUNS=arange(0.0,t,1); legs=[];

for i in range(t):
    xnxt=xnxtm[argmin(funs)].reshape(-1,1); FUNS[i]=funs[argmin(funs)];
    ynxt=f(3.65*xnxt-1); xinit=vstack([xinit,xnxt]); yinit=vstack([yinit,ynxt]);
    kernel=GPy.kern.Matern52(1,variance=1,lengthscale=0.1)
    model=GPy.models.GPRegression(xinit,yinit,kernel);
    model.optimize(optimizer='lbfgsb');
    model.Gaussian_noise.variance.fix(noise);
    x0=random.uniform(0,1,2).reshape(-1,1);
    # Run Serially
    # for j in range(x0.shape[0]):
    #     af=minimize(LCB,x0[j],method='L-BFGS-B',bounds=bnds);#constraints=cons);
    #     funs[j]=af.fun; xnxtm[j]=af.x
    # Run in Parallel
    opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds) for start_point in x0)
    xnxtm=array([res.x for res in opt]);
    funs=array([atleast_1d(res.fun)[0] for res in opt]);
    ym,std=model.predict(x); std=std**0.5; af=ym-exp_w*std;
    pyp.figure('FigureAF'); pyp.plot(x,af); pyp.scatter(xnxtm,funs);
    pyp.xlim((0,1)); legs.append('Run '+str(i+1));
    pyp.legend(legs);
    
    # xnxts=x[argmin(afs)]; ynxts=f(3.65*xnxts-1);
    # xinits=vstack([xinits,xnxts]); yinits=vstack([yinits,ynxts]);
    # models=GPy.models.GPRegression(xinits,yinits,kernel)
    # models.Gaussian_noise.variance=noise;
    # models.Gaussian_noise.variance.fix();
    # yms,stds=models.predict(x); stds=stds**0.5; afs=yms-exp_w*stds;

ym,std=model.predict(x); x=3.65*x-1; xinit=3.65*xinit-1; #xinits=3.65*xinits-1
pyp.figure(); pyp.plot(x,y); pyp.plot(x,ym);
pyp.fill_between(x.reshape(-1),(ym-exp_w*std**0.5).reshape(-1),
                   (ym+exp_w*std**0.5).reshape(-1),alpha=0.1);
pyp.scatter(xinit,yinit,marker='x'); pyp.xlim((-1,2.65)); pyp.ylim((-3,13));
pyp.figure(); pyp.plot(x,ym-exp_w*std**0.5);
itr=arange(1,t+2,1);
pyp.figure();
pyp.plot(itr[1:t+1],FUNS); pyp.scatter(itr[1:t+1],FUNS,marker='x');
pyp.plot(itr,yinit); pyp.scatter(itr,yinit,marker='x');

# pyp.figure(); pyp.plot(x,y); pyp.plot(x,yms);
# pyp.fill_between(x.reshape(-1),(yms-2*stds).reshape(-1),
#                    (yms+2*stds).reshape(-1),alpha=0.1);
# pyp.scatter(xinits,yinits,marker='x'); pyp.xlim((-1,2.65)); pyp.ylim((-3,13));

#%% Multi Variable

xinit=random.uniform(0,1,6).reshape(3,2);
yinit=f1(0.5*xinit[:,0],0.5*xinit[:,1]).reshape(-1,1); ybest=min(yinit);
kernel=GPy.kern.Matern52(2,variance=30,lengthscale=1)
model=GPy.models.GPRegression(xinit,yinit,kernel)
model.Gaussian_noise.variance=noise;
model.Gaussian_noise.variance.fix();

x0=random.uniform(0,1,200).reshape(100,2); funs=0*x0; xnxtm=0*x0
# Run Serially
# for i in range(x0.shape[0]):
#     af=minimize(LCB,x0[i],method='L-BFGS-B',bounds=bnds2);#constraints=cons);
#     funs[i]=af.fun; xnxtm[i]=af.x
# Run in Parallel
opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds2) for start_point in x0)
xnxtm=array([res.x for res in opt]);
funs=array([atleast_1d(res.fun)[0] for res in opt])

for i in range(t):
    xnxt=xnxtm[argmin(funs)].reshape(1,2); ynxt=f1(0.5*xnxt[0,0],0.5*xnxt[0,1]);
    xinit=vstack([xinit,xnxt]); yinit=vstack([yinit,ynxt]);
    ybest=vstack([ybest,min(yinit)]); 
    model=GPy.models.GPRegression(xinit,yinit,kernel)
    model.Gaussian_noise.variance=noise;
    model.Gaussian_noise.variance.fix();
    
    x0=random.uniform(0,1,200).reshape(100,2);
    # Run Serially
    # for j in range(x0.shape[0]):
    #     af=minimize(LCB,x0[j],method='L-BFGS-B',bounds=bnds2);#constraints=cons);
    #     funs[j]=af.fun; xnxtm[j]=af.x
    # # Run in Parallel
    opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds2)  for start_point in x0)
    xnxtm=array([res.x for res in opt]);
    funs=array([atleast_1d(res.fun)[0] for res in opt]);
    
xinit=0.5*xinit; beta1=arange(0.0,1.01,0.01); beta2=arange(0.0,1.01,0.01);
beta1,beta2=meshgrid(beta1,beta2);
beta1=beta1.reshape(-1,1); beta2=beta2.reshape(-1,1)
ym,std=model.predict(hstack([beta1,beta2]))
    
#%% Parallelize example
# from joblib import Parallel, delayed
# import time, math
# def my_fun(i):
#     """ We define a simple function here.
#     """
#     time.sleep(1)
#     return math.sqrt(i**2)

# num = 10
# start = time.time()
# for i in range(num):
#     my_fun(i)
# end = time.time()
# print('{:.4f} s'.format(end-start))

# start = time.time()
# # n_jobs is the number of parallel jobs
# Parallel(n_jobs=-1)(delayed(my_fun)(i) for i in range(num))
# end = time.time()
# print('{:.4f} s'.format(end-start))
    
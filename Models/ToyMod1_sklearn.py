# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:41:20 2020

@author: leonardo
"""

from numpy import arange, random, array, argmin, vstack, atleast_1d, hstack
from matplotlib import pyplot as pyp
from scipy.optimize import minimize, Bounds, curve_fit
from joblib import Parallel, delayed
import sklearn.gaussian_process as gpr
#from pyGPGO.covfunc import matern52
#from pyGPGO.acquisition import Acquisition
#from pyGPGO.surrogates.GaussianProcess import GaussianProcess
#from pyGPGO.GPGO import GPGO
from collections import OrderedDict

exp_w=2.6; t=14; bnds=Bounds(-1,1); bnds2=Bounds((-1,-1),(1,1)); noise=1e-6
cons = ({'type':'ineq','fun':lambda x:x[0]-1},
        {'type':'ineq','fun':lambda x:1-x[0]})

def f(x):
    return x**6-3*x**5+8*x**2;

def f2(x):
    x=3.65*x-1;
    return -(x**6-3*x**5+8*x**2);

def LCB(x):
    x=array([x]).reshape(-1,1); x=x.reshape(1,x.shape[0])
    mu,std=model.predict(x,return_std=True); std=std.reshape(-1,1)
    return (mu-exp_w*std).flatten()

def LCBmod(x):
    x=array([x]).reshape(-1,1); x=x.reshape(1,x.shape[0])
    mu,std=modelerr.predict(x,return_std=True); std=std.reshape(-1,1)
    return (fref((x+1-2.00/3.65)*3.65/2.00)+mu-exp_w*std).flatten()

x=arange(-1,2.66,0.01).reshape(-1,1);
y=f(x)

#%% GP Regression
xgp=random.uniform(-1,2.65,(5,1)); ygp=f(xgp);

kernel=gpr.kernels.Matern(1,(0.1,0.1),nu=2.5);
model=gpr.GaussianProcessRegressor(kernel,alpha=1e-6,normalize_y=True,
                                    n_restarts_optimizer=0)
model.fit(xgp,ygp)
ym,std=model.predict(x,return_std=True); std=std.reshape(-1,1)
fig,ax1=pyp.subplots(1,1,figsize=(10,8.5));
ax1.grid(color='gray',axis='both',alpha=0.25); ax1.set_axisbelow(True);
ax1.set_xlim((-1,2.65)); ax1.set_ylim((-3,13));
ax1.scatter(xgp,ygp,marker='o',color='white',edgecolor='black',zorder=3,s=200);
ax1.set_xlabel(r'$x$',fontsize=24); pyp.xticks(fontsize=24);
ax1.set_ylabel(r'$f(x)$',fontsize=24); pyp.yticks(fontsize=24);
ax1.plot(x,ym,color='red',linewidth=3);
ax1.plot(x,y,color='green',linewidth=3,linestyle='--'); 
pyp.fill_between(x.reshape(-1),(ym-2*std).reshape(-1),
                   (ym+2*std).reshape(-1),alpha=0.25);
ax1.legend([r'$GP\ Model$', r'$True\ Model$',
            r'$Sample\ Points$',r'$2\sigma\ Confidence\ Interval$'],fontsize=24);
#pyp.savefig('Toy_GP_l_is_0pt1.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

kernel=gpr.kernels.Matern(1,(0.5,0.5),nu=2.5);
model=gpr.GaussianProcessRegressor(kernel,alpha=1e-6,normalize_y=True,
                                    n_restarts_optimizer=0)
model.fit(xgp,ygp)
ym,std=model.predict(x,return_std=True); std=std.reshape(-1,1)
fig,ax1=pyp.subplots(1,1,figsize=(10,8.5));
ax1.grid(color='gray',axis='both',alpha=0.25); ax1.set_axisbelow(True);
ax1.set_xlim((-1,2.65)); ax1.set_ylim((-3,13));
ax1.scatter(xgp,ygp,marker='o',color='white',edgecolor='black',zorder=3,s=200);
ax1.set_xlabel(r'$x$',fontsize=24); pyp.xticks(fontsize=24);
ax1.set_ylabel(r'$f(x)$',fontsize=24); pyp.yticks(fontsize=24);
ax1.plot(x,ym,color='red',linewidth=3);
ax1.plot(x,y,color='green',linewidth=3,linestyle='--'); 
pyp.fill_between(x.reshape(-1),(ym-2*std).reshape(-1),
                   (ym+2*std).reshape(-1),alpha=0.25);
ax1.legend([r'$GP\ Model$', r'$True\ Model$',
            r'$Sample\ Points$',r'$2\sigma\ Confidence\ Interval$'],fontsize=24);
#pyp.savefig('Toy_GP_l_is_0pt5.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

kernel=gpr.kernels.Matern(1,(1.0,1.0),nu=2.5);
model=gpr.GaussianProcessRegressor(kernel,alpha=1e-6,normalize_y=True,
                                    n_restarts_optimizer=0)
model.fit(xgp,ygp)
ym,std=model.predict(x,return_std=True); std=std.reshape(-1,1)
fig,ax1=pyp.subplots(1,1,figsize=(10,8.5));
ax1.grid(color='gray',axis='both',alpha=0.25); ax1.set_axisbelow(True);
ax1.set_xlim((-1,2.65)); ax1.set_ylim((-3,13));
ax1.scatter(xgp,ygp,marker='o',color='white',edgecolor='black',zorder=3,s=200);
ax1.set_xlabel(r'$x$',fontsize=24); pyp.xticks(fontsize=24);
ax1.set_ylabel(r'$f(x)$',fontsize=24); pyp.yticks(fontsize=24);
ax1.plot(x,ym,color='red',linewidth=3);
ax1.plot(x,y,color='green',linewidth=3,linestyle='--'); 
pyp.fill_between(x.reshape(-1),(ym-2*std).reshape(-1),
                   (ym+2*std).reshape(-1),alpha=0.25);
ax1.legend([r'$GP\ Model$', r'$True\ Model$',
            r'$Sample\ Points$',r'$2\sigma\ Confidence\ Interval$'],fontsize=24);
#pyp.savefig('Toy_GP_l_is_1pt0.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);


#%% Single Variable
x=(2.00/3.65)*x+(2.00/3.65)-1; xinit=random.uniform(-1,1,1).reshape(-1,1);
xinit=array([(2.00/3.65)*1.50571143+(2.00/3.65)-1]).reshape(-1,1);#array([[0.19288134],[0.24403575]]);#array([[0.2484003],[0.49789958]]);#array([0.54844054]).reshape(-1,1);
yinit=f((xinit+1-2.00/3.65)*3.65/2.00).reshape(-1,1); ybest=min(yinit);
kernel=gpr.kernels.Matern(1,(0.1,100),nu=2.5);
model=gpr.GaussianProcessRegressor(kernel,alpha=1e-6,normalize_y=True,
                                     n_restarts_optimizer=10);
model.fit(xinit,yinit); #model.kernel_.get_params();
ym,std=model.predict(x,return_std=True); std=std.reshape(-1,1);
x0=random.uniform(-1,1,100).reshape(-1,1); funs=0*x0; xnxtm=0*x0
# Run in Parallel
opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds) for start_point in x0)
xnxtm=array([res.x for res in opt]);
funs=array([atleast_1d(res.fun)[0] for res in opt]); 
FUNS=arange(0.0,t,1)
af=ym-exp_w*std;

for i in range(t):
    xnxt=xnxtm[argmin(funs)].reshape(-1,1); FUNS[i]=funs[argmin(funs)];
    #xnxt=x[argmin(af)]
    ynxt=f((xnxt+1-2.00/3.65)*3.65/2.00);
    xinit=vstack([xinit,xnxt]); yinit=vstack([yinit,ynxt]);
    model.fit(xinit,yinit); #model.kernel_.get_params();
    x0=random.uniform(-1,1,100).reshape(-1,1);
    # Run in Parallel
    opt=Parallel(n_jobs=-1)(delayed(minimize)(LCB,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds) for start_point in x0)
    xnxtm=array([res.x for res in opt]);
    funs=array([atleast_1d(res.fun)[0] for res in opt]);
    ym,std=model.predict(x,return_std=True); std=std.reshape(-1,1);
    af=ym-exp_w*std; #pyp.figure("AF Runs"); pyp.plot(x,af)

ym,std=model.predict(x,return_std=True); std=std.reshape(-1,1);
x=(x+1-2.00/3.65)*3.65/2.00; xinit=(xinit+1-2.00/3.65)*3.65/2.00;
af=ym-exp_w*std; itr=arange(1,t+2,1);

fig,(ax1,ax2)=pyp.subplots(2,1,figsize=(10,14.25));#10,8.45
ax1.grid(color='lightgray',axis='both',alpha=1); ax1.set_axisbelow(True);
ax1.set_xlim((-1,2.65)); ax1.set_ylim((-3,13));
ax1.scatter(xinit,yinit,marker='o',color='white',edgecolor='black',zorder=3,s=200);
# ax1.scatter(xinit[-1],yinit[-1],marker='o',color='red',edgecolor='black',zorder=3,s=200);
#ax1.set_xticklabels([])
ax1.set_xlabel(r'$\xi$',fontsize=30);
ax1.set_ylabel(r'$f(\xi)$',fontsize=30);
for label in ax1.xaxis.get_ticklabels():
    label.set_fontsize(30)
for label in ax1.yaxis.get_ticklabels():
    label.set_fontsize(30)
ax1.plot(x,ym,color='red',linewidth=3);
ax1.plot(x,y,color='green',linewidth=3,linestyle='--'); 
ax1.fill_between(x.reshape(-1),(ym-2*std).reshape(-1),
                    (ym+2*std).reshape(-1),alpha=1,color='lightblue');
# ax1.legend([r'$Gaussian\ Process\ Model$', r'$True\ Model$',
#            r'$Sample\ Points$',r'$2\sigma\ Confidence\ Interval$'],fontsize=24);
## This half is to generate figure3
# ax2.grid(color='gray',axis='both',alpha=0.25); ax1.set_axisbelow(True);
# ax2.set_xlim((-1,2.65)); #ax2.set_ylim((-3,13));
# ax2.scatter(x[argmin(af)],af[argmin(af)],marker='x',color='red',zorder=3,s=200);
# ax2.set_xlabel(r'$x$',fontsize=24); #pyp.xticks(fontsize=24);
# ax2.set_ylabel(r'$AF(x)$',fontsize=24); #pyp.yticks(fontsize=24);
# for label in ax2.xaxis.get_ticklabels():
#     label.set_fontsize(24)
# for label in ax2.yaxis.get_ticklabels():
#     label.set_fontsize(24)
# ax2.plot(x,af,color='black',linewidth=3)
# pyp.savefig('AF_prog0.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);
## This half is to generate figure4
ax2.grid(color='lightgray',axis='both',alpha=1); ax1.set_axisbelow(True);
ax2.set_xlim((1,t+1)); ax2.set_ylim((-3,13));
ax2.scatter(itr,yinit,marker='o',color='white',edgecolor='black',zorder=3,s=200);
ax2.set_xlabel(r'Sample number',fontsize=30); #pyp.xticks(fontsize=24);
ax2.set_ylabel(r'$f(\xi)$',fontsize=30); #pyp.yticks(fontsize=24);
for label in ax2.xaxis.get_ticklabels():
    label.set_fontsize(30)
for label in ax2.yaxis.get_ticklabels():
    label.set_fontsize(30)
ax2.plot(itr,yinit,color='black',linewidth=3)
pyp.savefig('BO_no_mod.eps',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

#pyp.figure(); pyp.plot(x,af);
#pyp.figure();
#pyp.plot(itr[1:t+5],FUNS); pyp.scatter(itr[1:t+1],FUNS,marker='x');
#pyp.plot(itr,yinit); pyp.scatter(itr,yinit,marker='x');

#%% Reference model

xmod=array([0.727424,-0.948933,1.93663,1.25595,0.0238854])
#xmod=array([0.832294,0.727424,-0.948933,-0.107621,0.0546054,
#           1.65232,1.93663,1.25595,0.0238854,0.819567])
ymod=f(xmod);
fmod=lambda x,a3,a2,a1,a0: (a3*x**3+a2*x**2+a1*x+a0)
a3,a2,a1,a0=curve_fit(fmod,xmod,ymod)[0]

def fref(x):
    return a3*x**3+a2*x**2+a1*x+a0

x=(2.00/3.65)*x+(2.00/3.65)-1;
xinit=array([(2.00/3.65)*1.50571143+(2.00/3.65)-1]).reshape(-1,1);#array([[0.2484003],[0.49789958]]).reshape(-1,1);#random.uniform(-1,1,1).reshape(-1,1);
yinit=f((xinit+1-2.00/3.65)*3.65/2.00);
yrinit=fref((xinit+1-2.00/3.65)*3.65/2.00);
err=yinit-yrinit;
kerneler=gpr.kernels.Matern(1,(0.1,100),2.5);
modelerr=gpr.GaussianProcessRegressor(kerneler,alpha=1e-6,normalize_y=True,
                                     n_restarts_optimizer=10);
modelerr.fit(xinit,err);
x0=random.uniform(-1,1,100).reshape(-1,1); funs=0*x0; xnxtm=0*x0
# Run in Parallel
opt=Parallel(n_jobs=-1)(delayed(minimize)(LCBmod,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds) for start_point in x0)
xnxtm=array([res.x for res in opt]);
funs=array([atleast_1d(res.fun)[0] for res in opt]); 
FUNS=arange(0.0,t,1)
errm,std=modelerr.predict(x,return_std=True); std=std.reshape(-1,1);
ypred=fref((x+1-2.00/3.65)*3.65/2.00)+errm;
af=ypred-exp_w*std

for i in range(t):
    xnxt=xnxtm[argmin(funs)].reshape(-1,1); FUNS[i]=funs[argmin(funs)];
    #xnxt=x[argmin(af)].reshape(-1,1)
    ynxt=f((xnxt+1-2.00/3.65)*3.65/2.00);
    yrnxt=fref((xnxt+1-2.00/3.65)*3.65/2.00);
    errnxt=ynxt-yrnxt
    xinit=vstack([xinit,xnxt]); yinit=vstack([yinit,ynxt]); err=vstack([err,errnxt]);    
    modelerr.fit(xinit,err); #model.kernel_.get_params();
    x0=random.uniform(-1,1,100).reshape(-1,1);
    # Run in Parallel
    opt=Parallel(n_jobs=-1)(delayed(minimize)(LCBmod,x0=start_point,method='L-BFGS-B',
                                          bounds=bnds) for start_point in x0)
    xnxtm=array([res.x for res in opt]);
    funs=array([atleast_1d(res.fun)[0] for res in opt]);
    errm,std=modelerr.predict(x,return_std=True); std=std.reshape(-1,1);
    ypred=fref((x+1-2.00/3.65)*3.65/2.00)+errm;
    af=ypred-exp_w*std

errm,std=modelerr.predict(x,return_std=True); std=std.reshape(-1,1);
x=(x+1-2.00/3.65)*3.65/2.00; xinit=(xinit+1-2.00/3.65)*3.65/2.00;
ypred=fref(x)+errm
af=ypred-exp_w*std; itr=arange(1,yinit.shape[0]+1,1);    

fig,(ax1,ax2)=pyp.subplots(2,1,figsize=(10,14.25));
ax1.grid(color='lightgray',axis='both',alpha=1); ax1.set_axisbelow(True);
ax1.set_xlim((-1,2.65)); ax1.set_ylim((-25,13));
ax1.set_xlabel(r'$\xi$',fontsize=30);
ax1.set_ylabel(r'$g(\xi)$',fontsize=30);
for label in ax1.xaxis.get_ticklabels():
    label.set_fontsize(30)
for label in ax1.yaxis.get_ticklabels():
    label.set_fontsize(30)
ax1.plot(x,fref(x),color='blue',linewidth=3);
ax2.grid(color='lightgray',axis='both',alpha=1); ax1.set_axisbelow(True);
ax2.set_xlim((-1,2.65)); ax2.set_ylim((-3,13));
ax2.set_xlabel(r'$\xi$',fontsize=27); #pyp.xticks(fontsize=24);
ax2.set_ylabel(r'$\epsilon(\xi)$',fontsize=27); #pyp.yticks(fontsize=24);
for label in ax2.xaxis.get_ticklabels():
    label.set_fontsize(30)
for label in ax2.yaxis.get_ticklabels():
    label.set_fontsize(30)
ax2.plot(x,f(x)-fref(x),color='black',linewidth=3)
pyp.savefig('Ref_and_res_mod.eps',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);

fig,(ax1,ax2)=pyp.subplots(2,1,figsize=(10,14.25)); #12.5
ax1.grid(color='lightgray',axis='both',alpha=1); ax1.set_axisbelow(True);
ax1.set_xlim((-1,2.65)); ax1.set_ylim((-3,13));
ax1.scatter(xinit,yinit,marker='o',color='white',edgecolor='black',zorder=3,s=200);
#ax1.scatter(xinit[-1],yinit[-1],marker='o',color='red',edgecolor='black',zorder=3,s=200);
#ax1.set_xticklabels([])
ax1.set_xlabel(r'$\xi$',fontsize=30);
ax1.set_ylabel(r'$f(\xi)$',fontsize=30);
for label in ax1.xaxis.get_ticklabels():
    label.set_fontsize(30)
for label in ax1.yaxis.get_ticklabels():
    label.set_fontsize(30)
ax1.plot(x,ypred,color='red',linewidth=3);
ax1.plot(x,y,color='green',linewidth=3,linestyle='--'); 
ax1.fill_between(x.reshape(-1),(ypred-2*std).reshape(-1),
                   (ypred+2*std).reshape(-1),alpha=1,color='lightblue');
ax2.grid(color='lightgray',axis='both',alpha=1); ax1.set_axisbelow(True);
ax2.set_xlim((1,t+1)); ax2.set_ylim((-3,13));
ax2.scatter(itr,yinit,marker='o',color='white',edgecolor='black',zorder=3,s=200);
ax2.set_xlabel(r'Sample number',fontsize=30); #pyp.xticks(fontsize=24);
ax2.set_ylabel(r'$f(\xi)$',fontsize=30); #pyp.yticks(fontsize=24);
for label in ax2.xaxis.get_ticklabels():
    label.set_fontsize(30)
for label in ax2.yaxis.get_ticklabels():
    label.set_fontsize(30)
ax2.plot(itr,yinit,color='black',linewidth=3)
pyp.savefig('BO_mod.eps',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);


## This half is to generate figure3
fig,(ax1,ax2,ax3,ax4)=pyp.subplots(4,1,figsize=(10.5,18));
ax1.grid(color='gray',axis='both',alpha=0.25); ax1.set_axisbelow(True);
ax1.set_xlim((-1,2.65)); ax1.set_ylim((-3,13));
ax1.scatter(xinit,yinit,marker='o',color='white',edgecolor='black',zorder=3,s=200);
ax1.scatter(xinit[-1],yinit[-1],marker='o',color='red',edgecolor='black',zorder=3,s=200);
ax1.set_xticklabels([])
#ax1.set_xlabel(r'$x$',fontsize=24);
ax1.set_ylabel(r'$f(x)$',fontsize=24);
for label in ax1.xaxis.get_ticklabels():
    label.set_fontsize(24)
for label in ax1.yaxis.get_ticklabels():
    label.set_fontsize(24)
ax1.plot(x,y,color='k',linewidth=3);
ax2.grid(color='gray',axis='both',alpha=0.25); ax1.set_axisbelow(True);
ax2.set_xlim((-1,2.65)); ax2.set_ylim((-25,13));
ax2.scatter(xinit,yinit-err,marker='o',color='white',edgecolor='black',zorder=3,s=200);
ax2.scatter(xinit[-1],yinit[-1]-err[-1],marker='o',color='red',edgecolor='black',zorder=3,s=200);
ax2.set_xticklabels([])
#ax2.set_xlabel(r'$x$',fontsize=24);
ax2.set_ylabel(r'$g(x)$',fontsize=24);
for label in ax2.xaxis.get_ticklabels():
    label.set_fontsize(24)
for label in ax2.yaxis.get_ticklabels():
    label.set_fontsize(24)
ax2.plot(x,fref(x),color='k',linewidth=3);
ax3.grid(color='gray',axis='both',alpha=0.25); ax1.set_axisbelow(True);
ax3.set_xlim((-1,2.65)); #ax3.set_ylim((-3,13));
ax3.scatter(xinit,err,marker='o',color='white',edgecolor='black',zorder=3,s=200);
ax3.scatter(xinit[-1],err[-1],marker='o',color='red',edgecolor='black',zorder=3,s=200);
ax3.set_xticklabels([])
#ax3.set_xlabel(r'$x$',fontsize=24);
ax3.set_ylabel(r'$\epsilon(x)$',fontsize=24);
for label in ax3.xaxis.get_ticklabels():
    label.set_fontsize(24)
for label in ax3.yaxis.get_ticklabels():
    label.set_fontsize(24)
ax3.plot(x,errm,color='red',linewidth=3);
ax3.plot(x,y-fref(x),color='green',linewidth=3,linestyle='--'); 
ax3.fill_between(x.reshape(-1),(errm-2*std).reshape(-1),
                   (errm+2*std).reshape(-1),alpha=0.25);
ax4.grid(color='gray',axis='both',alpha=0.25); ax1.set_axisbelow(True);
ax4.set_xlim((-1,2.65)); #ax4.set_ylim((-3,13));
ax4.scatter(x[argmin(af)],af[argmin(af)],marker='x',color='red',zorder=3,s=200);
ax4.set_xlabel(r'$x$',fontsize=24); #pyp.xticks(fontsize=24);
ax4.set_ylabel(r'$AF(x)$',fontsize=24); #pyp.yticks(fontsize=24);
for label in ax4.xaxis.get_ticklabels():
    label.set_fontsize(24)
for label in ax4.yaxis.get_ticklabels():
    label.set_fontsize(24)
ax4.plot(x,af,color='black',linewidth=3)
#pyp.savefig('AF_prog_ref2.jpg',dpi=300,edgecolor='white',bbox_inches='tight',pad_inches=0.1);



#%% pyGPGO
kernel=matern52(l=1)
gp=GaussianProcess(kernel)
acq=Acquisition(mode='UCB',beta=2.6)
param=OrderedDict()
param['x']=('cont',[0, 1])
# param['demandRates'] = ('cont',[-2, 2])
# param['frReserveCap'] = ('cont',[-2, 2])

gpgo=GPGO(gp,acq,f2,param,n_jobs=-1)
gpgo._firstRun(n_eval=1) # 3, 10

for i in range(t):
    gpgo._optimizeAcq(method='L-BFGS-B', n_start=2)
    print('k=',i,', x_opt: ',gpgo.best,'y_opt', f2(gpgo.best[0]),'\n')
    gpgo.updateGP()
x=arange(0,1.001,0.001).reshape(-1,1); y2=-f2(x)
mean,cov=gpgo.GP.predict(x,return_std=True); #cov=cov**0.5
pyp.figure(); pyp.plot(x,y2); pyp.plot(x,-mean)
pyp.fill_between(x.reshape(-1),(-mean-2*cov).reshape(-1),
                   (-mean+2*cov).reshape(-1),alpha=0.1);
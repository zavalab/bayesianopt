# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:33:01 2020

@author: leonardo
"""

from numpy import array,arange,random,hstack,vstack,round,intersect1d,zeros
from numpy import delete,loadtxt,argmin
import GPy
from GPyOpt.methods import BayesianOptimization
from matplotlib import pyplot as pyp
import time

#pyp.close('all')
Traw=loadtxt('Rxtr_Temp_Data.txt').reshape(-1,1);
Ctraw=loadtxt('Rxtr_Cost_Data.txt').reshape(-1,1);

trainset_size=10; noiset=0.001#*random.normal(loc=0,scale=1);

def train(x,noise=noiset):
    v=x[:,0]; l=x[:,1]; #b=x[:,2];
    kernel=GPy.kern.Matern52(1,variance=v,lengthscale=l)
#    kernel=GPy.kern.RBF(1,variance=v,lengthscale=l)
#    kernel=GPy.kern.PeriodicMatern52(1,variance=v,lengthscale=l)
#    kernel1=GPy.kern.Poly(1,variance=v,scale=l,order=4)
#    kernel=GPy.kern.MLP(1,variance=v,weight_variance=l,bias_variance=b)
#    kernel2=GPy.kern.Bias(1,v)
#    kernel=kernel1+kernel2
#    kernel2=GPy.kern.Linear(1,variances=v)
#    kernel=kernel1+kernel2
    model=GPy.models.GPRegression(Ttest,Cttest,kernel)
    model.Gaussian_noise.variance=noise;
    model.Gaussian_noise.variance.fix()
#    model.optimize();
    Ctm,std=model.predict(Tm); std=std**0.5;
    a,b,c=intersect1d(Ttest,Tm,return_indices=True)
    Ctmod=Ctm[c]; Cthat=Cttest[b]; res=sum((100*(Ctmod-Cthat)/Cthat)**2);
    return res

Traw=round(Traw,4); Ctraw=round(Ctraw,4)
Dat=hstack([Traw,Ctraw]); #Dat=Dat[Dat[:,0].argsort()];
sets=int(round(Dat.shape[0]/trainset_size)); 
idx=zeros((sets,2),dtype=int); RES=[]; V=[]; L=[];
Tm=arange(0.373,0.97301,0.0001).reshape(-1,1); Tm=round(Tm,4)

for i in range(sets):
    idx[i,0]=(i*trainset_size)
    idx[i,1]=((i+1)*trainset_size)-1
    if i==sets-1:
        idx[i,1]=Dat.shape[0]-1
mu_str=[]

#%% Training
for i in range(sets):
    tidx=arange(idx[i,0],idx[i,1]+1,1)
    Datest=Dat[tidx,:]
    Datval=delete(Dat,tidx,0) #delete tidx rows
    Ttest=Datest[:,0].reshape(-1,1); Cttest=Datest[:,1].reshape(-1,1);
    Tval=Datval[:,0].reshape(-1,1); Ctval=Datval[:,1].reshape(-1,1);


#pyp.figure(); pyp.plot(Tm,Ctm);
#pyp.scatter(Ttest,Cttest,color='k',marker='x')
#pyp.fill_between(Tm.reshape(-1),(Ctm-2*std).reshape(-1),(Ctm+2*std).reshape(-1)
#                 ,alpha=0.1);
#pyp.xlim((0.373,0.973)); pyp.ylim((-4,-0.5));
    domain1=[{'name':'v','type':'continuous','domain':(25,50)},
             {'name':'l','type':'continuous','domain':(0.2,1)}]
    results=BayesianOptimization(f=train,domain=domain1,
                                 initial_design_numdata=10,acqusition_type='LCB')

    t1=time.perf_counter();
    results.run_optimization(max_iter=15,max_time=1800,eps=1e-8,verbosity=False)
#    results.plot_convergence()
#    results.plot_acquisition()
    ins=results.get_evaluations()[0]
    outs=results.get_evaluations()[1]
    evals=hstack([ins,outs])
#    print("The minimum value obtained was %f at v=%f, l=%f"
#          %(results.fx_opt,results.x_opt[0],results.x_opt[1]))#,results.x_opt[2]))
    t2=time.perf_counter(); #print(t2-t1);

    v=results.x_opt[0]; l=results.x_opt[1]; V.append(v); L.append(l);
    kernel=GPy.kern.Matern52(1,variance=v,lengthscale=l)
    model=GPy.models.GPRegression(Ttest,Cttest,kernel)
    model.Gaussian_noise.variance=noiset
    model.Gaussian_noise.variance.fix()
    #model.optimize();
#    Ctm,std=model.predict(Tm); std=std**0.5;
#    pyp.figure(); pyp.plot(Tm,Ctm);
#    pyp.scatter(Ttest,Cttest,color='k',marker='x')
#    pyp.fill_between(Tm.reshape(-1),(Ctm-2*std).reshape(-1),(Ctm+2*std).reshape(-1)
#                     ,alpha=0.1);
#    pyp.xlim((0.373,0.973)); pyp.ylim((-4,-0.5));

#%% Validation

    model=GPy.models.GPRegression(Tval,Ctval,kernel)
    model.Gaussian_noise.variance=noiset;
    model.Gaussian_noise.variance.fix()
    Ctm,std=model.predict(Tm); std=std**0.5;
    a,b,c=intersect1d(Tval,Tm,return_indices=True)
    Ctmod=Ctm[c]; Cthat=Ctval[b]; mu_str.append(min(Ctm))
#    pyp.figure(); pyp.plot(Tm,Ctm);
#    pyp.scatter(Tval,Ctval,color='k',marker='x')
#    pyp.fill_between(Tm.reshape(-1),(Ctm-2*std).reshape(-1),(Ctm+2*std).reshape(-1)
#                     ,alpha=0.1);
#    pyp.xlim((0.373,0.973)); pyp.ylim((-4,-0.5));
    res=sum((100*(Ctmod-Cthat)/Cthat)**2); RES.append(res)
#    print(res)

# for i in range(test_sets):
#     il=i*trainset_size; iu=(i+1)*trainset_size-1
#     test_Dat=Dat[r[il:iu+1],:]
#     Tval=test_Dat[:,0:test_Dat.shape[1]-1];
#     Ctval=test_Dat[:,test_Dat.shape[1]-1].reshape(-1,1)
#     model=GPy.models.GPRegression(Tval,Ctval,kernel)
#     model.Gaussian_noise.variance=noiset;
#     model.Gaussian_noise.variance.fix()
#     model.plot()
#     Ctm,std=model.predict(Tm); std=std**0.5;
#     a,b,c=intersect1d(Tval,Tm,return_indices=True)
#     Ctmod=Ctm[c]; Cthat=Ctval[b]; res=sum((100*(Ctmod-Cthat)/Cthat)**2);
#     print(res)

RES=array(RES); V=array(V).reshape(-1,1); L=array(L).reshape(-1,1);
RSLTS=hstack([RES,V,L]);
BST=argmin(RSLTS[:,0]); BST=RSLTS[BST];
print("The minimum value across sets obtained was %f at v=%f, l=%f"
      %(BST[0],BST[1],BST[2]))

mu_AVGstr=sum(mu_str)/len(mu_str)
from numpy import sin, pi, arange, array, random, argmax, asarray
from numpy import append, zeros, shape
from matplotlib import pyplot as pyp
from sklearn.gaussian_process import GaussianProcessRegressor as GP
import warnings
from scipy.stats import norm

#################### SET UP BOUNDS AND EVALUATE TEST f(x) #####################
# x0=0; xf=1; dt=1e-2;
# xset=arange(x0,xf+dt,dt).reshape(-1,1)
# yset=array(list(map(lambda s:s**2*sin(5*pi*s)**6,xset))).reshape(-1,1);

################ OBJECTVE FUNCTION WITH RANDOM NOISE~N(0,1) ##################
def objective(x,noise=0.01):
    noise=random.normal(loc=0,scale=noise)
    return x**2*sin(5*pi*x)**6+noise

##### EVALUATE NOISY f(x) & PLOT SCATTER ALONG REAL f(x) #####
# ynoise=array([objective(x) for x in xset]).reshape(-1,1)
# pyp.plot(xset,yset); pyp.scatter(xset,ynoise); pyp.show()
# idx=argmax(yset);
# print('Optimum is x=%.3f, y=%.3f' %(xset[idx],yset[idx]));

####################### CREATE SURROGATE MODEL ###############################
# Create a function that will ignore warnings
def surrogate(model,x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict(x, return_std=True)

# Plot real observations with surrogate functions
def plot(x,y,model):
    pyp.scatter(x,y)
    xsamples=asarray(arange(0,1,0.001)).reshape(-1,1)
    ysamples,std=surrogate(model,xsamples)
    pyp.plot(xsamples, ysamples); pyp.show()

# model=GP();
# # Generate random, noisy samples
# x=random.rand(100).reshape(-1,1)
# y=array([objective(x) for x in x]).reshape(-1,1)
# # fit and plot GP model with samples
# model.fit(x,y)
# plot(x,y,model);

########################### ACQUISTION FUNCTION ##############################
# Create acquisition function using prob. of improvement
def acquisition(x, xsamples, model):
    yhat,_=surrogate(model,x)
    best=max(yhat)
    mu, std=surrogate(model,xsamples)
    mu=mu[:,0]
    probs=norm.cdf((mu-best)/(std+1e-9))
    return probs

# Create a function to mininimize acquisition function
def opt_acquisition(x,y,model):
    xsamples=random.rand(100).reshape(-1,1)
    scores=acquisition(x,xsamples,model)
    idx=argmax(scores)
    return xsamples[idx]

########################## IMPROVEMENT ITERATIONS ############################
# test=100
# for i in range(test):
#     xnxt=opt_acquisition(x,y,model)
#     yhat=objective(xnxt)
#     x=append(x,xnxt).reshape(-1,1)
#     y=append(y,yhat).reshape(-1,1)
#     model.fit(x,y)
# plot(x,y,model)
# #r=shape(x)[0]; c=shape(x)[1];
# #D=zeros((r,c+1)); D[:,0:c]=x[:,0:c]; D[:,c]=y[:,0]
# #D=D[D[:,0].argsort()]; x=D[:,0:c]; y=D[:,c]; pyp.plot(x,y); pyp.show()
# idx=argmax(y);
# print('Optimum is x=%.3f, y=%.3f' %(x[idx],y[idx]));

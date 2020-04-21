#%%
# AR example
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ar_model import AR
from random import random
#%%
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = AR(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

#%%
data = [1, 3, 6, 25, 73, 222, 294]
#%%
def time_dependent_variable(t):
    if t<=1:
        return 2
    elif t<=2:
        return -2
    elif t<=3:
        return 3
    else :
        return -3
def test(t,y):
    c=time_dependent_variable(t)
    dydt=c
    return[dydt]
sol = solve_ivp(test,[0,10],[0],dense_output=True)
#%%
fig,ax = plt.subplots(figsize=(12,12))
plt.plot(sol.t,sol.y[0])
#plt.plot(np.arange(0,7),data,"k*:")
plt.grid("True")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

#plt.legend(["Susceptible","Infected","Removed","Original Data"])
# %%
def lotkavolterra(t, z, a, b, c, d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]
sol = solve_ivp(lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1),
                dense_output=True)
#%%
beta,gamma = [0.01,0.1]

def SIR(t,y):
    S = y[0]
    I = y[1]
    R = y[2]
    return([-beta*S*I, beta*S*I-gamma*I, gamma*I])

sol = solve_ivp(SIR,[0,0.01],[762,1,0],t_eval=np.arange(0,0.01,0.0001))

fig = plt.figure(figsize=(12,4))
plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1])
plt.plot(sol.t,sol.y[2])
plt.plot(np.arange(0,7),data,"k*:")
plt.grid("True")
plt.legend(["Susceptible","Infected","Removed","Original Data"])

#%%
def sumsq(p):
    beta, gamma = p
    def SIR(t,y):
	    S = y[0]
	    I = y[1]
	    R = y[2]
	    return([-beta*S*I, beta*S*I-gamma*I, gamma*I])
    sol = solve_ivp(SIR,[0,14],[762,1,0],t_eval=np.arange(0,14,1))
    return(sum((sol.y[1][::5]-data)**2))

from scipy.optimize import minimize

msol = minimize(sumsq,[0.001,1],method='Nelder-Mead')
msol.x
#%%
beta,gamma = msol.x
def SIR(t,y):
    S = y[0]
    I = y[1]
    R = y[2]
    return([-beta*S*I, beta*S*I-gamma*I, gamma*I])
sol = solve_ivp(SIR,[0,14],[762,1,0],t_eval=np.arange(0,14.2,0.2))
fig = plt.figure(figsize=(10,4))
plt.plot(sol.t,sol.y[0],"b-")
plt.plot(sol.t,sol.y[1],"r-")
plt.plot(sol.t,sol.y[2],"g-")
plt.plot(np.arange(0,15),data,"k*:")
plt.legend(["Susceptible","Infected","Removed","Original Data"])
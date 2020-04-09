#%%
## SIR 모형은 3개의 비선형 ODE로 구성
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
#%%
data = [1, 3, 6, 25, 73, 222, 294]


# %%
beta,gamma = [0.01,0.1]

def SIR(t,y):
    S = y[0]
    I = y[1]
    R = y[2]
    return([-beta*S*I, beta*S*I-gamma*I, gamma*I])

sol = solve_ivp(SIR,[0,7],[762,1,0],t_eval=np.arange(0,7.2,0.2))

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
    sol = solve_ivp(SIR,[0,14],[762,1,0],t_eval=np.arange(0,14.2,0.2))
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
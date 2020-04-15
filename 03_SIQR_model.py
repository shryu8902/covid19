#%%
def SIQR(t,y):
    S = y[0] 
    I = y[1]
    Q = y[2]
    R = y[3]
    T = y[4]
    return([-beta*I*S, beta*I*S-gamma*I-((mu*T)**alpha)*(I**(1-alpha)), ((mu*T)**alpha)*(I**(1-alpha))-theta*Q, gamma*I+theta*Q, 0])


beta, gamma, mu, alpha, theta = [0.01,0.001,0.01,0.001,0.01]
sol = solve_ivp(SIQR,[0,14],[10000,1,0,0,10],t_eval=np.arange(0,14,0.1))

fig = plt.figure(figsize=(12,4))
plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1])
plt.plot(sol.t,sol.y[2])
plt.plot(sol.t,sol.y[3])
plt.plot(sol.t,sol.y[4])
plt.plot(np.arange(0,14,0.1),data,"k*:")
plt.grid("True")


# %%

#%%
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
#%%
def dy_dx(y,x):
    return x-y
#%%
xs = np.linspace(0,5,100)
y0=1.0
ys = odeint(dy_dx, y0, xs)
ys = np.array(ys).flatten()
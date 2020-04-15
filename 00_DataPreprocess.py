#%%
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import scipy
#%%


#%%
DATA = pd.read_csv('./coronavirusdataset/Time.csv')\
# Columns : date, time, test, negative, confirmed, released, deceased 

DATA['confirmed_diff'] = DATA.confirmed.diff()   
#%%
def fn(x, a, b, c):
    return a+b*x[0]+c*x[1]

# y(x0,x1) data:
#  x0 = 0 1 2
# ___________
# x1=0 |0 1 2
# x1=1 |1 2 3
# x1=2 |2 3 4    
x= np.array([[0,1,2,0,1,2,0,1,2],[0,0,0,1,1,1,2,2,2]])
#y= np.array(np.log(DATA.confirmed_diff))
y=np.array([0,1,2,1,2,3,2,3,4])
popt, pcov = curve_fit(fn, x, y)
print(popt)

x_data = np.linspace(0,4,50)
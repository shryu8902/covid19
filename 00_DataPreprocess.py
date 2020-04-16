#%%
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import scipy
import matplotlib
from matplotlib import pyplot as plt
#%%
import pickle
with open('US_pop.pkl', 'rb') as f:
    US_pop = pickle.load(f)
#%%
DATA = pd.read_csv('./daily.csv') ## 미국 주별 데이터
DATA.date = pd.to_datetime(DATA.date, format = '%Y%m%d')
DATA=DATA.reindex(index=DATA.index[::-1])
DATA_US = pd.read_csv('./us-daily 0416.csv')
DATA_US.date = pd.to_datetime(DATA_US.date, format = '%Y%m%d')
DATA_state={}
DATA_state['US']=DATA_US.reindex(index=DATA_US.index[::-1])
for i in list(set(DATA.state))+['US']:
    DATA_state[i]=DATA[DATA.state==i].copy()
    DATA_state[i]=DATA_state[i].reset_index(drop=True).drop_duplicates(['date'])
    DATA_state[i]['C_']=abs(DATA_state[i].positiveIncrease)/US_pop[i]*100
    DATA_state[i]['T_']=abs(DATA_state[i].totalTestResultsIncrease)/US_pop[i]*100
    DATA_state[i]['CC_'] = DATA_state[i].positive/US_pop[i]*100
    DATA_state[i]['TT_'] = DATA_state[i].totalTestResults/US_pop[i]*100

#%%
for i in list(set(DATA.state)):
    fig, ax = plt.subplots()
    DATA_state[i].CC_.plot(title=str(i))
    DATA_state[i].TT_.plot.line(ax=ax)
    plt.savefig('./'+str(i)+'_cum.png')
    plt.close()
#%%
for i in list(set(DATA.state)):
    print(i)
#%%
DATA = pd.read_csv('./coronavirusdataset/Time.csv')\
# Columns : date, time, test, negative, confirmed, released, deceased 
DATA['country'] = 'US'
DATA['confirmed_diff'] = DATA.confirmed.diff()   
DATA

for i in list(set(DATA.state)):
    print(i,DATA[DATA.state==i].positiveIncrease.sum())

for i in list(US_pop.keys()):
    if not i in set(DATA.state):
        print(i)
#%% save US population data
import pickle
f = open("US_pop.pkl","wb")
pickle.dump(US_pop,f)
f.close()
#%%
US_pop = {'US': 327e6,
    'AS':55216,#America Samoa
    'GU':168461,#Guam
    'MP':57487,#Mariana islnad
    'VI':104456,#Virgin Islands
    'CA':39937489,
    'TX':29472295,
    'FL':21992985,
    'NY':19440469,
    'PA':12820878,
    'IL':12659682,
    'OH':11747694,
    'GA':10736059,
    'NC':10611862,
    'MI':10045029,
    'NJ':8936574,
    'VA':8626207,
    'WA':7797095,
    'AZ':7378494,
    'MA':6976597,
    'TN':6897576,
    'IN':6745354,
    'MO':6169270,
    'MD':6083116,
    'WI':5851754,
    'CO':5845526,
    'MN':5700671,
    'SC':5210095,
    'AL':4908621,
    'LA':4645184,
    'KY':4499692,
    'OR':4301089,
    'OK':3954821,
    'CT':3563077,
    'UT':3282115,
    'IA':3179849,
    'NV':3139658,
    'AR':3038999,
    'PR':3032165,
    'MS':2989260,
    'KS':2910357,
    'NM':2096640,
    'NE':1952570,
    'ID':1826156,
    'WV':1778070,
    'HI':1412687,
    'NH':1371246,
    'ME':1345790,
    'MT':1086759,
    'RI':1056161,
    'DE':982895,
    'SD':903027,
    'ND':761723,
    'AK':734002,
    'DC':720687,
    'VT':628061,
    'WY':567025}

Population = {
    'China-': 1386e6,
    'US-': 327e6,
    'EU-': 512e6 + (10+9+5+3+3+2+0.5+0.4)*1e6,


    
    'EU-Vatican City':801,
    'EU-United Kingdom':67886011,
    'EU-Ukraine':43733762,
    'EU-Turkey':84339067,
    'EU-Switzerland':8654622,
    'EU-Sweden':10099265,
    'EU-Spain':46754778,
    'EU-Slovenia':2078938,
    'EU-Slovakia':5459642,
    'EU-Serbia':8737371,
    'EU-San Marino':33931,
    'EU-Russia':145934462,
    'EU-Romania':19237691,
    'EU-Portugal':10196709,
    'EU-Poland':37846611,
    'EU-Norway':5421241,
    'EU-Netherlands':17134872,
    'EU-Montenegro':628066,
    'EU-Monaco':39242,
    'EU-Moldova':4033963,
    'EU-Malta':441543,
    'EU-Luxembourg':625978,
    'EU-Lithuania':2722289,
    'EU-Liechtenstein':38128,
    'EU-Latvia':1886198,
    'EU-Kazakhstan':18776707,
    'EU-Italy':60461826,
    'EU-Ireland':4937786,
    'EU-Iceland':341243,
    'EU-Hungary':9660351,
    'EU-Greece':10423054,
    'EU-Germany':83783942,
    'EU-Georgia':3989167,
    'EU-France':65273511,
    'EU-Finland':5540720,
    'EU-Faroe Islands':48863,
    'EU-Estonia':1326535,
    'EU-Denmark':5792202,
    'EU-Czech Republic':10708981,
    'EU-Cyprus':1207359,
    'EU-Croatia':4105267,
    'EU-Bulgaria':6948445,
    'EU-Bosnia and Herzegovina':3280819,
    'EU-Belgium':11589623,
    'EU-Belarus':9449323,
    'EU-Azerbaijan':10139177,
    'EU-Austria':9006398,
    'EU-Armenia':2963243,
    'EU-Andorra':77265,
    'EU-Albania':2877797,
    
    'China-Hubei':59e6, #wuhan=11, hubei=59 59e6
    'Singapore-': 5.6e6, #not enough data to calibrate
    'Japan-': 127e6
}

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
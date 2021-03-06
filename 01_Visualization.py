#%%
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
import scipy
import matplotlib
from matplotlib import pyplot as plt
import time

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%% 미국 주별 인구데이터 불러오기
import pickle
with open('US_pop.pkl', 'rb') as f:
    US_pop = pickle.load(f)

#%% 데이터 불러오기
# C_ : 일별 신규 코로나 확진자
# T_ : 일별 신규 검사자
# CC_: 누적 확진자
# TT_: 누적 검사자
# positive 
# totalTestResults
# positiveIncrease
# totalTestResultsIncrease

DATA = pd.read_csv('./daily.csv') ## 미국 주별 데이터
DATA.date = pd.to_datetime(DATA.date, format = '%Y%m%d')
DATA=DATA.reindex(index=DATA.index[::-1])
DATA_US = pd.read_csv('./us-daily 0416.csv') ## 미국 합산데이터
DATA_US.date = pd.to_datetime(DATA_US.date, format = '%Y%m%d')

DATA_US['state']='US'
DATA_US=DATA_US.reindex(index=DATA_US.index[::-1])
main_columns = ['date','state','positive','positiveIncrease','totalTestResults','totalTestResultsIncrease']

CORE_DATA = pd.concat([DATA_US[main_columns],DATA[main_columns]]).fillna(0).reset_index(drop=True)
CORE_DATA['Pop']=CORE_DATA.state.apply(lambda x: US_pop[x])

CORE_DATA['C_'] = abs(CORE_DATA.positiveIncrease.values)/CORE_DATA.Pop.values
CORE_DATA['T_'] = abs(CORE_DATA.totalTestResultsIncrease.values)/CORE_DATA.Pop.values
CORE_DATA['CC_'] = CORE_DATA.positive.values/CORE_DATA.Pop.values*100
CORE_DATA['TT_'] = CORE_DATA.totalTestResults.values/CORE_DATA.Pop.values*100

C_scaler = MinMaxScaler()

T_scaler = MinMaxScaler()
CC_scaler = MinMaxScaler()
TT_scaler = MinMaxScaler()
CORE_DATA['CC_s'] = CC_scaler.fit_transform(CORE_DATA[['CC_']])
CORE_DATA['TT_s'] = TT_scaler.fit_transform(CORE_DATA[['TT_']])
CORE_DATA['C_s'] = C_scaler.fit_transform(CORE_DATA[['C_']])
CORE_DATA['T_s'] = T_scaler.fit_transform(CORE_DATA[['T_']])
CORE_DATA['CoverT'] = pd.Series(CORE_DATA.C_.values/CORE_DATA.T_.values).fillna(0)
state_list = list(set(DATA.state))
state_list.sort()
state_list=state_list+['US']
#%%
CORE_DATA.pivot('date','state','CC_').plot(legend=False, title='Cumulative_Confirmed')
plt.savefig('./CC.png',bbox_inches = 'tight')
CORE_DATA.pivot('date','state','TT_').plot(legend=False, title='Cumulative_Tests')
plt.savefig('./TT.png',bbox_inches = 'tight')
CORE_DATA.pivot('date','state','C_').plot(legend=False, title='Daily_Confirmed')
plt.savefig('./C.png',bbox_inches = 'tight')
CORE_DATA.pivot('date','state','T_').plot(legend=False, title='Daily_Tests')
plt.savefig('./T.png',bbox_inches = 'tight')

#%%
for STATE in state_list:
    fig, ax = plt.subplots()
    CORE_DATA[CORE_DATA.state==STATE].plot(x='date',y='CoverT',ax=ax)
    plt.savefig('./CoverT/'+STATE+'.png',bbox_inches = 'tight')
    plt.close()
#%%
CORE_DATA.pivot('date','state','CoverT').plot(legend=False,ylim=[0,2])
#%%

CORE_DATA[CORE_DATA.state=='AK'][['C_','T_']].plot()

# %%

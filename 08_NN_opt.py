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
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda,LSTM, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

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
temp = C_scaler.fit_transform(CORE_DATA[['CC_','TT_']])
CORE_DATA['CC_s'] = C_scaler.fit_transform(CORE_DATA[['CC_']])
CORE_DATA['TT_s'] = T_scaler.fit_transform(CORE_DATA[['TT_']])
#%%
state_list = list(set(DATA.state))
state_list.sort()
state_list=state_list+['US']
#%%

test = np.arange(40).reshape(-1,10,2)
#%%
def non_adder(x):
    y=x[0]+x[1]
    return(y)
test = Lambda(non_adder)

K.clear_session()
model = Sequential()
model.add(InputLayer(input_shape=(time_len,2)))
model.add(Lambda(non_adder, output_shape=[1]))
model.compile(loss='mean_squared_error')

model.predict(test )





# %%

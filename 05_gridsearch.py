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
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

def normal_augmentation(input, output, seed, std=0.2, times=5):
    aug_input = np.vstack([(input*std*np.random.normal(size=input.shape)+input) for i in range(times)])
    aug_input = np.vstack([input, aug_input])
    aug_output = np.vstack([output for i in range(times+1)])
    return(aug_input, aug_output)

os.chdir(os.path.dirname(os.path.realpath(__file__)))

#%% 미국 주별 인구데이터 불러오기
import pickle
with open('US_pop.pkl', 'rb') as f:
    US_pop = pickle.load(f)
# with open('GridSearch.pkl', 'rb') as f:
#     grid_result = pickle.load(f)
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

state_list = list(set(DATA.state))
state_list.sort()
state_list=state_list+['US']

#%% 주별 입/출력 데이터 생성 (과거 10일로 다음날 예측)
X_state={}
Y_C_state={}
Y_T_state={}

time_len = 10 #lookback window

for STATE in state_list:
    df = CORE_DATA[CORE_DATA.state==STATE]
    X_state[STATE]=[]
    Y_C_state[STATE]=[]
    Y_T_state[STATE]=[]
    for i in range(len(df) - time_len):
        X_state[STATE].append(df[['CC_s','TT_s']].iloc[i:(i+time_len)].values)
#        Y_state[STATE].append(df[['CC_']].iloc[(i+1):(i+1+time_len)].values)
        Y_C_state[STATE].append(df[['CC_s']].iloc[(i+time_len)].values) 
        Y_T_state[STATE].append(df[['TT_s']].iloc[(i+time_len)].values) 
    X_state[STATE] = np.array(X_state[STATE])
    Y_C_state[STATE] = np.array(Y_C_state[STATE])
    Y_T_state[STATE] = np.array(Y_T_state[STATE])
#%% 최근 5일을 validation으로 사용
X_all = np.concatenate([X_state[STATE] for STATE in state_list])
Y_C_all = np.concatenate([Y_C_state[STATE] for STATE in state_list])
Y_T_all = np.concatenate([X_state[STATE] for STATE in state_list])
val_index = [[-1]*(len(X_state[STATE])-5)+[0]*5 for STATE in state_list]
val_index = np.array([x for indexes in val_index for x in indexes]) 

X_all_aug, Y_C_all_aug = normal_augmentation(X_all, Y_C_all,seed = 1, times = 20)
val_index_aug = np.tile(val_index,21)



#%%
def create_model(optimizer='RMSprop', init='glorot_uniform', lr=0.001, dropout=0.2, epoch = 10, time_len = 10,n_hidden=10,n_layer =1,loss='mean_squared_error'):
    K.clear_session()
    model = Sequential()
    model.add(InputLayer(input_shape=(time_len,2)))
#    model.add(BatchNormalization())
    for i in range(n_layer):
        if i==n_layer-1:
            model.add(LSTM(n_hidden, dropout=dropout, kernel_initializer = init, return_sequences=False))
        else:
            model.add(LSTM(n_hidden, dropout=dropout, kernel_initializer = init, return_sequences=True))
    model.add(Dense(1))
    model.add(Activation("relu"))
    if optimizer=='RMSprop':
        opt = optimizers.RMSprop(lr=lr)
    elif optimizer=='Adam':
        opt = optimizers.Adam(lr=lr)
    else:
        opt = optimizers.SGD(lr=lr)
    model.compile(loss=loss,optimizer=opt)
    return model
#%%
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

model_c = KerasRegressor(build_fn=create_model,shuffle=True,verbose=0)
param_grid = {
    'optimizer':['RMSprop','Adam','SGD'],
    'loss':['mean_squared_error','mean_squared_logarithmic_error'],
    'n_layer':[1,2],
    'dropout':[0.1, 0.3, 0.5],
    'epochs':[10,20,40,60],
    'n_hidden':[10,50,100,200],
    'batch_size':[256,512,1024],
    'lr':[0.01,0.001,0.0001],}
# param_grid = {
#    'optimizer':['Adam'],
   
#    'dropout':[0.1],
#    'epochs':[20, 10],
#    'n_hidden':[10],
#    'batch_size':[31017],}

#%%
from sklearn.model_selection import GridSearchCV, PredefinedSplit
ps = PredefinedSplit(test_fold=val_index_aug)
grid = GridSearchCV(estimator=model_c,cv=ps,param_grid=param_grid,scoring='neg_mean_absolute_error', n_jobs=1,return_train_score=True,verbose=2)
grid_result = grid.fit(X_all_aug,Y_C_all_aug)
#%%
import pickle
f = open("GridSearch2.pkl","wb")
pickle.dump([grid_result.best_params_, grid_result.cv_results_],f)
f.close()


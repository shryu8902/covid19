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
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer, Lambda, Concatenate, concatenate,Reshape, RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score

#%%
os.chdir(os.path.dirname(os.path.realpath(__file__)))
def normal_augmentation(input, output, seed, std=0.2, times=5):
    aug_input = np.vstack([(input*std*np.random.normal(size=input.shape)+input) for i in range(times)])
    aug_input = np.vstack([input, aug_input])
    aug_output = np.vstack([output for i in range(times+1)])
    return(aug_input, aug_output)
#%% 미국 주별 인구데이터 불러오기
import pickle
with open('US_pop.pkl', 'rb') as f:
    US_pop = pickle.load(f)
#%%
DATA = pd.read_csv('./daily_0508.csv') ## 미국 주별 데이터
DATA.date = pd.to_datetime(DATA.date, format = '%Y%m%d')
DATA=DATA.reindex(index=DATA.index[::-1])
DATA_US = pd.read_csv('./us-daily_0508.csv') ## 미국 합산데이터
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

CORE_DATA['lnC'] = np.log(CORE_DATA['C_']+1e-8)*100
CORE_DATA['lnT'] = np.log(CORE_DATA['T_']+1e-8)*100
C_scaler = MinMaxScaler()
T_scaler = MinMaxScaler()
CORE_DATA['CC_s'] = C_scaler.fit_transform(CORE_DATA[['CC_']])
CORE_DATA['TT_s'] = T_scaler.fit_transform(CORE_DATA[['TT_']])

state_list = list(set(DATA.state))
state_list.sort()
state_list=state_list+['US']
#%% 
X_lnC = []
X = []
Y_dlnC = []
Y_lnC = []
for STATE in state_list:
    df=CORE_DATA[CORE_DATA.state==STATE].copy()
    df['dlnC'] = df['lnC'].diff().shift(-1).fillna(0)
    df['dlnT'] = df['lnT'].diff().shift(-1).fillna(0)
    df['C/T'] = (df['C_']+1e-8) / (df['T_']+1e-8) 
    df['lnC_nxt'] = df['lnC'].shift(-1)

    if len(df)>60:
        X.append(df[['dlnT','C/T']].iloc[-61:-1].values)
        X_lnC.append(df[['lnC']].iloc[-61:-1].values)
        Y_dlnC.append(df[['dlnC']].iloc[-61:-1].values)
        Y_lnC.append(df[['lnC_nxt']].iloc[-61:-1].values)

X_lnC = np.array(X_lnC)
X = np.array(X)
Y_dlnC = np.array(Y_dlnC)
Y_lnC = np.array(Y_lnC)
#%% Augmentation
X_aug, Y_dlnC_aug = normal_augmentation(X,Y_dlnC,seed = 1, times = 20)
X_lnC_aug, Y_lnC_aug = normal_augmentation(X_lnC, Y_lnC, seed = 1, times = 20)

# %%

# optimizer='RMSprop', init='glorot_uniform', lr=0.001, dropout=0.2, epoch = 10, time_len = 10,n_hidden=10,n_layer =1,loss='mean_squared_error'
def CALC_dlnC(x):
    alpha = x[:,0]
    rho = x[:,1]
    A = x[:,2]
    B = x[:,3]
    dlnT = x[:,4]
    CT = x[:,5]
    return alpha*dlnT + A*K.pow(CT,rho)+B


main_input = Input(shape = (60,2),name = 'main')
lnC_input = Input(shape = (60,1), name ='lnC')

# params = LSTM(50,dropout=0.2,kernel_initializer = 'glorot_uniform', return_sequences = True)(main_input)
# params = TimeDistributed(Dense(4))(params)

params = LSTM(10,dropout=0.2,kernel_initializer = 'glorot_uniform')(main_input)
params2 = Dense(4, activation = 'tanh')(params)
params3 = RepeatVector(60)(params2)

merge = concatenate([params3, main_input],axis=-1)
dlnC_hat = TimeDistributed(Lambda(CALC_dlnC))(merge)
dlnC_hat2 = Reshape((60,1))(dlnC_hat)

model = Model(inputs = [main_input,lnC_input], outputs = [dlnC_hat2, params3])

model.compile(loss='mean_squared_error',optimizer='Adam',loss_weights=[1., 0.0])
#%%
d= np.repeat(Y_lnC, 4, axis=2) ## dummy

history = model.fit([X,X_lnC], [Y_dlnC,d], epochs = 1000, batch_size = 5, shuffle = True,verbose = 1)
history = model.fit([X[0].reshape(1,60,2),X_lnC[0].reshape(1,60,1)], [Y_dlnC[0].reshape(1,60,1),d[0].reshape(1,60,4)], epochs = 1000, batch_size = 1, shuffle = True,verbose = 1)

#%%
Y_C_hat, values = model.predict([X[50].reshape(1,60,2),X_lnC[50].reshape(1,60,1)])

Y_dlnC_hat, params_hat = model.predict([X, X_lnC])

r2_score(Y_dlnC[0,:,:].reshape(-1), Y_dlnC_hat[1,:,:].reshape(-1))
for i in range(1):
    print(r2_score(Y_dlnC[i,:,:].reshape(-1), Y_dlnC_hat[i,:,:].reshape(-1))
)


model.predict()



#%%
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')

loss_ax.legend(loc='upper left')
plt.show()



#%%
tf.keras.utils.plot_model(model)

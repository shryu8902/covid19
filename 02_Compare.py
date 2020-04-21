#%%
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import layers, Sequential
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from sklearn.model_selection import GridSearchCV, PredefinedSplit

#%%
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
with open('GridSearch2.pkl', 'rb') as f:
    grid_result = pickle.load(f)

#%%
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
CORE_DATA['CC_s'] = C_scaler.fit_transform(CORE_DATA[['CC_']])
CORE_DATA['TT_s'] = T_scaler.fit_transform(CORE_DATA[['TT_']])

state_list = list(set(DATA.state))
state_list.sort()
state_list=state_list+['US']
#%% 비교용 신규 데이터 불러오기
NewDATA = pd.read_csv('./daily_0419.csv')
NewDATA.date = pd.to_datetime(NewDATA.date, format = '%Y%m%d')
NewDATA=NewDATA[NewDATA.date>='2020-04-16'].copy()
NewDATA.reindex(index=NewDATA.index[::-1])
NewDATA_US = pd.read_csv('./us-daily_0419.csv') ## 미국 합산데이터
NewDATA_US.date = pd.to_datetime(NewDATA_US.date, format = '%Y%m%d')
NewDATA_US=NewDATA_US[NewDATA_US.date>='2020-04-16'].copy()
NewDATA_US['state']='US'
NewDATA_US.reindex(index=NewDATA_US.index[::-1])

NewCORE_DATA = pd.concat([NewDATA_US[main_columns],NewDATA[main_columns]]).fillna(0).reset_index(drop=True)
NewCORE_DATA['Pop']=NewCORE_DATA.state.apply(lambda x: US_pop[x])

NewCORE_DATA['C_'] = abs(NewCORE_DATA.positiveIncrease.values)/NewCORE_DATA.Pop.values
NewCORE_DATA['T_'] = abs(NewCORE_DATA.totalTestResultsIncrease.values)/NewCORE_DATA.Pop.values
NewCORE_DATA['CC_'] = NewCORE_DATA.positive.values/NewCORE_DATA.Pop.values*100
NewCORE_DATA['TT_'] = NewCORE_DATA.totalTestResults.values/NewCORE_DATA.Pop.values*100

NewCORE_DATA['CC_s'] = C_scaler.transform(NewCORE_DATA[['CC_']])
NewCORE_DATA['TT_s'] = T_scaler.transform(NewCORE_DATA[['TT_']])

#%%
#%%
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

#%%
X_all = np.concatenate([X_state[STATE] for STATE in state_list])
Y_C_all = np.concatenate([Y_C_state[STATE] for STATE in state_list])
Y_T_all = np.concatenate([Y_T_state[STATE] for STATE in state_list])
val_index = [[-1]*(len(X_state[STATE])-5)+[0]*5 for STATE in state_list]
val_index = np.array([x for indexes in val_index for x in indexes]) 

X_all_aug, Y_C_all_aug = normal_augmentation(X_all, Y_C_all,seed = 1, times = 20)
_, Y_T_all_aug = normal_augmentation(X_all, Y_T_all,seed = 1, times = 20)

val_index_aug = np.tile(val_index,21)

#%%
#%%
def create_model(optimizer='rmsprop', init='glorot_uniform',dropout=0.2, epoch = 10, time_len = 10,n_hidden=10,n_layer =1,loss='mean_squared_error'):
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
    model.compile(loss=loss,optimizer=optimizer)
    return model

#%%
{'batch_size': 1024,
 'dropout': 0.1,
 'epochs': 10,
 'loss': 'mean_squared_logarithmic_error',
 'n_hidden': 100,
 'n_layer': 1,
 'optimizer': 'RMSprop'}

model_c= create_model(dropout=0.1,n_hidden=100,n_layer=1,loss='mean_squared_logarithmic_error',optimizer='RMSprop')
model_t= create_model(dropout=0.1,n_hidden=100,n_layer=1,loss='mean_squared_logarithmic_error',optimizer='RMSprop')
#%%
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1,
                              patience = 5, min_lr = 1e-5, verbose = 1)
MODEL_CHECKPOINT_DIR = './{}'.format(time.strftime('%y%m%d_%H-%M', time.localtime(time.time())))
if not os.path.exists(MODEL_CHECKPOINT_DIR):
    os.mkdir(MODEL_CHECKPOINT_DIR)
CHECKPOINT_callback_C = ModelCheckpoint(filepath = MODEL_CHECKPOINT_DIR+'/best_m_C.hdf5', monitor = 'val_loss',
                                      verbose = 1, save_best_only = True, mode = 'min') 
CHECKPOINT_callback_T = ModelCheckpoint(filepath = MODEL_CHECKPOINT_DIR+'/best_m_T.hdf5', monitor = 'val_loss',
                                      verbose = 1, save_best_only = True, mode = 'min') 
#%%
#%% 모델 학습
history_c = model_c.fit(X_all_aug, Y_C_all_aug, epochs = 60, batch_size = 1024,
                    shuffle = True, validation_split = 0.2, verbose = 1,
                    callbacks = [CHECKPOINT_callback_C, reduce_lr] )
history_t = model_t.fit(X_all_aug, Y_T_all_aug, epochs = 60, batch_size = 1024,
                    shuffle = True, validation_split = 0.2, verbose = 1,
                    callbacks = [CHECKPOINT_callback_T, reduce_lr] )
#%%
model_c.load_weights(MODEL_CHECKPOINT_DIR+'/best_m_C.hdf5')     
model_t.load_weights(MODEL_CHECKPOINT_DIR+'/best_m_T.hdf5')

#%%
#%% forecast from future 30 days
test_df = pd.DataFrame(columns=['date','state','C_','T_','CC_','TT_','CC_s','TT_s','Pop'])
for STATE in state_list:
    test_state=CORE_DATA[CORE_DATA.state==STATE][['date','state','C_','T_','CC_','TT_','CC_s','TT_s','Pop']].iloc[-time_len:-1].copy()
    test_state=test_state.append(CORE_DATA[CORE_DATA.state==STATE][['date','state','C_','T_','CC_','TT_','CC_s','TT_s','Pop']].iloc[-1])
    test_state['predict']=False                     
    for j in range(30):
        ilocation = j+time_len
        X_input = test_state[['CC_s','TT_s']].iloc[j:j+time_len].values.reshape(1,time_len,2)
        Y_C_hat = model_c.predict(X_input)[0][0]
        Y_T_hat = model_t.predict(X_input)[0][0]
        test_state = test_state.append({'date':test_state.date.iloc[ilocation-1]+pd.DateOffset(1),'state':STATE,'CC_s':Y_C_hat,'TT_s':Y_T_hat,'Pop':US_pop[STATE],'predict':True},ignore_index=True)
    test_df=test_df.append(test_state)


#%%
for target in state_list:
    for target_y in ['CC_s','TT_s']:
        fig, ax = plt.subplots()
        test_df[(test_df.state== target)& (test_df.predict==True)].plot(x='date',y=target_y,ax=ax,title=target)
        CORE_DATA[CORE_DATA.state==target].plot(x='date',y=target_y,ax=ax)
        NewCORE_DATA[NewCORE_DATA.state==target].plot(x='date',y=target_y,ax=ax,marker='o',linewidth=2)
        L=ax.legend(['forecast','history','real'],loc='upper left')
        plt.savefig('./temp_long/'+target_y+'/'+target+'.png',bbox_inches = 'tight')
        plt.close()

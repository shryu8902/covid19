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

# DATA_state={}
# DATA_state['US']=DATA_US
state_list = list(set(DATA.state))
state_list.sort()
state_list=state_list+['US']
# for i in state_list:
#     if i !='US':
#         DATA_state[i]=DATA[DATA.state==i].copy()
#     DATA_state[i]=DATA_state[i].reset_index(drop=True).drop_duplicates(['date'])
#     DATA_state[i]['C_'] = abs(DATA_state[i].positiveIncrease.fillna(0))/US_pop[i]*100
#     DATA_state[i]['T_'] = abs(DATA_state[i].totalTestResultsIncrease.fillna(0))/US_pop[i]*100
#     DATA_state[i]['CC_'] = DATA_state[i].positive.fillna(0)/US_pop[i]*100
#     DATA_state[i]['TT_'] = DATA_state[i].totalTestResults.fillna(0)/US_pop[i]*100

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
from sklearn.model_selection import train_test_split
X_train_st={}
Y_C_train_st={}
Y_T_train_st={}

X_test_st={}
Y_C_test_st={}
Y_T_test_st={}
for STATE in state_list:
    X_train, X_test, Y_C_train, Y_C_test = train_test_split(X_state[STATE],Y_C_state[STATE], test_size=5, shuffle = False)
    _, __, Y_T_train, Y_T_test = train_test_split(X_state[STATE],Y_T_state[STATE], test_size=5, shuffle = False)
    X_train_st[STATE] = X_train
    X_test_st[STATE] = X_test
    Y_C_train_st[STATE] = Y_C_train
    Y_T_train_st[STATE] = Y_T_train
    Y_C_test_st[STATE] = Y_C_test
    Y_T_test_st[STATE] = Y_T_test

X_train = np.concatenate([X_train_st[STATE] for STATE in state_list])
Y_C_train = np.concatenate([Y_C_train_st[STATE] for STATE in state_list])
Y_T_train = np.concatenate([Y_T_train_st[STATE] for STATE in state_list])

X_test = np.concatenate([X_test_st[STATE] for STATE in state_list])
Y_C_test = np.concatenate([Y_C_test_st[STATE] for STATE in state_list])
Y_T_test = np.concatenate([Y_T_test_st[STATE] for STATE in state_list])

#%%
X_all = np.concatenate([X_state[STATE] for STATE in state_list])
Y_C_all = np.concatenate([Y_C_state[STATE] for STATE in state_list])
Y_T_all = np.concatenate([X_state[STATE] for STATE in state_list])
val_index = [[-1]*(len(X_state[STATE])-5)+[0]*5 for STATE in state_list]
val_index = np.array([x for indexes in val_index for x in indexes]) 

X_all_aug, Y_C_all_aug = normal_augmentation(X_all, Y_C_all,seed = 1, times = 20)
val_index_aug = np.tile(val_index,21)
#%%
scaler = StandardScaler()
train_df["ConfirmedCases_std"] = scaler.fit_transform(X_Train)


#%% Data augmentation
def normal_augmentation(input, output, seed, std=0.2, times=5):
    aug_input = np.vstack([(input*std*np.random.normal(size=input.shape)+input) for i in range(times)])
    aug_input = np.vstack([input, aug_input])
    aug_output = np.vstack([output for i in range(times+1)])
    return(aug_input, aug_output)
X_aug, Y_C_aug = normal_augmentation(X_train, Y_C_train, seed = 1, times = 20)
X_aug, Y_T_aug = normal_augmentation(X_train, Y_T_train, seed = 1, times = 20)

#%% Create model
batch_size = 100
n_hidden = 100
n_in = 2
model_c = Sequential()
model_c.add(InputLayer(input_shape=(time_len,2)))
model_c.add(BatchNormalization())
#model_c.add(LSTM(n_hidden, batch_input_shape =(None, time_len, 2),
model_c.add(LSTM(n_hidden, dropout=0.2, kernel_initializer = 'glorot_uniform',return_sequences=False))
#model_c.add(LSTM(n_hidden, dropout=0.2,kernel_initializer = 'glorot_uniform',return_sequences=True))
#model_c.add(TimeDistributed(Dense(1)))
model_c.add(Dense(1))
model_c.add(Activation("relu"))
ADAM = optimizers.Adam()
model_c.compile(loss='mean_squared_error',optimizer=ADAM)
model_c.summary()


#%%
model_t = Sequential()
model_t.add(InputLayer(input_shape=(time_len,2)))
model_t.add(BatchNormalization())
#model_t.add(LSTM(n_hidden, batch_input_shape =(None, time_len, 2),
model_t.add(LSTM(n_hidden, dropout=0.2, kernel_initializer = 'glorot_uniform',return_sequences=False))
#model_t.add(LSTM(n_hidden, dropout=0.2,kernel_initializer = 'glorot_uniform',return_sequences=True))
#model_t.add(TimeDistributed(Dense(1)))
model_t.add(Dense(1))
model_t.add(Activation("relu"))
ADAM = optimizers.Adam()
model_t.compile(loss='mean_squared_error',optimizer=ADAM)
model_t.summary()
#%% Callback(reduced learning rate, model check point)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1,
                              patience = 5, min_lr = 1e-5, verbose = 1)
MODEL_CHECKPOINT_DIR = './{}'.format(time.strftime('%y%m%d_%H-%M', time.localtime(time.time())))
if not os.path.exists(MODEL_CHECKPOINT_DIR):
    os.mkdir(MODEL_CHECKPOINT_DIR)
CHECKPOINT_callback_C = ModelCheckpoint(filepath = MODEL_CHECKPOINT_DIR+'/best_m_C.hdf5', monitor = 'val_loss',
                                      verbose = 1, save_best_only = True, mode = 'min') 
CHECKPOINT_callback_T = ModelCheckpoint(filepath = MODEL_CHECKPOINT_DIR+'/best_m_T.hdf5', monitor = 'val_loss',
                                      verbose = 1, save_best_only = True, mode = 'min') 

#%% 모델 학습
history_c = model_c.fit(X_aug, Y_C_aug, epochs = 100, batch_size = 512,
                    shuffle = True, validation_split = 0.2, verbose = 1,
                    callbacks = [CHECKPOINT_callback_C, reduce_lr] )
history_t = model_t.fit(X_aug, Y_T_aug, epochs = 100, batch_size = 512,
                    shuffle = True, validation_split = 0.2, verbose = 1,
                    callbacks = [CHECKPOINT_callback_T, reduce_lr] )
#%%
history_c = model_c.fit(X_aug, Y_C_aug, epochs = 100, batch_size = 512,
                    shuffle = True, validation_data = (X_test, Y_C_test), verbose = 1,
                    callbacks = [CHECKPOINT_callback_C, reduce_lr] )
history_t = model_t.fit(X_aug, Y_T_aug, epochs = 100, batch_size = 512,
                    shuffle = True, validation_data=(X_test,Y_T_test), verbose = 1,
                    callbacks = [CHECKPOINT_callback_T, reduce_lr] )

#%%
model_c.load_weights(MODEL_CHECKPOINT_DIR+'/best_m_C.hdf5')     
model_t.load_weights(MODEL_CHECKPOINT_DIR+'/best_m_T.hdf5')

model_c.evaluate(X_train,Y_C_train)
#%% forecast from future 10 days
test_df = pd.DataFrame(columns=['date','state','C_','T_','CC_','TT_','Pop'])
for STATE in state_list:
    test_state=CORE_DATA[CORE_DATA.state==STATE][['date','state','C_','T_','CC_','TT_','Pop']].iloc[-time_len:-1].copy()
    test_state=test_state.append(CORE_DATA[CORE_DATA.state==STATE][['date','state','C_','T_','CC_','TT_','Pop']].iloc[-1])
    test_state['predict']=False                     
    for j in range(10):
        ilocation = j+time_len
        X_input = test_state[['CC_','TT_']].iloc[j:j+time_len].values.reshape(1,time_len,2)
        Y_C_hat = model_c.predict(X_input)[0][0]
        Y_T_hat = model_t.predict(X_input)[0][0]
        test_state = test_state.append({'date':test_state.date.iloc[ilocation-1]+pd.DateOffset(1),'state':STATE,'CC_':Y_C_hat,'TT_':Y_T_hat,'Pop':US_pop[STATE],'predict':True},ignore_index=True)
    test_df=test_df.append(test_state)


#%%
fig, ax = plt.subplots()
test_df[(test_df.state== 'DC')& (test_df.predict==True)].plot(x='date',y='CC_',ax=ax)
CORE_DATA[CORE_DATA.state=='DC'].plot(x='date',y='CC_',ax=ax)
L=ax.legend(['forecast','history'],loc='upper left')

#%%
#pred_df = CORE_DATA[(CORE_DATA.state=='US')].groupby(['state', 'date']).agg({'C_': 'sum'}).reset_index()

Y_C_hat = model_c.predict(X_test)
Y_T_hat = model_t.predict(X_test)

result= pd.DataFrame({'C_pred':Y_C_hat.reshape(-1),'C_':Y_C_test.reshape(-1),'T_pred':Y_T_hat.reshape(-1),'T_':Y_T_test.reshape(-1)})
result.columns = ['C_pred','C_','T_pred','T_']

#model_c.evaluate(X_test,Y_test)

plt.figure()    
result[['C_','C_pred']][30:60].plot.bar(title = "ConfirmedCases_std")
plt.show()


#%%
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history_c.history['loss'], 'y', label='train loss')
loss_ax.plot(history_c.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')
plt.show()
#%%





#%%
MODEL_CHECKPOINT_DIR = './base/{}'.format(time.strftime('%y%m%d_%H-%M', time.localtime(time.time())))
if not os.path.exists(MODEL_CHECKPOINT_DIR):
    os.mkdir(MODEL_CHECKPOINT_DIR)

CHECKPOINT_callback = ModelCheckpoint(filepath = MODEL_CHECKPOINT_DIR+'/best_m.hdf5', monitor = 'val_loss',
                                      verbose = 1, save_best_only = True, mode = 'min') 

tensorboard = TensorBoard(log_dir = "logs/{}".format(time.strftime('%y%m%d_%H-%M', time.localtime(time.time()))))
#%% 모델 학습
history = model.fit(X_aug, Y_aug, epochs = 50, batch_size = 512,
                    shuffle = True, validation_split = 0.2, verbose = 1,
                    callbacks = [CHECKPOINT_callback, tensorboard, reduce_lr] )
#%%
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')
loss_ax.set_ylim(0., 0.2)
plt.show()

#%%
from tensorflow.keras.models import load_model
model = load_model(MODEL_CHECKPOINT_DIR+'/best_m.hdf5', custom_objects={'customLoss': customLoss})
score_test = model.evaluate(X_test_scale, Y_test)
#score_blind = model.evaluate(X_blind_scale, Y_blind_scale)

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
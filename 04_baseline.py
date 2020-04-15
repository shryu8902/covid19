#%%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#%%
wk2_train = pd.read_csv('./train_wk2.csv')
wk4_train = pd.read_csv('./train_wk4.csv')
#%%
wk2_train["Province_State"] = wk2_train["Province_State"].fillna("No State")
# %%
#wk2_train[wk2_train.Province_State=="Arizona"].ConfirmedCases.diff().plot()
train_df=wk2_train

#%%
#Ratio of test data
test_rate = 0.1

#Length of time series data
time_series_len = 20

#Number of dates that can be used as training data(1/22～3/18 = 57!)
train_data_date_count = len(set(train_df["Date"]))

ss_c = StandardScaler()
train_df["ConfirmedCases_std"] = ss_c.fit_transform(train_df["ConfirmedCases"].values.reshape(len(train_df["ConfirmedCases"].values),1))


ss_f = StandardScaler()
train_df["Fatalities_std"] = ss_f.fit_transform(train_df["Fatalities"].values.reshape(len(train_df["Fatalities"].values),1))

#%%
X, Y_c, Y_f = [],[],[]

for state,country in train_df.groupby(["Province_State","Country_Region"]).sum().index:
    df = train_df[(train_df["Country_Region"] == country) & (train_df["Province_State"] == state)]
    
    if df["ConfirmedCases"].sum() != 0 or df["Fatalities"].sum() != 0:
        
        for i in range(len(df) - time_series_len):
            if (df[['ConfirmedCases']].iloc[i+time_series_len-1].values != 0 or df[['Fatalities']].iloc[i+time_series_len-1].values != 0):
                X.append(df[['ConfirmedCases_std','Fatalities_std']].iloc[i:(i+time_series_len)].values)
                Y_c.append(df[['ConfirmedCases_std']].iloc[(i+1):(i+1+time_series_len)].values)
                Y_f.append(df[['Fatalities_std']].iloc[i+time_series_len].values)

X=np.array(X)
Y_f=np.array(Y_f)
Y_c=np.array(Y_c)
#%%
X_train, X_test, Y_c_train, Y_c_test = train_test_split(X, Y_c, test_size=test_rate, shuffle = True ,random_state = 0)
X_train, X_test, Y_f_train, Y_f_test = train_test_split(X, Y_f, test_size=test_rate, shuffle = True ,random_state = 0)

#%%
batch_size = 10
n_hidden = 100
n_in = 2
model_c = Sequential()
model_c.add(LSTM(n_hidden, batch_input_shape =(None, 20, 2),
kernel_initializer = 'random_uniform',return_sequences=True))
model_c.add(TimeDistributed(Dense(1)))
model_c.add(Activation("linear"))
ADAM = optimizers.Adam()
model_c.compile(loss='mean_squared_error',optimizer=ADAM)

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())




# %%
Population = {
    'China-': 1386e6,
    'US-': 327e6,
    'EU-': 512e6 + (10+9+5+3+3+2+0.5+0.4)*1e6,

    'US-California':39937489,
    'US-Texas':29472295,
    'US-Florida':21992985,
    'US-New York':19440469,
    'US-Pennsylvania':12820878,
    'US-Illinois':12659682,
    'US-Ohio':11747694,
    'US-Georgia':10736059,
    'US-North Carolina':10611862,
    'US-Michigan':10045029,
    'US-New Jersey':8936574,
    'US-Virginia':8626207,
    'US-Washington':7797095,
    'US-Arizona':7378494,
    'US-Massachusetts':6976597,
    'US-Tennessee':6897576,
    'US-Indiana':6745354,
    'US-Missouri':6169270,
    'US-Maryland':6083116,
    'US-Wisconsin':5851754,
    'US-Colorado':5845526,
    'US-Minnesota':5700671,
    'US-South Carolina':5210095,
    'US-Alabama':4908621,
    'US-Louisiana':4645184,
    'US-Kentucky':4499692,
    'US-Oregon':4301089,
    'US-Oklahoma':3954821,
    'US-Connecticut':3563077,
    'US-Utah':3282115,
    'US-Iowa':3179849,
    'US-Nevada':3139658,
    'US-Arkansas':3038999,
    'US-Puerto Rico':3032165,
    'US-Mississippi':2989260,
    'US-Kansas':2910357,
    'US-New Mexico':2096640,
    'US-Nebraska':1952570,
    'US-Idaho':1826156,
    'US-West Virginia':1778070,
    'US-Hawaii':1412687,
    'US-New Hampshire':1371246,
    'US-Maine':1345790,
    'US-Montana':1086759,
    'US-Rhode Island':1056161,
    'US-Delaware':982895,
    'US-South Dakota':903027,
    'US-North Dakota':761723,
    'US-Alaska':734002,
    'US-District of Columbia':720687,
    'US-Vermont':628061,
    'US-Wyoming':567025,
    
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
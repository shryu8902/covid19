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
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Activation, BatchNormalization, InputLayer, Lambda, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

#%% 

DATA = np.array([[[1,2],[3,4],[5,6]],[[2,1],[4,3],[6,5]],[[-1,-2],[-3,-4],[-5,-6]],[[-2,-1],[-4,-3],[-6,-5]]])
DATA2 = np.array([[[1,2,3],[3,4,5],[5,6,7]],[[2,1,0],[4,3,2],[6,5,4]],[[-1,-2,-3],[-3,-4,-5],[-5,-6,-7]],[[-2,-1,0],[-4,-3,-2],[-6,-5,-4]]])

# %%

def part(x):
# params : shape : None, 10, 2 [None, 10, 2]
# inputs : shape : None, 10, 2 [None, 10, 2]
    params = x[:,0]
    inputs = x[:,2]

    # x1 = K.exp(params)
    # x2 = inputs + x1
    x2 = params-inputs
    return x2

def part2(x):
    params = x[0]
    inputs = x[1]

    x1 = params[:,:,0]
    x2 = inputs[:,:,0]
    return(x1+x2)
    
main_input = Input(shape = (3,2),name = 'main')
aux_input = Input(shape = (3,3), name ='aux')
mer = concatenate([main_input, aux_input],axis=-1)
merged=Concatenate([main_input,aux_input])
Out = TimeDistributed(Lambda(part))(mer)


layer_1 = -main_input

Out = TimeDistributed(Lambda(part2))([main_input,layer_1])


model = Model(inputs = [main_input,aux_input], outputs = Out)
model.compile(loss='mean_squared_error',optimizer='RMSprop')
model.predict([DATA,DATA2])
#%%
from tensorflow.keras.utils import plot_model
plot_model(model)

model = Sequential()
x = Input
model.add(InputLayer(input_shape=(3,2)))
model.add(TimeDistributed(Lambda(lambda x: x[:,0])))
model.compile(loss='mean_squared_error',optimizer='RMSprop')
out = model.predict(DATA)

model.predict()
#    model.add(BatchNormalization())
for i in range(n_layer):
    if i==n_layer-1:
        model.add(LSTM(n_hidden, dropout=dropout, kernel_initializer = init, return_sequences=False))
    else:
        model.add(LSTM(n_hidden, dropout=dropout, kernel_initializer = init, return_sequences=True))
model_t.add(TimeDistributed(Dense(1)))

model.add(Dense(1))
model.add(Activation("relu"))
model.compile(loss=loss,optimizer=optimizer)
return model


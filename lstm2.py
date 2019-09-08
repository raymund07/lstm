import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
df = pd.read_csv('data/EURUSD.csv')
from datetime import datetime, timedelta
df.rename(columns={'Gmt time' : 'timestamp', 'Open' : 'open', 'Close' : 'close', 
                   'High' : 'high', 'Low' : 'low', 'Close' : 'close', 'Volume' : 'volume'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)  - timedelta(hours=4,minutes=0,seconds=0)

df.set_index('timestamp', inplace=True)
df = df.astype(float)
df.head()


def heikin_ashi(df):
    heikin_ashi_df = pd.DataFrame(index=df.index.values, columns=['open', 'high', 'low', 'close'])
    
    heikin_ashi_df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    for i in range(len(df)):
        if i == 0:
            heikin_ashi_df.iat[0, 0] = df['open'].iloc[0]
        else:
            heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i-1, 0] + heikin_ashi_df.iat[i-1, 3]) / 2
        
    heikin_ashi_df['high'] = heikin_ashi_df.loc[:, ['open', 'close']].join(df['high']).max(axis=1)
    
    heikin_ashi_df['low'] = heikin_ashi_df.loc[:, ['open', 'close']].join(df['low']).min(axis=1)
    
    return heikin_ashi_df

df1=heikin_ashi(df)


df1['ema8'] = pd.Series.ewm(df1['close'], span=8).mean()
df1['ema20'] = pd.Series.ewm(df1['close'], span=20).mean()
df1['ema50'] = pd.Series.ewm(df1['close'], span=50).mean()
df1['ema200'] = pd.Series.ewm(df1['close'], span=200).mean()
df1['future']=df1['close'].shift(-8)

df2=df1
df1.dropna(how="any",inplace=True)

X_train = []
y_train = []

sc=MinMaxScaler(feature_range=(0,1))
df2_scaled=sc.fit_transform(df2)

from sklearn import preprocessing

#from collections import deque
#sequential_data = []  # this is a list that will CONTAIN the sequences
#prev_days = deque(maxlen=60)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in
#for col in df2.columns:  
##    pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
##    df.dropna(inplace=True)  # remove the nas created by pct_change
#    df2[col] = preprocessing.scale(df[col].values)

X_train = []
y_train = []

df2_t=pd.DataFrame(df2_scaled)
from collections import deque

sequential_data = []  # this is a list that will CONTAIN the sequences
prev_days = deque(maxlen=60)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

for i in df2_t.values:  # iterate over the values
    prev_days.append([n for n in i[:-1]])  # store all but the target
    if len(prev_days) == 60:  # make sure we have 60 sequences!
        sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!
import random
random.shuffle(sequential_data)

X = []
y = []

for seq, target in sequential_data:  # going over our new sequential data
    X.append(seq)  # X is the sequences
    y.append(target)
        


#def create_ts(df2, series):
#    X, Y =[], []
#    for i in range(len(ds)-series - 1):
#        item = df2[i:(i+series), 0]
#        X.append(item)
#        Y.append(df2[i+series, 0])
#    return np.array(X), np.array(Y)
#
#series = 7
#
#trainX, trainY = create_ts(df2, series)
#testX, testY = create_ts(df2, series)
#
#
#
#import random
#
#random.shuffle(sequential_data)
#sc=MinMaxScaler(feature_range=(0,1))
#sequential_data=sc.fit_transform(sequential_data)
#
#
#sequential_data = scaler.fit_transform(sequential_data)
#
#x_train=df1.iloc[:,0:8]
#y_train=df1['future']
X_train=np.array(X)
Y_train=np.array(y)

#
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape=(X_train.shape[1:])))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
regressor.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
print(regressor.summary())
# Fitting the RNN to the Training set
#regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir="logs2\EURUSD")
checkpoint = ModelCheckpoint("weights_2.best.hdf5", monitor='mean_squared_error', verbose=1, save_best_only=True, mode='max')
# Fit
callbacks_list = [checkpoint]
history = regressor.fit(X_train, Y_train, epochs=200, batch_size=500, verbose=1, callbacks=[tensorboard, checkpoint], validation_split=0.1)


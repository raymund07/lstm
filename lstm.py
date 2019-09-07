# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 01:35:41 2019

@author: User
"""

import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing
SEQ_LEN=60
FUTURE_PERIOD_PREDICT=3
RATIO_TO_PREDICT="EURUSD"

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def preprocess_df(df):
    df = df.drop("future", 1)  # don't need this anymore.
    print(df.head())

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            try:
                df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
                df.dropna(how='all' ,inplace=True)  # remove the nas created by pct_change
                df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.
            except:
                print(df[col])
                
    df.dropna(how='all',inplace=True)  # cleanup again... jic.
    print(df.tail(10))


    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets
    holds=[]

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!
        else:
            holds.append([seq,target])#its hold or do nothing

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(holds)  # shuffle the sells!
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells),len(holds))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.
    holds = holds[:lower]  # make sure both lists are only up to the shortest length.
    sequential_data = buys+sells+holds # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!

def classify(current,future):
    
    if RATIO_TO_PREDICT=='USDJPY':
       pip_value=float(future)-float(current)
       if (pip_value>0.10):
           return 1
       elif(pip_value<-0.10):
           return 0
       else:
           return 2
    else:
        pip_value=float(future)-float(current)
        if (pip_value>.0010):
            return 1
        elif(pip_value>-.0010):
            return 0
        else:
            return 2
        
        

main_df=pd.DataFrame()
ratios=["AUDUSD","EURUSD","GBPUSD","NZDUSD","USDCAD","USDCHF","USDJPY"]
from datetime import datetime, timedelta

for ratio in ratios:
    dataset=f"data/{ratio}.csv"
    df=pd.read_csv(dataset)
    df.rename(columns={"Gmt time":"time","Close":f"{ratio}_close","Volume":f"{ratio}_volume"},inplace=True)
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)  - timedelta(hours=4,minutes=0,seconds=0)
    df['time']=  pd.to_datetime(df['time']) 
    
    df.set_index("time", inplace=True)
    
    df=df[[f"{ratio}_close",f"{ratio}_volume"]]
    print(df.head())
    
    if len(main_df)==0:
        main_df=df
    else:
        main_df=main_df.join(df)
        
main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(how='any', inplace=True)
print(main_df.head())  # how did we do??

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

print(main_df.head(10))
times = sorted(main_df.index.values)  # get the times
main_df.dropna(inplace=True)
main_df.dropna(how='any', inplace=True)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]  # get the last 5% of the times
validation_main_df = main_df[(main_df.index >= last_5pct)]  # make the validation data where the index is in the last 5%
main_df=main_df[(main_df.index <= last_5pct)]

main_df.dropna(how='any', inplace=True)
print(main_df.tail(20))
print(validation_main_df.tail(20))

print(validation_main_df.head())
#last_5pct = sorted(main_df.index.values)[-int(0.95*len(times))]  # get the last 5% of the times
#main_df=main_df.replace([np.inf, -np.inf], np.nan)
#main_df.replace([np.inf, -np.inf], np.nan).dropna(how="all",inplace=True)
print(main_df.head())
train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)



print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)},hold:{train_y.count(2)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)},hold:{validation_y.count(2)}")


import time

EPOCHS = 3000  # how many passes through our data
BATCH_SIZE = 500  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the mode

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


model.summary()
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)
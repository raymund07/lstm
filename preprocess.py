# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:15:43 2019

@author: User
"""
from sklearn import preprocessing
import pandas as pd
import os
df=pd.read_csv('data/AUDUSD.csv')
print(df.head(10))
df["Volume"]=round(df["Volume"]/1000000)
print(df.head(10))
df['Gmt time']=  pd.to_datetime(df['Gmt time']) 
df.set_index("Gmt time", inplace=True)
df = df.replace(0, np.nan)
df.dropna(how='any', inplace=True)

for col in df.columns:  
        # go through all of the columns
        if col != "target":
            # normalize all ... except for the target itself!
            df[col] = df[col].pct_change() 
     
#           pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.
            print(df[col])
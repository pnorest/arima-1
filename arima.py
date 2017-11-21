#coding=utf-8
import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import read_csv
import matplotlib.pyplot as plt

filename='log-insight-2016.csv'
#data = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sunspot.year.csv')
data = pd.read_csv(filename)
#data.index = data['time'].values
data_index=[]
for i in range(len(data['time'].values)):
    data_index.append(i)
data.index=data_index

plt.figure(figsize=(15,5))
plt.plot(data.index,data['num'])
plt.ylabel('Sunspots')
plt.title('Yearly Sunspot Data')
plt.show()

model=pf.ARIMA(data=data,ar=20,ma=20,integ=0,target='num')
x=model.fit("MLE")#Maximum Likelihood Estimation
x.summary()

#model.plot_z(indices=range(1,20))
model.plot_fit(figsize=(15,5))
model.plot_predict_is(h=30,past_values=100,figsize=(15,5))
#model.plot_predict(h=20,past_values=20,figsize=(15,5))
model.predict(h=20)
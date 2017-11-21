#coding=utf-8
import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt

data = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sunspot.year.csv')
data.index = data['time'].values

plt.figure(figsize=(15,5))
plt.plot(data.index,data['sunspot.year'])
plt.ylabel('Sunspots')
plt.title('Yearly Sunspot Data')
plt.show()

model=pf.ARIMA(data=data,ar=4,ma=4,integ=0,target='sunspot.year')
x=model.fit("MLE")#Maximum Likelihood Estimation
x.summary()

model.plot_z(indices=range(1,9))
model.plot_fit(figsize=(15,5))
model.plot_predict_is(50,figsize=(15,5))
model.plot_predict(h=20,past_values=20,figsize=(15,5))
model.predict(h=20)
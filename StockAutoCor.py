# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:08:45 2018

@author: Laptop
"""

#help from - https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
from sklearn.linear_model import LinearRegression

#==============================================================================
# Process Data
#==============================================================================
#Load data
df = pd.read_csv('INTU.csv',header=0)
df['ret'] = df['Open']/df['Close'] - 1
series = df['ret']
#Create Lagged Dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
# split into train and test sets :: last 7 data points are for training
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-5:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

#==============================================================================
# Prelimiary Plots
#==============================================================================
print(series.head())
series.plot()
pyplot.show()
lag_plot(series)
pyplot.show()
print(result)
autocorrelation_plot(series)
pyplot.show()
plot_acf(series, lags=365)
pyplot.show()
#print(dataframe.values)
#print(values)
print("<------------------------------>")

#==============================================================================
# Autoregression Modeling
#==============================================================================
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-5:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.8f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
#==============================================================================
# Possible outcomes after certaint Returns/Prices
#==============================================================================
pyplot.hist(df['ret'].values,bins=100)
pyplot.show()
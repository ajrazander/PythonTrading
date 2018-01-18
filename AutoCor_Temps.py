# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 21:03:53 2018

@author: Laptop
"""
#help from - https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

from pandas import Series
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR

#==============================================================================
# Process Data
#==============================================================================
#Load data
series = Series.from_csv('MinTemps.csv', header=0)
#Create Lagged Dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
# split into train and test sets :: last 7 data points are for training
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
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
plot_acf(series, lags=31)
pyplot.show()
#print(dataframe.values)
#print(values)
print("<------------------------------>")
#==============================================================================
# Persistence Modeling
#==============================================================================
# persistence model
def model_persistence(x):
	return x
# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()
print("<------------------------------>")
#==============================================================================
# Autoregression Modeling
#==============================================================================
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
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
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
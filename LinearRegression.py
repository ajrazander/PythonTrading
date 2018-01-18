# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 21:45:55 2018

@author: Laptop
"""

import statsmodels.formula.api as smf
import pandas as pd
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
#import numpy as np

dataframe = pd.read_csv('AAPL.csv')
df = dataframe[5701:9351]

reg0 = smf.ols('Price ~ Date',data=df).fit()
print(reg0.summary())

prstd, iv_l, iv_u = wls_prediction_std(reg0)

fig, ax = plt.subplots(figsize=(8,6))
fig, bx = plt.subplots(figsize=(8,6))
x = df[["Date"]].values
y = df[["Price"]].values

ax.plot(x, y, 'o', label="data")
#ax.plot(x, y_true, 'b-', label="True")
ax.plot(x, reg0.fittedvalues, 'r--.', label="OLS")
#ax.plot(x, iv_u, 'r--')
#ax.plot(x, iv_l, 'r--')
ax.legend(loc='best');

bx.plot(x, reg0.resid, label="Residuals")

residuals = reg0.resid.values

df2 = pd.concat([x, residuals], axis=1)
print(df2.head())
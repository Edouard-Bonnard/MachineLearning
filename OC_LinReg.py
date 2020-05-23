# Linear Regression multi features

# -*- Python 3.7.6 -*-
# coding UTF-8

# Comment : in progress, problem with NaN values in house_data_X, LinReg not working


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd

#data importation
#price, surface, arrondissement
house_data = pd.read_csv('/Users/edouard/Documents/Code/MachineLearning/house_data.csv')

house_data = house_data[house_data['price'] < 10000]

price_y = house_data['price'] #target
house_data_X = house_data.drop(columns='price') #features

#split in validation and tests sets 80%/20%
house_data_X_train, house_data_X_test, price_y_train, price_y_test = train_test_split(house_data_X, price_y, train_size=0.8)

#convertion to Numpy arrays transposed
X = (house_data_X_train.to_numpy()).T
y = (price_y_train.to_numpy()).T

regr = linear_model.LinearRegression()
regr.fit(X,y)

print(regr.intercept_)
print(regr.coef_)

print('end')

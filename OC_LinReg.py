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

house_data = house_data[house_data['price'] < 10000] #high values removal
house_data = house_data.dropna() #NaN values removal

price_y = house_data['price'] #target
house_data_X = house_data.drop(columns='price') #features

#split in validation and tests sets 80%/20%
house_data_X_train, house_data_X_test, price_y_train, price_y_test = train_test_split(house_data_X, price_y, train_size=0.8)

#check for NaN values in dataframe
# X.isnull().values.any() existence of at least a NaN value
# X.isnull().sum().sum() numbers of NaN values

#train
regr_m = linear_model.LinearRegression()
regr_m.fit(house_data_X_train,price_y_train)

#regr_s = linear_model.LinearRegression()
#regr_s.fit(house_data_X_train_only-surface, price_y_train)

#test
print(regr_m.score(house_data_X_test, price_y_test))
#regr_s.score(house_data_X_train_only-surface, price_y_test)

plt.figure(1)
plt.plot(house_data_X_train['surface'], price_y_train, 'ro', markersize=4)

plt.figure(2)
plt.plot(house_data_X_train['arrondissement'], price_y_train, 'bo', markersize=4)

plt.show()



print('end')

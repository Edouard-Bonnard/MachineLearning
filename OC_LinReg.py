# Linear Regression multi features

# -*- Python 3.8.2 -*-
# coding UTF-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd

#data importation
# house_data : price, surface, arrondissement
house_data = pd.read_csv('/Users/edouard/Documents/Code/MachineLearning/house_data.csv')

house_data = house_data[house_data['price'] < 10000] #high values removal
house_data = house_data.dropna() #NaN values removal

house_data_X = house_data.drop(columns='price') #features
price_y = house_data['price'] #target

#split in validation and tests sets 80%/20%
house_data_X_train, house_data_X_test, price_y_train, price_y_test = train_test_split(house_data_X, price_y, train_size=0.8)

#check for NaN values in dataframe
# X.isnull().values.any() existence of at least a NaN value
# X.isnull().sum().sum() numbers of NaN values

##1st analysis
#train
regr_m = linear_model.LinearRegression()
regr_m.fit(house_data_X_train,price_y_train)

regr_s = linear_model.LinearRegression()
regr_s.fit(house_data_X_train.drop(columns='arrondissement'), price_y_train)

#test des algorithmes
m_score = regr_m.score(house_data_X_test, price_y_test)
s_score = regr_s.score(house_data_X_test.drop(columns='arrondissement'), price_y_test)

#Affichage des performances
print('single feature lin reg accuracy = ', "{0:.0%}".format(s_score))
print('multi features lin reg accuracy = ', "{0:.0%}".format(m_score))

#plt.figure(1)
#plt.plot(house_data_X_train['surface'], price_y_train, 'ro', markersize=4)
#plt.figure(2)
#plt.plot(house_data_X_train['arrondissement'], price_y_train, 'bo', markersize=4)

##2nd analysis
pos = house_data['arrondissement'].unique() #liste des arrondissements

data_set = []

fig, axs = plt.subplots()

for i in pos:
    data = (house_data_X_train[house_data_X_train['arrondissement'] == i]['surface']).to_numpy()
    data_set.append(data)


axs.violinplot(data_set, pos, showmeans=True, showextrema=False)
plt.show()



print('end')

# -*- Python 3.7.6 -*-
# coding UTF-8
# See https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4121986-programmez-votre-premiere-regression-lineaire

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

house_data = pd.read_csv('/Users/edouard/Documents/Code/MachineLearning/house.csv')
house_data = house_data[house_data['loyer'] < 10000] #higher values removal

#classic plot
plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
plt.show()

X = house_data['surface']
y = house_data['loyer']

from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X,y)



#regr.predict(<des données de test)

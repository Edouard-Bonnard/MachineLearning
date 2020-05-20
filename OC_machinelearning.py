# -*- Python 3.7.6 -*-
# coding UTF-8
# See https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning/4121986-programmez-votre-premiere-regression-lineaire

## Linear Regression with sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

##Données d'entrainement
house_data = pd.read_csv('/Users/edouard/Documents/Code/MachineLearning/house.csv')
house_data = house_data[house_data['loyer'] < 10000] #higher values removal

##méthode litérale

#vecteur de entrées
#on ajoute une colonne de 1 afin de simplifier le calcul matriciel
#le coef. theta[0] est alors l'ordonnée à l'origine
#attention à la transposition
X = np.array([np.ones(house_data.shape[0]),house_data['surface']]).T 

#vecteur des sorties
y = np.array(house_data['loyer']).T #sorties

#formule litérale
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta)

plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4) #affichage données

#droite de regression entre 0 et 250
plt.plot([0,250], [theta.item(0),theta.item(0) + 250 * theta.item(1)], linestyle='--', c='#000000')
plt.show()

##méthode avec SkLearn
A = np.array([house_data['surface']]).T #ici pas de colonnne de 1
b = np.array(house_data['loyer']).T

regr = linear_model.LinearRegression()
regr.fit(A,b)

#regr.intercept_ : ordonnée à l'origine
#regr.coef_ : coefficient de pente pour chacune des variables observées

print('end')


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

#sample = np.random.randint(house_data.shape[0],size=house_data.shape[0])
sample = np.arange(house_data.shape[0])
np.random.shuffle(sample)

train_ratio = 0.8
train_len = round(train_ratio * house_data.shape[0])

train = sample[:train_len]
test = sample[train_len:]

house_data_train = house_data.drop(index=test)

house_data_train = house_data_train[house_data_train['loyer'] < 10000] #higher values removal

##entrainement méthode litérale

#vecteurs de entrées
#on ajoute une colonne de 1 afin de simplifier le calcul matriciel
#le coef. theta[0] est alors l'ordonnée à l'origine
#attention à la transposition
X = np.array([np.ones(house_data_train.shape[0]),house_data_train['surface']]).T 

#vecteur des sorties
y = np.array(house_data_train['loyer']).T #sorties

#formule litérale
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta)

plt.plot(house_data_train['surface'], house_data_train['loyer'], 'ro', markersize=4) #affichage données

#droite de regression entre 0 et 250
plt.plot([0,250], [theta.item(0),theta.item(0) + 250 * theta.item(1)], linestyle='--', c='#000000')


##méthode avec SkLearn
A = np.array([house_data_train['surface']]).T #ici pas de colonnne de 1
b = np.array(house_data_train['loyer']).T

regr = linear_model.LinearRegression()
regr.fit(A,b)

#regr.intercept_ : ordonnée à l'origine
#regr.coef_ : coefficient de pente pour chacune des variables observées

## Validation du modèle

house_data_test = house_data.drop(index=train) #suppression des valeurs d'entrainement

predict = house_data_test['surface']*theta[1]+theta[0] #prédictions

plt.plot(house_data_test['surface'], house_data_test['loyer'], 'bo', markersize=4)
plt.plot(house_data_test['surface'], predict, 'yo', markersize=4)

plt.show()

#Solidité du modèle (inspiration personnelle)

Erreur = house_data_test['loyer'] - predict
Erreur_quad = Erreur*Erreur

Erreur_moy = np.mean(Erreur)
Erreur_moy_quad = np.mean(Erreur_quad)

print('end')


## k-NN with sklearn

# -*- Python 3.7.6 -*-
# coding UTF-8

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

#resampling
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample] #data => données
target = mnist.target[sample] #target => annotations

#split du jeu de données train/test 80%/20%
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

#k-NN
from sklearn import neighbors

#Entrainement pour k=3
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)
#Erreur du classifieur
error = 1 - knn.score(xtest, ytest)

#Entrainement pour k de 2 à 14
errors=[]
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    knn.fit(xtrain, ytrain)
    errors.append(1-knn.score(xtest, ytest))

plt.plot(range(2,15),errors,'-o')
plt.show()






## k-NN with sklearn

# -*- Python 3.7.6 -*-
# coding UTF-8

import numpy as np

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

#Entrainement
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)

#Erreur du classifieur
error = 1 - knn.score(xtest, ytest)
print('Erreur: %f' % error)

print(mnist.data.shape)

# Choosing the right k in k-NN
# -*- Python 3.8.2 -*-
# coding UTF-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection #for spliting dataset
from sklearn import preprocessing #for centering and scaling
from sklearn import neighbors, metrics #for choosing the right k

#dataset import
data = pd.read_csv('winequality-red+(7).csv', sep=";") 
print(data.head())

#extract in Numpy arrays
X = data[data.columns[:-1]].values #all columns exepted the last
y = data['quality'].values #last column

#dataset visualization 
fig = plt.figure(figsize=(9, 6))
for feat_idx in range(X.shape[1]):
    ax = fig.add_subplot(3,4, (feat_idx+1))
    h = ax.hist(X[:, feat_idx], bins=50, color='steelblue', density=True, edgecolor='none')
    ax.set_title(data.columns[feat_idx], fontsize=14)

# plt.show()
# comment : we can see that some value are within small intervalls, and other within large ones.
# => we need to standardize theses values


# Let's tranform the problem in a classification problem :
# a wine ranked >6 will be "good" ; =1
# a wine ranked <6 will be " not good " ; =0 
y_class = np.where(y<6, 0, 1)

# train/test dataset split
X_train, X_test, y_train, y_test = \
	model_selection.train_test_split(X, y_class,
                                	test_size=0.3 # 30% ratio
                                	)

# Centering and scaling of data
# Warning : only works for gaussian datasets
std_scale = preprocessing.StandardScaler().fit(X_train) #Parameters calc
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

# Visualization of data after centering and scaling
fig = plt.figure(figsize=(9, 6))
for feat_idx in range(X_train_std.shape[1]):
    ax = fig.add_subplot(3,4, (feat_idx+1))
    h = ax.hist(X_train_std[:, feat_idx], bins=50, color = 'steelblue', density=True, edgecolor='none')
    ax.set_title(data.columns[feat_idx], fontsize=14)

#plt.show()

# Crossed validation of k for k-NN
param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15]} #k values to be tested
score = 'accuracy' #score to be optimised

# Build of classifyer
clf = model_selection.GridSearchCV(
    neighbors.KNeighborsClassifier(), # K-NN estimator
    param_grid,     # dict of parameters to test
    cv=5,           # folds number
    scoring=score   # score to optimise
)

# Optimization on the train set
clf.fit(X_train_std, y_train)

# Display of best parameters
print("Best parameters on training set:")
print(clf.best_params_)

# Display of crossed validation
print("Result for crossed validation:")
for mean, std, params in zip(
        clf.cv_results_['mean_test_score'], # mean score
        clf.cv_results_['std_test_score'],  # std deviation score
        clf.cv_results_['params']           # hyper parameter value
    ):

    print("{} = {:.3f} (+/-{:.03f}) for {}".format(
        score,
        mean,
        std*2,
        params
    ) )

# Generalization on test set
# Comment : gridsearchCV automatically train the best k-NN on the whole train set
y_pred = clf.predict(X_test_std)
print("\nOn test set : {:.3f}".format(metrics.accuracy_score(y_test, y_pred)))







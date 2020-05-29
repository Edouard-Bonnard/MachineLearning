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
print('-----','1st study : linear regressions','-----')
print('single feature lin reg accuracy = ', "{0:.0%}".format(s_score))
print('multi features lin reg accuracy = ', "{0:.0%}".format(m_score))

#plt.figure(1)
#plt.plot(house_data_X_train['surface'], price_y_train, 'ro', markersize=4)
#plt.figure(2)
#plt.plot(house_data_X_train['arrondissement'], price_y_train, 'bo', markersize=4)

##Visualisation des surfaces par arrondissement
arr_list = house_data['arrondissement'].unique() #liste des arrondissements
data_set = [] #vecteur pour affichage violin plot

for i in arr_list: #remplissage data_set
    data = (house_data_X_train[house_data_X_train['arrondissement'] == i]['surface']).to_numpy()
    data_set.append(data)

fig, axs = plt.subplots()
axs.violinplot(data_set,showmeans=True, showextrema=False)

arr_list_str = list( map(str, arr_list) ) #Mise en forme
arr_list_str.insert(0,'0') #ajout d'un '0' pour indexage xticklabels
axs.set_xticklabels(arr_list_str)

#plt.show()

##Segmentation par arrondissement puis single feature lin reg

print('-----','2st study : segmentation','-----')

regr_seg = linear_model.LinearRegression()

fig2, axs2 = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
pos = 0

for i in arr_list: #segmentation par arrondissement
    X_train = pd.DataFrame(house_data_X_train[house_data_X_train['arrondissement'] == i]['surface'])
    y_train = pd.DataFrame(price_y_train[house_data_X_train['arrondissement'] == i])
    
    regr_seg.fit(X_train,y_train)

    X_test = pd.DataFrame(house_data_X_test[house_data_X_test['arrondissement'] == i]['surface'])
    y_test = pd.DataFrame(price_y_test[house_data_X_test['arrondissement'] == i])
    seg_score = regr_seg.score(X_test, y_test)

    print('Arrondissement :',i,'single feature lin reg accuracy = ', "{0:.0%}".format(seg_score))

    axs2[pos].scatter(X_test,y_test) #affichage des valeurs de test
    axs2[pos].plot([0,250], [regr_seg.intercept_[0],regr_seg.intercept_[0] + 250 * regr_seg.coef_[0][0]], linestyle='--', c='#000000') #affichage du modèle de prédiction

    
    title = 'Arrondissement : '+ str(i) + ' eff. = ' + "{0:.0%}".format(seg_score)
    axs2[pos].set_title(title, fontsize=10)
    pos = pos+1 #position de la figure


plt.show()
print('end')

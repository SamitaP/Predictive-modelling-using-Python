# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 09:28:40 2017

@author: Samita
"""

import numpy as np
from sklearn import cross_validation, neighbors
import pandas as pd
#import pdb
 
df = pd.read_csv('D:\data\BCTrain.csv')  #Read data from csv file to train the model
df.replace('?', -99999, inplace=True)   #Handling missing data
df.drop(['id'], 1, inplace=True)        #Removing id since it is not required to train the model

X = np.array(df.drop(['Class'], 1))     #Avoid class attribute while training the model
y = np.array(df['Class'])               

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25) #Separating data in train and test

classifier = neighbors.KNeighborsClassifier()
    
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)     #Confidence of algorithm
print(accuracy)

unknown = np.array([15.92,14.97,14.65,16.58,18.77,15.18,17.91,20.78,20.7,15.34,13.08,15.34,17.94,20.74,19.46,12.74,12.96,20.18,15.94,18.15,22.22,22.04,19.76,9.71,18.8,12.39,19.63,11.89,14.71,15.15])
unknown = unknown.reshape(len(unknown), -1)

prediction = classifier.predict(unknown)        #predicting class value for unknown data
print(prediction)

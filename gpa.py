# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 00:22:15 2018

@author: Isaac Reynaldo
"""

#variables
#sat. combined SAT score
#tothrs. total hours through fall semest
#colgpa. GPA after fall semester
#athlete. =1 if athlete
#verbmath. verbal/math SAT score
#hsize. size grad. class, 100s
#hsrank. rank in grad. class
#hsperc. high school percentile, from top
#female. =1 if female
#white. =1 if white
#black. =1 if black
#hsizesq. hsize^2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Importing the Keras Libraries and packages
import keras
from keras.models import Sequential #to initialize ann
from keras.layers import Dense # to create layers ann
from keras.layers import Dropout #dropout to regulate when some of the neuron become off

# Importing the dataset
dataset = pd.read_csv('gpa.csv')
X = dataset.iloc[:, 1:13].values #independent variables
y = dataset.iloc[:, 13].values #dependent variable, exited



# Encoding categorical data, for example country or gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#for country making it into categories or number
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#for gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# X is the array of independent variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling to ease calculations, standard normal distribution with a mean of zero and a standard deviation of one 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
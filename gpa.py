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

# Importing the Keras Libraries and packages
import keras
from keras.models import Sequential #to initialize ann
from keras.layers import Dense # to create layers ann
from keras.layers import Dropout #dropout to regulate when some of the neuron become off

# Importing the dataset
dataset = pd.read_csv('gpa.csv')
X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,10,11,12]].values #independent variables
y = dataset.iloc[:, 9].values #dependent variable, exited

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling to ease calculations, standard normal distribution with a mean of zero and a standard deviation of one 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initializing ANN
# model of classification, cuz its a clasifier from sequential
classifier = Sequential()

# Adding the input layer and first hidden layer
#(PARAMETERS) of Dense
#1 parameter of dense function will be average between input and ouput nodes
#2 initializing the weights randomly close to zero.
#3 activation relu is rectifier function
#4 input dim is how many nodes are input or independent. with dropout
classifier.add(Dense(activation="relu", input_dim=12, units=7, kernel_initializer="uniform")) # first hidden layer

# Adding second hidden layer
classifier.add(Dense(activation="relu", units=7, kernel_initializer="uniform")) # first hidden layer, it doesnt need input cuz it was already specified
#classifier.add(Dropout(p = 0.1)) #try under 0.5

# output Layer, softmax is if more than one category for output instead of sigmoid
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform")) #using sigmoid function, and 1 output node

# Compiling the ANN, using Stichastic Gradient Descent
#(PARAMETERS) of compile
#1 optimizer will be the algorithm to find ultimate set of weights on ann, Stichastic Gradient Descent. called adam
#2 loss function within the Stichastic Gradient Descent, because is sigmoid will use logarithmic loss function, because one output binary
#3 metric argument is to evaluate the model, we'll use accurate to improve the performance, until it reaches top accuracy. is a list of metrics thats why []
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting ANN to the training set
#(PARAMETERS) of fit
#1 the train independent
#2 the train dependent
#3 batch_size , maybe experiment in it.
#4 how many epochs
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #threshold if y_pred > 0.5 say true

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 22:14:22 2018

@author: peter
"""

#import csv
from sklearn import metrics
from sklearn.neural_network import MLPClassifier 
from numpy import genfromtxt
#import numpy as np

dataset = genfromtxt('mnist_train.csv', delimiter=",")[1:] #loads the MNIST data set form a .csv file into an array
labels = [x[0] for x in dataset] #Creating a list of labels for each data value, in this case a label will be either a 0, 1,...9
data = [x[1:] for x in dataset] #Load each piece of data (each image) into the array, note we have converted the mnist images into 28 by 28 grid of the gray scale values , this is represented in te .csv file  

n_samples = len(labels)  #Number of labels 
n_features = len(data[0]) #Number of how many pieces of data/features we have

print("Number of samples: " + str(n_samples) + ", number of features: "+ str(n_features))

# a support vector classifier
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(70), random_state=1)  #Create a Neural Network with 1 hidden layer consisting of 30 neurons 

split_point = int(n_samples * 0.9)  #Take 2/3 of the training data and use it for training the rest wil be used for testing 


labels_learn = labels[:split_point] #Load in 2/3 of the labels 
data_learn = data[:split_point] #load 2/3 of the features/data 

labels_test = labels[split_point:] #Do the same for the testing features and labels
data_test = data[split_point:]

print("Training: " + str(len(labels_learn)) + " Test: " + str(len(labels_test)))

# Learning Phase
classifier.fit(data_learn, labels_learn) #Train the model 

# Predict Test Set
predicted = classifier.predict(data_test) #Start predicting, 

# classification report
print("Classification report for classifier %s:n%sn" % (classifier, metrics.classification_report(labels_test, predicted)))

# confusion matrix
print("Confusion matrix:n%s" % metrics.confusion_matrix(labels_test, predicted))

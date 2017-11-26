# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:24:43 2017
@author: peter
"""

from sklearn import datasets
from sklearn import tree
from sklearn.cross_validation import train_test_split 


iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
Y = iris.target
 
features_train, features_test, labels_train, labels_test = train_test_split(X, Y, test_size = 0.5)
clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train) 


accuracy = clf.score(features_test, labels_test)  
print(accuracy)

#Accuracy 0.93 (< than SVM)

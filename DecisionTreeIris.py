# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:24:43 2017
@author: peter
"""

from sklearn import datasets
from sklearn import tree

iris = datasets.load_iris()
features = iris.data[:, :2]  # we only take the first two features.
labels = iris.target

num = int(0.66 * len(features)) 
X = features[:num] 
Y = labels[:num] 

clf = tree.DecisionTreeClassifier()
clf.fit(X,Y) 

testFeatures = features[num:]
testLabels = labels[num:] 

acc = clf.score(testFeatures, testLabels) 
print(acc)

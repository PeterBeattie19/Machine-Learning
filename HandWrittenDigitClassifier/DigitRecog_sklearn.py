# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:42:27 2017

@author: Peter Beattie

Description:
    This program uses three different supervised learning classifiers to classify
    handwritten digits, using sklearn's hand written digit dataset,
    each piece of data is a 7 by 7 pixel image of a number
    therefore there are 49 dimensions. 
"""

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import tree

digits = load_digits()

'''plt.gray()
plt.matshow(digits.images[50])
plt.show

print(len(digits.data)) '''

train = int(len(digits.data)*0.9)

features_train =  digits.data[:train]
labels_train = digits.target[:train] 

features_test = digits.data[train:]
labels_test = digits.target[train:]

clf = tree.DecisionTreeClassifier(min_samples_split = 22) #22 is the best min sample split, gives greatst accuracy of 80%

clf.fit(features_train, labels_train) 

accuracy = clf.score(features_test, labels_test) 

print(accuracy) 

from sklearn import svm

clf = svm.SVC(gamma = 0.0001) 


clf.fit(features_train, labels_train) 

accuracy = clf.score(features_test, labels_test)  #Support Vector Machine accuracy: 92%

print(accuracy)

from sklearn.naive_bayes import GaussianNB 
clf = GaussianNB()    

clf.fit(features_train, labels_train) 

accuracy = clf.score(features_test, labels_test) 

print(accuracy) #Naive Bayes accuracy: 80%


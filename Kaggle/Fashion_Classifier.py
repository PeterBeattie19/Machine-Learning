# 70'000 greyscale pictures of trouser, T-Shirts, etc. Each 24 by 24 pixels.
# 1 layer of 100 neurons. 2 mins 30 sec to train. 
# MNIST Fashion

import numpy as np 
import pandas as pd 
from sklearn.neural_network import MLPClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

dataset = pd.read_csv("../input/fashion-mnist_train.csv") 

Label_df = dataset[['label']] #First column contains the labels

features = dataset.as_matrix() #Convert data frame to numpy array
features = np.delete(features, 0, 1) #remove first column, this column contains the labels 

temp = Label_df.as_matrix() 

labels = np.array([])

for i in temp:
    labels = np.append(labels, i)


clf = MLPClassifier(solver = "adam",hidden_layer_sizes = (100)) #adam solver beats Stochastic Gradient Descent 

clf.fit(features, labels) 

test_data = pd.read_csv("../input/fashion-mnist_test.csv") 


Label_df = test_data[['label']]

features = test_data.as_matrix() #Convert data frame to numpy array
features = np.delete(features, 0, 1) #remove first column, this column contains the labels 

temp = Label_df.as_matrix() 

labels = np.array([])

for i in temp:
    labels = np.append(labels, i)

print(clf.score(features,labels))

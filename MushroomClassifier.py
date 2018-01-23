# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/mushrooms.csv")

#print(dataset)
#print(dataset.iloc[0:10,0:1]) #rows then columns 

clf = MLPClassifier(hidden_layer_sizes = (30,20))

y_data = dataset[['class']]

labels = np.array([])

for i in range(0,8124):
    char = y_data.iloc[i,0]

    if char == 'p':
        labels = np.append(labels, 1)
    elif char == 'e':
        labels = np.append(labels, 0)
    
#print(labels)
#8124 rows and 23 columns

features = dataset.as_matrix()
features = np.delete(features, 0, 1)
print(features)

clf = clf.fit(features,labels)

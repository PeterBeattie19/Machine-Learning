import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPClassifier
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print("reading train csv")
dataset = pd.read_csv("../input/train.csv") 
print("Finished")

Label_df = dataset[['label']]

features = dataset.as_matrix() #Convert data frame to numpy array
features = np.delete(features, 0, 1) #remove first column, this column contains the labels 

temp = Label_df.as_matrix() 

labels = np.array([])

print("putting lables into a numpy array")
for i in temp:
    labels = np.append(labels, i)
print("Finished")


clf = MLPClassifier(solver = "adam", hidden_layer_sizes = (70))

print("Trainng classifier") 
clf.fit(features, labels) 
print("Finished")

print("reading test data")
test_data = pd.read_csv("../input/test.csv") 
print("Finished")



features = test_data.as_matrix() #Convert data frame to numpy array

    
print("predicting labels")
results = clf.predict(features)
print("Finished")

print(results.size)

ImageIds = np.array([])

for i in range(0,28000):
    ImageIds = np.append(ImageIds, i)
    

print("creating Dictionary for csv file")
d = {"ImageID":ImageIds, "Label":results} 
print("Finished")

print("Putting dictionary into Data Frame")
df = pd.DataFrame(data=d)
print("Finished")

print("Saving to csv file")
df.to_csv("sample_submission.csv", index=False, header=False)
print("Finished")

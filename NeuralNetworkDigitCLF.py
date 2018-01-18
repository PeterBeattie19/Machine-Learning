''' Training a Neural Network to recognize handwritten digits,
    using SKlearn's digits dataset.
    The dataset consists of 1797 8 by 8 pixel images of handwritten digits.
    The Neural Net has 64 input neurons 70, hidden layer has 70 neurons, output layer has 10 neurons.
    The network ache=ieves an accuracy of 94%, this is higher than using just 30 neurons in the hidden layer '''
    

from sklearn.neural_network import MLPClassifier
from sklearn import datasets

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(70), random_state=1)


digits = datasets.load_digits()

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

test_features = data[n_samples // 2:]
test_labels = digits.target[n_samples // 2:]

pred = clf.predict(test_features)
numRight = 0

for i in range(0,len(pred)):
    #print pred[i] , test_labels[i]
    if pred[i] == test_labels[i]:
        numRight += 1

print numRight , " classifications correct " , "out of ", len(pred)

# 94% accuracy score  


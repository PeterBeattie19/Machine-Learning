from sklearn.neural_network import MLPClassifier
from sklearn import datasets

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30,40), random_state=1)


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

# 92% accuracy score  


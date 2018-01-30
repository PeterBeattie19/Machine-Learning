import numpy as np

input_data = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) #Features

output_labels = np.array([[0],[1],[1],[0]])

def sigmoid(x, derivative = False):
    if derivative == True:
        return x*(1-x)
    return (1/(1+np.exp(-x)))

weight_matrix_1 = 2*np.random.random((3,4)) - 1
weight_matrix_2 = 2*np.random.random((3,4)) - 1

print(weight_matrix_1)
print(weight_matrix_2)

for _ in range(6000):
    input_layer = input_data

    layer1 = sigmoid(np.dot(input_layer, weight_matrix_1))
    layer2 = sigmoid(np.dot(layer1, weight_matrix_2))

    layer2_error = output_labels - layer2

    layer2_gradient = layer2_error * activate(layer2, derivative = True)

    layer1_error = layer2_gradient.dot(weight_matrix_1.T)

    layer1_gradient = layer1_error * activate(layer1, derivative = True)

    weight_matrix_1 += layer1.T.dot(layer2_gradient)
    weight_matrix_2 += input_layer.T.dot(layer1_gradient)

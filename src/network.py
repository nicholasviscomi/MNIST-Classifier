import numpy as np
import scipy.optimize
import pandas as pd

# Structure of data:
# first column is the label of what the number actually is
# All the other columns are the values for each pixel, either 0 or 1
# Thus, one row is one image
raw_data = pd.read_csv('data/train.csv')
data = np.array(raw_data)
# data = np.load('data/mod_train.npy', allow_pickle=True)
rows, x3 = data.shape # rows, cols
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:x3]
# X_dev = X_dev / 255.

data_train = data[1000:rows].T
Y_train = data_train[0] # labels for all of the images
X_train = data_train[1:x3]
# X_train = X_train / 255.
_,m_train = X_train.shape

def tanh(x):
    return np.tanh(x)

def deriv_tanh(x):
    return 1 / (np.cosh(x) ** 2)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# structure is a tuple with the number of nuerons per layer
# (784, 16, 10)
def init_network(structure):
    all_weights = []
    all_biases  = []
    prev = -1
    for n in structure:
        if prev == -1: 
            prev = n
            continue

        # if len(all_weights) == 0:
        #     all_weights = np.random.rand(n, prev) - 0.5
        # if len(all_biases) == 0:
        #     all_weights = np.random.rand(n, 1) - 0.5

        all_weights.append(np.random.rand(n, prev) - 0.5)
        all_biases.append(np.random.rand(n, 1) - 0.5)
        prev = n

    return (all_weights, all_biases)


# network is a tuple of all weights and biases
# same data type as returned by init_network
# returns tuple of the sums, activations (size will be 1 less than the number of layers)
# inputs need to be shape (n, 1). Can be done by transposing the input
# activations will have the inputs as the first element
def feed_forward(inputs, network: tuple[list, list]):
    sums = [inputs]
    activations = [inputs]
    weights, biases = network[0], network[1]
    assert len(weights) == len(biases)

    prev_layer = np.array(inputs)
    for i in range(len(weights)):
        W = np.array(weights[i])
        B = np.array(biases[i])

        A = []
        # W = np.array(W).T # need to transpose to make the matrix dimensions work
        # print(f"{W.shape} @ {prev_layer.shape} + {B.shape}")
        Z = W @ prev_layer
        for i, (z, b) in enumerate(zip(Z, B)):
            Z[i] = z + b

        sums.append(Z)
        if i == len(weights) - 1:
            A = softmax(Z)
        else:
            A = tanh(Z)
        activations.append(A) 
        prev_layer = A

    return sums, activations

def one_hot(Y):
    # make an array that has 10 numbers. Everything is 0, 
    # except for the position of the correct label, which is one

    # When doing backpropagation, we want to turn off the bright neurons
    # that were wrong and keep on the correct ones. Thus, when calculating what
    # needs to change the most, the correct neuron needs to change by 0 and all
    # the others need to change by their value so that they become 0. This means
    # that the correct position of the correct label needs to be 1 so it doesn't
    # change, and all the other ones do.
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def backward_propagation(Y, sums, activations, weights, biases):
    # y = correct label of the image

    # first one will be that of the final layer
    sums = sums[::-1]
    activations = activations[::-1] 
    weights = weights[::-1]
    biases = biases[::-1]

    dWeights = [] 
    dBiases  = []
    
    dZ = activations[0] - one_hot(Y)
    for i, W in enumerate(weights):
        if not (i < len(activations) - 1): break
        dW = (dZ @ activations[i + 1].T) / rows 
        dWeights.append(dW)

        dB = np.sum(dZ) / rows
        dBiases.append(dB)

        dZ = (W.T @ dZ) * deriv_tanh(sums[i + 1])


    # dZ2 = A2 - one_hot(Y) 
    # dW2 = (dZ2 @ A1.T) / rows
    # db2 = np.sum(dZ2) / rows 

    # dZ1 = (W2.T @ dZ2) * deriv_tanh(Z1)
    # dW1 = (dZ1 @ X.T) / rows
    # db1 = np.sum(dZ1) / rows

    # these are in reverse order. They need to be iterated through in reverse
    return dWeights, dBiases

def update_params(weights, biases, dWeights, dBiases, lr):
    new_weights, new_biases = [], []
    
    for w, b, dw, db in zip(weights, biases, dWeights[::-1], dBiases[::-1]): # reverse dWeights & dBiases into correct order
        new_weights.append(w - (lr * dw))
        new_biases.append (b - (lr * db))

    return new_weights, new_biases

def gradient_descent(X, Y, learning_rate: float, iterations: int, network: tuple[list, list]):
    weights, biases = network[0], network[1]

    for i in range(iterations):
        sums, activations = feed_forward(X, (weights, biases))
        dWeights, dBiases = backward_propagation(Y, sums, activations, weights, biases)
        weights, biases = update_params(weights, biases, dWeights, dBiases, learning_rate)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(activations[-1])
            print(get_accuracy(predictions, Y))

    return weights, biases

def get_predictions(output):
    return np.argmax(output, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def make_predictions(inputs, weights, biases):
    _, activations = feed_forward(inputs, (weights, biases))
    predictions = get_predictions(activations[-1])
    return predictions

def print_image_and_label(image, label, prediction):
    print(f"Label: {label}")
    print(f"Prediction: {prediction}")
    
    count = 0
    for pixel in image.T.flatten():
        if pixel != 0:
            print("⬜️", end='')
        else:
            print("⬛️", end='')
            
        count += 1

        if count % 28 == 0:
            print("\n", end='') # add a new line

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print_image_and_label(current_image, label, prediction)

def download_model(W1, B1, W2, B2, name):
    master = np.array([W1, B1, W2, B2], dtype=object)
    np.save(f"models/{name}/master.npy", master)
    return

def flatten_output(O):
    flat = np.zeros(10)
    for i, _ in enumerate(O):
        flat[i] = O[i][0]
    return flat


# name = "100hidden"
# W1, B1, W2, B2 = np.load(f'models/{name}/master.npy', allow_pickle=True)
# def loss(X, target):
#     one_hot = np.zeros((10, 1))
#     one_hot[target] = 1   
#     _,_,_,out = feed_forward(X, W1, B1, W2, B2)
#     out = flatten_output(out)
#     for i, _ in enumerate(out):
#         out[i] = (one_hot[i] - out[i]) ** 2

#     return out.sum()

# def reverse_engineer_image(target):
    x0 = np.zeros(784)
    return scipy.optimize.minimize(loss, x0, args=target) # don't need bounds, just normalize the values after
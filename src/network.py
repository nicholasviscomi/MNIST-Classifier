import pandas as pd
import numpy as np

# Structure of data:
# first column is the label of what the number actually is
# All the other columns are the values for each pixel, either 0 or 1
# Thus, one row is one image

# raw_data = pd.read_csv('data/train.csv')
# data = np.array(raw_data)
data = np.load('data/mod_train.npy', allow_pickle=True)
m, n = data.shape # rows, cols
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
# X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0] # labels for all of the images
X_train = data_train[1:n]
# X_train = X_train / 255.
_,m_train = X_train.shape

# 748 inputs, 16 hidden, 10 output
W1 = [] # first weights = 748 * 16
B1 = [] # first biases = 16 for hidden nodes

W2 = [] # 16 * 10
B2 = [] # second biases = 10 for output nodes

def init_weights_biases(n_hidden):
    W1 = np.random.rand(n_hidden, 784) - 0.5 # 16 arrays with 748 random numbers from -0.5 to 0.5
    B1 = np.random.rand(n_hidden, 1) - 0.5
    W2 = np.random.rand(10, n_hidden) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2

def tanh(x):
    return np.tanh(x)

def deriv_tanh(x):
    return 1 / (np.cosh(x) ** 2)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def feed_forward(inputs, W1, B1, W2, B2):
    Z1 = W1.dot(inputs) + B1 # (nh x 784) * (784 x 1) = 16 x 1
    A1 = tanh(Z1) 
    Z2 = W2.dot(A1) + B2 # (10 x nh) * (nh x 1) = 10 x 1
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

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

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # y = correct label of the image

    # see how much the output layer, A2 needs to change compared to correct answer
    dZ2 = A2 - one_hot(Y)
    dW2 = dZ2.dot(A1.T) / m
    db2 = np.sum(dZ2) / m

    dZ1 = W2.T.dot(dZ2) * deriv_tanh(Z1)
    dW1 = dZ1.dot(X.T) / m
    db1 = np.sum(dZ1) / m
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, learning_rate, iterations, n_hidden):
    W1, b1, W2, b2 = init_weights_biases(n_hidden)

    for i in range(iterations):
        Z1, A1, Z2, A2 = feed_forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def make_predictions(inputs, W1, B1, W2, B2):
    _, _, _, A2 = feed_forward(inputs, W1, B1, W2, B2)
    predictions = get_predictions(A2)
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

import pandas as pd
import numpy as np

data = pd.read_csv("data/train.csv")
data = np.array(data)
# Structure of data:
# first column is the label of what the number actually is
# All the other columns are the values for each pixel, either 0 or 1
# Thus, one row is one image

def print_image_and_label(row):
    label = row[0]
    print(f"Label: {label}")
    
    img = row[1:]
    count = 0
    for pixel in img:
        if pixel != 0:
            print("⬜️", end='')
        else:
            print("⬛️", end='')
            
        count += 1

        if count % 28 == 0:
            print("\n", end='') # add a new line

# 748 inputs, 16 hidden, 10 output
w1 = [] # first weights = 748 * 16
b1 = [] # first biases = 16 for hidden nodes

w2 = [] # 16 * 10
b2 = [] # second biases = 10 for output nodes

def init_weights_biases():
    w1 = np.random.rand(16, 784) - 0.5 # 16 arrays with 748 random numbers from -0.5 to 0.5
    b1 = np.random.rand(16, 1) - 0.5
    w2 = np.random.rand(10, 16)
    b2 = np.random.rand(10, 1)
    return w1, b1, w2, b2

def tanh(x):
    return np.tanh(x)

def feed_forward(inputs, w1, b1, w2, b2):
    # run the inputs through the network
    h1 = []
    for i in range(16): # i = hidden layer node index
        # val = [inputs[j] + w1[i][j] for j in range(w1[i])] + b1[i]
        # val = (inputs @ w1[i]) + b1[i]
        total = 0
        for j in range(784):
            total += inputs[j] * w1[i][j]

        h1.append(tanh(total + b1[i])) # activation of the total with bias

    o1 = []
    for i in range(10):
        # val = (h1[j] + w2[i][j] for j in range(w2[i])) + b2[i]
        # val = (h1 @ w2[i]) + b2[i]
        total = 0
        for j in range(16):
            total += h1[j] * w2[i][j]

        o1.append(tanh(total + b2[i]))

    return o1

# @param o1 = the returned output layer, a list of 10 floats
# @param correct = the label of the image
def cost(o1, correct):
    total = 0
    for i, v in enumerate(o1):
        if i == correct:
            total += (v - 1) ** 2
        else:
            total += v ** 2 
    return total

for i in range(3):
    img = data[i]
    print_image_and_label(img)

    inputs = img[1:] # everything but the label
    w1, b1, w2, b2 = init_weights_biases()
    res = feed_forward(inputs, w1, b1, w2, b2)
    res = list(res)
    print(f"Prediction: { res.index(max(res)) }")
    print(f"Cost: {cost(res, img[0])}")
    print("————————————————————————————————————————————————————————\n")
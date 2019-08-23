import numpy as np
import random
from CreateData import makeDatoids

class NeuralNetwork:
    def __init__(self, layers):
        # Number of neurons in each layer in the form of an array i.e [8, 16, 16, 128]
        self.layers = layers
        self.num_layers = len(layers)

        # Create the biases in a format of a [y x 1] matrix for each layer
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        # Create the weights for each layer in the format [[neurons in layer 1 x neurons in layer 2][n in 1 x n in 3]...]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

        # Create the desired data set with given size and range of numbers
        # The range of each number in first layer: 1/2 (of the total first layer nodes) - 1 (bit reserved for - or +)
        self.training_data = makeDatoids(10, self.layers[0] // 2 - 1)
        self.learning_data = makeDatoids(3000, self.layers[0] // 2 - 1)




    # Take in an array of 1s and 0s as input and then return the output of network
    # This function's credit (as well as code for weights and biases) to https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
    def feedforward(self, net):

        for l_count, layer in enumerate(net):
            # Start at layer 1, and still have access to layer 0
            if l_count == 0:
                continue
            for a_count, a in enumerate(layer):
                w = self.weights[l_count - 1][a_count]
                b = self.biases[l_count - 1][a_count]
                # print('a: \n', net[l_count][a_count])
                # print('w: \n', weights[l_count - 1][a_count])
                # print('b: \n', biases[l_count - 1][a_count])
                z = np.dot(w, a) + b
                net[l_count][a_count] = sigmoid(add_array_elements(z))

        return net

    # Find the cost function for the results of the network
    def cost_function(self, output):
        cost = []
        # Compare expected result to the measured result
        for expected in self.training_data:
            for measured in output:
                cost.append(np.square(float(measured) - float(expected[1])))

        return cost

    # Learning: Using calculus in order to decrease our cost function
    def backprop(self):
        return None




    # # Return the output of the network if 'a' is input
    # def feedforward(self, a):
    #     for b, w in zip(self.biases, self.weights):
    #         # the input of the next layer is the output of the last layer
    #         a = sigmoid(np.dot(w, a) + b)
    #     return a
    #
    #

    #
    # def SGD(self, training_data, epochs, mini_batch_size, eta,
    #         test_data=None):
    #     """Train the neural network using mini-batch stochastic
    #     gradient descent.  The ``training_data`` is a list of tuples
    #     ``(x, y)`` representing the training inputs and the desired
    #     outputs.  The other non-optional parameters are
    #     self-explanatory.  If ``test_data`` is provided then the
    #     network will be evaluated against the test data after each
    #     epoch, and partial progress printed out.  This is useful for
    #     tracking progress, but slows things down substantially."""
    #     if test_data: n_test = len(test_data)
    #     n = len(training_data)
    #     for j in range(epochs):
    #         random.shuffle(training_data)
    #         mini_batches = [
    #             training_data[k:k + mini_batch_size]
    #             for k in range(0, n, mini_batch_size)]
    #         for mini_batch in mini_batches:
    #             self.update_mini_batch(mini_batch, eta)
    #         if test_data:
    #             print
    #             "Epoch {0}: {1} / {2}".format(
    #                 j, self.evaluate(test_data), n_test)
    #         else:
    #             print
    #             "Epoch {0} complete".format(j)
    #
    # def update_mini_batch(self, mini_batch, eta):
    #     """Update the network's weights and biases by applying
    #     gradient descent using backpropagation to a single mini batch.
    #     The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    #     is the learning rate."""
    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]
    #     for x, y in mini_batch:
    #         delta_nabla_b, delta_nabla_w = self.backprop(x, y)
    #         nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
    #         nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    #     self.weights = [w - (eta / len(mini_batch)) * nw
    #                     for w, nw in zip(self.weights, nabla_w)]
    #     self.biases = [b - (eta / len(mini_batch)) * nb
    #                    for b, nb in zip(self.biases, nabla_b)]
    #
    # def backprop(self, x, y):
    #     """Return a tuple ``(nabla_b, nabla_w)`` representing the
    #     gradient for the cost function C_x.  ``nabla_b`` and
    #     ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    #     to ``self.biases`` and ``self.weights``."""
    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]
    #     # feedforward
    #     activation = x
    #     activations = [x]  # list to store all the activations, layer by layer
    #     zs = []  # list to store all the z vectors, layer by layer
    #     for b, w in zip(self.biases, self.weights):
    #         z = np.dot(w, activation) + b
    #         zs.append(z)
    #         activation = sigmoid(z)
    #         activations.append(activation)
    #     # backward pass
    #     delta = self.cost_derivative(activations[-1], y) * \
    #             sigmoid_prime(zs[-1])
    #     nabla_b[-1] = delta
    #     nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    #     # Note that the variable l in the loop below is used a little
    #     # differently to the notation in Chapter 2 of the book.  Here,
    #     # l = 1 means the last layer of neurons, l = 2 is the
    #     # second-last layer, and so on.  It's a renumbering of the
    #     # scheme in the book, used here to take advantage of the fact
    #     # that Python can use negative indices in lists.
    #     for l in range(2, self.num_layers):
    #         z = zs[-l]
    #         sp = sigmoid_prime(z)
    #         delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
    #         nabla_b[-l] = delta
    #         nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
    #     return (nabla_b, nabla_w)
    #
    # def evaluate(self, test_data):
    #     """Return the number of test inputs for which the neural
    #     network outputs the correct result. Note that the neural
    #     network's output is assumed to be the index of whichever
    #     neuron in the final layer has the highest activation."""
    #     test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    #     return sum(int(x == y) for (x, y) in test_results)
    #
    # def cost_derivative(self, output_activations, y):
    #     """Return the vector of partial derivatives \partial C_x /
    #     \ partial a for the output activations."""
    #     return (output_activations - y)



# Adds elements of arrays together
def add_array_elements(arr):
    total = 0
    for n in range(0, len(arr)):
        total = total + arr[n]
    return total


# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# derivative of sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

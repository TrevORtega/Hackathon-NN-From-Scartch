import numpy as np
import random

from CreateData import *


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
        """ The range for each number in first layer is:
        1/2 of the total first layer nodes - 1 (the bit reserved for - or +)"""
        self.training_data = makeDatoids(10, self.layers[0] // 2 - 1)
        self.testing_data = makeDatoids(10, self.layers[0] // 2 - 1)

    # Take in an array of 1s and 0s as input and then return the output of network
    def feedforward(self, net):
        for l_count, layer in enumerate(net):
            # Start at layer 1, and still have access to layer 0
            if l_count == 0:
                continue
            for a_count, a in enumerate(layer):
                w = self.weights[l_count - 1][a_count]
                b = self.biases[l_count - 1][a_count]
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

    # Recursively move through the network, "teaching" the network how it should improve
    def backprop(self, net, desired, layers):
        """ Compares the cost function of ONE piece of training data given,
         then record the changes to the weights / biases for this example
         that would decrease the cost function """
        # There must be 2 layers in order to keep back propagating the network
        if len(layers) < 2:
            return None, None
        # Start on the last layer of the network
        l_count = len(layers) - 1
        # This is the actual output of neurons for the last layer
        actual = net[l_count]
        # Derivative of cost with respect to weight. Fill with empty zeros (n in layer L * n in layer L-1)
        d_weight = []
        # Derivative of cost with respect to the bias. Fill with empty zeros (n in layer L)
        d_bias = []
        #print('Net: \n', net[0], '\n', net[1], '\n', net[2])

        # Iterate through each neuron in the last layer
        for a_count, a in enumerate(actual):
            # The desired output for a given neuron
            y = desired[a_count]

            # The shorthand activation for this neuron
            z = self.get_activation(net, l_count, a_count)

            # The neurons in the previous layer
            a_1 = net[l_count - 1]

            # ADJUSTING THE WEIGHTS

            # Iterate through the weights of a single neuron in the last layer
            for w_count, w in enumerate(self.weights[l_count - 1][a_count]):
                # Record this neuron's desired changes to the weights. (dCost / dWeight)
                d_weight.append(a_1[w_count] * sigmoid_prime(z[0]) * (2 * (a - y)))

            # ADJUSTING THE BIAS

            # Record this neuron's desired change to the bias
            d_bias.append(sigmoid_prime(z) * (2 * (a - y)))

            # ADJUST PREVIOUS NEURONS

            """ For each weight in the last layer, 
            we record adjustments all the neurons in the previous layer"""
            temp_d_a = []
            # Iterate through each weight for a node in the last layer
            for w_count, w in enumerate(self.weights[l_count - 1]):
                neuron_changes = w * sigmoid_prime(z) * (2 * (a - y))
                # Can't add to an empty array
                if w_count == 0:
                    temp_d_a = neuron_changes
                    continue
                """ These represent the changes that the current neuron, a, wants to make
                to all the neurons in the previous layer (dCost / dA_1)"""
                temp_d_a = np.add(temp_d_a, neuron_changes)

        # Average the changes to the weir
        d_weight = np.divide(d_weight, len(actual))

        # Average the changes to the biases by dividing by number of neurons in last layer
        # For some reason deleting this ruins it
        d_bias = np.divide(d_bias, len(actual))
        # Average the changes to the previous layer neurons by dividing by # of neurons in last layer
        d_a = np.divide(temp_d_a, len(actual))

        # Back propagate the rest of the network and return the desired changes
        dw_1, db_1 = self.backprop(net, d_a, layers[:-1])

        # If nothing was returned, we just return our weight and bias changes
        if dw_1 is None or db_1 is None:
            return d_weight, d_bias

        ''' Add the changes from the rest of the network to our
        arrays for the changes to the weights and biases'''
        d_weight = [d_weight, dw_1]
        d_bias = [d_bias, db_1]

        # Return the proposed changes to the weights and biases
        return d_weight, d_bias

    # Calculate and return the activation for a single neuron
    def get_activation(self, net, l_count, a_count):
        # The shorthand for the activation of a neuron
        z = 0
        # The bias for the neuron we want
        b = self.biases[l_count - 1][a_count]
        # Iterate through every weight attached to a (the neuron's activation we want)
        for w_count, w in enumerate(self.weights[l_count - 1]):
            # The neuron attached to this weight
            a_1 = net[l_count][w_count]
            z += add_array_elements(np.dot(w, a_1))
        z += b
        return z

    # Using stochastic gradient descent in order to more efficiently minimize our overall cost function
    def SGD(self, mini_batch_size):
        """ Using mini batches of data, for each piece of training data,
        save the changes to the weights that the back propagation makes.
        Then average the changes over that specific batch of data and save it.
        Then average the changes to the weights over all mini batches
        for our final results. For this example, our data is already
        randomized so there is no need to shuffle it"""
        # The changes to the weights and biases
        weights = []
        biases = []
        num_mini_batches = len(self.training_data) // mini_batch_size

        for i in range(num_mini_batches):
            # The start and end of the next mini batch
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            # The learning rate of the network
            lr = 0.1
            mini_weights = []
            mini_biases = []

            # Creating the average changes to weights/biases for one mini-batch
            for d_count, data in enumerate(self.training_data[start:end]):
                input_neurons = self.make_addition_input(data[0])
                # Feed the network the data, and get back the neurons in the network
                net = self.feedforward(input_neurons)
                # Back propagate the network with the input data
                w, b = self.backprop(net, data[0], self.layers)
                # Add each w/b for each training example in this mini batch
                if d_count == 0:
                    mini_weights = w
                    mini_biases = b
                else:
                    mini_weights = np.add(mini_weights, w)
                    mini_biases = np.add(mini_biases, b)

            # Add the weights and biases of each mini batch together
            if i == 0:
                weights = mini_weights
                biases = mini_biases
            else:
                weights = np.add(weights, mini_weights * lr)
                biases = np.add(biases, mini_biases * lr)
        # Average the weights and biases for every mini batch
        weights = np.divide(weights, num_mini_batches)
        biases = np.divide(biases, num_mini_batches)

        # Reformat the weights and biases in the same shape as the network weights and biases
        weights, biases = self.reformat_weights_biases(weights, biases)

        # Update the network weights and biases
        self.weights = np.add(self.weights, weights)
        self.biases = np.add(self.biases, biases)

    """ Reformat the final weight and bias adjustment arrays 
    in order to subtract from the network weights and biases"""
    def reformat_weights_biases(self, w, b):
        # Reverse the order of the arrays
        w = w[::-1]
        b = b[::-1]

        # Make w the same shape as the network weights
        for i, weights in enumerate(self.weights):
            shape = np.shape(weights)
            w[i] = np.reshape(w[i], shape)

        # Make b the same shape as the network biases
        for j, biases in enumerate(self.biases):
            shape = np.shape(biases)
            b[j] = np.reshape(b[j], shape)

        return w, b

    # Format the addition binary data as input for the neural network
    def make_addition_input(self, num):
        # 2 number input in binary
        input_layer = [np.array(num)]

        # Random values for the neurons of the other layers
        random_layers = [np.random.randn(y) for y in self.layers[1:]]

        # Make sure the neurons are between 0 and 1
        # Not computationally efficient for now, but will work
        for array in random_layers:
            for i, x in enumerate(array):
                array[i] = sigmoid(x)

        # Combine the two arrays and return
        return input_layer + random_layers

    # Feed the network input and regurgitate the results
    def feed_and_regurgitate(self):
        # All the neurons in the network
        neurons = []
        # The results of the network (last layer of neurons)
        outin = []

        # Iterate through each tuple of numbers that are to be added together
        for nums in self.testing_data:
            # Feed the (2) numbers into the neural net as input, then return the network of neurons
            neurons = self.feedforward(self.make_addition_input(nums[0]))
            out = []
            # For each neuron in the last layer of the network
            for n in neurons[self.num_layers - 1]:
                out.append(sigmoid_activation(n))
            # Append the the last layers of neurons to output array, with the original input
            outin.append((np.array(out), nums[2]))

        # Return the results
        return outin

    # Calculates the percentage of correct answers
    def test_network(self):
        # Output and input of the network
        outin = self.feed_and_regurgitate()

        amount_correct = 0
        total = (len(outin) - 1)
        # Iterate through each tuple of output and input
        for tup in outin:
            if np.array_equal(tup[0], tup[1]):
                amount_correct += 1
        return round((amount_correct / total) * 100, 3), outin


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


# Determines if a sigmoid value should be activated
def sigmoid_activation(x):
    return 1 if x > 0.5 else 0

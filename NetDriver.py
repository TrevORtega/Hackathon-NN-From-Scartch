import NeuralNetwork
import numpy as np
from CreateData import Datoid

def main():
    #num1 = int(input("Enter 1st number to add"))
    #num2 = int(input("Enter 2nd number to add"))
    num1 = 5
    num2 = 3
    make_addition_network(num1, num2)


def make_addition_network(n1, n2):
    # Turn numbers into binary data
    #datoid = Datoid(n1, n2, 11)


    # Make a Neural Network
    nn = NeuralNetwork.NeuralNetwork([5, 4, 3])
    #nn = NeuralNetwork.NeuralNetwork([len(datoid.nums), 16, (len(datoid.nums) / 2) + 1])

    # First layer < 4. One byte for each number + One byte for negative or positive = 4 bytes = [0101]
    # Also last layer > 3. Must be able to write a 2 (e.g 1+1 = 2 = [001])
    if nn.layers[0] < 4 or nn.layers[len(nn.layers) - 1] < 3:
        raise Exception("First layer must have at least 4 neurons; Last layer must have at least 3 neurons")

    #inital_neurons = make_addition_input(nn.training_data[0], nn)

    # Iterate through each tuple of numbers that are to be added together
    for nums in nn.training_data:
        # Feed the numbers into the neural net as input
        nn.feedforward(make_addition_input(nums[0], nn))

# Format the addition binary data as input for the neural network
def make_addition_input(tup, nn):
    # 2 number input in binary
    input_layer = [np.array(tup[0], ndmin=2).transpose()]

    # Random values for the neurons of the other layers
    random_layers = [np.random.randn(y, 1) for y in nn.layers[1:]]

    # Make sure the neurons are between 0 and 1
    # Not computationally efficient for now, but will work
    for array in random_layers:
        for i, x in enumerate(array):
            array[i] = sigmoid(x)

    # Combine the two arrays and return
    return np.array([input_layer + random_layers])


# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# derivative of sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))



if __name__ == '__main__':
    main()

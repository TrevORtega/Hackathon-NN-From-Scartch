import NeuralNetwork
import numpy as np
import random
from CreateData import Datoid, binary_decode, binary_encode

def main():
    #num1 = int(input("Enter 1st number to add"))
    #num2 = int(input("Enter 2nd number to add"))
    num1 = 4
    num2 = 3
    make_addition_network(num1, num2)


def make_addition_network(n1, n2):
    # Turn numbers into binary data
    # datoid = Datoid(n1, n2, 11)

    # Make a Neural Network
    nn = NeuralNetwork.NeuralNetwork([4, 6, 3])
    # nn = NeuralNetwork.NeuralNetwork([len(datoid.nums), 16, (len(datoid.nums) / 2) + 1])

    # First layer >= 4. One bit for each number + One bit for negative or positive = 4 bits = [0101]
    # Also last layer > 3. Must be able to write a 2 (e.g 1+1 = 2 = [001])
    if nn.layers[0] < 4 or nn.layers[len(nn.layers) - 1] < 3:
        raise Exception("First layer must have at least 4 neurons; Last layer must have at least 3 neurons")

    # Feed the network the training data and get back the results
    output = feed_and_regurgitate(nn)

    cost_function = nn.cost_function(output)
    print(cost_function)

# Format the addition binary data as input for the neural network
def make_addition_input(tup, nn):
    # 2 number input in binary
    input_layer = [np.array(tup)]

    # Random values for the neurons of the other layers
    random_layers = [np.random.randn(y) for y in nn.layers[1:]]

    # Make sure the neurons are between 0 and 1
    # Not computationally efficient for now, but will work
    for array in random_layers:
        for i, x in enumerate(array):
            array[i] = sigmoid(x)

    # Combine the two arrays and return
    return input_layer + random_layers


# Feed the network input and regurgitate the results
def feed_and_regurgitate(nn):
    output = []

    # Iterate through each tuple of numbers that are to be added together
    for nums in nn.training_data:
        # Feed the (2) numbers into the neural net as input, get new (1) number out as output
        neurons = nn.feedforward(make_addition_input(nums[0], nn))
        # Append the final number to output array
        output.append(neurons[nn.num_layers - 1])

    # Translate the output of the network back into binary numbers and decode the binary numbers back to base 10
    for num_count, number in enumerate(output):
        for bit_count, bit in enumerate(number):
            # Turn number array to binary
            number[bit_count] = activation(bit)
        # Turn binary to base 10
        output[num_count] = binary_decode(number)

    # Return the results
    return output

# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# derivative of sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

# Enacts activation of sigmoid values
def activation(x):
    if x > 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    main()

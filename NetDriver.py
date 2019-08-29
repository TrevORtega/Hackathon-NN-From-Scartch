import NeuralNetwork
import numpy as np


def main():
    #num1 = int(input("Enter 1st number to add"))
    #num2 = int(input("Enter 2nd number to add"))
    num1 = 4
    num2 = 3
    # Initialize the network
    nn = create_addition_network(num1, num2)
    epoch = 5
    for i in range(epoch):
        results, outin = nn.test_network()
        print(results)
        nn.SGD(2)





    #np.save(w, b)

def create_addition_network(n1, n2):
    # Turn numbers into binary data
    # datoid = Datoid(n1, n2, 11)

    # Make a Neural Network
    nn = NeuralNetwork.NeuralNetwork([4, 6, 3])
    # nn = NeuralNetwork.NeuralNetwork([len(datoid.nums), 16, (len(datoid.nums) / 2) + 1])

    # First layer >= 4. One bit for each number + One bit for negative or positive = 4 bits = [0101]
    # Also last layer > 3. Must be able to write a 2 (e.g 1+1 = 2 = [001])
    if nn.layers[0] < 4 or nn.layers[len(nn.layers) - 1] < 3:
        raise Exception("First layer must have at least 4 neurons; Last layer must have at least 3 neurons")

    return nn

# sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# derivative of sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


if __name__ == '__main__':
    main()

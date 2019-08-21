import numpy as np
import NetDriver as nd
import NeuralNetwork
nn = NeuralNetwork.NeuralNetwork([3, 4, 2])

layers = [3, 4, 2]

# array slicing makes a new array
#print('layers[:-1]: ', layers[:-1])
#print('\nlayers[1:]', layers[1:])

# randn creates a new (m,n) array
biases = [np.random.randn(y,1) for y in layers[1:]]

#print("Biases:\n", biases)
#print("biases[0][3]:", biases[0][3])

zipLayer = zip(layers[:-1], layers[1:])

# print(set(zipLayer))

# zip maps elements of the same index together
weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

#print("Weights:\n", weights)

#print("weights[0]:", weights[0])

def binary_encode(num, num_range):
    # get the sign of the number and if it is negative, make it positive
    if (num >= 0):
        sign = 0
    else:
        sign = 1
        num = num * -1
    arr = np.array([sign] + [num >> d & 1 for d in range(num_range)])
    return arr


def binary_decode(num):
    # last digit is 1 if negative, else 0
    sign = num[0]

    # combined total of the binary digits
    total = 0

    # if num has a one in the ith place, add 2^i to the total
    for i in range(len(num)):
        if num[i] == 1:
            total = np.exp2(i-1) + total

    # make total negative if the sign == 1
    if sign == 1:
        return int(total * -1)

    return int(total)


num = binary_encode(-57, 8)
num1 = binary_encode(6, 8)


final = [num, num1]

a = nd.make_addition_input(nn.training_data[0], nn)

for w, b in weights, biases:
    print(np.dot(w, a) + b)

#print(np.dot(a, weights[0][0]))

#newTweet = '13/10 would push anytimepic.twitter.com/Tyzbjj5IWx I believe that this 21 Savage arrest is the most at peace while swinging.'

#newTweet.split(' ')

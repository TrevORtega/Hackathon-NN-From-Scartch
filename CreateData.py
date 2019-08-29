import random, os
import numpy as np

# CREATE DATA TO TRAIN A NEURAL NET CALCULATOR


class Datoid:
    def __init__(self, num1, num2, num_range):
        # Regular additon result
        self.result = num1 + num2
        # Result in binary
        self.binary_result = binary_encode(self.result, num_range + 1)

        # The amount of bits for each num in binary
        self.num_range = num_range

        self.num1 = binary_encode(num1, self.num_range)
        self.num2 = binary_encode(num2, self.num_range)

        self.nums = [self.num1, self.num2]


def binary_encode(num, num_range):
    # get the sign of the number and if it is negative, make it positive
    if num >= 0:
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
    for i in range(1, len(num)):
        if num[i] == 1:
            total = np.exp2(i - 1) + total

    # make total negative if the sign == 1
    if sign == 1:
        return int(total * -1)

    return int(total)


def dataToFile(datoids):

    # Add datoids to text file
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    f = open(os.path.join(__location__, "MathData.txt"), 'w')

    for tup in datoids:
        # Write the tuple
        f.write('\n' + ' '.join(str(s) for s in tup) + '\n')
        f.write(str(binary_decode(tup[0][:len(tup[0])//2])))
        f.write('           ')
        f.write(str(binary_decode(tup[0][len(tup[0])//2:])))

    f.close()


def makeDatoids(amount, binary_range):
    # Where we will store our data pieces
    datoids = []

    # Where we store the final combined binary arrays
    final = []

    # Stores 2 numbers that add to an answer
    for i in range(amount):

        # Create two random numbers in a desired range
        n1 = random.randint(-1 * (np.exp2(binary_range) - 1), np.exp2(binary_range) - 1)
        n2 = random.randint(-1 * (np.exp2(binary_range) - 1), np.exp2(binary_range) - 1)

        dato = Datoid(n1, n2, binary_range)

        # Go through the 2 binary number arrays in the datoid
        for arr in dato.nums:
            # Go through the numbers in the binary array
            for j in arr:
                final.append(j)

        # Append a tuple to datoids
        datoids.append((final, dato.result, dato.binary_result))
        final = []

    return datoids


dataToFile(makeDatoids(20, 3))
# Decode "nums" from Datoid (a 2 number binary array)
# print(binary_decode(final[:len(final)//2]), binary_decode(final[len(final)//2:]))

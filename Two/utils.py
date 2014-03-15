import sys
import math

def sigmoid(alpha):
    """Returns the value of the sigmoid function at a point alpha.
    """
    # print ("Alpha: " + str(alpha))
    if alpha > 20.:
        return 1. - 1.e-9

    if alpha < -20.:
        return 1.e-9

    return float(1) / (1. + math.exp(-alpha))

def scalar_product(one, two, d):
    """Returns scalar product of two vectors of dimension d.
    """

    res = 0.
    for i in range(d):
        if i in one and i in two:
            res += float(one[i]) * two[i]

    return res

def vector_sum(one, two, d):
    """Returns a sum of two vectors.
    """

    sum_ = {}
    for i in range(d):
        try:
            sum_[i] = one[i] + two[i]
        except:
            sum_[i] = 0.

    return sum_

def multiply_scalar(scalar, vector, d):
    """Returns a new vector that represents scalar * vector.
    """

    res = {}
    for i in range(d):
        try:
            res[i] = scalar * vector[i]
        except:
            res[i] = 0.

    return res

def vector_diff(one, two, d):
    """Returns difference of two vectors as a new vector.
    """

    res = {}
    for i in range(d):
        first = 0.
        second = 0.
        if i in one:
            first = one[i]
        if i in two:
            second = two[i]

        res[i] = first - second

    return res 

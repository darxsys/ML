import sys
import math
import numpy as np

def euclid(one, two):
    # sum_ = 0
    # for i in range(len(one)):
    #     sum_ += (one[i] - two[i]) ** 2
    # return sum_
    return np.linalg.norm(one - two)

def calc_error(k, classes, centers):
    """Calculates Kmeans error.
    """

    sum_ = 0.
    for i in range(k):
        for j in range(len(classes[i])):
            sum_ += euclid(centers[i][0], classes[i][j][0]) ** 2

    return sum_

def multivariate_probability(x, mi, sigma, dimension):
    """Calculates Gauss multivariate probability for x.
    """

    # print (x)
    # print (mi)
    # print (sigma)
    # print ("$$$")
    denominator = pow(pow(2 * np.pi, dimension), 0.5) * pow(np.linalg.det(sigma), 0.5)
    numerator = math.exp(-0.5 * np.dot(np.dot(np.transpose(x - mi), np.array(np.linalg.inv(sigma))), (x - mi)))

    # print (float(denominator))
    return float(numerator) / float(denominator)

def get_centers(k, examples):
    # if k == 2:
    centers = []
    center = next((obj for obj in examples if obj[1] == 'opel'), -1)
    if not center == -1:
        centers.append(center[:])

    center = next((obj for obj in examples if obj[1] == 'bus'), -1)
    if not center == -1:
        centers.append(center[:])

    if k >= 3:
        center = next((obj for obj in examples if obj[1] == 'van'), -1)
        if not center == -1:
            centers.append(center[:])

    if k == 4:
        center = next((obj for obj in examples if obj[1] == 'saab'), -1)
        if not center == -1:
            centers.append(center[:])

    if k == 5:
        num = 0
        for obj in examples:
            if obj[1] == 'saab':
                centers.append(obj[:])
                num += 1

                if num == 2:
                    break

    if not len(centers) == k:
        raise ValueError("Error while looking for centers. Not enough centers. K: " + str(k))
    return centers
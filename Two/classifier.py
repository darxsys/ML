import sys
import math
import numpy as np

import utils

def train(set_, dimension, lambda_):
    temp_w = np.zeros(dimension)
    w0 = 0.
    w = temp_w

    # print ("Lambda: " + str(lambda_))

    prev_error = 0.

    h = [0.5] * len(set_)
    current_error = calc_error(set_, w, lambda_, h, dimension)
    # num_iter = 0

    while abs(current_error - prev_error) > 0.001:
        delta_w0 = 0.
        delta_w = np.zeros(dimension)

        # print ("Current error: " + str(current_error))

        for i in range(len(set_)):
            h = utils.sigmoid(np.dot(set_[i][1], w) + w0)
            y = set_[i][0]

            delta_w0 += float(h) - y
            temp_x = (float(h) - y) * set_[i][1]
            delta_w = delta_w + temp_x

        n, error, w, w0 = line_search(set_, dimension, lambda_, w, w0,
            delta_w, delta_w0, current_error)

        # print ("Line search result: ")
        # print (str(w0) + " " + str(w))

        if n == 0:
            break

        # num_iter += 1

        prev_error = current_error
        current_error = error
        # print (current_error)

    # print ("Num iter: " + str(num_iter))
    # print ("params found: " + str(w0) + str(w))
    # print ("-------------------------------")
    # print()
    h = [float(utils.sigmoid(np.dot(tup[1], w) + w0)) 
        for tup in set_]
    return w, w0, calc_error(set_, w, 0, h, dimension)

def line_search(set_, dimension, lambda_, w, w0, delta_w, delta_w0, start_error):
    # print ("Lambda: " + str(lambda_))
    delta = 0
    n = 0.01
    step = 0.01
    if lambda_ >= 100:
        step = n / float(lambda_)
        n /= float(lambda_)
        
    last_error = start_error
    new_w = w
    new_w0 = w0
    last_w = w
    last_w0 = w0

    while True:
        # move
        new_w = (1-n*lambda_) * new_w - n * delta_w
        new_w0 -= n * delta_w0

        h = [float(utils.sigmoid(np.dot(tup[1], new_w) + new_w0)) 
            for tup in set_]
            
        error = calc_error(set_, new_w, lambda_, h, dimension)

        if error > last_error:
            break

        last_error = error
        last_w = new_w
        last_w0 = new_w0

        new_w0 = w0
        new_w = w

        delta += 1

        n += step
        if n > 1:
            break

    # print ("n: " + str(n))
    return delta, last_error, last_w, last_w0

def calc_error(set_, w, lambda_, h, dimension):
    """Calculates the error of the classifier with parameters w.
    """

    error = 0.
    for i in range(len(set_)):
        error += set_[i][0] * math.log(h[i]) + (1.-set_[i][0]) * math.log(1.-h[i])

    error = -error
    if lambda_ > 0:
        error += lambda_ / 2. * np.dot(w, w)

    return error

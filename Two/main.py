# -*- coding: utf-8 -*-
import sys
import numpy as np

from classifier import train
import utils

def parse_input(file_, dimension):
    """Inputs all data from a set file.
    """

    set_ = []
    with open(file_, "r") as f:
        for line in f:
            line = line.split()
            attr = np.zeros(dimension)
            for l in line[1:]:
                l = l.split(":")
                attr[int(l[0])] = float(l[1])

            set_.append((int(line[0]), np.array(attr)))

    return set_

def main(args):
    """Calls training and other functions. Needs 5 arguments.
    """

    word_file = args[0]
    training_file = args[1]
    validation_file = args[2]
    test_file = args[3]
    out_path = args[4]

    words = {}
    with open(word_file, "r") as f:
        i = 0
        for line in f:
            line = line.split()
            words[i] = line[0]
            i += 1

    dimension = len(words)
    training_set = parse_input(training_file, dimension)
    validation_set = parse_input(validation_file, dimension)
    test_set = parse_input(test_file, dimension)

    # first part
    w, w0, error = train(training_set, dimension, 0.)
    emp_error = calc_num_wrong(w, w0, training_set, dimension)
    print_weights(out_path + "tezine1.dat", w, w0, error, emp_error)

    # second part
    l = [0.1, 1., 5., 10., 100., 1000.]

    with open(out_path + "optimizacija.dat", "w") as f:
        best_error = error
        best_w = w
        best_w0 = w0

        num_wrong = calc_num_wrong(w, w0, validation_set, dimension)
        optimal = 0.
        f.write("\u03BB" + " = " + str(0) + ", " + str(num_wrong) + "\n")

        for lambda_ in l:
            w, w0, error = train(training_set, dimension, lambda_)
            num = calc_num_wrong(w, w0, validation_set, dimension)
            # if lambda_ == 1.:
            #     output_predictions(out_path + "pred_proba.dat", validation_set, w, w0)
            #     top_five = w.argsort()[-5:][::-1]
            #     with open(out_path + "rijeci_proba.txt", "w") as f2:
            #         for x in top_five:
            #             f2.write(words[x] + "\n")

            f.write("\u03BB" + " = " + str(lambda_) + ", " + str(num) + "\n")
            if num <= num_wrong:
                num_wrong = num
                optimal = lambda_

        f.write("optimalno: " + "\u03BB = " + str(optimal) + "\n")

    # third part
    training_set.extend(validation_set)
    w, w0, error = train(training_set, dimension, optimal)
    emp_error = calc_num_wrong(w, w0, training_set, dimension)
    print_weights(out_path + "tezine2.dat", w, w0, error, emp_error)

    top_twenty = w.argsort()[-20:][::-1]
    with open(out_path + "rijeci.txt", "w") as f:
        for x in top_twenty:
            f.write(words[x] + "\n")

    output_predictions(out_path + "ispitni-predikcije.dat", test_set, w, w0)

def calc_num_wrong(w, w0, set_, d):
    """Calculates percentage of wrong classifications on a set.
    """

    N = len(set_)
    wrong = 0.
    for ex in set_:
        h = utils.sigmoid(np.dot(w, ex[1]) + w0)
        if h >= 0.5:
            h = 1
        else:
            h = 0

        wrong += abs(h - ex[0])

    return wrong / float(N)

def output_predictions(out_path, set_, w, w0):
    """Outputs predictions of a set to a file.
    """

    N = len(set_)
    wrong = 0.
    # i = 1

    with open(out_path, "w") as f:
        for ex in set_:
            h = utils.sigmoid(np.dot(w, ex[1]) + w0)
            # f.write(str(h) + " ")
            if h >= 0.5:
                h = 1
            else:
                h = 0

            # if i == 55:
            #     print (utils.sigmoid(np.dot(w, ex[1]) + w0))
            # i += 1
            f.write(str(h) + "\n")
            wrong += abs(h - ex[0])
        f.write("Greška: %.2lf\n" % (wrong / float(N)))

def print_weights(path, w, w0, error, emp_error):
    """Prints weights along with EE and CEE to a file.
    """

    with open(path, "w") as f:
        f.write("%.2f\n" % w0)
        for x in w:
            f.write("%.2f\n" % x)

        f.write("EE: %.2lf\n" % emp_error)
        f.write("CEE: %.2lf\n" % error)

if __name__ == "__main__":
    if not len(sys.argv) == 6:
        raise ValueError("Not a good number of arguments.")

    main(sys.argv[1:])
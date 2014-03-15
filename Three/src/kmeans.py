import sys
import numpy as np 

import utils

EPS = 1e-3

def solve(list_k, examples, out_path):
    """ Does k-means for each k in the list of k_s for a particular set of examples.
    """

    # start_centers = utils.get_centers
    ret_centers = []
    size = len(examples[0][0])
    out_all = open(out_path + "/kmeans-all.dat", "w")
    out_k4 = open(out_path + "/kmeans-k4.dat", "w")
    buffer_ = ""
    buffer_2 = "#iteracije: J\n--\n"
    for k in list_k:
        centers = utils.get_centers(k, examples)
        num_iter = 0
        # labels = {}
        while True:
            # print (centers)
            num_iter += 1
            changed = 0
            classes = []
            for cent in centers:
                classes.append([])

            for count in range(len(examples)):
                ex = examples[count]
                bk = 0
                dist = utils.euclid(ex[0], centers[0][0])
                # labels 

                for i in range(1, len(classes)):
                    dist2 = utils.euclid(ex[0], centers[i][0])
                    if dist2 < dist:
                        bk = i
                        dist = dist2

                classes[bk].append(ex)
                if not bk == ex[2]:
                    changed = 1
                    examples[count][2] = bk
        
            if k == 4:
                buffer_2 += "#%d: " % (num_iter-1) + "%.2lf"\
                    % (utils.calc_error(k, classes, centers)) + "\n"

            if changed == 0:
                break

            for i in range(k):
                # print (i)
                # new_centers = []
                a = np.zeros(size, dtype=np.float64)
                for j in range(len(classes[i])):
                    a = a + classes[i][j][0]
                a = a / len(classes[i])
                centers[i][0] = a

        # print (sum_)
        sum_ = utils.calc_error(k, classes, centers)
        if k == 4:
            buffer_2 += "--\n"

        buffer_ += "K = " + str(k) + "\n"
        for i in range(k):
            buffer_ += ("c%d: " % (i+1))
            if k == 4:
                buffer_2 += "Grupa %d: " % (i+1)
                count = dict()
                for j in range(len(classes[i])):
                    if classes[i][j][1] not in count:
                        count[classes[i][j][1]] = 1
                    else:
                        count[classes[i][j][1]] += 1

                count = sorted(count.items(), key=lambda x:x[1], reverse=True)
                # print (count)
                for (key, val) in count:
                    # print (key,val)
                    buffer_2 += str(key) + " " + str(val) + ", "

                buffer_2 = buffer_2[:-2] + "\n"


            for j in range(size):
                buffer_ += "%.2lf" % centers[i][0][j]
                buffer_ += " "
            buffer_ = buffer_[:-1] + "\n"
            buffer_ += "grupa %d: " % (i+1)

            buffer_ += str(len(classes[i])) + " primjera\n"
        buffer_ += ("#iter: " + str(num_iter) + "\n")
        buffer_ += ("J: %.2lf" % (sum_) + "\n")
        buffer_ += ("--\n")

        if k == 4:
            out_k4.write(buffer_2)
            ret_centers = centers[:]

    out_all.write(buffer_[:-3])
    return ret_centers

        # out_all.write()

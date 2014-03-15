import sys
import numpy as np
import math

import utils

EPS = 1e-5

def calc_log(examples, mi, sigma, pi):
    # print (pi)
    dim = len(examples[0][0])
    sum_ln = 0.
    for i in range(len(examples)):
        sum_p = 0.
        for j in range(len(pi)):
            # print (sigma[j])
            # if len(pi) == 3:
            #     print (i, j)
            sum_p += pi[j] * utils.multivariate_probability(examples[i][0], mi[j][0], sigma[j], dim)
            # pass

        # print ("sum_p: " + str(sum_p))
        sum_ln += math.log(sum_p)

    return sum_ln
# 
def solve(list_k, examples, out_path, centers_=None,
        conf_num=None, out_minus=True, out_iterations=False):
    """Solves the clustering problem using EM algorithm.
    """

    # start_centers = utils.get_centers()
    dimension = len(examples[0][0])
    # print (dimension)
    N = len(examples)    # print(N)
    for k in list_k:
        # sys.stderr.write(str(k) + "\n")
        h = np.zeros((N, k), dtype=np.float64)
        # responsibilities.fill(1/float(k))
        pi_k = np.zeros(k, dtype=np.float64)
        pi_k.fill(1/float(k))
        # print
        configuration = False
        if centers_ == None:
            centers = utils.get_centers(k, examples)
        else:
            centers = centers_
            configuration = True

        matrix = np.identity(dimension, dtype=np.float64)
        matrices = np.array([matrix] * k)
        # print (matrices)
        log = calc_log(examples, centers, matrices, pi_k)
        # print (log)

        # output stuff
        num_iter = 0
        # if k == 4:
        # groups = [list([])] * k
        groups = []
        for i in range(k):
            groups.append([])
        probabilities = dict()

        # if k == 4:
        #     print (h)
        #     print (pi_k)
        #     print (matrices)
        #     print (centers)

        if out_iterations:
            iter_out = open(out_path + "em-kmeans.dat", "w")
            iter_out.write("#iteracije: log-izglednost\n")
            iter_out.write("--\n")
            iter_out.write("#0: %.2lf\n" % log)
        # buffer_all = "K = %d\n" % k
        # number of members in a certain group

        while True:
            num_iter += 1
            group_count = [0] * k
            # init = dict()
            group_dicts = []
            for i in range(k):
                group_dicts.append({})
            # group_dicts = [] * k
            # E step
            # print (num_iter)
            # changed = 0
            for i in range(N):
                x = examples[i][0]
                mi_j = centers[0][0]
                sigma_j = matrices[0]
                p = utils.multivariate_probability(x, mi_j, sigma_j, dimension) * pi_k[0]
                h[i][0] = p
                sum_h = h[i][0]
                probabilities[i] = p
                group = examples[i][1]
                if group not in group_dicts[0]:
                    group_dicts[0][group] = 1
                else:
                    group_dicts[0][group] += 1

                # if p > 
                group_count[0] += 1
                examples[i][2] = 0

                for j in range(1,k):
                    mi_j = centers[j][0]
                    sigma_j = matrices[j]
                    # print (sigma_j)
                    p2 = utils.multivariate_probability(x, mi_j, sigma_j, dimension) * pi_k[j]
                    if p2 > p:
                        group_count[examples[i][2]] -= 1
                        group_dicts[examples[i][2]][group] -= 1
                        if group not in group_dicts[j]:
                            group_dicts[j][group] = 1
                        else:
                            group_dicts[j][group] += 1
                        examples[i][2] = j
                        group_count[j] += 1
                        p = p2
                        probabilities[i] = p

                    h[i][j] = p2
                    sum_h += h[i][j]

                h[i] /= sum_h
                probabilities[i] /= sum_h
            # print ("$$$$$$$")
            # M step
            # if num_iter == 1:
            #     print (h)
            #     sys.exit(1)
            for i in range(k):
                new_mi = np.zeros(dimension, dtype=np.float64)
                sum_h = 0.
                new_sigma = np.zeros((dimension, dimension), dtype=np.float64)

                for j in range(N):
                    x = examples[j][0]
                    sum_h += h[j][i]
                    new_mi += h[j][i] * x

                new_mi /= sum_h
                # if utils.euclid(new_mi, centers[i][0]) > EPS:
                #     changed = 1
                centers[i] = (new_mi, centers[i][1], centers[i][2])

                # mi = np.matrix(new_mi)
                mi = new_mi
                for j in range(N):
                    x = examples[j][0]
                    # print ((x-mi).T * (x - mi))
                    diff = np.matrix(x - mi)
                    # print (diff.T * diff)
                    new_sigma += h[j][i] * (diff.T * diff)
                    # break

                new_sigma /= sum_h
                matrices[i] = new_sigma
                new_pi = sum_h / float(N)
                pi_k[i] = new_pi

            # print (matrices)
            # if num_iter == 1:
            #     sys.exit(1)

            new_log = calc_log(examples, centers, matrices, pi_k)

            # print (new_log)
            if abs(new_log - log) < EPS:
                break
            log = new_log
            # if k == 4:
            #     print (log)
            if out_iterations:
                iter_out.write("#%d: %.2lf\n" % (num_iter, log))
            # if changed == 0:
            #     break

        # set the groups for K = 4
        # print (groups)
        if not configuration and not out_iterations:
            if k == 4:
                for i in range(N):
                    # print (examples[i][2])
                    # print (groups)
                    groups[examples[i][2]].append((examples[i][1], probabilities[i]))

                with open(out_path + "/em-k4.dat", "a") as f:
                    for i in range(k):
                        # print (len(groups[i]))
                        groups[i] = sorted(groups[i], key=lambda tup: tup[1], reverse=True)
                        f.write("Grupa %d:\n" % (i+1))
                        for j in range(len(groups[i])):
                            f.write(groups[i][j][0] + " %.2lf\n" % groups[i][j][1])
                        if i < k-1:
                            f.write("--\n")

            with open(out_path + "/em-all.dat", "a") as f:
                f.write("K = %d\n" % k)
                for i in range(k):
                    f.write("c%d:" % (i+1))
                    for j in range(dimension):
                        f.write(" %.2lf" % centers[i][0][j])
                    f.write("\n")
                    f.write("grupa %d: %d primjera\n" % ((i+1), group_count[i]))
                f.write("#iter: %d\n" % num_iter)
                f.write("log-izglednost: %.2lf\n" % log)
                if k < 5:
                    f.write("--\n")
        elif not out_iterations:
            with open(out_path + "/em-konf.dat", "a") as f:
                f.write("Konfiguracija %d:\n" % conf_num)
                f.write("log-izglednost: %.2lf\n" % log)
                f.write("#iteracija: %d\n" % num_iter)
                if out_minus == True:
                    f.write("--\n")
                # if i < k - 1:
        else:
            iter_out.write("--\n")
            for i in range(len(group_dicts)):
                iter_out.write("Grupa %d:" % (i+1))
                buff = ""
                for key in group_dicts[i]:
                    buff += " " + key + " %d" % group_dicts[i][key] + ","

                buff = buff[:-1]
                iter_out.write(buff + "\n")
            # pass

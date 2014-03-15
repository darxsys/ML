import sys
import numpy as np 

import kmeans
import em

def main(argv):
    """Does initialization and calling of other methods for clustering.
    """

    list_of_k = [2,3,4,5]
    # list_of_k = [4,]
    silh_file = argv[1]
    conf_file = argv[2]
    out_path = argv[3]
    # examples = []
    with open(silh_file, 'r') as f:
        lines = f.readlines()
        examples = [[np.array(line.strip().split()[:-1], dtype=np.float64), \
            line.strip().split()[-1].strip(), 0] for line in lines]

    # print (examples)
    out_all = open(out_path + "/em-all.dat", "w")
    out_k4 = open(out_path + "/em-k4.dat", "w")
    out_all.close()
    out_k4.close()

    centers_k4 = kmeans.solve(list_of_k, examples, out_path)
    em.solve(list_of_k, examples, out_path)
    # configurations
    list_of_k = [4,]
    out = open(out_path + "/em-konf.dat", "w")
    out.close()

    with open(conf_file, "r") as f:
        lines = f.readlines()
        centers = [None] * 4
        for i in range(0, len(lines), 6):
            c_num = int(lines[i].strip().split()[-1][:-1])
            centers[0] = [np.array(lines[i+1].strip().split()[:-1], dtype=np.float64), lines[i+1].strip().split()[-1], 0]
            centers[1] = [np.array(lines[i+2].strip().split()[:-1], dtype=np.float64), lines[i+2].strip().split()[-1], 1]
            centers[2] = [np.array(lines[i+3].strip().split()[:-1], dtype=np.float64), lines[i+3].strip().split()[-1], 2]
            centers[3] = [np.array(lines[i+4].strip().split()[:-1], dtype=np.float64), lines[i+4].strip().split()[-1], 3]

            # try:
            if i + 6 >= len(lines) - 1:
                em.solve(list_of_k, examples, out_path, centers_=centers, conf_num = c_num, out_minus=False)
            else:
                em.solve(list_of_k, examples, out_path, centers_=centers, conf_num = c_num)
            # except:
                # continue
    # em.solve(list_of_k, examples, out_path, centers_=centers_k4, out_iterations=True)

if __name__ == "__main__":
    if not len(sys.argv) == 4:
        raise ValueError("Not a good number of arguments.")

    main(sys.argv)
    
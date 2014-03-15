import sys
import utils

with open("treniranje.dat", "r") as f:
    lines = f.readlines()
    out = open("modified.dat", "w")

    for line in lines[1:]:
        x = line.strip().split()
        x[2] = str(utils.default_classes.index(x[2]))
        out.write(" ".join(x) + "\n")


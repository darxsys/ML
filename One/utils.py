# -*- coding: utf-8 -*- 
import sys
import math
import numpy

default_classes = ["Narančasta", "Žuta", "Zelena", "Plava", "Tirkizna", "Indigo", "Modra", "Magenta"]

def multivariateProbability(x, mi, sigma, dimension):	
	denominator = pow(pow(2*numpy.pi, dimension), 0.5) * pow(numpy.linalg.det(sigma), 0.5)
	numerator = math.exp(-0.5 * numpy.dot(numpy.dot(numpy.transpose(x - mi),\
        numpy.array(numpy.linalg.inv(sigma))), (x - mi)))

	return float(numerator) / float(denominator)


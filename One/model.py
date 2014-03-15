# -*- coding: utf-8 -*- 
import sys
import numpy

import utils

class Model(object):
	"""Class that is a general super class for Bayesian classifier models."""

	def __init__(self, training_set, classes):
		self.training_set = training_set
		self.classes = classes
		self.class_probabilities = {}


		self.data_dimension = len(self.training_set[0]) - 1

		self.mi = {}
		self.generateParMi()
		# print (self.mi)

		self.S = {}
		self.generateParMatrix()
		# print (self.S)

	def generateParMi(self):
		mi = {}
		num_points = len(self.training_set)

		for class_ in self.classes:
			N = 0
			n = self.data_dimension
			mi_ = numpy.array([0.] * n)

			for point in self.training_set:
				if not class_ == point[-1]:
					continue

				N += 1
				point = numpy.array(point[:-1])
				mi_ = mi_ + point

				"""TODO: if N == 0"""
			mi_ = mi_ / float(N)
			mi[class_] = mi_

			# print ("Class " + class_ + " appears: " + str(N))
			self.class_probabilities[class_] = float(N) / num_points

		# print (mi)
		self.mi = mi

	def generateParMatrix(self):
		S = {}

		for class_ in self.classes:
			N = 0
			mi = numpy.matrix(self.mi[class_])
			# print (mi)

			n = self.data_dimension
			temp = [0.] * n
			S_ = numpy.matrix([temp] * n)

			for point in self.training_set:
				if not class_ == point[-1]:
					continue

				N += 1
				point = numpy.matrix(point[:-1])

				# print ("*****")
				# print ((point - mi))
				# print ((point - mi).T)
				# print ((point - mi).T * (point - mi))
				# print ("*****")
				
				S_ = S_ + (point - mi).T * (point - mi)

				# print (temp)
			# print (S_)
			S_ = S_ / float(N)
			S[class_] = S_

		self.S = S
		# print (S)

	def classify(self, example):
		p_x = 0.
		probs = {}
		max_ = 0
		max_class = ""

		for class_ in self.classes:
			mi = self.mi[class_]
			sigma = self.S[class_]
			# print (sigma)
			p_j = self.class_probabilities[class_]

			map_ = utils.multivariateProbability(example, mi, sigma, self.data_dimension) * p_j
			probs[class_] = map_
			p_x += map_

			if map_ > max_:
				max_ = map_
				max_class = class_

		for key in probs:
			probs[key] = probs[key] / float(p_x)

		return max_class, max_ / float(p_x), probs

class GeneralBayesianModel(Model):
	"""Class that models a general Bayesian classifier without simplifications."""

	def __init__(self, training_set, classes):
		Model.__init__(self, training_set, classes)
		# print (self.S)

	def generateParMi(self):
		Model.generateParMi(self)

	def generateParMatrix(self):
		Model.generateParMatrix(self)
		# print ("General matrix:")
		# print (self.S)

class SharedMatrixBayesianModel(Model):
	"""Class that models a Bayesian classifier in which all classes share one same 
	covariation matrix. Calculated like S = Sum_over_j P(Cj) * Sj. """

	def __init__(self, training_set, classes):
		Model.__init__(self, training_set, classes)
		self.setMatrix()
		# print (self.S)
		# print ("Shared matrix:")
		# print (self.S["Tirkizna"])

	def generateParMi(self):
		Model.generateParMi(self)

	def generateParMatrix(self):
		Model.generateParMatrix(self)	

	def setMatrix(self):
		n = self.data_dimension
		temp = [0.] * n
		S = numpy.matrix([temp,] * n)

		for class_ in self.classes:
			P = self.class_probabilities[class_]
			S_ = self.S[class_]

			S += P * S_

		for class_ in self.classes:
			self.S[class_] = S

class DiagonalMatrixBayesianModel(SharedMatrixBayesianModel):
	"""Class that models a Bayesian classifier that assumes variables are independent.
	Basically a naive Bayes for multivariat case."""

	def __init__(self, training_set, classes):
		SharedMatrixBayesianModel.__init__(self, training_set, classes)
		self.setMatrix()
		# print ("Diagonal matrix:")
		# print (self.S["Tirkizna"])
		# print (self.S)

	def generateParMi(self):
		Model.generateParMi(self)

	def generateParMatrix(self):
		Model.generateParMatrix(self)	

	def setMatrix(self):
		SharedMatrixBayesianModel.setMatrix(self)

		for class_ in self.classes:
			S_ = self.S[class_]
			n = S_.shape[0]

			for i in range(n):
				for j in range(n):
					if not i == j:
						S_[(i, j)] = 0.

			self.S[class_] = S_

class IsotropicMatrixBayesianModel(DiagonalMatrixBayesianModel):
	
	def __init__(self, training_set, classes):
		DiagonalMatrixBayesianModel.__init__(self, training_set, classes)
		self.setMatrix()
		# print ("Isotropic matrix:")
		# print (self.S["Tirkizna"])
		# print (self.S)

	def generateParMi(self):
		Model.generateParMi(self)

	def generateParMatrix(self):
		Model.generateParMatrix(self)	

	def setMatrix(self):
		DiagonalMatrixBayesianModel.setMatrix(self)

		for class_ in self.classes:
			S_ = self.S[class_]
			n = S_.shape[0]
			sum_ = 0.

			for i in range(n):
				for j in range(n):
					if not i == j:
						S_[(i, j)] = 0.
					else:
						sum_ += S_[(i,j)]

			sum_ /= float(n)
			for i in range(n):
				for j in range(n):
					if i == j:
						S_[(i,j)] = sum_

			self.S[class_] = S_
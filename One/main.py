# -*- coding: utf-8 -*- 
import sys
import getopt
import numpy

import utils
import model

def parseData(lines, num_classes=8):
	result = []
	classes = {}
	class_list = []

	for line in lines:
		line = line.split()
		elem = []

		# Insert all attributes
		for i in range(len(line[:-1])):
			elem.append(float(line[i]))

		line[-1] = line[-1].title()
		elem.append(line[-1])

		if num_classes > 8:
			if line[-1] not in classes:
				class_list.append(line[-1])

		classes[line[-1]] = 0
		result.append(elem)

	return result, class_list

def evaluateModel(model, sample_set, filename=None, class_list=[]):
	N = 0
	N_wrong = 0
	least_certain = []
	header = "\t".join(class_list)

	if not filename == None:
		f = open(filename, "w")
		print("%s\tklasa" % header, file=f)

	for sample in sample_set:
		N += 1
		sample_class = sample[-1]
		sample = numpy.array(sample[:-1])
		out_class, probability, probs = model.classify(sample)

		if not filename == None:
			buff = ""
			for clas in class_list:
				buff += "%.2lf\t" % probs[clas]
			buff += out_class

			print("%s" % buff, file=f)

			least_certain.append((probability, buff))

		if not sample_class == out_class:
			N_wrong += 1

	if filename == "../output/opceniti.dat":
		least_certain = sorted(least_certain, key=lambda tup:tup[0])
		least_certain = least_certain[:5]

		f = open("../output/nejednoznacne.dat", "w")
		print("%s\tklasa" % header, file=f)

		for (x, y) in least_certain:
			print("%s" % y, file=f)

	return N_wrong / float(N)

def createAndEvaluateModels(training_set, generalization_set, classes):
	one = model.GeneralBayesianModel(training_set, classes)
	two = model.SharedMatrixBayesianModel(training_set, classes)
	three = model.DiagonalMatrixBayesianModel(training_set, classes)
	four = model.IsotropicMatrixBayesianModel(training_set, classes)


	# for class_ in classes:
	# 	print (class_)
	# 	print ("mi")
	# 	print (one.mi[class_])
	# 	print ("Sigma")
	# 	print (one.S[class_])



	# print (two.S["Tirkizna"])
	# print (three.S["Tirkizna"])
	# print (four.S["Tirkizna"])


	f = open("../output/greske.dat", "w")

	print("opceniti\t%.2lf\t%.2lf" % (evaluateModel(one, training_set), 
		evaluateModel(one, generalization_set, "../output/opceniti.dat", classes)), file=f)
	# print ("%.2lf\n" % (evaluateModel(one, training_set),)) 

	print("dijeljena\t%.2lf\t%.2lf" % (evaluateModel(two, training_set), 
		evaluateModel(two, generalization_set, "../output/dijeljena.dat", classes)), file=f)

	print("dijagonalna\t%.2lf\t%.2lf" % (evaluateModel(three, training_set), 
		evaluateModel(three, generalization_set, "../output/dijagonalna.dat", classes)), file=f)

	print("izotropna\t%.2lf\t%.2lf" % (evaluateModel(four, training_set), 
		evaluateModel(four, generalization_set, "../output/izotropna.dat", classes)), file=f)

	# # print (evaluateModel(two, training_set))
	# print (evaluateModel(three, training_set))
	# print (evaluateModel(four, training_set))

	# print (evaluateModel(two, generalization_set))
	# print (evaluateModel(three, generalization_set))
	# print (evaluateModel(four, generalization_set))

def getArguments(argv):
	# print(argv)
	try:
		opts, args = getopt.getopt(argv,"input0:input1",["input0=","input1="])
	except getopt.GetoptError:
		print ('main.py --input0=<path>  --input1=<path>')

	training = ""
	generalization = ""

	for opt, arg in opts:
		# print (opt)
		if opt == '-h':
			print ('main.py -input0=<path>  -input1=<path>')
			sys.exit(1)
		elif opt == "--input0":
			training = arg
		elif opt == "--input1":
			generalization = arg

	return training, generalization

def main():
	if not len(sys.argv) == 3:
		print ("Need a file with a test and a generalization set.")
		sys.exit(1)

	training_file, generalization_file = getArguments(sys.argv[1:])

	training = open(training_file, mode="r", encoding='utf-8')
	generalization = open(generalization_file, mode="r", encoding='utf-8')
	data = training.readline()

	# print (lines)
	num_points = int(data.split()[0])
	num_classes = int(data.split()[1])

	default_classes = utils.default_classes
	training_set, class_list = parseData(training.readlines())

	generalization.readline()
	generalization_set, class_list = parseData(generalization.readlines(), num_classes)

	# TODO: Isprobaj ovaj dio
	temp = []
	if num_classes > 8:
		for class_ in class_list:
			if class_ not in default_classes:
				temp.append(class_)

	temp.sort()
	default_classes += temp
	# print (default_classes)
	createAndEvaluateModels(training_set, generalization_set, default_classes)

if __name__ == "__main__":
	main()


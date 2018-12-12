import pandas as pd
import numpy as np
from numpy import genfromtxt
import math
import sys
import csv
import os.path

RED = '\033[1;38;2;225;20;20m'
WHITE = '\033[1;38;2;255;251;214m'
YELLO = '\033[1;38;2;255;200;0m'
ORANGE = '\033[1;38;2;255;120;10m'
GREEN = '\033[1;38;2;0;175;117m'

EPS = 0.1

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))

def d_sigmoid(x):
	return x * (1 - x)

def softmax(x):
    return np.exp(x) / float(sum(np.exp(x)))

def min(arr):
	m = float('inf')
	for i in range(len(arr)):
		if arr[i] < m:
			m = arr[i]
	return m

def max(arr):
	m = float('-inf')
	for i in range(len(arr)):
		if arr[i] > m:
			m = arr[i]
	return m

def normalize(arr):
	mul = (arr.max() - arr.min()) / 2
	arr /= mul
	shift = arr.min() + 1
	arr -= shift
	return (arr, mul, shift)

class NeuralNetwork:
	def __init__(self, inp, expt):
		self.NLayer1 = 16
		self.NLayer2 = 9
		self.input = inp
		self.L1Weights = 2 * np.random.rand(26, self.NLayer1) - 1
#		self.L1Bias = 2 * np.random.rand(1, self.NLayer1) - 1
		self.L2Weights = 2 * np.random.rand(self.NLayer1, self.NLayer2) - 1
#		self.L2Bias = 2 * np.random.rand(1, self.NLayer2) - 1
		self.L3Weights = 2 * np.random.rand(self.NLayer2, expt.shape[1]) - 1
#		self.L3Bias = 2 * np.random.rand(1, expt.shape[1]) - 1
		self.y = expt
		self.output = np.zeros(self.y.shape)

	def feedforward(self):
		self.z1 = np.dot(self.input, self.L1Weights)# + self.L1Bias
		self.layer1 = sigmoid(self.z1)
		self.z2 = np.dot(self.layer1, self.L2Weights)# + self.L2Bias
		self.layer2 = sigmoid(self.z2)
		self.z3 = np.dot(self.layer2, self.L3Weights)# + self.L3Bias

#		print(self.L3Bias);
		self.output = sigmoid(self.z3)
		self.diff = self.y - self.output
		print("Cost ")
		print(np.dot(self.diff.T, self.diff))

	def backprop(self):
		l3e = 2 * (self.y - self.output)
		d_weights3 = l3e * d_sigmoid(self.z3)
		l2e = np.dot(d_weights3, self.L3Weights.T)
		d_weights2 = l2e * d_sigmoid(self.z2)
		l1e = np.dot(d_weights2, self.L2Weights.T)
		d_weights1 = l1e * d_sigmoid(self.z1)

		# print(self.input.T.dot(d_weights1))
		self.L1Weights += self.input.T.dot(d_weights1)
		self.L2Weights += self.layer1.T.dot(d_weights2)
		self.L3Weights += self.layer2.T.dot(d_weights3)
		print(self.L3Weights)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(ORANGE + "Usage: " + sys.argv[0] + " file.csv")
		exit(1)

	sys.stdout.write(WHITE)
	print("...Importing Data...")

	if os.path.isfile(sys.argv[1]) == 0:
		print(RED + "404 File not found >.< Can't help you there...")
		sys.exit(1)

	data = pd.read_csv(sys.argv[1], names = ["Index", "Type", "1", "2", "3", "4",
		"5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
		"18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"])

	print("...Cleaning Data...")

	data = data.drop(['Index', '3', '4', '23', '24'], axis = 1)

	f = open("result", "w")

	attr = ['1','2','5','6','7','8','9','10','11','12','13','14','15','16','17',
	'18','19','20','21','22','25','26','27','28','29','30']

	m = dict()
	c = dict()
	for att in attr:
		data[att], m[att], c[att] = normalize(data[att])
		print(m[att], file = f)
		print(c[att], file = f)

	X = data.as_matrix(columns=attr)
	y = np.zeros((data.shape[0], 1), dtype=float)
	i = 0
	for (i, row) in data.iterrows():
		if row['Type'] is 'M':
			y[i] = [1]
		else:
			y[i] = [0]
		i += 1


	nn = NeuralNetwork(X, y)
	for i in range(5):
		nn.feedforward()
		nn.backprop()

	# print(nn.y)
	# print(nn.output)

	f.close()
	print("Done!")

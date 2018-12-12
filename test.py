import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from numpy import genfromtxt
import sys
import csv
import os.path

RED = '\033[1;38;2;225;20;20m'
WHITE = '\033[1;38;2;255;251;214m'
YELLO = '\033[1;38;2;255;200;0m'
ORANGE = '\033[1;38;2;255;120;10m'
GREEN = '\033[1;38;2;0;175;117m'

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
f = open(sys.argv[1], 'rt')

print("...Computing Graphs...")

sns.set(style="ticks")
g = sns.PairGrid(data, hue="Type")
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter, s=3)
g = g.add_legend()

sys.stdout.write(WHITE)
print("...Outputting...")

g.savefig("pairplot.png")

sys.stdout.write(GREEN)
print("DONE!!!Output saved as pairplot.png")

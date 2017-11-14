import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("paths", nargs="+")
args = parser.parse_args()
for path in args.paths:
	csv = pd.read_csv(path)
	csv = csv.drop(['Seconds', 'LearningRate'], axis=1)
	csv.plot(x=0, title=path)
plt.show()


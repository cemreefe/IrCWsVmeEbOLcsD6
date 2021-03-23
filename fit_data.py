import argparse
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from functions import fit_classifiers
from functions import show_results

parser = argparse.ArgumentParser(description='Apziva Ignite Project - I', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-p", "--plot", help="use matplotlib and seaborn to display data", action='store_true')
parser.add_argument("-d", "--data", help="path of the dataset (in csv format)", default="data/ACME-HappinessSurvey2020.csv")

args = parser.parse_args()

# if plotting will be done or not
plotting = getattr(args, "plot")

# read data
df = pd.read_csv(args.data)

# labels
y = np.array(df['Y'])

# delete labels from the dataset
del df['Y']

# features
X = np.array(df)
X = StandardScaler().fit_transform(X)

# feature names
ftr_names = list(df.columns)

print("""
==========================================
FEATURE CORRELATIONS
==========================================
""")

correlation = df.corr()
print(correlation)

if plotting:
	# Correlation heatmap
	plt.figure(figsize=(10, 10))
	sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
	plt.title("Correlation matrix of provided features")
	plt.show()
	
print()
print("Calculating the average of all correlations, we can approximate the feature importances.")
print(df.mean(axis = 0))

# delete dataframe
del df

print()
print("X and y shapes:", X.shape, y.shape)


	
print("""
==========================================
TRAINING USING ALL POINTS IN THE DATASET
==========================================
""")
	
accuracies, classifiers, fi = fit_classifiers(X, y, X, y)

show_results(accuracies, classifiers, fi, ftr_names, 'all', plotting)

print(f"Feature '{ftr_names[np.argmin(fi)]}' provides the least information (for XGB approach), therefore it is the best candidate to be removed.")

print("""
==========================================
TRAINING ON TRAIN/TEST SPLIT DATASET (0.2)
==========================================
""")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

accuracies, classifiers, fi = fit_classifiers(X_train, y_train, X_test, y_test)

show_results(accuracies, classifiers, fi, ftr_names, '0.2 train/test split', plotting)

print()
print("We can see that when the data is split into train and test sets, there is a dignificant decrease in accuracy (to around 50%). This might be a sign that we are not actually learning a valid classification on the data, but instead learning the data itself by overfitting.")
print("")




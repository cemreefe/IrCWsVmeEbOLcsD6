import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# read data
df = pd.read_csv("data/ACME-HappinessSurvey2020.csv")

# labels
y = np.array(df['Y'])

# delete labels from the dataset
del df['Y']

# features
X = np.array(df)
X = StandardScaler().fit_transform(X)

# feature names
ftr_names = list(df.columns)

# Correlation heatmap
correlation = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title("Correlation matrix of provided features")
plt.show()

# delete dataframe
del df

print("X and y shapes:", X.shape, y.shape)

# classifiers to feed the data
classifiers = {
    "KNearest":      KNeighborsClassifier(3), 
    "RBF SVM":       SVC(gamma=2, C=1),
    "Decision Tree": DecisionTreeClassifier(max_depth=5), 
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), 
    "Neural Net":    MLPClassifier(alpha=1, max_iter=1000), 
    "XGB":           xgb.XGBClassifier(),   
}

accuracies = []

# print accuracies by model
print("accuracies:")
for key, model in classifiers.items():
    model.fit(X, y)
    accuracies.append(model.score(X, y)*100)
    print(f"{accuracies[-1]:.2f}% - {key}")
    
    
# plot accuracies
x_pos = range(len(accuracies))
plt.gca().yaxis.grid()
plt.bar(x_pos, accuracies, color='thistle')

plt.xticks(x_pos, classifiers.keys(), rotation=20)
plt.title("Accuracies by Model")
plt.xlabel("model")
plt.ylabel("accuracy (%)")
plt.ylim(0,100)
plt.show()

# print feature importances
print("\nfeature\timportance")
for ftr in list(zip(ftr_names, model.feature_importances_)):
    print(f"{ftr[0]}:\t{ftr[1]:.2f}")

# plot feature importances
x_pos = range(len(model.feature_importances_))
plt.bar(x_pos, model.feature_importances_, color='lightblue')
plt.xticks(x_pos, ftr_names)
plt.title("Feature Importances (for XGBClassifier)")
plt.xlabel("feature")
plt.ylabel("importance")
plt.show()

print(f"Feature '{ftr_names[np.argmin(model.feature_importances_)]}' provides the least information (for XGB approach), therefore it is the best candidate to be removed.")




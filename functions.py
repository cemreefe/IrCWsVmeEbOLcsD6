from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import matplotlib.pyplot as plt

def bars(value, hi, lo, length=20):
	bars = int(value)//int((hi-lo)/length)
	return '▄'*bars + '_'*(20-bars)

def fit_classifiers(X_train, y_train, X_test, y_test):

	# classifiers to feed the data
	classifiers = {
	    "KNearest":      KNeighborsClassifier(3), 
	    "RBF SVM":       SVC(gamma=2, C=1),
	    "Decision Tree": DecisionTreeClassifier(max_depth=5), 
	    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), 
	    "Neural Net":    MLPClassifier(alpha=1, max_iter=1000), 
	    "XGB":           xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),   
	}

	accuracies = []

	# print accuracies by model

	for key, model in classifiers.items():
	    model.fit(X_train, y_train)
	    accuracies.append(model.score(X_test, y_test)*100)
	    
	return accuracies, list(classifiers.keys()), model.feature_importances_

def show_results(accuracies, classifiers, fi, ftr_names, data_mode='all', plotting=False):
	
	print(f"Accuracies for different approaches (trained on {data_mode} data):")
	for accuracy, classifier_name in zip(accuracies, classifiers):
		print(f"{bars(accuracy, 100, 0)} {accuracy:.2f}% - {classifier_name}")
	    
	
	if plotting:    
		# plot accuracies
		x_pos = range(len(accuracies))
		plt.gca().yaxis.grid()
		plt.bar(x_pos, accuracies, color='thistle')

		plt.xticks(x_pos, classifiers, rotation=20)
		plt.title(f"Accuracies by Model (trained on {data_mode} data)")
		plt.xlabel("model")
		plt.ylabel("accuracy (%)")
		plt.ylim(0,100)
		plt.show()

	# print feature importances
	print(f"\nfeature\timportance (trained on {data_mode} data)")
	for name, importance in list(zip(ftr_names, fi)):
	    print(f"{bars(importance*100, 100, 0)} {importance:.2f} - {name}")

	if plotting:
		# plot feature importances
		x_pos = range(len(fi))
		plt.bar(x_pos, fi, color='lightblue')
		plt.xticks(x_pos, ftr_names)
		plt.title(f"Feature Importances (for XGBClassifier, trained on {data_mode} data)")
		plt.xlabel("feature")
		plt.ylabel("importance")
		plt.show()

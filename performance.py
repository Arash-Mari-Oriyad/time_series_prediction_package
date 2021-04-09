# performance function
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn import metrics
import matplotlib.pyplot as plt

# mean absolute error
def mae(true_values, predicted_values):
	return np.mean(np.abs(true_values - predicted_values))

# mean absolute percentage error
def mape(true_values, predicted_values):
	return np.mean(np.abs((true_values - predicted_values) / true_values))

# mean squared error
def mse(true_values, predicted_values):
	return np.mean(np.square(true_values - predicted_values))

# mean absolute scaled error
def mase(true_values, predicted_values, trivial_values):
	return (mae(true_values, predicted_values))/(mae(true_values, trivial_values))

# ROC AUC for binary, multiclass and multilabel classification
def auc(true_values, predicted_values, labels=None):
	auc = metrics.roc_auc_score(true_values, predicted_values, labels)
	false_positive_rate, true_positive_rate, _ = metrics.roc_curve(true_values, predicted_values)

	plt.figure(dpi=100)
	plt.axis('scaled')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.title("AUC & ROC Curve")
	plt.plot(false_positive_rate, true_positive_rate, 'g')
	plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
	plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.show()

	return auc

def aupr(true_values, predicted_values, pos_label):
	# Data to plot precision - recall curve
	precision, recall, _ = metrics.precision_recall_curve(true_values, predicted_values, pos_label)
	# Use AUC function to calculate the area under the curve of precision recall curve
	auc_precision_recall = metrics.auc(recall, precision)

	plt.figure(dpi=100)
	plt.axis('scaled')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.title("PR ROC Curve")
	plt.plot(recall, precision, 'g')
	plt.fill_between(recall, precision, facecolor='lightgreen', alpha=0.7)
	plt.text(0.95, 0.05, 'AUPR = %0.4f' % auc_precision_recall, ha='right', fontsize=12, weight='bold', color='blue')
	plt.xlabel("Recall/True Positive Rate")
	plt.ylabel("Precision")
	plt.show()

	return auc_precision_recall

def performance(true_values, predicted_values, performance_measures=['MAPE'], trivial_values=[], method="normal", period=7, 
	labels=None, pos_label=None):
	"""
	This function receives true_values and predicted_values
	and a list of performance measures as input from the user
	and returns the performance measures between true_values and predicted_values.

	*****
	Parameters:

	true_values : list<float>
		ground truth for target values.
	predicted_values : list<float>
		predicted values for target.
	performance_measures : list<string>, default=['MAPE']
		a list of performance measures.
	verbose : int
		the level of produced detailed logging information.
	*****

	*****
	Returns:

	errors : list<float>
		a list of errors based on the ‘performace_measures’ input and between true_values and predicted_values.
		if ture_values contains at least one zero value then for ‘MAPE’, a warning for division by zero will be displayed.
	*****
	"""
	if method.lower() == "cumulative":
		# calculate cumulative sum
		true_values = np.cumsum(true_values)
		predicted_values = np.cumsum(predicted_values)
	elif method.lower() == "moving_average":
		# calculate moving average
		true_values = pd.Series(true_values).rolling(window=period).mean().iloc[period-1:].values.tolist()
		predicted_values = pd.Series(predicted_values).rolling(window=period).mean().iloc[period-1:].values.tolist()
	
	errors = []

	true_values = np.asarray(true_values)
	predicted_values = np.asarray(predicted_values)
	trivial_values = np.asarray(trivial_values)

	for error_type in performance_measures:
		if error_type.lower() == 'mae':
			errors.append(mae(true_values, predicted_values))
		elif error_type.lower() == 'mape':
			errors.append(mape(true_values, predicted_values))
		elif error_type.lower() == 'mse':
			errors.append(mse(true_values, predicted_values))
		elif error_type.lower() == 'r2_score':
			errors.append(r2_score(true_values, predicted_values))
		elif error_type.lower() == 'mase':
			errors.append(mase(true_values, predicted_values, trivial_values))
		elif error_type.lower() == 'auc':
			errors.append(auc(true_values, predicted_values, labels))
		elif error_type.lower() == 'aupr':
			errors.append(aupr(true_values, predicted_values, labels))

	return errors
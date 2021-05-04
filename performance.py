# performance function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import metrics
from math import log

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

def aic_regression(y_true, y_pred, k):
	# k = number of independent variables to build model
	mse_error = mse(y_true, y_pred)
	aic = 2*k - 2*log(mse_error)
	return aic

def bic_regression(y_true, y_pred, k):
	# k = number of independent variables to build model
	# n = sample size (#observations)
	n = len(y_true)
	mse_error = mse(y_true, y_pred)
	bic = k*log(n) - 2*log(mse_error)
	return bic

def aic_classification(y_true, y_pred, k):
	# k = number of independent variables to build model
	mse_error = metrics.log_loss(y_true, y_pred)
	aic = 2*k - 2*log(mse_error)
	return aic

def bic_classification(y_true, y_pred, k):
	# k = number of independent variables to build model
	# n = sample size (#observations)
	n = len(y_true)
	mse_error = metrics.log_loss(y_true, y_pred)
	bic = k*log(n) - 2*log(mse_error)
	return bic

def performance(true_values, predicted_values, performance_measures=['MAPE'], trivial_values=[], model_type='regression', num_params=1, labels=None, pos_label=None):
	"""
	true_values : ground truth for target values (list or array)
	predicted_values : predicted values for target (list or array)
	performance_measures : a list of performance measures (list or array)
	trivial_values : just use this when want to calculate 'MASE' (list or array)
	model_type : type of model used for solving the problem, just needed when want to calculate 'AIC' or 'BIC' (string)
	num_params : number of independent variables to build model, just use it when want to calculate 'AIC' or 'BIC' (int)
	labels : just for multiclass classification and when want to calculate 'AUC' (List of labels that index the classes in predicted_values)
	pos_label : the label of positive class, just use it when want to calculate 'AUPR' (int or string)
	"""
	
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
			errors.append(aupr(true_values, predicted_values, pos_label))
		elif error_type.lower() == 'aic':
			if model_type == 'regression':
				errors.append(aic_regression(true_values, predicted_values, num_params))
			elif model_type == 'classification':
				errors.append(aic_classification(true_values, predicted_values, num_params))
		elif error_type.lower() == 'bic':
			if model_type == 'regression':
				errors.append(bic_regression(true_values, predicted_values, num_params))
			elif model_type == 'classification':
				errors.append(bic_classification(true_values, predicted_values, num_params))

	return errors

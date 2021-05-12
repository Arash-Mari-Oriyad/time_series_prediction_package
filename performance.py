# performance function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.preprocessing import label_binarize
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

# ROC AUC for binary and multiclass classification
def auc(true_values, y_score, labels=None):
	# check if the labels are related to binary classification or not
	binary_classification = (len(labels) == 1 or len(labels) == 2)

	if binary_classification == True:
		auc = metrics.roc_auc_score(
			y_true=true_values, 
			y_score=y_score[:, 1], 
			labels=labels
		)
	else:
		auc = metrics.roc_auc_score(
			y_true=true_values, 
			y_score=y_score, 
			multi_class='ovo', 
			labels=labels
		)

	# Data to plot roc curve
	false_positive_rate, true_positive_rate, _ = metrics.roc_curve(true_values, y_score)

	# plot roc curve
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

def aupr(true_values, probas_pred, labels):
	# detecting how many classes the classification problem have
	number_of_classes = len(labels)
	
	# multiclass classification
	if number_of_classes > 2:
		# labelize the true_values to be usable in 'precision_recall_curve' from 'sklearn.metrics'
		true_values = label_binarize(true_values, classes=labels)
	
		auc_precision_recall = 0
		precision = dict()
		recall = dict()

		for i in range(number_of_classes):
			# Data to plot precision - recall curve
			precision[i], recall[i], _ = metrics.precision_recall_curve(true_values[:, i], probas_pred[:, i])
			# Use AUC function to calculate the area under the curve of precision recall curve
			auc_precision_recall += metrics.auc(recall[i], precision[i])

		# the final aupr value is mean of aupr between all classes
		auc_precision_recall /= (i+1)

		# plot the precision - recall graph
		plt.figure(dpi=100)
		plt.axis('scaled')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.title("PR ROC Curve")
		for i in range(number_of_classes):
			plt.plot(recall[i], precision[i], color='g', label='class {}'.format(labels[i]))
			plt.fill_between(recall[i], precision[i], facecolor='lightgreen', alpha=0.7)
		plt.text(0.95, 0.05, 'AUPR = %0.4f' % auc_precision_recall, ha='right', fontsize=12, weight='bold', color='blue')
		plt.xlabel("Recall/True Positive Rate")
		plt.ylabel("Precision")
		plt.show()
	
		return auc_precision_recall
	
	# binary classification
	else:
		# Data to plot precision - recall curve
		precision, recall, _ = metrics.precision_recall_curve(true_values, probas_pred)
		# Use AUC function to calculate the area under the curve of precision recall curve
		auc_precision_recall = metrics.auc(recall, precision)

		# plot the precision vs recall graph
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

def performance(
		true_values, predicted_values, 
		performance_measures=['MAPE'], trivial_values=[], 
		model_type='regression', num_params=1, 
		labels=None):
	"""
	Parameters:
		true_values:	list or array
			ground truth for target values
		
		predicted_values:	list or array
			predicted values for target
		
		performance_measures:	list or array
			a list of performance measures
		
		trivial_values:	list or array
			just use this when want to calculate 'MASE'
		
		model_type:	{'regression' or 'classification'}
			type of model used for solving the problem, just needed when want to calculate 'AIC' or 'BIC'
		
		num_params: int
			number of independent variables to build model, just use it when want to calculate 'AIC' or 'BIC'
		
		labels: list or array
			list of labels for classification problems

	Returns:
		errors:	list
			list of values for errors specified in 'performance_measures'
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
			errors.append(aupr(true_values, predicted_values, labels))
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

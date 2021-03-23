# performance function
import numpy as np
from sklearn.metrics import r2_score

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

def performance(true_values, predicted_values, performance_measures=['MAPE'], trivial_values=[]):
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

	return errors
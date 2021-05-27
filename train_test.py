import pandas as pd
import numpy as np
import pathlib
import configurations

from performance import performance
from select_features import select_features
from split_data import split_data
from train_evaluate import inner_train_evaluate
from scaling import data_scaling
from scaling import target_descale
from get_target_quantities import get_target_quantities
from get_normal_target import get_normal_target
from apply_performance_mode import apply_performance_mode
from get_trivial_values import get_trivial_values

def train_test(
		data, instance_testing_size, 
		forecast_horizon, feature_or_covariate_set, 
		history_length, model='knn', 
		model_type='regression', model_parameters=None, 
		feature_scaler='logarithmic', target_scaler='logarithmic', 
		labels=None, performance_measures=['MAPE'], 
		performance_mode='normal', performance_report=True, 
		save_predictions=True, verbose=0):
	
	"""
	Parameters:
		data:	Pandas DataFrame
			a preprocessed DataFrame to be used for training the model and making predictions on the test part
		
		instance_testing_size:	int or float
			the size of testing instances
		
		forecast_horizon:	int
			forecast horizon to gap consideration in data splitting process by the gap, we mean the number of temporal units
			which are excluded from data to simulate the situation of real prediction in which we do not have access to the
			information of forecast horizon-1 units before the time point of the target variable.

		feature_or_covariate_set:	list<string>
			a list of covariates or features which feature selection process will be based on them if historical data is provided, 
			the input will be considered as a feature list, otherwise as a covariate list

		history_length:	int
			history length of the input "data", history length is just used for the reports in "train_test"

		model:	string or callable or dict
			string: one of the pre-defined model names 
			function: a user-defined function
			dict: pre-defined model names and corresponding hyper parameters
			pre-defined model names: 'knn', 'nn' , 'gbm', 'glm'

		model_type:	string

		model_parameters:	list<int> or None

		feature_scaler:	string

		target_scaler:	string

		labels:	list<int> or None

		performance_measures:	list<string>
			a list of performance measures that the user wants to calculate the errors on predictions of test dataset 
		
		performance_mode:	string

		performance_report:	bool
			if True, some tables containing a report on models and their corresponding errors (based on performance_measurements) 
			will be saved in the same directory
		
		save_predictions:	bool
			if True, the prediction values of trained models for training data and validation data through train_and_evaluate 
			process will be saved in the same directory as your program is running as in ‘.csv’ format
		
		verbose:	int
			the level of produced detailed logging information
			available options:
			0: no logging
			1: only important information logging 
			2: all details logging


	Returns:
		model:	string or callable or dict
			exactly same as the 'model' parameter

		model_parameters:	list<int>
	"""

	################################ checking for TypeError and other possible mistakes in the inputs
	if not(isinstance(data, pd.DataFrame)):
		raise TypeError("Expected a pandas DataFrame for data.")

	if not(isinstance(instance_testing_size, int) or isinstance(instance_testing_size, float)):
		raise TypeError("Expected an integer or a float number for instance_testing_size.")
	
	if not(isinstance(forecast_horizon, int)):
		raise TypeError("Expected an integer for forecast_horizon.")
	
	if not(isinstance(feature_or_covariate_set, list)):
		raise TypeError("Expected a list of strings for feature_or_covariate_set.")
	
	if not(isinstance(history_length, int)):
		raise TypeError("Expected an integer for history_length.")
	
	if not(isinstance(model, str) or callable(model) or isinstance(model, dict)):
		raise TypeError("Expected a string or function or a dictionary of model parameters for model.")
	
	if not(isinstance(model_type, str)):
		raise TypeError("Expected a string for model_type.")
	
	if not(isinstance(model_parameters, list) or model_parameters == None):
		raise TypeError("Expected a list or None value for model_parameters.")
	
	if not(isinstance(feature_scaler, str) or feature_scaler == None):
		raise TypeError("Expected a string or None value for feature_scaler.")
	
	if not(isinstance(target_scaler, str) or target_scaler == None):
		raise TypeError("Expected a string or None value for target_scaler.")

	if not(isinstance(labels, list) or labels == None):
		raise TypeError("Expected a list or None value for labels.")
	
	if not(isinstance(performance_measures, list)):
		raise TypeError("Expected a list for performance_measures.")
	
	if not(isinstance(performance_mode, str)):
		raise TypeError("Expected a string for performance_mode.")
	
	if not(isinstance(performance_report, bool)):
		raise TypeError("Expected a bool variable for performance_report.")
	
	if not(isinstance(save_predictions, bool)):
		raise TypeError("Expected a bool variable for save_predictions.")
	
	if not(isinstance(verbose, int)):
		raise TypeError("Expected an integer (0 or 1 or 2) for verbose.")
	################################

	# classification checking
	labels = None
	if model_type == 'classification':
		if not set(performance_measures) <= set(configurations.CLASSIFICATION_PERFORMANCE_MEASURES):
			raise Exception("Error: The input 'performance_measures' is not valid according to 'model_type=classification'.")
		if performance_mode != 'normal':
			performance_mode = 'normal'
			print("Warning: The input 'performance_mode' is set to 'normal' according to model_type=classification'.")
		if target_scaler is not None:
			target_scaler = None
			print("Warning: The input 'target_scaler' is set to None according to model_type=classification'.")

	# get some information of the data
	target_mode, target_granularity, granularity, data = get_target_quantities(data=data.copy())

	# check rows related to future prediction are removed and if not then remove them
	temp_data = data.sort_values(by = ['temporal id','spatial id']).copy()
	number_of_spatial_units = len(temp_data['spatial id'].unique())
	if all(temp_data.tail(granularity*forecast_horizon*number_of_spatial_units)['Target'].isna()):
		data = temp_data.iloc[:-(granularity*forecast_horizon*number_of_spatial_units)]

	# check if model is a string or function
	model_name = ''
	if isinstance(model, str) == False:
		model_name = model.__name__
		if model_name in ['nn', 'knn', 'glm', 'gbm']:
			raise TypeError("Name of the user defined model matches the name of one of our predefined models.")
	else:
		model_name = model

	# find labels for classification problem
	if labels == None:
		if model_type == 'regression':	# just an empty list
			labels = []
		elif model_type == 'classification':	# unique values in 'Target' column of data
			labels = data.Target.unique()
			labels.sort()

	# select features
	processed_data = select_features(
		data=data.copy(), 
		ordered_covariates_or_features=feature_or_covariate_set
	)

	# splitting data in the way is set for train_test
	training_data, _, testing_data, gap_data = split_data(
		data=processed_data.copy(), 
		splitting_type='instance', 
		instance_testing_size=instance_testing_size, 
		instance_validation_size=None, 
		instance_random_partitioning=False, 
		fold_total_number=0, 
		fold_number=0, 
		forecast_horizon=forecast_horizon, 		
		granularity=granularity, 
		verbose=verbose
	)

	# separate some data which are needed later
	base_data = training_data['Target'].values.tolist()
	training_target = training_data[['spatial id', 'temporal id', 'Target', 'Normal target']]
	test_target = testing_data[['spatial id', 'temporal id', 'Target', 'Normal target']]

	# scaling data
	training_data, testing_data = data_scaling(
		train_data=training_data.copy(), 
		test_data=testing_data.copy(), 
		feature_scaler=feature_scaler, 
		target_scaler=target_scaler
	)

	# training model with processed data	
	training_predictions, testing_predictions, _, number_of_parameters = inner_train_evaluate(
		training_data=training_data.copy(), 
		validation_data=testing_data.copy(), 
		model=model, 
		model_type=model_type, 
		model_parameters=model_parameters, 
		labels=labels, 
		verbose=verbose
	)

	# target descale
	training_predictions = target_descale(
		scaled_data=list(training_predictions), 
		base_data=base_data, 
		scaler=target_scaler
	)

	testing_predictions = target_descale(
		scaled_data=list(testing_predictions), 
		base_data=base_data, 
		scaler=target_scaler
	)

	# checking for some files to exit which will be used in the next phases
	test_process_backup_file_name = 'test_process_backup.csv'
	if pathlib.Path(test_process_backup_file_name).is_file() == False:
		df = pd.DataFrame(columns=['spatial id', 'temporal id', 'Target', 'Normal target', 'prediction'])
		df.to_csv(test_process_backup_file_name, index=False)


	# getting back previous points (useful for one-by-one method, also works for one-as-whole method)
	previous_test_points = pd.read_csv(test_process_backup_file_name)
	previous_testing_predictions = previous_test_points['prediction'].tolist()

	# append current point to previous points
	test_target = test_target.append(previous_test_points[['spatial id', 'temporal id', 'Target', 'Normal target']], ignore_index=True)
	testing_predictions = list(testing_predictions) + previous_testing_predictions

	# saving test_target+testing_predictions into a backup file to be used in the next point
	df_for_backup = test_target.copy()
	df_for_backup.insert(loc=len(df_for_backup.columns), column='prediction', value=testing_predictions)
	df_for_backup.to_csv(test_process_backup_file_name, index=False)

	# get normal data
	training_target, test_target, training_prediction, test_prediction = get_normal_target(
		training_target=training_target.append(gap_data[['spatial id', 'temporal id', 'Target', 'Normal target']], ignore_index=True), 
		test_target=test_target.copy(), 
		training_prediction=list(training_predictions) + gap_data['Target'].tolist(), 
		test_prediction=testing_predictions, 
		target_mode=target_mode, 
		target_granularity=target_granularity
	)

	# make copy of some data to be stores later
	test_target_normal, test_prediction_normal = test_target.copy(), test_prediction.copy()

	# including performance_mode
	training_target, test_target, training_prediction, test_prediction = apply_performance_mode(
		training_target=training_target.copy(), 
		test_target=test_target.copy(), 
		training_prediction=list(training_prediction), 
		test_prediction=test_prediction, 
		performance_mode=performance_mode
	)

	# computing trivial values for the test set
	_, _, _, testing_true_values, testing_predicted_values, testing_trivial_values = get_trivial_values(
		train_true_values_df=training_target.copy(), 
		validation_true_values_df=test_target.copy(), 
		train_prediction=list(training_prediction), 
		validation_prediction=test_prediction, 
		forecast_horizon=forecast_horizon, 
		granularity=granularity
	)

	# computing performnace on test dataset
	test_prediction_errors = performance(
		true_values=testing_true_values, 
		predicted_values=testing_predicted_values, 
		performance_measures=performance_measures, 
		trivial_values=testing_trivial_values, 
		model_type=model_type, 
		num_params=number_of_parameters, 
		labels=labels)
	
	# checking for existance of some directories for logging purpose
	if pathlib.Path('prediction/test process').is_dir() == False:
		pathlib.Path('prediction/test process').mkdir(parents=True, exist_ok=True)
	if pathlib.Path('performance/test process').is_dir() == False:
		pathlib.Path('performance/test process').mkdir(parents=True, exist_ok=True)

	# saving predictions based on model_type
	pred_file_name = 'prediction/test process/test prediction forecast horizon = %s.csv' % (forecast_horizon)
	testing_predictions = np.array(testing_predictions)

	if save_predictions == True:
		if model_type == 'regression':
			df = pd.DataFrame()
			df['real'] = test_target_normal['Normal target'].values.tolist()
			df['prediction'] = list(test_prediction_normal)
			df.insert(0, 'temporal id', test_target_normal['temporal id'].values.tolist(), True)
			df.insert(0, 'spatial id', test_target_normal['spatial id'].values.tolist(), True)
			df.insert(0, 'model name', model_name, True)
			df.to_csv(pred_file_name, index=False)
		elif model_type == 'classification':
			df = pd.DataFrame()
			df['real'] = test_target_normal['Normal target'].values.tolist()
			for i in range(len(labels)):
				col_name = str(labels[i])
				df[col_name] = testing_predictions[:, i]
			df.insert(0, 'temporal id', test_target_normal['temporal id'].values.tolist(), True)
			df.insert(0, 'spatial id', test_target_normal['spatial id'].values.tolist(), True)
			df.insert(0, 'model name', model_name, True)
			df.to_csv(pred_file_name, index=False)
	
	# saving performance (same approach for both regression and classification)
	performance_file_name = 'performance/test process/test performance report forecast horizon = %s.csv' % (forecast_horizon)
	if performance_report == True:
		df_data = {
			'model name': list([model_name]), 
			'history length': list([history_length]), 
			'feature or covariate set': ', '.join(feature_or_covariate_set)
		}
		df = pd.DataFrame(df_data, columns=list(df_data.keys()))
		for i in range(len(performance_measures)):
			df[performance_measures[i]] = list([float(test_prediction_errors[i])])
		df.to_csv(performance_file_name, index=False)
	
	return model, model_parameters

import pandas as pd

from pathlib import Path
from performance import performance
from select_features import select_features
from split_data import split_data
from train_evaluate import train_evaluate
from scaling import data_scaling
from scaling import target_descale
from get_target_quantities import get_target_quantities
from get_normal_target import get_normal_target
from apply_performance_mode import apply_performance_mode
from get_trivial_values import get_trivial_values

def train_test(data, instance_testing_size, forecast_horizon, feature_or_covariate_set, history_length, model='knn', model_type='regression', 
	model_parameters=None, input_scaler='logarithmic', output_scaler='logarithmic', performance_measures=['MAPE'], performance_mode='normal', 
	performance_report=True, save_predictions=True, verbose=0):

	# check if model is a string or function
	model_name = ''
	if isinstance(model, str) == False:
		model_name = model.__name__
		if model_name in ['nn', 'knn', 'glm', 'gbm']:
			raise TypeError("Name of the user defined model matches the name of one of our predefined models.")
	else:
		model_name = model
	
	# select features
	processed_data = select_features(
		data=data.copy(), 
		ordered_covariates_or_features=feature_or_covariate_set
	)

	target_mode, target_granularity, granularity, processed_data = get_target_quantities(data=processed_data.copy())

	# splitting data in the way is set for train_test
	training_data, validation_data, testing_data, gap_data = split_data(
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
		input_scaler=input_scaler, 
		output_scaler=output_scaler
	)

	# drop the columns ("Normal target", "spatial id", "temporal id") from data
	training_data = training_data.drop(['Normal target', 'spatial id', 'temporal id'], axis=1)
	testing_data = testing_data.drop(['Normal target', 'spatial id', 'temporal id'], axis=1)

	# training model with processed data	
	training_predictions, testing_predictions, trained_model = train_evaluate(
		training_data=training_data.copy(), 
		validation_data=testing_data.copy(), 
		model=model, 
		model_type=model_type, 
		model_parameters=model_parameters, 
		verbose=verbose
	)

	# target descale
	training_predictions = target_descale(
		scaled_data=list(training_predictions), 
		base_data=base_data, 
		scaler=output_scaler
	)

	testing_predictions = target_descale(
		scaled_data=list(testing_predictions), 
		base_data=base_data, 
		scaler=output_scaler
	)

	# checking for some files to exit which will be used in the next phases
	test_process_backup_file_name = 'test_process_backup.csv'
	if Path(test_process_backup_file_name).is_file() == False:
		df = pd.DataFrame(columns=['spatial id', 'temporal id', 'Target', 'Normal target'])
		df.to_csv(test_process_backup_file_name, index=False)


	# getting back previous points (useful for one-by-one method, also works for one-as-whole method)
	previous_test_target = pd.read_csv(test_process_backup_file_name)
	previous_testing_predictions = previous_test_target['Target'].tolist()

	# append current point to previous points
	test_target = test_target.append(previous_test_target, ignore_index=True)
	testing_predictions = testing_predictions + previous_testing_predictions

	# saving test_target into a backup file to be used in the next point
	test_target.to_csv(test_process_backup_file_name, index=False)

	# get normal data
	training_target, test_target, training_prediction, test_prediction = get_normal_target(
		training_target=training_target.append(gap_data[['spatial id', 'temporal id', 'Target', 'Normal target']], ignore_index=True), 
		test_target=test_target.copy(), 
		training_prediction=list(training_predictions), 
		test_prediction=list(testing_predictions), 
		target_mode=target_mode, 
		target_granularity=target_granularity
	)

	# make copy of some data to be stores later
	training_target_normal, test_target_normal, training_prediction_normal, test_prediction_normal = \
		training_target.copy(), test_target.copy(), training_prediction.copy(), test_prediction.copy()

	# including performance_mode
	training_target, test_target, training_prediction, test_prediction = apply_performance_mode(
		training_target=training_target.append(gap_data[['spatial id', 'temporal id', 'Target', 'Normal target']], ignore_index=True), 
		test_target=test_target.copy(), 
		training_prediction=training_prediction, 
		test_prediction=test_prediction, 
		performance_mode=performance_mode
	)

	# computing trivial values for the test set
	trivial_values = None
	if any(item.lower() == 'mase' for item in performance_measures):
		trivial_values = get_trivial_values(
			train_true_values_df=training_target.copy(), 
			validation_true_values_df=test_target.copy(), 
			train_prediction=training_prediction, 
			validation_prediction=test_prediction, 
			forecast_horizon=forecast_horizon, 
			granularity=granularity
		)

	# computing performnace on test dataset
	test_prediction_errors = performance(
		true_values=test_target['Normal target'].values.tolist(), 
		predicted_values=test_prediction, 
		performance_measures=performance_measures, 
		trivial_values=trivial_values, 
		labels=None, 
		pos_label=None
	)
	
	# saving predictions
	pred_file_name = 'prediction/test process/test prediction forecast horizon = %s.csv' % (forecast_horizon)
	if save_predictions == True:
		df = pd.DataFrame()
		df['real'] = test_target_normal['Normal target'].values.tolist()
		df['prediction'] = list(test_prediction_normal)
		df.insert(0, 'temporal id', test_target_normal['temporal id'].values.tolist(), True)
		df.insert(0, 'spatial id', test_target_normal['spatial id'].values.tolist(), True)
		df.insert(0, 'model name', model_name, True)
		df.to_csv(pred_file_name, index=False)
	
	# saving performance
	performance_file_name = 'performance/test process/test performance report forecast horizon = %s.csv' % (forecast_horizon)
	if performance_report == True:
		df_data = {
			'model name': list([model_name]), 
			'history length': list([history_length]), 
			'feature or covariate set': ', '.join(feature_or_covariate_set)
		}
		df = pd.DataFrame(df_data, columns=list(df_data.keys()))
		for i in range(len(performance_measures)):
			df[performance_measures[i]] = list(test_prediction_errors[i])
		df.to_csv(performance_file_name, index=False)
	
	return model, model_parameters

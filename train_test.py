import pandas as pd

from performance import performance
from select_features import select_features
from split_data import split_data
from train_evaluate import train_evaluate
from scaling import data_scaling
from scaling import target_descale
from get_normal_target import get_normal_target
from get_trivial_values import get_trivial_values

def train_test(data, instance_testing_size, forecast_horizon, ordered_covariates_or_features, granularity, target_mode='normal', 
	target_granularity=1, best_model='knn', model_type='regression', model_parameters=None, input_scaler='logarithmic', 
	output_scaler='logarithmic', performance_measures=['MAPE'], performance_report=True, save_predictions=True, verbose=0):
	
	processed_data = select_features(data.copy(), ordered_covariates_or_features)

	# deleting last rows from each group (by spatial id) which meet the condition (Target == NULL)
	# processed_data = processed_data.groupby('spatial id').apply(lambda x: x.drop(x.tail(forecast_horizon*granularity).index))
	processed_data = processed_data.sort_values(by = ['temporal id', 'spatial id'])
	number_of_spatial_units = len(data['spatial id'].unique())
	processed_data = processed_data.iloc[:-(forecast_horizon*granularity*number_of_spatial_units)].copy()

	# splitting data in the way is set for train_test
	training_data, validation_data, testing_data = split_data(
		data=processed_data, forecast_horizon=forecast_horizon, instance_testing_size=instance_testing_size, 
		instance_validation_size=None, fold_total_number=0, fold_number=0, splitting_type='instance', 
		instance_random_partitioning=False, granularity=granularity, verbose=verbose
	)

	# separate some data which are needed later
	base_data = training_data['Normal target'].values.tolist()
	training_target = training_data['spatial id', 'temporal id', 'Target', 'Normal target']
	test_target = testing_data['spatial id', 'temporal id', 'Target', 'Normal target']

	# scaling data
	training_data, testing_data = data_scaling(training_data, testing_data, input_scaler=input_scaler, output_scaler=output_scaler)

	# drop the columns ("Normal target", "spatial id", "temporal id") from data
	training_data = training_data.drop(['Normal target', 'spatial id', 'temporal id'], axis=1)
	testing_data = testing_data.drop(['Normal target', 'spatial id', 'temporal id'], axis=1)

	# training model with processed data	
	training_predictions, testing_predictions, trained_model = train_evaluate(
		training_data=training_data, validation_data=testing_data, model=best_model, model_type=model_type, 
		model_parameters=model_parameters, verbose=verbose
	)

	# target descale
	training_predictions = target_descale(scaled_data=list(training_predictions), base_data=base_data, scaler=output_scaler)
	testing_predictions = target_descale(scaled_data=list(testing_predictions), base_data=base_data, scaler=output_scaler)

	# get normal data
	training_target, test_target, training_prediction, test_prediction = get_normal_target(
		training_target=training_target, test_target=test_target, 
		training_prediction=list(training_predictions), test_prediction=list(testing_predictions), 
		target_mode=target_mode, target_granularity=target_granularity
	)

	# computing trivial values for the test set
	trivial_values = None
	if any(item.lower() == 'mase' for item in performance_measures):
		trivial_values = get_trivial_values(
			train_true_values_df=training_target, 
			validation_true_values_df=test_target, 
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

	if performance_report == True:	# should be completed
		pass

	if save_predictions == True:
		# predictions_df = pd.DataFrame()
		# predictions_df['True Values'] = testing_data.values.tolist()
		# predictions_df['Predicted Values'] = testing_predictions

		# predictions_df.to_csv('train_test_predictions.csv', index=False)
		pass

	return best_model, model_parameters

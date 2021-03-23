from performance import performance
from select_features import select_features
import pandas as pd

def train_test(data, best_trained_model, forecast_horizon, instance_testing_size, best_history_length, ordered_covariates_or_features, 
	best_model='knn', best_feature_set_indices=[1, 2, 3], performance_measures=['MAPE'], performance_report=True, save_predictions=True, 
	verbose=0):
	
	processed_data = select_features(data, ordered_covariates_or_features, best_feature_set_indices)

	training_data, validation_data, testing_data = split_data(processed_data, forecast_horizon, 'instance', instance_testing_size,
		instance_validation_size=0, instance_random_partitioning=False, fold_total_number=0, fold_number=0, verbose=verbose)

	training_predictions, testing_prediction, trained_model = train_evaluate(training_data, testing_data, best_model, verbose)

	errors = performance(testing_data.values.reshape(-1), testing_prediction, performance_measures)

	if performance_report == True:	# should be completed
		pass

	if save_predictions == True:
		predictions_df = pd.DataFrame()
		predictions_df['True Values'] = testing_data.values.tolist()
		predictions_df['Predicted Values'] = training_predictions

		predictions_df.to_csv('train_test_predictions.csv', index=False)

	return trained_model
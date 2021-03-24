def future_predict(data, best_model, best_trained_model, forecast_horizon, best_history_length, best_feature_sets_indices,
                   ordered_covariates_or_features, performance_measure, performance_report, save_predictions, verbose):
    
    if type(data) == dict:
        if best_history_length in data.keys():
            if type(data[best_history_length]) == str:
                try:
                    data = pd.read_csv(data[best_history_length])
                except FileNotFoundError:
                    sys.exit("The address '{0}' is not valid.".format(data[best_history_length]))
                    
            elif type(data[best_history_length]) == pd.DataFrame:
                data = pd.read_csv(data[best_history_length])
                
            else: sys.exit("The values in input data dictionary must be of type DataFrame or string of data address.")
        else: sys.exit("The DataFrame corresponding to the best_history_length is not included in the input data dictionary.")
    else: sys.exit("The data argument must be a dictionary of DataFrames or data addresses.")
                
    data = select_features(data = data, ordered_covariates_or_features = ordered_covariates_or_features, best_feature_sets_indices = best_feature_sets_indices, verbose = 0)
    
    training_data, _ , testing_data = split_data(data, forecast_horizon=forecast_horizon, instance_testing_size=forecast_horizon, instance_validation_size=None, fold_total_number = None,
               fold_number = None, splitting_type = 'instance', instance_random_partitioning = False, verbose = 0)
    
    train_predictions, testing_predictions, trained_model = train_evaluate(training_data = training_data, validation_data = testing_data, model = best_model, verbose = 0)
    
    return trained_model
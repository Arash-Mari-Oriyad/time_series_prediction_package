def train_validate(data, max_history_length, forecast_horizon, ordered_covariates_or_features, feature_sets_indices,
                   models, splitting_type, instance_validation_size, instance_testing_size, instance_random_partitioning,
                   fold_total_number, performance_measure, benchmark,
                   performance_report, save_predictions, verbose):
    
    supported_models_name = ['nn', 'knn', 'glm', 'gbm']
    supported_performance_measures = ['MAE', 'MAPE', 'MASE', 'MSE', 'R2']
    models_list = [] # list of models (str or callable)
    models_parameter_list = [] # list of models' parameters (dict or None)
    models_name_list = [] # list of models' names (str)
    best_history_length = None
    best_feature_sets_indices = None
    best_model = None
    best_trained_model = None
    
    ################## reading and validating inputs
    
    # data input
    
    data_list = []
    if type(data) == list:
        for history in range(len(data+1)):
            if type(data[history]) == pd.DataFrame:
                data_list[history] = data[history]
            elif type(data[history]) == str:
                try:
                    data_list[history] = pd.read_csv(data[history])
                except FileNotFoundError:
                    sys.exit("The address '{0}' is not valid.".format(data[history]))
            else:
                sys.exit("The input data must be a list of DataFrames or strings of data addresses.")
    else:
        sys.exit("The input data must be a list of DataFrames or strings of data addresses.")
        
    # models input
    
    if type(models) != list:
        sys.exit("Warning: The models must be of type list.")
        
    callable_model_number = 1
    
    for model in models:
        
        if type(model) == dict:       
            model_name = list(model.keys())[0]
            
            if (len(model) == 1) and (model_name in supported_models_name):
                
                if model_name not in models_list:
                    
                    models_list.append(model_name)
                    models_name_list.append(model_name)
                    if type(model[model_name]) == dict:
                        models_parameter_list.append(model[model_name])
                    else:
                        models_parameter_list.append(None)
                        print("Warning: The values in the dictionary items of models list must be a dictionary of the model hyper parameter names and values. Other values will be ignored.")
            else:
                print("Warning: Each dictionary item in models list must contain only one item with a name of one of the supported models as a key and the parameters of that model as value. The incompatible cases will be ignored.")
        
        elif type(model) == str:
            if (model in supported_models_name) and (model not in models_list):
                models_list.append(model)
                models_name_list.append(model)
                models_parameter_list.append(None)
            else:
                print("Warning: The string items in the models list must be one of the supported models names. The incompatible cases will be ignored.")
                
        elif callable(model):
            models_list.append(model)
            models_name_list.append('user_defined_' + str(callable_model_number))
            models_parameter_list.append(None)
            callable_model_number += 1
            
        else:
            print("Warning: The items in the models list must be of type string, dict or callable. The incompatible cases will be ignored.")
    
    if len(models_list) < 1:
        sys.exit("There is no item in the models list or the items are invalid.")
        
    # performance measure input
    
    if type(performance_measure) != list:
        sys.exit("The performance_measure must be of type list.")
        
    unsupported_measures = list(set(performance_measure)-set(supported_performance_measures))
    if len(unsupported_measures) > 0:
        print("Warning: Some of the specified measures are not valid:\n{0}\nThe supported measures are: ['MAE', 'MAPE', 'MASE', 'MSE', 'R2']".format(unsupported_measures))
    
    performance_measure = list(set([measure for measure in supported_performance_measures if measure in performance_measure]))
    
    if len(performance_measure) < 1:
        sys.exit("No valid measure is specified.")
        
    # benchmark input
    
    if benchmark not in supported_performance_measures:
        print("Warning: The specified benchmark must be one of the supported performance measures: ['MAE', 'MAPE', 'MASE', 'MSE', 'R2']\nThe incompatible cases will be ignored and replaced with 'MAPE'.")
        benchmark = 'MAPE'
    if benchmark in ['MAE', 'MAPE', 'MASE', 'MSE']:
        overall_min_validation_error = float('Inf')
    else:
        overall_min_validation_error = float('-Inf')
        
    # splitting_type, fold_total_number, fold_number inputs, instance_testing_size, and instance_validation_size
        
    if splitting_type == 'cross-validation':

        if (fold_total_number is None) or (fold_number is None):
            sys.exit("if the splitting_type is 'cross-validation', the fold_total_number and fold_number must be specified.")
        if (type(fold_total_number) != int) or (type(fold_number) != int):
            sys.exit("The fold_total_number and fold_number must be of type int.")
        elif (fold_number > fold_total_number) or (fold_number < 1):
            sys.exit("The fold_number must be a number in a range between 1 and fold_total_number.")
            
    elif splitting_type in ['training-validation', 'training-validation-testing']:
    
        if type(instance_validation_size) == float:
            if instance_validation_size > 1:
                sys.exit("The float instance_validation_size will be interpreted to the proportion of data which is considered as validation set and must be less than 1.")
                
        elif (type(instance_validation_size) != int):
            sys.exit("The type of instance_validation_size must be int or float.")
            
        if splitting_type == 'training-validation-testing':
        
            # check the type of instance_testing_size and instance_validation_size
            if type(instance_testing_size) == float:
                if instance_testing_size > 1:
                    sys.exit("The float instance_testing_size will be interpreted to the proportion of data that is considered as the test set and must be less than 1.")
            elif (type(instance_testing_size) != int):
                sys.exit("The type of instance_testing_size must be int or float.")
    else:
        sys.exit("The specified splitting_type is ambiguous. The supported values are 'training-validation', 'training-validation-testing', and 'cross-validation'.")
    #############################################################
    
    # initializing
    
    if splitting_type != 'cross-validation':
        fold_total_number = 1
        
    # the outputs of running the models 
    training_predictions = {model_name : {} for model_name in models_name_list} # train prediction result for each fold
    validation_predictions = {model_name : {} for model_name in models_name_list} # validation prediction result for each fold
#     all_training_predictions = {model_name : {} for model_name in models_name_list} # train prediction result for all folds
#     all_validation_predictions = {model_name : {} for model_name in models_name_list} # validation prediction result for all folds
    trained_model = {model_name : {} for model_name in models_name_list}
    target_real_values = {fold_number : {'training':None,'validation':None} for fold_number in range(1, fold_total_number + 1)}
    
    validation_errors = {measure: {model_name: {} for model_name in models_name_list} for measure in performance_measure}
    training_errors = {measure: {model_name: {} for model_name in models_name_list} for measure in performance_measure}
    
    
    ########## main part (loop over history_length, feature_sets_indices, and folds)
    
    for history in range(1,len(data_list)+1):
        data = data_list[history]
        
        if splitting_type == 'training-validation-testing' :
            raw_train_data, _ , raw_testing_data = split_data(data = data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = instance_testing_size,
                                                      instance_validation_size = None, fold_total_number = None, fold_number = None, splitting_type = 'instance',
                                                      instance_random_partitioning = instance_random_partitioning, verbose = 0)
        else:
            raw_train_data = data.copy()
        
        loom = ProcessLoom(max_runner_cap = len(feature_sets_indices) * fold_total_number * len(models_list) + 5)
        
        for feature_set_index in range(len(feature_sets_indices)):
            train_data = select_features(data = raw_train_data.copy(), ordered_covariates_or_features = ordered_covariates_or_features,
                                        feature_set_indices = feature_set_indices[feature_set_index])
            
            if splitting_type == 'cross-validation':
                split_data_splitting_type = 'fold'
            else:
                split_data_splitting_type = 'instance'
                
            for fold_number in range(1, fold_total_number + 1):
                training_data, validation_data, _ = split_data(data = train_data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = None,
                                              instance_validation_size = instance_validation_size, fold_total_number = fold_total_number, fold_number = fold_number,
                                              splitting_type = split_data_splitting_type, instance_random_partitioning = instance_random_partitioning, verbose = 0)
                
                # saving real values
                target_real_values[fold_number]['training'] = list(np.array(training_data['Target']).reshape(-1))
                target_real_values[fold_number]['validation'] = list(np.array(validation_data['Target']).reshape(-1))
                
                for model_number, model in enumerate(models_list):
                    
                    model_parameters = models_parameter_list[model_number]
                    model_name = models_name_list[model_number]
                    iter_key = str(feature_set_index) + ' ' + str(fold_number) + ' ' + model_name
                    
                    loom.add_function(train_evaluate, [training_data, validation_data, model, model_parameters, 0], {}, iter_key)
                    
        # run the processes in parallel
        parallel_outputs = loom.execute()
        
        # get outputs, calculate and save the performance
        for feature_set_index in range(len(feature_sets_indices)):
            for model_number, model in enumerate(models_list):
                
                model_name = models_name_list[model_number]
                fold_validation_error = {fold_number : {measure: None for measure in supported_performance_measures} for fold_number in range(1, fold_total_number + 1)}
                fold_training_error = {fold_number : {measure: None for measure in supported_performance_measures} for fold_number in range(1, fold_total_number + 1)}
                
                for fold_number in range(1, fold_total_number + 1):
                    
                    iter_key = str(feature_set_index) + ' ' + str(fold_number) + ' ' + model_name
                    
                    training_predictions[model_name][(fold_number, history, feature_set_index)], validation_predictions[model_name][(fold_number, history, feature_set_index)], trained_model[model_name][(fold_number, history, feature_set_index)] = parallel_outputs[iter_key][
                    'output']
                    
                    # calculate and store the performance measure for the current fold
                    for measure in performance_measure:
                        fold_validation_error[fold_number][measure] = performance(target_real_values[fold_number]['validation'], 
                                                                                 validation_predictions[model_name][(fold_number, history, feature_set_index)],
                                                                                 list([measure]), verbose = 0)
                        fold_training_error[fold_number][measure] = performance(target_real_values[fold_number]['training'], 
                                                                                 training_predictions[model_name][(fold_number, history, feature_set_index)],
                                                                                 list([measure]), verbose = 0)
                
                # calculating and storing the cross-validation final performance measure by taking the average of the folds performance measure
                for measure in performance_measure:
                    
                    validation_errors[measure][model_name][(history, feature_set_index)] = np.mean(list([fold_validation_error[fold_number][measure] for fold_number in range(1, fold_total_number + 1)]))
                    training_errors[measure][model_name][(history, feature_set_index)] = np.mean(list([fold_training_error[fold_number][measure] for fold_number in range(1, fold_total_number + 1)]))
                    
                    if measure == benchmark:
                        if measure in ['MAE', 'MAPE', 'MASE', 'MSE']:
                            if validation_errors[measure][model_name][(history, feature_set_index)] < overall_min_validation_error:
                                overall_min_validation_error = validation_errors[measure][model_name][(history, feature_set_index)]
                                best_model = model
                                best_history_length = history
                                best_feature_sets_indices = feature_sets_indices[feature_set_index]
                                best_trained_model = trained_model[model_name][(1, history, feature_set_index)] ##### why???
                        else:
                            if validation_errors[measure][model_name][(history, feature_set_index)] > overall_min_validation_error:
                                overall_min_validation_error = validation_errors[measure][model_name][(history, feature_set_index)]
                                best_model = model
                                best_history_length = history
                                best_feature_sets_indices = feature_sets_indices[feature_set_index]
                                best_trained_model = trained_model[model_name][(1, history, feature_set_index)]
                                
    return best_model, best_history_length, best_feature_sets_indices, best_trained_model
        
                                
import pandas as pd 
import numpy as np
import sys
import datetime
from models import KNN,NN,GLM,GBM
from sklearn.model_selection import KFold
from multiprocessing import Pool
from functools import partial

def save_prediction_data_frame(models_name_list, fold_total_number, target_real_values, fold_validation_predictions,
                           fold_training_predictions, models_best_history_length, models_best_feature_set_number,
                           prediction_type):

    for model_number, model_name in enumerate(models_name_list):

        best_hist_length = models_best_history_length[model_name]
        best_feat_set_number = models_best_feature_set_number[model_name]

        prediction_data_frame = pd.DataFrame(columns = ['fold', 'spatial id', 'temporal id','real value', 'predicted value'])


        for fold_number in range(1, fold_total_number + 1):
            if prediction_type == 'validation':
                temp = target_real_values[best_hist_length][fold_number]['validation']
                temp['predicted value'] = fold_validation_predictions[model_name][(fold_number, best_hist_length, best_feat_set_number)]
            else:
                temp = target_real_values[best_hist_length][fold_number]['training']
                temp['predicted value'] = fold_training_predictions[model_name][(fold_number, best_hist_length, best_feat_set_number)]

            temp['fold'] = fold_number
            temp.rename(columns = {'Target':'real value'}, inplace = True)
            temp = temp[['fold', 'spatial id', 'temporal id','real value', 'predicted value']]

            prediction_data_frame = prediction_data_frame.append(temp)

        if fold_total_number == 1:
            prediction_data_frame.drop(['fold'], axis = 1, inplace = True)

        prediction_data_frame.to_csv(prediction_type + '_prediction '+model_name+'.csv', index = False)
    
    
def train_validate(data, max_history_length, forecast_horizon, ordered_covariates_or_features, feature_sets_indices,
                   models, splitting_type, instance_validation_size, instance_testing_size, instance_random_partitioning,
                   fold_total_number, performance_measure, benchmark,
                   performance_report, save_predictions, verbose):
    
    
    supported_models_name = ['nn', 'knn', 'glm', 'gbm']
    supported_performance_measures = ['MAE', 'MAPE', 'MASE', 'MSE', 'R2']
    models_list = [] # list of models (str or callable)
    models_parameter_list = [] # list of models' parameters (dict or None)
    models_name_list = [] # list of models' names (str)
    

    ############################ reading and validating inputs
    
    ############## data input
    
    data_list = []
    if type(data) == list:
        for history in range(len(data)):
            if type(data[history]) == pd.DataFrame:
                data_list.append(data[history])
            elif type(data[history]) == str:
                try:
                    data_list.append(pd.read_csv(data[history]))
                except FileNotFoundError:
                    sys.exit("The address '{0}' is not valid.".format(data[history]))
            else:
                sys.exit("The input data must be a list of DataFrames or strings of data addresses.")
    else:
        sys.exit("The input data must be a list of DataFrames or strings of data addresses.")
        
    ############## models input
    
    if type(models) != list:
        sys.exit("Warning: The models must be of type list.")
        
    # keep the number of user defined models to distinguish them
    callable_model_number = 1
    for item in models:
        
        # if the item is the dictionary of model name and its parameters
        if type(item) == dict:       
            model_name = list(item.keys())[0]
            
            # if the dictionary contain only one of the supported models
            if (len(item) == 1) and (model_name in supported_models_name):
                
                # if model is not duplicate 
                if model_name not in models_list:
                    models_list.append(model_name)
                    models_name_list.append(model_name)
                    # if the value of the model name is dictionary of models parameter list
                    if type(item[model_name]) == dict:
                        models_parameter_list.append(item[model_name])
                    else:
                        models_parameter_list.append(None)
                        print("Warning: The values in the dictionary items of models list must be a dictionary of the model hyper parameter names and values. Other values will be ignored.")
            else:
                print("Warning: Each dictionary item in models list must contain only one item with a name of one of the supported models as a key and the parameters of that model as value. The incompatible cases will be ignored.")
        
        # if the item is only name of model whithout parameters
        elif type(item) == str:
            if (item in supported_models_name):
                if (item not in models_list):
                    models_list.append(item)
                    models_name_list.append(item)
                    models_parameter_list.append(None)
            else:
                print("Warning: The string items in the models list must be one of the supported models names. The incompatible cases will be ignored.")
        
        # if the item is user defined function
        elif callable(item):
            models_list.append(item)
            models_name_list.append('user_defined_' + str(callable_model_number))
            models_parameter_list.append(None)
            callable_model_number += 1
            
        else:
            print("Warning: The items in the models list must be of type string, dict or callable. The incompatible cases will be ignored.")
    
    if len(models_list) < 1:
        sys.exit("There is no item in the models list or the items are invalid.")
        
    ############## performance measure input
    
    if type(performance_measure) != list:
        sys.exit("The performance_measure must be of type list.")
        
    unsupported_measures = list(set(performance_measure)-set(supported_performance_measures))
    if len(unsupported_measures) > 0:
        print("Warning: Some of the specified measures are not valid:\n{0}\nThe supported measures are: ['MAE', 'MAPE', 'MASE', 'MSE', 'R2']".format(unsupported_measures))
    
    performance_measure = list(set([measure for measure in supported_performance_measures if measure in performance_measure]))
    
    if len(performance_measure) < 1:
        sys.exit("No valid measure is specified.")
        
    ############## benchmark input
    
    if benchmark not in supported_performance_measures:
        print("Warning: The specified benchmark must be one of the supported performance measures: ['MAE', 'MAPE', 'MASE', 'MSE', 'R2']\nThe incompatible cases will be ignored and replaced with 'MAPE'.")
        benchmark = 'MAPE'
    # set the appropriate min error based on benchmark measure
    if benchmark in ['MAE', 'MAPE', 'MASE', 'MSE']:
        overall_min_validation_error = float('Inf')
        models_min_validation_error = {model_name : float('Inf') for model_name in models_name_list}
    else:
        overall_min_validation_error = float('-Inf')
        models_min_validation_error = {model_name : float('-Inf') for model_name in models_name_list}
    
    # checking validity of benchmark
    for history in range(0,len(data_list)):
        data = data_list[history].copy()
        if (len(data[data['Target']==0]) > 0) and (benchmark == 'MAPE'):
                benchmark = 'MAE'
                print("Warning : The input data contain some zero values for Target variable. Therefore 'MAPE' can not be used as a benchmark and the benchmark will be set to 'MAE'.")
        
        
    ############## splitting_type, fold_total_number, instance_testing_size, and instance_validation_size inputs
    
    # check validity of fold_total_number
    if splitting_type == 'cross-validation':

        if fold_total_number is None:
            sys.exit("if the splitting_type is 'cross-validation', the fold_total_number must be specified.")
        if (type(fold_total_number) != int) or (fold_total_number <= 1):
            sys.exit("The fold_total_number must be an integer greater than 1.")

    # check validity of instance_validation_size and instance_testing_size
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
    
    # for non cross val splitting_type, the fold_total_number  will be set to 1, to perform the prediction process only one time
    if splitting_type != 'cross-validation':
        fold_total_number = 1
    
    # setting the splitting_type of split_data function according to user specified splitting_type in train_validate function
    if splitting_type == 'cross-validation':
        split_data_splitting_type = 'fold'
    else:
        split_data_splitting_type = 'instance'
        
    #################################################### initializing
        
    models_best_history_length = {model_name : None for model_name in models_name_list} # best_history_length for each model
    models_best_feature_set_number = {model_name : None for model_name in models_name_list} # index of best_feature_set in feature_set_indices for each model
    # models_best_trained_model = {model_name : None for model_name in models_name_list} # trained model with best history and feature set
    best_model = None # overall best model
    
        
    ####### the outputs of running the models 
    fold_training_predictions = {model_name : {} for model_name in models_name_list} # train prediction result for each fold
    fold_validation_predictions = {model_name : {} for model_name in models_name_list} # validation prediction result for each fold
    # trained_model = {model_name : {} for model_name in models_name_list} # trained model
    
    # training and validation target real values for different history lengths and fold number
    target_real_values = {history:{fold_number : {'training':None,'validation':None} for fold_number in range(1, fold_total_number + 1)} for history in range(0,len(data_list))}
    
    # train_data for each history and feature set index (will be used to train best model using train data with the best history length and feature set)
    train_data_dict = {} 

    # validation and training error of different measures for each model
    validation_errors = {measure: {model_name: {} for model_name in models_name_list} for measure in performance_measure}
    training_errors = {measure: {model_name: {} for model_name in models_name_list} for measure in performance_measure}
    
    
    #################################################### main part
    # (loop over history_length, feature_sets_indices, and folds)
    
    for history in range(0,len(data_list)):
        
        # get the data with specific history length
        data = data_list[history].copy()
        
        # separating the test part
        if splitting_type == 'training-validation-testing' :
            raw_train_data, _ , raw_testing_data = split_data(data = data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = instance_testing_size,
                                                      instance_validation_size = None, fold_total_number = None, fold_number = None, splitting_type = 'instance',
                                                      instance_random_partitioning = instance_random_partitioning, verbose = 0)
        else:
            raw_train_data = data.copy()
        
        # initializing the pool for parallel run
        prediction_pool = Pool(processes = len(feature_sets_indices) * fold_total_number * len(models_list) + 5)
        pool_list = [] # list of all the different combination of the arguments of pool function
        
        for feature_set_number in range(len(feature_sets_indices)):
            
            # select the features
            train_data = select_features(data = raw_train_data.copy(), ordered_covariates_or_features = ordered_covariates_or_features,
                                        feature_set_indices = feature_sets_indices[feature_set_number])
            
            # holding train data with different histories and feature_set_indices to train best model in last step of function
            train_data_dict[(history,feature_set_number)] = train_data.copy()
                
            for model_number, model in enumerate(models_list):
                    
                model_parameters = models_parameter_list[model_number]
                model_name = models_name_list[model_number]

                for fold_number in range(1, fold_total_number + 1):
                    
                    # get the current fold training and validation data
                    training_data, validation_data, _ = split_data(data = train_data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = None,
                                                  instance_validation_size = instance_validation_size, fold_total_number = fold_total_number, fold_number = fold_number,
                                                  splitting_type = split_data_splitting_type, instance_random_partitioning = instance_random_partitioning, verbose = 0)

                    
                    # saving the target real values of each fold for data with different history lengths to use in
                    # calculation of performance and saving real and predicted target values in csv files
                    target_real_values[history][fold_number]['training'] = training_data[['spatial id', 'temporal id', 'Target']]#list(np.array(training_data['Target']).reshape(-1))
                    target_real_values[history][fold_number]['validation'] = validation_data[['spatial id', 'temporal id', 'Target']]#list(np.array(validation_data['Target']).reshape(-1))
                    
                    # add the current fold data, model name and model parameters to the list of pool function arguments
                    pool_list.append(tuple((training_data, validation_data, model, model_parameters, 0)))
                    
                    
        # running the processes in parallel
        parallel_output = prediction_pool.map(partial(parallel_run),list(pool_list))
        prediction_pool.close()
        prediction_pool.join()
        
        ####################### get outputs, calculate and save the performance
        
        pool_index = 0 # the index of pool results
        
        for feature_set_number in range(len(feature_sets_indices)):
            
            for model_number, model in enumerate(models_list):
                
                model_name = models_name_list[model_number]
                # initializing a dictionary for hold each folds training and validation error for this iteration model
                fold_validation_error = {fold_number : {measure: None for measure in supported_performance_measures} for fold_number in range(1, fold_total_number + 1)}
                fold_training_error = {fold_number : {measure: None for measure in supported_performance_measures} for fold_number in range(1, fold_total_number + 1)}
                
                for fold_number in range(1, fold_total_number + 1):
                    
                    
                    # save models prediction output for the current fold in the dictionary of predicted values
                    fold_training_predictions[model_name][(fold_number, history, feature_set_number)] = parallel_output[pool_index][0]
                    fold_validation_predictions[model_name][(fold_number, history, feature_set_number)] = parallel_output[pool_index][1]
                    # trained_model[model_name][(fold_number, history, feature_set_number)] = parallel_outputs[pool_index][2]
                    
                    pool_index = pool_index + 1
                    
                    # calculate and store the performance measure for the current fold
                    for measure in performance_measure:
                        fold_validation_error[fold_number][measure] = performance(list(np.array(target_real_values[history][fold_number]['validation']['Target']).reshape(-1)), 
                                                                                 fold_validation_predictions[model_name][(fold_number, history, feature_set_number)],
                                                                                 list([measure]))
                        fold_training_error[fold_number][measure] = performance(list(np.array(target_real_values[history][fold_number]['training']['Target']).reshape(-1)), 
                                                                                 fold_training_predictions[model_name][(fold_number, history, feature_set_number)],
                                                                                 list([measure]))
            
                # calculating and storing the cross-validation final performance measure by taking the average of the folds performance measure
                for measure in performance_measure:
                    
                    validation_errors[measure][model_name][(history, feature_set_number)] = np.mean(list([fold_validation_error[fold_number][measure][0] for fold_number in range(1, fold_total_number + 1)]))
                    training_errors[measure][model_name][(history, feature_set_number)] = np.mean(list([fold_training_error[fold_number][measure][0] for fold_number in range(1, fold_total_number + 1)]))
                    
                    # update the best history length and best feature set based on the value of benchmark measure
                    if measure == benchmark:
                        if measure in ['MAE', 'MAPE', 'MASE', 'MSE']:
                            if validation_errors[measure][model_name][(history, feature_set_number)] < models_min_validation_error[model_name]:
                                models_min_validation_error[model_name] = validation_errors[measure][model_name][(history, feature_set_number)]
                                models_best_history_length[model_name] = history
                                models_best_feature_set_number[model_name] = feature_set_number
                                # models_best_trained_model[model_name] = trained_model[model_name][(1, history, feature_set_number)] ##### why??? 
                                
                        else:
                            if validation_errors[measure][model_name][(history, feature_set_number)] > models_min_validation_error[model_name]:
                                models_min_validation_error[model_name] = validation_errors[measure][model_name][(history, feature_set_number)]
                                models_best_history_length[model_name] = history
                                models_best_feature_set_number[model_name] = feature_set_number
                                # models_best_trained_model[model_name] = trained_model[model_name][(1, history, feature_set_number)]
    
    #################################################### saving predictions
    
    # save the real and predicted value of target variable in training and validation set for each model
    
    if save_predictions == True:
        
        save_prediction_data_frame(models_name_list, fold_total_number, target_real_values, fold_validation_predictions,
                                   fold_training_predictions, models_best_history_length, models_best_feature_set_number,
                                   'training')
        save_prediction_data_frame(models_name_list, fold_total_number, target_real_values, fold_validation_predictions,
                                   fold_training_predictions, models_best_history_length, models_best_feature_set_number,
                                   'validation')       
    
    
    #################################################### finding best model and overall best history length and feature set
    
    for model_number, model_name in enumerate(models_name_list):
        if benchmark in ['MAE', 'MAPE', 'MASE', 'MSE']:
            if models_min_validation_error[model_name] < overall_min_validation_error:
                overall_min_validation_error = models_min_validation_error[model_name]
                best_history_length = models_best_history_length[model_name]
                best_feature_sets_indices = feature_sets_indices[models_best_feature_set_number[model_name]]
                # best_trained_model = models_best_trained_model[model_name]
                best_model = models_list[model_number]
                best_model_number = model_number
        else:
            if models_min_validation_error[model_name] > overall_min_validation_error:
                overall_min_validation_error = models_min_validation_error[model_name]
                best_history_length = models_best_history_length[model_name]
                best_feature_sets_indices = feature_sets_indices[models_best_feature_set_number[model_name]]
                # best_trained_model = models_best_trained_model[model_name]
                best_model = models_list[model_number]
                best_model_number = model_number
                
    # training the best model using the data with the overall best history length and feature set
    best_feature_set_number = models_best_feature_set_number[models_name_list[best_model_number]]
    best_train_data = train_data_dict[(best_history_length,best_feature_set_number)]
    
    best_model_parameters = models_parameter_list[best_model_number]
    
    _, _, best_trained_model = train_evaluate(training_data = best_train_data,
                                              validation_data = best_train_data,
                                              model = best_model, model_parameters = best_model_parameters,
                                              verbose = 0)
                
    return best_model, best_history_length, best_feature_sets_indices, best_trained_model

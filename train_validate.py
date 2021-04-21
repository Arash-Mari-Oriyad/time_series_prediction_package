import pandas as pd 
import numpy as np
import sys
import datetime
from models import KNN_REGRESSOR,KNN_CLASSIFIER,NN_REGRESSOR,NN_CLASSIFIER,GLM_REGRESSOR,GLM_CLASSIFIER,GBM_REGRESSOR,GBM_CLASSIFIER
from sklearn.model_selection import KFold
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import datetime



#####################################################################################################


def report_performance(errors_dict, max_history, ordered_covariates_or_features,
                      feature_sets_indices, column_identifier, performance_measure, models_name_list, report_type):
    
    output_data_frame = pd.DataFrame(columns = ['model name', 'history length', 'feature or covariate set'] + performance_measure)

    for model_name in models_name_list:
        for history in range(1, max_history+1):
            
            feature_sets = {feature_set_number:[] for feature_set_number in range(len(feature_sets_indices[history-1]))}
            for feature_set_number in range(len(feature_sets_indices[history-1])):
                # get the name of features in a current selected feature set which are in
                # a format 'temporal covariate x' or 'spatial covariate x'
                for index in feature_sets_indices[history-1][feature_set_number]:
                    feature_original_name = ordered_covariates_or_features[history-1][index]
                    feature_sets[feature_set_number].append(feature_original_name)

            temp = pd.DataFrame(columns = ['model name', 'history length', 'feature or covariate set'] + list(performance_measure))
            temp.loc[:,('feature or covariate set')] = list([feature_sets[feature_set_number] for feature_set_number in range(len(feature_sets_indices[history-1]))])
            temp.loc[:,('model name')] = model_name
            temp.loc[:,('history length')] = history
            
            for measure in performance_measure:
                errors_list = []
                for feature_set_number in range(len(feature_sets_indices[history-1])):
                    errors_list.append(errors_dict[measure][model_name][(history, feature_set_number)])
                temp.loc[:,(measure)] = list(errors_list)
            
            output_data_frame = output_data_frame.append(temp)
    
    address = './performance/validation process/'
    if not os.path.exists(address):
                os.makedirs(address)
    output_data_frame.to_csv(address + report_type + ' performance report.csv', index = False)
    
#############################################################

def parallel_run(prediction_arguments):
    train_predictions, validation_predictions, trained_model = train_evaluate(training_data = prediction_arguments[0],
                                                                              validation_data = prediction_arguments[1],
                                                                              model = prediction_arguments[2], 
                                                                              model_type = prediction_arguments[3],
                                                                              model_parameters = prediction_arguments[4], 
                                                                              verbose = prediction_arguments[5])
    return train_predictions, validation_predictions

    


def save_prediction_data_frame(models_name_list, fold_total_number, target_real_values, fold_validation_predictions,
                               fold_training_predictions, models_best_history_length, models_best_feature_set_number,
                               prediction_type):
    
    prediction_data_frame = pd.DataFrame(columns = ['model', 'spatial id', 'temporal id','real value', 'predicted value'])
    fold_number = 1
    
    for model_number, model_name in enumerate(models_name_list):

        model_best_history_length = models_best_history_length[model_name]
        model_best_feature_set_number = models_best_feature_set_number[model_name]

        if prediction_type == 'validation':
            temp = target_real_values[model_best_history_length][fold_number]['validation']
            temp.loc[:,('predicted value')] = fold_validation_predictions[model_name][(fold_number, model_best_history_length, model_best_feature_set_number)]
        else:
            temp = target_real_values[model_best_history_length][fold_number]['training']
            temp.loc[:,('predicted value')] = fold_training_predictions[model_name][(fold_number, model_best_history_length, model_best_feature_set_number)]
        
        temp.loc[:,('model')] = model_name
        temp = temp.drop(['Target'], axis = 1)
        temp.rename(columns = {'Normal target':'real value'}, inplace = True)
        temp = temp[['model', 'spatial id', 'temporal id','real value', 'predicted value']]

        prediction_data_frame = prediction_data_frame.append(temp)
    
    address = './predictions/validation process/'
    if not os.path.exists(address):
                os.makedirs(address)
    prediction_data_frame.to_csv(address + prediction_type + '_prediction.csv', index = False)
    

def get_trivial_values(train_true_values_df, validation_true_values_df, train_prediction, validation_prediction, forecast_horizon, granularity):
    
    train_true_values_df.loc[:,('prediction')] = train_prediction
    validation_true_values_df.loc[:,('prediction')] = validation_prediction
    number_of_spatial_units = len(train_true_values_df['spatial id'].unique())
    train_true_values_df = train_true_values_df.sort_values(by = ['temporal id', 'spatial id'])
    accessible_train_df = train_true_values_df.copy().iloc[(forecast_horizon * granularity * number_of_spatial_units):,:]
    train_true_values = list(np.array(accessible_train_df['Normal target']).reshape(-1))
    train_predicted_values = list(np.array(accessible_train_df['prediction']).reshape(-1))
    validation_true_values = list(np.array(validation_true_values_df['Normal target']).reshape(-1))
    validation_predicted_values = list(np.array(validation_true_values_df['prediction']).reshape(-1))
    val_size = len(validation_true_values)
    train_size = len(train_true_values)
    
    base_df = train_true_values_df.append(validation_true_values_df)
    base_df = base_df.sort_values(by = ['temporal id', 'spatial id'])
    base_df = base_df.iloc[:-(forecast_horizon * granularity * number_of_spatial_units),:]
    validation_trivial_values = list(np.array(base_df.tail(val_size)['Normal target']).reshape(-1))
    base_df = base_df.iloc[:-(val_size),:]
    train_trivial_values = list(np.array(base_df.tail(train_size)['Normal target']).reshape(-1))
    
    return train_true_values, train_predicted_values, train_trivial_values, validation_true_values, validation_predicted_values, validation_trivial_values

###########################################################################################
    
    
def train_validate(data, ordered_covariates_or_features, instance_validation_size, instance_testing_size,
                   fold_total_number, instance_random_partitioning = False, max_history_length = 1,
                   forecast_horizon = 1, models = ['knn'],  model_type = 'regression', splitting_type = 'training-validation',
                   performance_measure = ['MAPE'], benchmark = 'MAPE', input_scaler = None, output_scaler = None,
                   performance_report = True, granularity = 1, target_mode = 'normal', target_granularity = 1, column_identifier = None,
                   save_predictions = True, verbose = 1):
    
    
    supported_models_name = ['nn', 'knn', 'glm', 'gbm']
    supported_performance_measures = ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score']
    models_list = [] # list of models (str or callable)
    models_parameter_list = [] # list of models' parameters (dict or None)
    models_name_list = [] # list of models' names (str)
    

    ############################ reading and validating inputs
    
    ############## data input
    
    data_list = []
    if type(data) == list:
        for history in range(1,len(data)+1):
            if type(data[history-1]) == pd.DataFrame:
                data_list.append(data[history-1])
            elif type(data[history-1]) == str:
                try:
                    data_list.append(pd.read_csv(data[history-1]))
                except FileNotFoundError:
                    sys.exit("File '{0}' does not exist.".format(data[history-1]))
            else:
                sys.exit("The input data must be a list of DataFrames or strings of data addresses.")
    else:
        sys.exit("The input data must be a list of DataFrames or strings of data addresses.")
        
    max_history = len(data_list)
        
    ############## models input
    
    if type(models) != list:
        sys.exit("The models must be of type list.")
        
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
                        print("\nWarning: The values in the dictionary items of models list must be a dictionary of the model hyper parameter names and values. Other values will be ignored.\n")
            else:
                print("\nWarning: Each dictionary item in models list must contain only one item with a name of one of the supported models as a key and the parameters of that model as value. The incompatible cases will be ignored.\n")
        
        # if the item is only name of model whithout parameters
        elif type(item) == str:
            if (item in supported_models_name):
                if (item not in models_list):
                    models_list.append(item)
                    models_name_list.append(item)
                    models_parameter_list.append(None)
            else:
                print("\nWarning: The string items in the models list must be one of the supported models names. The incompatible cases will be ignored.\n")
        
        # if the item is user defined function
        elif callable(item):
            models_list.append(item)
            models_name_list.append('user_defined_' + str(callable_model_number))
            models_parameter_list.append(None)
            callable_model_number += 1
            
        else:
            print("\nWarning: The items in the models list must be of type string, dict or callable. The incompatible cases will be ignored.\n")
    
    if len(models_list) < 1:
        sys.exit("There is no item in the models list or the items are invalid.")
        
    ############## performance measure input
    
    if type(performance_measure) != list:
        sys.exit("The performance_measure must be of type list.")
        
    unsupported_measures = list(set(performance_measure)-set(supported_performance_measures))
    if len(unsupported_measures) > 0:
        print("\nWarning: Some of the specified measures are not valid:\n{0}\nThe supported measures are: ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score']\n".format(unsupported_measures))
    
    performance_measure = list(set([measure for measure in supported_performance_measures if measure in performance_measure]))
    
    if (splitting_type == 'cross-validation') and ('MASE' in performance_measure):
        performance_measure.remove('MASE')
        print("\nWarning: In cross-validation splitting mode, the MASE measure could not be calculated.\n")
    if len(performance_measure) < 1:
        sys.exit("No valid measure is specified.")
        
    ############## benchmark input
    
    if benchmark not in supported_performance_measures:
        print("\nWarning: The specified benchmark must be one of the supported performance measures: ['MAE', 'MAPE', 'MASE', 'MSE', 'R2_score']\nThe incompatible cases will be ignored and replaced with 'MAPE'.\n")
        benchmark = 'MAPE'
    # set the appropriate min error based on benchmark measure
    if benchmark in ['MAE', 'MAPE', 'MASE', 'MSE']:
        overall_min_validation_error = float('Inf')
        models_min_validation_error = {model_name : float('Inf') for model_name in models_name_list}
    else:
        overall_min_validation_error = float('-Inf')
        models_min_validation_error = {model_name : float('-Inf') for model_name in models_name_list}
    
    # checking validity of benchmark
    for history in range(1,max_history+1):
        data = data_list[history-1].copy()
        if (len(data[data['Target']==0]) > 0) and (benchmark == 'MAPE'):
                benchmark = 'MAE'
                print("\nWarning : The input data contain some zero values for Target variable. Therefore 'MAPE' can not be used as a benchmark and the benchmark will be set to 'MAE'.\n")
    
    ############## ordered_covariates_or_features and feature_sets_indices
    
    if type(ordered_covariates_or_features) == list:
        second_level_type_list = [isinstance(item,list) for item in ordered_covariates_or_features]
        second_level_type_str = [isinstance(item,str) for item in ordered_covariates_or_features]
        if all(second_level_type_list):
            feature_selection_type = 'feature'
            if len(ordered_covariates_or_features) != max_history:
                sys.exit("The number of feature lists in ordered_covariates_or_features does not match the number of input data.")
        elif all(second_level_type_str):
            feature_selection_type = 'covariate'
            repeated_list = [ordered_covariates_or_features for history in range(max_history)]
            ordered_covariates_or_features = repeated_list
    else:
        sys.exit("The ordered_covariates_or_features must be of type list.")
    
    feature_sets_indices = [] # feature_set_indices set of all history lengths
    for history in range(max_history):
        history_feature_sets_indices = [] # feature_set_indices for specific history length
        for number_of_features in range(1,len(ordered_covariates_or_features[history])+1):
            history_feature_sets_indices.append(list(range(number_of_features)))
        feature_sets_indices.append(history_feature_sets_indices)
                
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
    target_real_values = {history:{fold_number : {'training':None,'validation':None} for fold_number in range(1, fold_total_number + 1)} for history in range(1,max_history+1)}
    
    # train_data for each history and feature set index (will be used to train best model using train data with the best history length and feature set)
    train_data_dict = {} 

    # validation and training error of different measures for each model
    validation_errors = {measure: {model_name: {} for model_name in models_name_list} for measure in performance_measure}
    training_errors = {measure: {model_name: {} for model_name in models_name_list} for measure in performance_measure}
    
    knn_alert_flag = 0
    
    #################################################### main part
    
    # (loop over history_length, feature_sets_indices, and folds)
    
    for history in range(1,max_history+1):
        
        print("\n"+"-"*55+"\nValidation process is running for history length = {0}.\n".format(history)+"-"*55+"\n")
        
        # get the data with specific history length
        data = data_list[history-1].copy()
        
        # removing the last part of the data which is related to the last temporal units and contains
        # null values for target
        number_of_spatial_units = len(data['spatial id'].unique())
        data = data.iloc[:-(forecast_horizon * granularity * number_of_spatial_units)].copy()
        
        # separating the test part
        if splitting_type == 'training-validation-testing' :
            raw_train_data, _ , raw_testing_data = split_data(data = data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = instance_testing_size,
                                                      instance_validation_size = None, fold_total_number = None, fold_number = None, splitting_type = 'instance',
                                                      instance_random_partitioning = instance_random_partitioning, granularity = granularity, verbose = 0)
        else:
            raw_train_data = data.copy()
            
        
        # initializing the pool for parallel run
        prediction_pool = Pool(processes = len(feature_sets_indices[history-1]) * fold_total_number * len(models_list) + 5)
        pool_list = [] # list of all the different combination of the arguments of pool function
        
        for feature_set_number in range(len(feature_sets_indices[history-1])):
            
            indices = feature_sets_indices[history-1][feature_set_number]
            names_to_select = [ordered_covariates_or_features[history-1][index] for index in indices]
            
            # select the features
            train_data = select_features(data = raw_train_data.copy(), ordered_covariates_or_features = names_to_select)
            
            # holding train data with different histories and feature_set_indices to train best model in last step of function
            train_data_dict[(history,feature_set_number)] = train_data.copy()
                
            for model_number, model in enumerate(models_list):
                    
                model_parameters = models_parameter_list[model_number]
                model_name = models_name_list[model_number]

                for fold_number in range(1, fold_total_number + 1):
                    
                    # get the current fold training and validation data
                    training_data, validation_data, _ = split_data(data = train_data.copy(), forecast_horizon = forecast_horizon, instance_testing_size = None,
                                                  instance_validation_size = instance_validation_size, fold_total_number = fold_total_number, fold_number = fold_number,
                                                  splitting_type = split_data_splitting_type, instance_random_partitioning = instance_random_partitioning, 
                                                  granularity = granularity, verbose = 0)

                    
                    if (model_parameters is not None) and ('n_neighbors' in model_parameters.keys()) and (type(model_parameters['n_neighbors']) == int):
                        if (model_parameters['n_neighbors']<len(training_data)) and (knn_alert_flag == 0):
                            print("\nWarning: The number of neighbors for KNN algorithm is not specified or is too large for input data shape.")
                            print("The number of neighbors will be set to the best number of neighbors obtained by grid search in the range [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200]\n")
                            knn_alert_flag = 1
                    
                    # saving the target real values of each fold for data with different history lengths to use in
                    # calculation of performance and saving real and predicted target values in csv files
                    needed_columns = ['spatial id', 'temporal id', 'Target']
                    if 'Normal target' in training_data.columns:
                        needed_columns = needed_columns + ['Normal target']
                    
                    target_real_values[history][fold_number]['training'] = training_data[needed_columns]
                    target_real_values[history][fold_number]['validation'] = validation_data[needed_columns]
                    
                    # scaling features and target based on input_scaler and output_scaler
                    training_data, validation_data = data_scaling(train_data = training_data, test_data = validation_data, input_scaler = input_scaler, output_scaler = output_scaler)
                    
                    # add the current fold data, model name and model parameters to the list of pool function arguments
                    pool_list.append(tuple((training_data, validation_data, model, model_type, model_parameters, 0)))
                    
                    
                    
        # running the models in parallel
        parallel_output = prediction_pool.map(partial(parallel_run),tqdm(list(pool_list)))
        prediction_pool.close()
        prediction_pool.join()
        
        ####################### get outputs, calculate and save the performance
        
        pool_index = 0 # the index of pool results
        
        for feature_set_number in range(len(feature_sets_indices[history-1])):
            
            for model_number, model in enumerate(models_list):
                
                model_name = models_name_list[model_number]
                # initializing a dictionary for hold each folds training and validation error for the current model
                fold_validation_error = {fold_number : {measure: None for measure in supported_performance_measures} for fold_number in range(1, fold_total_number + 1)}
                fold_training_error = {fold_number : {measure: None for measure in supported_performance_measures} for fold_number in range(1, fold_total_number + 1)}
                
                for fold_number in range(1, fold_total_number + 1):
                    
                    # save the predefined models prediction output for the current fold
                    fold_training_predictions[model_name][(fold_number, history, feature_set_number)] = parallel_output[pool_index][0]
                    fold_validation_predictions[model_name][(fold_number, history, feature_set_number)] = parallel_output[pool_index][1]
                    # trained_model[model_name][(fold_number, history, feature_set_number)] = parallel_output[pool_index][2]
                    
                    # descale the predictions
                    fold_training_predictions[model_name][(fold_number, history, feature_set_number)] = target_descale(
                                                                                                        scaled_data = fold_training_predictions[model_name][(fold_number, history, feature_set_number)],
                                                                                                        base_data = list(np.array(target_real_values[history][fold_number]['training']['Target']).reshape(-1)), 
                                                                                                        scaler = output_scaler)
                    fold_validation_predictions[model_name][(fold_number, history, feature_set_number)] = target_descale(
                                                                                                        scaled_data = fold_validation_predictions[model_name][(fold_number, history, feature_set_number)], 
                                                                                                        base_data = list(np.array(target_real_values[history][fold_number]['training']['Target']).reshape(-1)), 
                                                                                                        scaler = output_scaler)
                   
                    # get the normal values of the target variable and predictions for the cumulative, differential,
                    # and moving average modes
                    
                    target_real_values[history][fold_number]['training'], target_real_values[history][fold_number]['validation'],\
                    fold_training_predictions[model_name][(fold_number, history, feature_set_number)],\
                    fold_validation_predictions[model_name][(fold_number, history, feature_set_number)] = \
                                            get_normal_target(training_target = target_real_values[history][fold_number]['training'],
                                                               test_target = target_real_values[history][fold_number]['validation'],
                                                               training_prediction = fold_training_predictions[model_name][(fold_number, history, feature_set_number)],
                                                               test_prediction = fold_validation_predictions[model_name][(fold_number, history, feature_set_number)],
                                                               target_mode = target_mode, target_granularity = target_granularity)
                    
                    
                    
                    pool_index = pool_index + 1

                    
                    # calculate and store the performance measure for the current fold
                    for measure in performance_measure:

                        train_df = target_real_values[history][fold_number]['training']
                        validation_df = target_real_values[history][fold_number]['validation']


                        if measure != 'MASE':
                            train_true_values = list(np.array(train_df['Normal target']).reshape(-1))
                            train_predicted_values = fold_training_predictions[model_name][(fold_number, history, feature_set_number)]
                            train_trivial_values = []
                            validation_true_values = list(np.array(validation_df['Normal target']).reshape(-1))
                            validation_predicted_values = fold_validation_predictions[model_name][(fold_number, history, feature_set_number)]
                            validation_trivial_values = []

                        if measure == 'MASE':

                            train_true_values, train_predicted_values, train_trivial_values,\
                            validation_true_values, validation_predicted_values,\
                            validation_trivial_values = get_trivial_values(
                            train_true_values_df = train_df, validation_true_values_df = validation_df,
                                train_prediction = fold_training_predictions[model_name][(fold_number, history, feature_set_number)],
                                validation_prediction = fold_validation_predictions[model_name][(fold_number, history, feature_set_number)], 
                                forecast_horizon = forecast_horizon, granularity = granularity)
                            print(validation_trivial_values)


                        fold_validation_error[fold_number][measure] = performance(true_values = validation_true_values,
                                                                                  predicted_values = validation_predicted_values, 
                                                                                  performance_measures = list([measure]), 
                                                                                  trivial_values = validation_trivial_values)
                        fold_training_error[fold_number][measure] = performance(true_values = train_true_values,
                                                                                  predicted_values = train_predicted_values,
                                                                                  performance_measures = list([measure]),
                                                                                  trivial_values = train_trivial_values)
            
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
                        else:
                            if validation_errors[measure][model_name][(history, feature_set_number)] > models_min_validation_error[model_name]:
                                models_min_validation_error[model_name] = validation_errors[measure][model_name][(history, feature_set_number)]
                                models_best_history_length[model_name] = history
                                models_best_feature_set_number[model_name] = feature_set_number
    
    
    #################################################### saving predictions
    
    # save the real and predicted value of target variable in training and validation set for each model
    
    if (save_predictions == True) and (fold_total_number == 1): # if cross validation mode is on, predictions are not saved
        
        save_prediction_data_frame(models_name_list, fold_total_number, target_real_values, fold_validation_predictions,
                                   fold_training_predictions, models_best_history_length, models_best_feature_set_number,
                                   'training')
        save_prediction_data_frame(models_name_list, fold_total_number, target_real_values, fold_validation_predictions,
                                   fold_training_predictions, models_best_history_length, models_best_feature_set_number,
                                   'validation')
        
    #################################################### reporting performance
    
    if performance_report == True:
        
        report_performance(errors_dict = validation_errors, max_history = max_history, ordered_covariates_or_features = ordered_covariates_or_features,
                          feature_sets_indices = feature_sets_indices, column_identifier = column_identifier, performance_measure = performance_measure, models_name_list = models_name_list, report_type = 'validation')
        report_performance(errors_dict = training_errors, max_history = max_history, ordered_covariates_or_features = ordered_covariates_or_features,
                          feature_sets_indices = feature_sets_indices, column_identifier = column_identifier, performance_measure = performance_measure, models_name_list = models_name_list, report_type = 'training')
        
    
    
    #################################################### finding best model and overall best history length and feature set
    
    for model_number, model_name in enumerate(models_name_list):
        if benchmark in ['MAE', 'MAPE', 'MASE', 'MSE']:
            if models_min_validation_error[model_name] < overall_min_validation_error:
                overall_min_validation_error = models_min_validation_error[model_name]
                best_history_length = models_best_history_length[model_name]
                best_feature_set_number = models_best_feature_set_number[model_name]
                best_feature_sets_indices = feature_sets_indices[best_history_length-1][best_feature_set_number]                # best_trained_model = models_best_trained_model[model_name]
                best_model = models_list[model_number]
                best_model_number = model_number
        else:
            if models_min_validation_error[model_name] > overall_min_validation_error:
                overall_min_validation_error = models_min_validation_error[model_name]
                best_history_length = models_best_history_length[model_name]
                best_feature_set_number = models_best_feature_set_number[model_name]
                best_feature_sets_indices = feature_sets_indices[best_history_length-1][best_feature_set_number]
                best_model = models_list[model_number]
                best_model_number = model_number
                
    # training the best model using the data with the overall best history length and feature set
    best_train_data = train_data_dict[(best_history_length,best_feature_set_number)]
    
    best_model_parameters = models_parameter_list[best_model_number]
    
    best_train_data, _ = data_scaling(train_data = best_train_data, test_data = best_train_data, input_scaler = input_scaler, output_scaler = output_scaler)
    
    _, _, best_trained_model = train_evaluate(training_data = best_train_data,
                                              validation_data = best_train_data,
                                              model = best_model, model_type = model_type,
                                              model_parameters = best_model_parameters,
                                              verbose = 0)
    
    best_feature_or_covariate_set = [ordered_covariates_or_features[best_history_length-1][index] for index in best_feature_sets_indices]
    
    return best_model, best_model_parameters, best_history_length, best_feature_or_covariate_set, best_trained_model
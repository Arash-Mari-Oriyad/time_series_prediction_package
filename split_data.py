import pandas as pd 
import numpy as np
import sys

def temporal_shuffle(data):
    # create df of temporal ids, shuffle the temporal ids in the df and
    # rearrange data based on order of temporal ids in the shuffled df
    temporal_id_df = data[['temporal id']].drop_duplicates()
    temporal_id_df = temporal_id_df.iloc[np.random.permutation(len(temporal_id_df))]
    temporal_id_df = temporal_id_df.reset_index(drop = True).reset_index()
    data = pd.merge(data, temporal_id_df, how = 'left', on=['temporal id'])
    data = data.sort_values(by = ['index','spatial id'])
    data = data.drop(['index'], axis = 1)
    
    return data


def split_data(data, forecast_horizon, instance_testing_size, instance_validation_size, fold_total_number,
               fold_number, splitting_type = 'instance', instance_random_partitioning = False,
               granularity = 1, verbose = 0):
    
    if type(data) == str:
        try:
            data = pd.read_csv(data)
        except FileNotFoundError:
            sys.exit("File '{0}' does not exist.".format(data))
        
    
    # initializing
    training_data = None
    validation_data = None
    testing_data = None
    
    gap = (forecast_horizon - 1) * granularity # number of temporal units to be removed
    
    if (splitting_type == 'fold') and (instance_testing_size is not None):
        sys.exit("The cross validation method must not to be used for the splitting data to the training and testing set.")
    
    
    number_of_spatial_units = len(data['spatial id'].unique())
    number_of_temporal_units = len(data['temporal id'].unique())
    
    data.sort_values(by = ['temporal id','spatial id'], inplace = True)
    
    if splitting_type == 'instance':
        
        # check the type of instance_testing_size and instance_validation_size
        if type(instance_testing_size) == float:
            if instance_testing_size > 1:
                sys.exit("The float instance_testing_size will be interpreted to the proportion of data that is considered as the test set and must be less than 1.")
            instance_testing_size = round(instance_testing_size * (number_of_temporal_units))
        elif (type(instance_testing_size) != int) and (instance_testing_size is not None):
            sys.exit("The type of instance_testing_size must be int or float.")

        if type(instance_validation_size) == float:
            if instance_validation_size > 1:
                sys.exit("The float instance_validation_size will be interpreted to the proportion of data which is considered as validation set and must be less than 1.")
            
            if instance_testing_size is not None:
                instance_validation_size = round(instance_validation_size * (number_of_temporal_units - instance_testing_size))
            else:
                instance_validation_size = round(instance_validation_size * (number_of_temporal_units))
                
        elif (type(instance_validation_size) != int) and (instance_validation_size is not None):
            sys.exit("The type of instance_validation_size must be int or float.")                
                
                
        if (instance_testing_size is not None) and (instance_validation_size is None):
            if instance_testing_size > len(data):
                sys.exit("The specified instance_testing_size is too large for input data.")
            testing_data = data.tail(instance_testing_size * number_of_spatial_units).copy()
            training_data = data.iloc[:-((instance_testing_size + gap) * number_of_spatial_units)].copy()
            if verbose > 0:
                print("The splitting of the data is running. The training set includes {0}, and the testing set includes {1} instances.\n".format(len(training_data),len(testing_data)))
        
        elif (instance_testing_size is None) and (instance_validation_size is not None):
            if (instance_validation_size* number_of_spatial_units) > len(data):
                sys.exit("The specified instance_validation_size is too large for input data.")
            # shuffling the temporal ids in the data for random partitioning
            if instance_random_partitioning == True:
                data = temporal_shuffle(data.copy())
            validation_data = data.tail(instance_validation_size * number_of_spatial_units).copy()
            training_data = data.iloc[:-(instance_validation_size * number_of_spatial_units)].copy()
            if verbose > 0:
                print("The splitting of the data is running. The training set includes {0}, and the validation set includes {1} instances.\n".format(len(training_data),len(validation_data)))
        
        elif (instance_testing_size is not None) and (instance_validation_size is not None):
            if ((instance_testing_size + instance_validation_size)* number_of_spatial_units) > len(data):
                sys.exit("The specified instance_testing_size and instance_validation_size are too large for input data.")
            testing_data = data.tail(instance_testing_size * number_of_spatial_units).copy()
            train_data = data.iloc[:-((instance_testing_size + gap) * number_of_spatial_units)].copy()
            if instance_random_partitioning == True:
                train_data = temporal_shuffle(train_data.copy())
            validation_data = train_data.tail(instance_validation_size * number_of_spatial_units).copy()
            training_data = train_data.iloc[:-(instance_validation_size * number_of_spatial_units)].copy()
            if verbose > 0:
                print("The splitting of the data is running. The training set, validation set, and testing set includes {0}, {1}, {2} instances respectively.\n".format(len(training_data),len(validation_data),len(testing_data)))
        
        else:
            sys.exit("If the type of splitting is 'instance' at least one of the instance_validation_size and instance_testing_size must have a value.")
            
    elif splitting_type == 'fold':
        
        if (fold_total_number is None) or (fold_number is None):
            sys.exit("if the splitting_type is 'fold', the fold_total_number and fold_number must be specified.")
        if (type(fold_total_number) != int) or (type(fold_number) != int):
            sys.exit("The fold_total_number and fold_number must be of type int.")
        elif (fold_number > fold_total_number) or (fold_number < 1):
            sys.exit("The fold_number must be a number in a range between 1 and fold_total_number.")
            
        
        temporal_unit_list = data['temporal id'].unique()
        k_fold = KFold(n_splits=fold_total_number)
        iteration = 0
        for training_index, validation_index in k_fold.split(temporal_unit_list):
            
            training_fold_temporal_units = temporal_unit_list[training_index]
            validation_fold_temporal_units = temporal_unit_list[validation_index]
            
            training_data = data[data['temporal id'].isin(training_fold_temporal_units)]
            validation_data = data[data['temporal id'].isin(validation_fold_temporal_units)]
            
            iteration += 1
            if iteration == fold_number:
                break
        if verbose == 1:
            print("The splitting of the data is running. The validation set is fold number {0} of the total of {1} folds. Each fold includes {2} instances.".format(fold_number, fold_total_number, len(validation_fold_temporal_units)))
        
    else:
        sys.exit("The splitting type must be 'instance' or 'fold'.")
        
    return training_data, validation_data, testing_data

def apply_performance_mode(training_target, test_target, training_prediction, test_prediction, performance_mode):
    
    '''
    ::: input :::
    training_target : a data frame including columns 'spatial id', 'temporal id', 'Normal target'
                        from the training set (extra columns allowed)
                        
    test_target : a data frame including columns 'spatial id', 'temporal id', 'Target', 'Normal target'
                        from the test set (extra columns allowed)
                        
    training_prediction : list of predicted values for the training set
    test_prediction : list of predicted values for the test set
    
    performance_mode = 'normal' , 'cumulative' , 'moving average + x' the desired mode of the target variable
                        when calculating the performance
                        
    ::: output :::
    
    training_target : a data frame with the same columns as input training_target with the modified
                        values for the 'Normal target' based on performance_mode
                        
    test_target : a data frame with the same columns as input test_target with the modified
                        values for the 'Normal target' based on performance_mode
                        
    training_prediction : list of modified predicted values for the training set based on performance_mode
    test_prediction : list of modified predicted values for the test set based on performance_mode
    
    '''
    column_names_list = list(training_target.columns)
    
    # decode performance mode to get the window size of the moving average
    if performance_mode.startswith('moving average'):
        if len(performance_mode.split(' + ')) > 1:
            window = performance_mode.split(' + ')[-1]
            performance_mode = 'moving average'
        else:
            sys.exit("For the moving average performance_mode the window size must also be specifid in the performance_mode with the format 'moving average + x' where x is the window size.")
        try:
            window = int(window)
        except ValueError:
            sys.exit("The specified window for the moving average performance_mode is not valid.")
    
    training_target.loc[:,('type')] = 1
    test_target.loc[:,('type')] = 2
    
    training_target.loc[:,('prediction')] = training_prediction
    test_target.loc[:,('prediction')] = test_prediction
    
    data = training_target.append(test_target)
    data = data.sort_values(by = ['temporal id','spatial id'])
    temporal_ids = data['temporal id'].unique()
    
    # if performance mode is cumulative, the cumulative values of the target and prediction is calculated
    if performance_mode == 'cumulative':
        
        target_df = data.pivot(index='temporal id', columns='spatial id', values='Normal target')
        prediction_df = data.pivot(index='temporal id', columns='spatial id', values='prediction')

        target_df = target_df.cumsum()
        prediction_df = prediction_df.cumsum()

        target_df = pd.melt(target_df.reset_index(), id_vars='temporal id', value_vars=list(target_df.columns),
                             var_name='spatial id', value_name='Normal target')
        prediction_df = pd.melt(prediction_df.reset_index(), id_vars='temporal id', value_vars=list(prediction_df.columns),
                             var_name='spatial id', value_name='prediction')


        data = data.drop(['Normal target','prediction'], axis = 1)
        data = pd.merge(data, target_df, how = 'left')
        data = pd.merge(data, prediction_df, how = 'left')
        
    elif performance_mode == 'moving average':
        if window > len(temporal_ids):
            sys.exit("The specified window for the moving average performance_mode is too large for the input data.")
        
        number_of_spatial_units = len(data['spatial id'].unique())
        
        target_df = data.pivot(index='temporal id', columns='spatial id', values='Normal target')
        prediction_df = data.pivot(index='temporal id', columns='spatial id', values='prediction')

        target_df = target_df.rolling(window).mean()
        prediction_df = prediction_df.rolling(window).mean()

        target_df = pd.melt(target_df.reset_index(), id_vars='temporal id', value_vars=list(target_df.columns),
                             var_name='spatial id', value_name='Normal target')
        prediction_df = pd.melt(prediction_df.reset_index(), id_vars='temporal id', value_vars=list(prediction_df.columns),
                             var_name='spatial id', value_name='prediction')


        data = data.drop(['Normal target','prediction'], axis = 1)
        data = pd.merge(data, target_df, how = 'left')
        data = pd.merge(data, prediction_df, how = 'left')
        
        data = data.sort_values(by = ['temporal id', 'spatial id'])
        
        data = data.iloc[(window-1)*number_of_spatial_units:]
        
    elif performance_mode != 'normal':
        sys.exit("Specified performance_mode is not valid.")
        
    data = data.sort_values(by=['temporal id', 'spatial id'])
    training_set = data[data['type'] == 1]
    test_set = data[data['type'] == 2]
    
    if (len(test_set) < 1) and (performance_mode == 'moving average'):
        sys.exit("The number of remaining instances in the test set is less than one when applying moving average performance_mode (the first 'window - 1' temporal units is removed in the process)")
    if (len(training_set) < 1) and (performance_mode == 'moving average'):
        print("\nWarning: The number of remaining instances in the training set is less than one when applying moving average performance_mode (the first 'window - 1' temporal units is removed in the process).\n")

    training_prediction = list(training_set['prediction'])
    test_prediction = list(test_set['prediction'])

    training_target = training_set.drop(['type','prediction'], axis = 1)
    test_target = test_set.drop(['type','prediction'], axis = 1)
    
    training_target = training_target[column_names_list]
    test_target = test_target[column_names_list]
    
    return training_target, test_target, training_prediction, test_prediction

def get_trivial_values(train_true_values_df, validation_true_values_df, train_prediction, validation_prediction, forecast_horizon, granularity):
    
    '''
    :::inputs:::
    
    train_true_values_df : a data frame including columns 'spatial id', 'temporal id', 'Target', 'Normal target'
                        from the training set
    validation_true_values_df : a data frame including columns 'spatial id', 'temporal id', 'Target', 'Normal target'
                        from the test set           
    train_prediction : list of predicted values for the training set
    validation_prediction : list of predicted values for the test set
    forecast_horizon : number of temporal units in the future to be forecasted
    granularity : number of smaller scale temporal units which is averaged to get the values 
                    of bigger scale unit in the temporal transformation process
    
    :::outputs:::
    
    train_true_values: a list target real values in the training set
    train_predicted_values: a list predicted values for the training set
    train_trivial_values: a list of trivial values for the training set
    validation_true_values: a list target real values in the validation set
    validation_predicted_values: a list predicted values for the validation set
    validation_trivial_values: a list of trivial values for the validation set
    
    '''
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

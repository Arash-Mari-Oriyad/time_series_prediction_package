import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def data_scaling(traing_data, testing_data, input_scaler = 'logarithmic', output_scaler = 'logarithmic'):
    
    '''
    output_scaler = 'logarithmic' or 'normalize' or 'standardize' or None : the scaler used to scale target variable
    input_scaler = 'logarithmic' or 'normalize' or 'standardize' or None : the scaler used to scale features
    '''
    if input_scaler is not None:
        
        columns = list(traing_data.columns)
        features = [column for column in columns if column not in ['spatial id','temporal id','Target']]
        train_data_features = traing_data[features]
        test_data_features = testing_data[features]
        
        if input_scaler == 'logarithmic':
            for feature in features:
                if (len(testing_data[testing_data[feature]<0]) > 0) or (len(traing_data[traing_data[feature]<0]) > 0) :
                    sys.exit("The features includes negative values. So the logarithmic input_scaler couldn't be applied.")
                testing_data.loc[:,(feature)] = list(np.log((testing_data[feature] + 1).astype(float)))
                traing_data.loc[:,(feature)] = list(np.log((traing_data[feature] + 1).astype(float)))

        else:
            if input_scaler == 'standardize':       
                scaleObject = StandardScaler()
            if input_scaler == 'normalize':        
                scaleObject = MinMaxScaler()
            
            scaleObject.fit(train_data_features)
            train_data_features = scaleObject.transform(train_data_features)
            test_data_features = scaleObject.transform(test_data_features)
            
            test_data = pd.concat([testing_data[['spatial id', 'temporal id', 'Target']].reset_index(drop = True), pd.DataFrame(test_data_features, columns = features).reset_index(drop = True)], axis=1)
            train_data = pd.concat([traing_data[['spatial id', 'temporal id', 'Target']].reset_index(drop = True), pd.DataFrame(train_data_features, columns = features).reset_index(drop = True)], axis=1)
        
        
    if output_scaler is not None:
        
        if output_scaler == 'logarithmic':
            if (len(test_data[test_data['Target']<0]) > 0) or (len(train_data[train_data['Target']<0]) > 0) :
                sys.exit("The target variable includes negative values. So the logarithmic output_scaler couldn't be applied.")
            test_data.loc[:,('Target')] = list(np.log((test_data['Target'] + 1).astype(float)))
            train_data.loc[:,('Target')] = list(np.log((train_data['Target'] + 1).astype(float)))
            
        else:
            if output_scaler == 'standardize':       
                scaleObject = StandardScaler()
            if output_scaler == 'normalize':        
                scaleObject = MinMaxScaler()
        
            train_data_target = np.array(train_data['Target']).reshape(len(train_data), 1)
            test_data_target = np.array(test_data['Target']).reshape(len(test_data), 1)
            scaleObject.fit(train_data_target)
            train_data_target = scaleObject.transform(train_data_target)
            test_data_target = scaleObject.transform(test_data_target)
            
            test_data.loc[:,('Target')] = list(test_data_target)
            train_data.loc[:,('Target')] = list(train_data_target)
            
    return train_data, test_data

def target_descale(scaled_data, base_data, scaler = 'logarithmic'):
    
    '''
    scaled_data : List of target variable predicted values in test set (scaled)
    base_data : List of target variable values in train set before scaling
    scaler = 'logarithmic' or 'normalize' or 'standardize' or None
    '''
    
    base_data = np.array(base_data).reshape(-1, 1)
    scaled_data = np.array(scaled_data).reshape(-1, 1)
    
    if scaler == 'logarithmic':
        output_data = np.exp(scaled_data) - 1
        
    elif scaler == 'standardize':        
        scaleObject = StandardScaler()
        scaleObject.fit_transform(base_data)
        output_data = scaleObject.inverse_transform(scaled_data)
        
    elif scaler == 'normalize':        
        scaleObject = MinMaxScaler()
        scaleObject.fit_transform(base_data)
        output_data = scaleObject.inverse_transform(scaled_data)
    else:
        output_data = scaled_data
    
    return list(output_data.ravel())
import os
import sys

import pandas as pd

import configurations
from get_normal_target import get_normal_target
from get_target_quantities import get_target_quantities
from scaling import data_scaling, target_descale
from select_features import select_features
from train_evaluate import train_evaluate
from get_target_temporal_ids import get_target_temporal_ids


def predict_future(data: pd.DataFrame or str,
                   future_data: pd.DataFrame or str,
                   forecast_horizon: int,
                   feature_scaler: str or None,
                   target_scaler: str or None,
                   feature_or_covariate_set: list,
                   model_type: str,
                   labels: list or None,
                   model: str or callable,
                   base_models: list,
                   model_parameters: dict or None,
                   scenario: str or None,
                   save_predictions: bool,
                   verbose: int):
    # input checking
    # data input checking
    if not (isinstance(data, pd.DataFrame) or isinstance(data, str)):
        sys.exit("Error: The input 'data' must be a dataframe or an address to a DataFrame.")
    # future_data input checking
    if not (isinstance(future_data, pd.DataFrame) or isinstance(future_data, str)):
        sys.exit("Error: The input 'future_data' must be a DataFrame or an address to a dataframe.")
    # forecast_horizon input checking
    if not (isinstance(forecast_horizon, int) and forecast_horizon >= 1):
        sys.exit("Error: The input 'forecast_horizon' must be an integer and greater than zero.")
    # feature_scaler input checking
    if feature_scaler not in configurations.FEATURE_SCALERS:
        sys.exit(f"Error: The input 'feature_scaler' is not valid. Valid options are: {configurations.FEATURE_SCALERS}")
    # target_scaler input checking
    if target_scaler not in configurations.TARGET_SCALERS:
        sys.exit(f"Error: The input 'target_scaler' is not valid. Valid options are: {configurations.TARGET_SCALERS}")
    # feature_or_covariate_set input checking
    if not isinstance(feature_or_covariate_set, list):
        sys.exit("Error: The input 'feature_or_covariate_set' must be a list of features or covariates.")
    for feature_or_covariate in feature_or_covariate_set:
        if not isinstance(feature_or_covariate, str):
            sys.exit("Error: The input 'feature_or_covariate_set' must be a list of features or covariates.")
    # model_type input checking
    if model_type not in configurations.MODEL_TYPES:
        sys.exit(f"Error: The input 'model_type' is not valid. Valid options are: {configurations.MODEL_TYPES}")
    # model input checking
    if not ((isinstance(model, str) and (model in configurations.PRE_DEFINED_MODELS or\
           model in ['mixed_'+name for name in configurations.PRE_DEFINED_MODELS])) or callable(model)):
        sys.exit(f"Error: The input 'model' must be whether one of the pre-defined models"
                 f" ({configurations.PRE_DEFINED_MODELS}) or a callable object as a custom model.")
    # model_parameters input checking
    if not (isinstance(model_parameters, dict) or model_parameters is None):
        sys.exit("Error: The input 'model_parameters' must be a dictionary of parameters or None.")
    # scenario input checking
    if scenario not in configurations.SCENARIOS:
        sys.exit(f"Error: The input 'scenario' is not valid. Valid options are {configurations.SCENARIOS}.")
    # save_predictions input checking
    if not isinstance(save_predictions, bool):
        sys.exit("Error: The input 'save_predictions' must be boolean.")
    # verbose
    if not (isinstance(verbose, int) and verbose in configurations.VERBOSE_OPTIONS):
        sys.exit(f"Error: The input 'verbose' is not valid. Valid options are {configurations.VERBOSE_OPTIONS}.")

    # data and future_data preparing
    if isinstance(data, str):
        try:
            data = pd.read_csv(data)
        except Exception as e:
            sys.exit(str(e))
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        sys.exit("The input 'data' must be a dataframe or an address to a DataFrame.")
    if isinstance(future_data, str):
        try:
            future_data = pd.read_csv(future_data)
        except Exception as e:
            sys.exit(str(e))
    elif isinstance(future_data, pd.DataFrame):
        pass
    else:
        sys.exit("The input 'future_data' must be a DataFrame or an address to a dataframe.")

    target_mode, target_granularity, granularity, training_data = get_target_quantities(data=data.copy())
    _, _, _, testing_data = get_target_quantities(data=future_data.copy())
    
    # get the target temporal ids as the temporal id
    if ('target temporal id' in training_data.columns) and ('target temporal id' in testing_data.columns):
        training_data = training_data.rename(columns={'target temporal id':'temporal id'})
        testing_data = testing_data.rename(columns={'target temporal id':'temporal id'})
    else:
        training_data['sort'] = 'train'
        testing_data['sort'] = 'test'
        
        training_data = get_target_temporal_ids(temporal_data = training_data, forecast_horizon = forecast_horizon,
                                               granularity = granularity)
        testing_data = get_target_temporal_ids(temporal_data = testing_data, forecast_horizon = forecast_horizon,
                                               granularity = granularity)
        

    training_data = select_features(data=training_data.copy(),
                                    ordered_covariates_or_features=feature_or_covariate_set)
    testing_data = select_features(data=testing_data.copy(),
                                   ordered_covariates_or_features=feature_or_covariate_set)

    futuristic_features = [column_name
                           for column_name in training_data.columns.values
                           if len(column_name.split()) > 1 and column_name.split()[1].startswith('t+')]

    if scenario:
        for spatial_id in testing_data['spatial id'].tolist():
            for futuristic_feature in futuristic_features:
                if scenario == 'max':
                    value = training_data.loc[training_data['spatial id'] == spatial_id, futuristic_feature].max()
                elif scenario == 'min':
                    value = training_data.loc[training_data['spatial id'] == spatial_id, futuristic_feature].min()
                elif scenario == 'mean':
                    value = training_data.loc[training_data['spatial id'] == spatial_id, futuristic_feature].mean()
                else:
                    value = training_data.loc[training_data['spatial id'] == spatial_id, futuristic_feature].values[-1]
                testing_data.loc[testing_data['spatial id'] == spatial_id, futuristic_feature] = value
    else:
        if len(futuristic_features) > 0 and not \
                all([testing_data.isna().sum()[futuristic_feature] == 0 for futuristic_feature in futuristic_features]):
            sys.exit("Error: The input 'scenario' is not provided and "
                     "some futuristic features have null values in the input 'future_data'.")

    scaled_training_data, scaled_testing_data = data_scaling(train_data=training_data.copy(),
                                                             test_data=testing_data.copy(),
                                                             feature_scaler=feature_scaler,
                                                             target_scaler=target_scaler)

    scaled_training_data.drop(configurations.NON_FEATURE_COLUMNS_NAMES, axis=1, inplace=True)
    scaled_training_data.drop(configurations.NORMAL_TARGET_COLUMN_NAME, axis=1, inplace=True)
    scaled_testing_data.drop(configurations.NON_FEATURE_COLUMNS_NAMES, axis=1, inplace=True)
    scaled_testing_data.drop(configurations.NORMAL_TARGET_COLUMN_NAME, axis=1, inplace=True)

    scaled_training_predictions, scaled_testing_predictions, trained_model = \
        train_evaluate(training_data=scaled_training_data,
                       validation_data=scaled_testing_data,
                       model_type=model_type,
                       model=model,
                       model_parameters=model_parameters,
                       labels=labels,
                       base_models=base_models,
                       verbose=verbose)

    training_predictions = target_descale(scaled_data=list(scaled_training_predictions),
                                          base_data=training_data['Target'].values.tolist(),
                                          scaler=target_scaler)
    testing_predictions = target_descale(scaled_data=list(scaled_testing_predictions),
                                         base_data=training_data['Target'].values.tolist(),
                                         scaler=target_scaler)

    normal_training_target, normal_testing_target, normal_training_predictions, normal_testing_predictions = \
        get_normal_target(
            training_target=training_data[['spatial id', 'temporal id', 'Target', 'Normal target']].copy(),
            test_target=testing_data[['spatial id', 'temporal id', 'Target', 'Normal target']].copy(),
            training_prediction=list(training_predictions),
            test_prediction=list(testing_predictions),
            target_mode=target_mode,
            target_granularity=target_granularity
        )

    testing_data_spatial_ids = normal_testing_target['spatial id'].copy()
    testing_data_temporal_ids = normal_testing_target['temporal id'].copy()

    data_to_save = pd.DataFrame()
    data_to_save.loc[:, 'spatial id'] = testing_data_spatial_ids
    data_to_save.loc[:, 'temporal id'] = testing_data_temporal_ids
    data_to_save.loc[:, 'model name'] = model if isinstance(model, str) else model.__name__
    data_to_save.loc[:, 'real'] = None
    if model_type == 'regression':
        data_to_save.loc[:, 'prediction'] = normal_testing_predictions
    else:
        converted_normal_testing_predictions = list(zip(*normal_testing_predictions))
        for index, label in enumerate(labels):
            data_to_save.loc[:, f'class {label}'] = list(converted_normal_testing_predictions[index])
            
    
    save_predictions_address = \
        f'prediction/future prediction/future prediction forecast horizon = {forecast_horizon}.csv'

    if save_predictions:
        if os.path.exists(save_predictions_address):
            old_saved_data = pd.read_csv(save_predictions_address)
            data_to_save = pd.concat([old_saved_data, data_to_save], ignore_index=True)
        else:
            if not os.path.exists('prediction'):
                os.mkdir('prediction')
                os.mkdir('prediction/future prediction')
            if not os.path.exists('prediction/future prediction'):
                os.mkdir('prediction/future prediction')
        data_to_save.to_csv(save_predictions_address, index=False)

    return trained_model
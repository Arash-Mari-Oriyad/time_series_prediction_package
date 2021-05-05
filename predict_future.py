import os
import sys

import pandas as pd

import configurations
from get_normal_target import get_normal_target
from get_target_quantities import get_target_quantities
from scaling import data_scaling, target_descale
from select_features import select_features
from train_evaluate import train_evaluate


def predict_future(data: pd.DataFrame or str,
                   future_data: pd.DataFrame or str,
                   forecast_horizon: int,
                   feature_scaler: str or None,
                   target_scaler: str or None,
                   feature_or_covariate_set: list,
                   model_type: str,
                   model: str or callable,
                   model_parameters: dict or None,
                   scenario: str or None,
                   save_predictions: bool,
                   verbose: int):
    # input checking
    # data input_checking
    if not (isinstance(data, pd.DataFrame) or isinstance(data, str)):
        sys.exit("data input format is not valid.")
    # future_data input checking
    if not (isinstance(future_data, pd.DataFrame) or isinstance(future_data, str)):
        sys.exit("future_data input format is not valid.")
    # forecast_horizon input checking
    if not (isinstance(forecast_horizon, int) and forecast_horizon >= 1):
        sys.exit("forecast_horizon is not valid.")
    # feature_scaler input checking
    if feature_scaler not in configurations.FEATURE_SCALERS:
        sys.exit("feature_scaler input is not valid.")
    # target_scaler input checking
    if target_scaler not in configurations.TARGET_SCALERS:
        sys.exit("target_scaler input is not valid.")
    # feature_or_covariate_set
    if not isinstance(feature_or_covariate_set, list):
        sys.exit("feature_or_covariate_set input format is not valid.")
    for feature_or_covariate in feature_or_covariate_set:
        if not isinstance(feature_or_covariate, str):
            sys.exit("feature_or_covariate_set input format is not valid.")
    # model_type
    if model_type not in configurations.MODEL_TYPES:
        sys.exit("model_type input is not valid.")
    # model
    if not ((isinstance(model, str) and model in configurations.PRE_DEFINED_MODELS) or callable(model)):
        sys.exit("model input is not valid.")
    # model_parameters
    if not (isinstance(model_parameters, dict) or model_parameters is None):
        sys.exit("model_parameters input format is not valid.")
    # scenario
    if not ((isinstance(scenario, str) and scenario in configurations.SCENARIOS) or scenario is None):
        sys.exit("scenario input is not valid.")
    # save_predictions
    if not isinstance(save_predictions, bool):
        sys.exit("save_predictions input format is not valid.")
    # verbose
    if not (isinstance(verbose, int) and verbose in configurations.VERBOSE_OPTIONS):
        sys.exit("verbose input format is not valid.")

    # data and future_data preparing
    if isinstance(data, str):
        try:
            data = pd.read_csv(data)
        except Exception as e:
            sys.exit(str(e))
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        sys.exit("data input format is not valid.")
    if isinstance(future_data, str):
        try:
            future_data = pd.read_csv(future_data)
        except Exception as e:
            sys.exit(str(e))
    elif isinstance(future_data, pd.DataFrame):
        pass
    else:
        sys.exit("data input format is not valid.")

    target_mode, target_granularity, granularity, training_data = get_target_quantities(data=data.copy())
    _, _, _, testing_data = get_target_quantities(data=future_data.copy())

    testing_data_spatial_ids = testing_data['spatial id'].copy()
    testing_data_temporal_ids = testing_data['temporal id'].copy()

    training_data = select_features(data=training_data.copy(),
                                    ordered_covariates_or_features=feature_or_covariate_set)
    testing_data = select_features(data=testing_data.copy(),
                                   ordered_covariates_or_features=feature_or_covariate_set)

    futuristic_features = [column_name
                           for column_name in data.columns.values
                           if len(column_name.split()) > 1 and column_name.split()[1].startswith('t+')]

    if scenario:
        for futuristic_feature in futuristic_features:
            if scenario == 'max':
                value = training_data[futuristic_feature].max()
            elif scenario == 'min':
                value = training_data[futuristic_feature].min()
            elif scenario == 'mean':
                value = training_data[futuristic_feature].mean()
            else:
                value = training_data[futuristic_feature].values[-1]
            testing_data[futuristic_feature].values[:] = value

    else:
        if not all([testing_data.isna().sum()[futuristic_feature] == 0 for futuristic_feature in futuristic_features]):
            sys.exit("scenario is not provided and some futuristic features have null values.")

    scaled_training_data, scaled_testing_data = data_scaling(train_data=training_data.copy(),
                                                             test_data=testing_data.copy(),
                                                             input_scaler=feature_scaler,
                                                             output_scaler=target_scaler)

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
                       verbose=verbose)

    training_predictions = target_descale(scaled_data=list(scaled_training_predictions),
                                          base_data=training_data['Target'].values.tolist(),
                                          scaler=target_scaler)
    testing_predictions = target_descale(scaled_data=list(scaled_testing_predictions),
                                         base_data=training_data['Target'].values.tolist(),
                                         scaler=target_scaler)

    normal_training_target, normal_testing_target, normal_training_prediction, normal_testing_prediction = \
        get_normal_target(
            training_target=training_data[['spatial id', 'temporal id', 'Target', 'Normal target']].copy(),
            test_target=testing_data[['spatial id', 'temporal id', 'Target', 'Normal target']].copy(),
            training_prediction=list(training_predictions),
            test_prediction=list(testing_predictions),
            target_mode=target_mode,
            target_granularity=target_granularity
        )

    data_to_save = pd.DataFrame()
    data_to_save.loc[:, 'spatial id'] = testing_data_spatial_ids
    data_to_save.loc[:, 'temporal id'] = testing_data_temporal_ids
    data_to_save.loc[:, 'model name'] = model if isinstance(model, str) else model.__name__
    data_to_save.loc[:, 'real'] = None
    data_to_save.loc[:, 'prediction'] = pd.Series(normal_testing_prediction)

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
        data_to_save.to_csv(save_predictions_address)

    return trained_model

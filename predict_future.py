import sys

import pandas as pd

import configurations
from scaling import data_scaling, target_descale
from select_features import select_features

# from split_data import split_data
# from train_evaluate import train_evaluate
from train_evaluate import train_evaluate


def predict_future(data: pd.DataFrame or str,
                   forecast_horizon: int,
                   feature_scaler: str or None,
                   target_scaler: str or None,
                   target_mode: str,
                   target_granularity: int,
                   granularity: int,
                   feature_or_covariate_set: list,
                   model_type: str,
                   model: str or callable,
                   model_parameters: list or None,
                   scenario: str or None,
                   save_predictions: bool,
                   verbose: int):
    # input checking

    if isinstance(data, str):
        try:
            data = pd.read_csv(data)
        except Exception as e:
            sys.exit(str(e))
    elif isinstance(data, pd.DataFrame):
        pass
    else:
        sys.exit("data input format is not valid.")

    for column_name in data.columns.values:
        if column_name.startswith('Target '):
            data.rename(columns={column_name: configurations.TARGET_COLUMN_NAME}, inplace=True)
            break

    data = select_features(data=data.copy(),
                           ordered_covariates_or_features=feature_or_covariate_set)

    print(data.columns.values)
    print(data.shape)

    data.sort_values(by=['temporal id', 'spatial id'], inplace=True)
    number_of_spatial_units = len(data['spatial id'].unique())
    testing_data = data.iloc[-(forecast_horizon * granularity * number_of_spatial_units):]
    training_data = data.iloc[:-(forecast_horizon * granularity * number_of_spatial_units)]

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

    base_data = training_data['Target'].values.tolist()

    scaled_training_data, scaled_testing_data = data_scaling(train_data=training_data,
                                                             test_data=testing_data,
                                                             input_scaler=feature_scaler,
                                                             output_scaler=target_scaler)

    scaled_training_data.drop(configurations.NON_FEATURE_COLUMNS_NAMES, axis=1, inplace=True)
    scaled_training_data.drop(configurations.NORMAL_TARGET_COLUMN_NAME, axis=1, inplace=True)
    scaled_testing_data.drop(configurations.NON_FEATURE_COLUMNS_NAMES, axis=1, inplace=True)
    scaled_testing_data.drop(configurations.NORMAL_TARGET_COLUMN_NAME, axis=1, inplace=True)

    scaled_training_predictions, scaled_testing_predictions, trained_model = train_evaluate(
        training_data=scaled_training_data,
        validation_data=scaled_testing_data,
        model_type=model_type,
        model=model,
        model_parameters=model_parameters,
        verbose=verbose)

    training_predictions = target_descale(scaled_data=list(scaled_training_predictions),
                                          base_data=base_data,
                                          scaler=target_scaler)
    testing_predictions = target_descale(scaled_data=list(scaled_testing_predictions),
                                         base_data=base_data,
                                         scaler=target_scaler)

    print(len(testing_predictions))
    print(testing_predictions)

    # return None


if __name__ == '__main__':
    predict_future(data='historical_data h=2.csv',
                   forecast_horizon=4,
                   feature_scaler=None,
                   target_scaler=None,
                   target_mode='normal',
                   target_granularity=1,
                   granularity=1,
                   feature_or_covariate_set=['virus-pressure t+2', 'virus-pressure t+3', 'area', 'temperature t'],
                   model_type='regression',
                   model='knn',
                   model_parameters=None,
                   scenario='max',
                   save_predictions=False,
                   verbose=0)

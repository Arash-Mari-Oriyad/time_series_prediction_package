import sys

import pandas as pd

import configurations
from scaling import data_scaling
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
                   model_parameters: list,
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

    print(training_data.shape, testing_data.shape)
    print(testing_data.isna().sum())

    # print(testing_data.head(10))

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
        pass

    # base_data = training_data['Target'].values.tolist()
    # training_target = training_data['spatial id', 'temporal id', 'Target', 'Normal target']
    # testing_target = testing_data['spatial id', 'temporal id', 'Target', 'Normal target']
    #
    # training_data, testing_data = data_scaling(training_data,
    #                                            testing_data,
    #                                            input_scaler=feature_scaler,
    #                                            output_scaler=target_scaler)
    #
    # training_data = training_data.drop(['Normal target', 'spatial id', 'temporal id'], axis=1)
    # testing_data = testing_data.drop(['Normal target', 'spatial id', 'temporal id'], axis=1)
    #
    # training_predictions, testing_predictions, trained_model = train_evaluate(training_data=training_data,
    #                                                                           validation_data=testing_data,
    #                                                                           model=model,
    #                                                                           model_type=model_type,
    #                                                                           model_parameters=model_parameters,
    #                                                                           verbose=verbose)

    # train_predictions, testing_predictions, trained_model = train_evaluate(training_data=training_data,
    #                                                                        validation_data=testing_data,
    #                                                                        model=best_model, verbose=0)
    #
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
                   model_parameters=[],
                   scenario='max',
                   save_predictions=False,
                   verbose=0)

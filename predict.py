import sys

import pandas as pd

import configurations
import rank_covariates


def predict(data: list,
            test_type: str,
            max_history_length: int,
            forecast_horizon: int,
            ranking_method: str,
            forced_covariates: list):
    # input checking
    if not isinstance(data, list):
        sys.exit("The input data must be a list of DataFrames or a list of strings of data addresses.")
    str_check = [isinstance(d, str) for d in data]
    df_check = [isinstance(d, pd.DataFrame) for d in data]
    if not (all(str_check) or all(df_check)):
        sys.exit("The input data must be a list of DataFrames or a list of strings of data addresses.")
    if test_type not in configurations.TEST_TYPES:
        sys.exit("Test type is not valid.")

    # data preparing
    if isinstance(data[0], str):
        try:
            data = [pd.read_csv(d) for d in data]
        except:
            sys.exit("Data addresses are invalid.")

    # data columns names manipulation
    target_mode, granularity = None, None
    target_column_name = list(filter(lambda x: x.startswith('Target '), data[0].columns.values))[0]
    temp = target_column_name.split(' ')[-1][1:-1]
    if temp.startswith('augmented'):
        granularity = int(temp.split(' ')[2])
        temp = temp.split(' ')[-1]
    if temp == 'normal':
        target_mode = 'normal'
    elif temp == 'cumulative':
        target_mode = 'cumulative'
    elif temp == 'differential':
        target_mode = 'differential'
    else:
        sys.exit("Target column name is not valid.")
    data = [d.rename(columns={target_column_name: target_column_name.split(' ')[0]}) for d in data]

    # ranking
    ordered_covariates_or_features = rank_covariates.rank_covariates(data=pd.DataFrame.copy(data[0]),
                                                                     ranking_method='correlation',
                                                                     forced_covariates=['virus-pressure',
                                                                                        'social-distancing-encounters'
                                                                                        '-grade'])
    print(ordered_covariates_or_features)

    # train_validate

    # train_test

    # predict_future

    return None


if __name__ == '__main__':
    predict(data=['historical_data h=1.csv', 'historical_data h=2.csv', 'historical_data h=3.csv'],
            test_type='whole-as-one',
            max_history_length=3,
            forecast_horizon=5,
            ranking_method='mRMR',
            forced_covariates=[])

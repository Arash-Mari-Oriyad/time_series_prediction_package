import os
import shutil
import sys

import pandas as pd

import configurations
from get_future_data import get_future_data
from rank_covariates import rank_covariates
from rank_features import rank_features
from train_validate import train_validate
from train_test import train_test
from predict_future import predict_future


def predict(data: list,
            forecast_horizon: int = 1,
            feature_scaler: str = None,
            target_scaler: str = None,
            test_type: str = 'whole-as-one',
            feature_sets: dict = {'covariate': 'mRMR'},
            model_type: str = 'regression',
            models: list = ['knn'],
            instance_testing_size: int or float = 0.2,
            splitting_type: str = 'training-validation',
            instance_validation_size: int or float = 0.3,
            instance_random_partitioning: bool = False,
            fold_total_number: int = 5,
            performance_benchmark: str = 'MAPE',
            performance_mode: str = 'normal',
            performance_measures: list = ['MAPE'],
            scenario: str or None = 'current',
            validation_performance_report: bool = True,
            testing_performance_report: bool = True,
            save_predictions: bool = True,
            verbose: int = 0):
    """

    Args:
        data:
        forecast_horizon:
        feature_scaler:
        target_scaler:
        test_type:
        feature_sets:
        model_type:
        models:
        instance_testing_size:
        splitting_type:
        instance_validation_size:
        instance_random_partitioning:
        fold_total_number:
        performance_benchmark:
        performance_mode:
        performance_measures:
        scenario:
        validation_performance_report:
        testing_performance_report:
        save_predictions:
        verbose:

    Returns:

    """
    # input checking
    # data
    if not isinstance(data, list):
        sys.exit("The input 'data' must be a list of DataFrames or a list of data addresses.")
    str_check = [isinstance(d, str) for d in data]
    df_check = [isinstance(d, pd.DataFrame) for d in data]
    if not (all(str_check) or all(df_check)):
        sys.exit("The input 'data' must be a list of DataFrames or a list data addresses.")
    # forecast_horizon
    if not (isinstance(forecast_horizon, int) and forecast_horizon >= 1):
        sys.exit("The input 'forecast_horizon' must be integer and greater than or equal to one.")
    # feature_scaler
    if feature_scaler not in configurations.FEATURE_SCALERS:
        sys.exit(f"The input 'feature_scaler' must be string and one of the following options:\n"
                 f"{configurations.FEATURE_SCALERS}")
    # target_scaler
    if target_scaler not in configurations.TARGET_SCALERS:
        sys.exit(f"The input 'target_scaler' must be string and one of the following options:\n"
                 f"{configurations.TARGET_SCALERS}")
    # test_type
    if test_type not in configurations.TEST_TYPES:
        sys.exit(f"The input 'test_type' must be string and one of the following options:\n"
                 f"{configurations.TEST_TYPES}")
    # feature_sets input checking
    if not (isinstance(feature_sets, dict) and len(feature_sets.keys()) == 1):
        sys.exit("feature_sets input format is not valid.")
    if not (list(feature_sets.keys())[0] in configurations.FEATURE_SELECTION_TYPES
            and list(feature_sets.values())[0] in configurations.RANKING_METHODS):
        sys.exit("feature_sets input is not valid.")
    # model_type input checking
    if model_type not in configurations.MODEL_TYPES:
        sys.exit("model_type input is not valid.")
    # models input checking
    if not isinstance(models, list):
        sys.exit("models input format is not valid.")
    for model in models:
        if isinstance(model, str):
            if model not in configurations.PRE_DEFINED_MODELS:
                sys.exit("models input is not valid.")
        elif isinstance(model, dict):
            if len(list(model.keys())) == 1:
                if list(model.keys())[0] not in configurations.PRE_DEFINED_MODELS:
                    sys.exit("models input is not valid.")
            else:
                sys.exit("models input is not valid.")
        elif callable(model):
            pass
        else:
            sys.exit("Models input is not valid.")
    # instance_testing_size input checking
    if not ((isinstance(instance_testing_size, float) and 0 < instance_testing_size < 1) or (
            isinstance(instance_testing_size, int) and instance_testing_size > 0)):
        sys.exit("instance_testing_size input is not valid.")
    # splitting_type input checking
    if splitting_type not in configurations.SPLITTING_TYPES:
        sys.exit("splitting_type input is not valid.")
    # instance_validation_size input checking
    if not ((isinstance(instance_validation_size, float) and 0 < instance_validation_size < 1) or (
            isinstance(instance_validation_size, int) and instance_validation_size > 0)):
        sys.exit("instance_validation_size input is not valid.")
    # instance_random_partitioning input checking
    if not isinstance(instance_random_partitioning, bool):
        sys.exit("instance_random_partitioning input is not valid.")
    # fold_total_number input checking
    if not (isinstance(fold_total_number, int), fold_total_number > 1):
        sys.exit("fold_total_number input is not valid.")
    # performance_benchmark input checking
    if performance_benchmark not in configurations.PERFORMANCE_BENCHMARKS:
        sys.exit("performance_benchmark input is not valid.")
    # performance_mode input checking
    if not isinstance(performance_mode, str):
        sys.exit("performance_mode input format is not valid.")
    if not any(performance_mode.startswith(performance_mode_starts_with)
               for performance_mode_starts_with in configurations.PERFORMANCE_MODES_STARTS_WITH):
        sys.exit("performance_mode input is not valid.")
    # performance_measures input checking
    if not (isinstance(performance_measures, list) and len(performance_measures) > 0):
        sys.exit("performance_measures input format is not valid.")
    for performance_measure in performance_measures:
        if performance_measure not in configurations.PERFORMANCE_MEASURES:
            sys.exit("performance_measures input is not valid.")
    # scenario
    if not ((isinstance(scenario, str) and scenario in configurations.SCENARIOS) or scenario is None):
        sys.exit("scenario input is not valid.")
    # validation_performance_report input checking
    if not isinstance(validation_performance_report, bool):
        sys.exit("validation_performance_report input is not valid.")
    # testing_performance_report input checking
    if not isinstance(testing_performance_report, bool):
        sys.exit("testing_performance_report input is not valid.")
    # save_predictions input checking
    if not isinstance(save_predictions, bool):
        sys.exit("save_predictions input is not valid.")
    # verbose input checking
    if verbose not in configurations.VERBOSE_OPTIONS:
        sys.exit("verbose input is not valid.")

    # removing prediction and performance directories and test_process_backup csv file
    if os.path.exists('prediction'):
        shutil.rmtree('prediction')
    if os.path.exists('performance'):
        shutil.rmtree('performance')
    if os.path.isfile('test_process_backup.csv'):
        os.remove('test_process_backup.csv')

    # data preparing
    if isinstance(data[0], str):
        try:
            data = [pd.read_csv(d).sort_values(by=['temporal id', 'spatial id']) for d in data]
        except Exception as e:
            sys.exit(str(e))

    # classification checking
    labels = None
    if model_type == 'classification':
        if not set(performance_measures) <= set(configurations.CLASSIFICATION_PERFORMANCE_MEASURES):
            sys.exit("Error: The input 'performance_measures' is not valid according to 'model_type=classification'.")
        if performance_benchmark not in configurations.CLASSIFICATION_PERFORMANCE_BENCHMARKS:
            sys.exit("Error: The input 'performance_benchmark' is not valid according to 'model_type=classification'.")
        if performance_mode != 'normal':
            performance_mode = 'normal'
            print("Warning: The input 'performance_mode' is set to 'normal' according to model_type=classification'.")
        if target_scaler is not None:
            target_scaler = None
            print("Warning: The input 'target_scaler' is set to None according to model_type=classification'.")
        target_column_name = list(filter(lambda x: x.startswith('Target'), data[0].columns.values))[0]
        labels = data[0].loc[:, target_column_name].unique().tolist()

    # one_by_one checking
    if test_type == 'one-by-one':
        splitting_type = 'training-validation'
        instance_validation_size = 1
        instance_random_partitioning = False
        if data[0]['spatial id'].nunique() == 1:
            if 'AUC' in performance_measures:
                performance_measures.remove('AUC')
            if 'R2_score' in performance_measures:
                performance_measures.remove('R2_score')
            if 'AUPR' in performance_measures:
                performance_measures.remove('AUPR')
            if len(performance_measures) == 0:
                sys.exit("Error: The input 'performance_measures' is not valid according to 'test_type=one-by-one'.")
            if 'AUC' in performance_benchmark:
                sys.exit("Error: The input 'performance_benchmark' is not valid according to 'test_type=one-by-one'.")
            if 'R2_score' in performance_measures:
                sys.exit("Error: The input 'performance_benchmark' is not valid according to 'test_type=one-by-one'.")
            if 'AUPR' in performance_measures:
                sys.exit("Error: The input 'performance_benchmark' is not valid according to 'test_type=one-by-one'.")

    data, future_data = get_future_data(data=[d.copy() for d in data],
                                        forecast_horizon=forecast_horizon)

    # ranking
    feature_selection_type = list(feature_sets.keys())[0]
    ranking_method = list(feature_sets.values())[0]
    ordered_covariates_or_features = []
    if feature_selection_type == 'covariate':
        ordered_covariates_or_features = rank_covariates(data=data[0].copy(),
                                                         ranking_method=ranking_method)
    else:
        for d in data:
            ordered_covariates_or_features.append(rank_features(data=d.copy(),
                                                                ranking_method=ranking_method))
    # ordered_covariates_or_features = ordered_covariates_or_features[:7]

    # main process
    if test_type == 'whole-as-one':
        # train_validate
        best_model, best_model_parameters, best_history_length, best_feature_or_covariate_set, _ = \
            train_validate(data=[d.copy() for d in data],
                           forecast_horizon=forecast_horizon,
                           feature_scaler=feature_scaler,
                           target_scaler=target_scaler,
                           ordered_covariates_or_features=ordered_covariates_or_features,
                           model_type=model_type,
                           labels=labels,
                           models=models,
                           instance_testing_size=instance_testing_size,
                           splitting_type=splitting_type,
                           instance_validation_size=instance_validation_size,
                           instance_random_partitioning=instance_random_partitioning,
                           fold_total_number=fold_total_number,
                           performance_benchmark=performance_benchmark,
                           performance_measures=performance_measures,
                           performance_report=validation_performance_report,
                           save_predictions=save_predictions,
                           verbose=verbose)

        # train_test
        best_model, best_model_parameters = train_test(data=data[best_history_length - 1].copy(),
                                                       forecast_horizon=forecast_horizon,
                                                       history_length=best_history_length,
                                                       feature_scaler=feature_scaler,
                                                       target_scaler=target_scaler,
                                                       feature_or_covariate_set=best_feature_or_covariate_set,
                                                       model_type=model_type,
                                                       labels=labels,
                                                       model=best_model,
                                                       model_parameters=best_model_parameters,
                                                       instance_testing_size=instance_testing_size,
                                                       performance_measures=performance_measures,
                                                       performance_mode=performance_mode,
                                                       performance_report=testing_performance_report,
                                                       save_predictions=save_predictions,
                                                       verbose=verbose)

        # predict_future
        best_model, best_model_parameters, best_history_length, best_feature_or_covariate_set, _ = \
            train_validate(data=[d.copy() for d in data],
                           forecast_horizon=forecast_horizon,
                           feature_scaler=feature_scaler,
                           target_scaler=target_scaler,
                           ordered_covariates_or_features=ordered_covariates_or_features,
                           model_type=model_type,
                           labels=labels,
                           models=models,
                           instance_testing_size=0,
                           splitting_type=splitting_type,
                           instance_validation_size=instance_validation_size,
                           instance_random_partitioning=instance_random_partitioning,
                           fold_total_number=fold_total_number,
                           performance_benchmark=performance_benchmark,
                           performance_measures=performance_measures,
                           performance_report=False,
                           save_predictions=False,
                           verbose=0)
        best_data = data[best_history_length - 1].copy()
        best_future_data = future_data[best_history_length - 1].copy()
        best_data_temporal_ids = best_data['temporal id'].unique()
        temp = forecast_horizon - 1

        trained_model = predict_future(data=best_data[best_data['temporal id'].isin((best_data_temporal_ids
                                                                                     if temp == 0
                                                                                     else best_data_temporal_ids[:-temp]
                                                                                     ))].copy(),
                                       future_data=best_future_data.copy(),
                                       forecast_horizon=forecast_horizon,
                                       feature_scaler=feature_scaler,
                                       target_scaler=target_scaler,
                                       feature_or_covariate_set=best_feature_or_covariate_set,
                                       model_type=model_type,
                                       labels=labels,
                                       model=best_model,
                                       model_parameters=best_model_parameters,
                                       scenario=scenario,
                                       save_predictions=save_predictions,
                                       verbose=verbose)

    elif test_type == 'one-by-one':
        # loop over test points
        data_temporal_ids = [d['temporal id'].unique() for d in data]
        if isinstance(instance_testing_size, float):
            instance_testing_size = int(instance_testing_size * len(data_temporal_ids[0]))
        for i in range(instance_testing_size):
            print(150 * '#')
            print('i =', i + 1)
            # train_validate
            print(100 * '-')
            print('Train Validate Process')
            best_model, best_model_parameters, best_history_length, best_feature_or_covariate_set, _ = \
                train_validate(data=
                               [d[d['temporal id'].isin((
                                   data_temporal_ids[index][:] if i == 0 else data_temporal_ids[index][:-i]))].copy()
                                for index, d in enumerate(data)],
                               forecast_horizon=forecast_horizon,
                               feature_scaler=feature_scaler,
                               target_scaler=target_scaler,
                               ordered_covariates_or_features=ordered_covariates_or_features,
                               model_type=model_type,
                               labels=labels,
                               models=models,
                               instance_testing_size=1,
                               splitting_type=splitting_type,
                               instance_validation_size=instance_validation_size,
                               instance_random_partitioning=instance_random_partitioning,
                               fold_total_number=fold_total_number,
                               performance_benchmark=performance_benchmark,
                               performance_measures=performance_measures,
                               performance_report=validation_performance_report,
                               save_predictions=save_predictions,
                               verbose=verbose)

            # train_test
            print(100 * '-')
            print('Train Test Process')
            d = data[best_history_length - 1].copy()
            best_model, best_model_parameters = train_test(data=d[d['temporal id'].isin(
                (data_temporal_ids[best_history_length - 1][:]
                 if i == 0
                 else data_temporal_ids[best_history_length - 1][:-i]
                 ))].copy(),
                                                           forecast_horizon=forecast_horizon,
                                                           history_length=best_history_length,
                                                           feature_scaler=feature_scaler,
                                                           target_scaler=target_scaler,
                                                           feature_or_covariate_set=best_feature_or_covariate_set,
                                                           model_type=model_type,
                                                           labels=labels,
                                                           model=best_model,
                                                           model_parameters=best_model_parameters,
                                                           instance_testing_size=1,
                                                           performance_measures=performance_measures,
                                                           performance_mode=performance_mode,
                                                           performance_report=testing_performance_report,
                                                           save_predictions=save_predictions,
                                                           verbose=verbose)

        # predict_future
        print(100 * '-')
        print('Train Validate Process')
        best_model, best_model_parameters, best_history_length, best_feature_or_covariate_set, _ = \
            train_validate(data=[d.copy() for d in data],
                           forecast_horizon=forecast_horizon,
                           feature_scaler=feature_scaler,
                           target_scaler=target_scaler,
                           ordered_covariates_or_features=ordered_covariates_or_features,
                           model_type=model_type,
                           labels=labels,
                           models=models,
                           instance_testing_size=0,
                           splitting_type=splitting_type,
                           instance_validation_size=instance_validation_size,
                           instance_random_partitioning=instance_random_partitioning,
                           fold_total_number=fold_total_number,
                           performance_benchmark=performance_benchmark,
                           performance_measures=performance_measures,
                           performance_report=False,
                           save_predictions=False,
                           verbose=0)
        best_data = data[best_history_length - 1].copy()
        best_future_data = future_data[best_history_length - 1].copy()
        best_data_temporal_ids = best_data['temporal id'].unique()
        best_future_data_temporal_ids = best_future_data['temporal id'].unique()
        for i in range(forecast_horizon):
            print(150 * '*')
            print('i =', i + 1)
            temp = forecast_horizon - i - 1
            print(100 * '-')
            print('Predict Future Process')
            trained_model = predict_future(data=best_data[best_data['temporal id'].isin(
                (best_data_temporal_ids if temp == 0
                 else best_data_temporal_ids[:-temp]))].copy(),
                                           future_data=best_future_data[best_future_data['temporal id'] ==
                                                                        best_future_data_temporal_ids[i]].copy(),
                                           forecast_horizon=forecast_horizon,
                                           feature_scaler=feature_scaler,
                                           target_scaler=target_scaler,
                                           feature_or_covariate_set=best_feature_or_covariate_set,
                                           model_type=model_type,
                                           labels=labels,
                                           model=best_model,
                                           model_parameters=best_model_parameters,
                                           scenario=scenario,
                                           save_predictions=save_predictions,
                                           verbose=verbose)

    return None


if __name__ == '__main__':
    # input_data = [f'usa_data/historical_data h={i}.csv' for i in range(1, 4)]
    input_data = [f'canada_data/historical_data h={i}.csv' for i in range(1, 11)]
    predict(data=input_data,
            forecast_horizon=14,
            feature_scaler='logarithmic',
            target_scaler=None,
            test_type='one-by-one',
            feature_sets={'covariate': 'mRMR'},
            model_type='regression',
            models=['knn', 'gbm'],
            instance_testing_size=0.2,
            splitting_type='training-validation',
            instance_validation_size=0.3,
            instance_random_partitioning=False,
            fold_total_number=5,
            performance_benchmark='MAE',
            performance_mode='normal',
            performance_measures=['MAPE', 'MAE'],
            scenario='current',
            validation_performance_report=True,
            testing_performance_report=True,
            save_predictions=True,
            verbose=0)

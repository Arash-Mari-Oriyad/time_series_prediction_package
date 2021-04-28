import sys

import pandas as pd

import configurations
import rank_covariates
import rank_features
import train_test
import train_validate


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
            performance_measures: str = ['MAPE'],
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
        performance_measures:
        validation_performance_report:
        testing_performance_report:
        save_predictions:
        verbose:

    Returns:

    """
    # input checking
    # data input checking
    if not isinstance(data, list):
        sys.exit("The input data must be a list of DataFrames or a list of strings of data addresses.")
    str_check = [isinstance(d, str) for d in data]
    df_check = [isinstance(d, pd.DataFrame) for d in data]
    if not (all(str_check) or all(df_check)):
        sys.exit("The input data must be a list of DataFrames or a list of strings of data addresses.")
    # forecast_horizon input checking
    if not (isinstance(forecast_horizon, int) and forecast_horizon >= 1):
        sys.exit("forecast_horizon is not valid.")
    # feature_scaler input checking
    if feature_scaler not in configurations.FEATURE_SCALERS:
        sys.exit("feature_scaler input is not valid.")
    # target_scaler input checking
    if target_scaler not in configurations.TARGET_SCALERS:
        sys.exit("target_scaler input is not valid.")  # test_type input checking
    if test_type not in configurations.TEST_TYPES:
        sys.exit("test_type is not valid.")
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
    # performance_measures input checking
    if not isinstance(performance_measures, list):
        sys.exit("performance_measures input format is not valid.")
    for performance_measure in performance_measures:
        if performance_measure not in configurations.PERFORMANCE_MEASURES:
            sys.exit("performance_measures input is not valid.")
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

    # data preparing
    if isinstance(data[0], str):
        try:
            data = [pd.read_csv(d) for d in data]
        except Exception as e:
            sys.exit(str(e))

    # data columns names manipulation
    target_mode, target_granularity, granularity = None, 1, 1
    target_column_name = list(filter(lambda x: x.startswith('Target '), data[0].columns.values))[0]
    temp = target_column_name.split(' ')[-1][1:-1]
    if temp.startswith('augmented'):
        granularity = int(temp.split(' ')[2])
        temp = temp[temp.index('-') + 2:]
    if temp.startswith('normal'):
        target_mode = 'normal'
    elif temp.startswith('cumulative'):
        target_mode = 'cumulative'
    elif temp.startswith('differential'):
        target_mode = 'differential'
    elif temp.startswith('moving'):
        target_mode = 'moving_average'
        target_granularity = int(temp.split(' ')[3])
    else:
        sys.exit("Target column name is not valid.")
    data = [d.rename(columns={target_column_name: target_column_name.split(' ')[0]}) for d in data]

    # ranking
    feature_selection_type = list(feature_sets.keys())[0]
    ranking_method = list(feature_sets.values())[0]
    ordered_covariates_or_features = []
    if feature_selection_type == 'covariate':
        ordered_covariates_or_features = rank_covariates.rank_covariates(data=pd.DataFrame.copy(data[0]),
                                                                         ranking_method=ranking_method)
    else:
        for d in data:
            ordered_covariates_or_features.append(rank_features.rank_features(data=d,
                                                                              ranking_method=ranking_method))

    # main process
    if test_type == 'whole-as-one':
        # train_validate
        best_model, best_model_parameters, best_history_length, best_feature_or_covariate_set, best_trained_model = \
            train_validate.train_validate(data=data,
                                          forecast_horizon=forecast_horizon,
                                          input_scaler=feature_scaler,
                                          output_scaler=target_scaler,
                                          target_mode=target_mode,
                                          target_granularity=target_granularity,
                                          granularity=granularity,
                                          ordered_covariates_or_features=ordered_covariates_or_features,
                                          model_type=model_type,
                                          models=models,
                                          instance_testing_size=instance_testing_size,
                                          splitting_type=splitting_type,
                                          instance_validation_size=instance_validation_size,
                                          instance_random_partitioning=instance_random_partitioning,
                                          fold_total_number=fold_total_number,
                                          performance_benchmark=performance_benchmark,
                                          performance_measure=performance_measures,
                                          performance_report=validation_performance_report,
                                          save_predictions=save_predictions,
                                          verbose=verbose)

        # train_test
        best_model, best_model_parameters = train_test.train_test(data=data[best_history_length-1].copy(),
                                                                  forecast_horizon=forecast_horizon,
                                                                  input_scaler=feature_scaler,
                                                                  output_scaler=target_scaler,
                                                                  target_mode=target_mode,
                                                                  target_granularity=target_granularity,
                                                                  granularity=granularity,
                                                                  feature_or_covariate_set=best_feature_or_covariate_set,
                                                                  model_type=model_type,
                                                                  model=best_model,
                                                                  model_parameters=best_model_parameters,
                                                                  instance_testing_size=instance_testing_size,
                                                                  performance_measures=performance_measures,
                                                                  performance_report=testing_performance_report,
                                                                  save_predictions=save_predictions,
                                                                  verbose=verbose)

        # predict_future

    elif test_type == 'ono-by-one':
        # loop over test points
        pass

    return None


if __name__ == '__main__':
    predict(data=['historical_data h=1.csv', 'historical_data h=2.csv', 'historical_data h=3.csv'],
            test_type='whole-as-one',
            forecast_horizon=5,
            feature_sets={'covariate': 'mRMR'})

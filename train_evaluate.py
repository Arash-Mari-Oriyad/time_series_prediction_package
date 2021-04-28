import pandas as pd
import numpy as np
import sys
import datetime
from models import KNN_REGRESSOR, KNN_CLASSIFIER, NN_REGRESSOR, NN_CLASSIFIER, GLM_REGRESSOR, GLM_CLASSIFIER, \
    GBM_REGRESSOR, GBM_CLASSIFIER


def train_evaluate(training_data, validation_data, model, model_type, model_parameters=None, verbose=1):
    supported_models_name = ['gbm', 'glm', 'knn', 'nn']
    train_predictions = None
    validation_predictions = None
    trained_model = None

    if type(training_data) == str:
        try:
            training_data = pd.read_csv(training_data)
        except FileNotFoundError:
            sys.exit("The address '{0}' is not valid.".format(training_data))
    if type(validation_data) == str:
        try:
            validation_data = pd.read_csv(validation_data)
        except FileNotFoundError:
            sys.exit("The address '{0}' is not valid.".format(validation_data))

    # split features and target
    if 'spatial id' in training_data.columns.values or 'temporal id' in training_data.columns.values:
        X_training = training_data.drop(['Target', 'spatial id', 'temporal id'], axis=1)
    else:
        X_training = training_data.drop(['Target'], axis=1)
    Y_training = np.array(training_data['Target']).reshape(-1)
    if 'spatial id' in validation_data.columns.values or 'temporal id' in validation_data.columns.values:
        X_validation = validation_data.drop(['Target', 'spatial id', 'temporal id'], axis=1)
    else:
        X_validation = validation_data.drop(['Target'], axis=1)
    Y_validation = np.array(validation_data['Target']).reshape(-1)

    if 'Normal target' in X_training.columns:
        X_training = X_training.drop(['Normal target'], axis=1)
        X_validation = X_validation.drop(['Normal target'], axis=1)

    if (type(model) == str) and (model in supported_models_name):

        if model == 'gbm':
            if model_type == 'classification':
                validation_predictions, train_predictions, trained_model = GBM_CLASSIFIER(X_training, X_validation,
                                                                                          Y_training, model_parameters,
                                                                                          verbose)
            if model_type == 'regression':
                validation_predictions, train_predictions, trained_model = GBM_REGRESSOR(X_training, X_validation,
                                                                                         Y_training, model_parameters,
                                                                                         verbose)
        elif model == 'glm':
            if model_type == 'classification':
                validation_predictions, train_predictions, trained_model = GLM_CLASSIFIER(X_training, X_validation,
                                                                                          Y_training, model_parameters,
                                                                                          verbose)
            if model_type == 'regression':
                validation_predictions, train_predictions, trained_model = GLM_REGRESSOR(X_training, X_validation,
                                                                                         Y_training, model_parameters,
                                                                                         verbose)
        elif model == 'knn':
            if model_type == 'classification':
                validation_predictions, train_predictions, trained_model = KNN_CLASSIFIER(X_training, X_validation,
                                                                                          Y_training, model_parameters,
                                                                                          verbose)
            if model_type == 'regression':
                validation_predictions, train_predictions, trained_model = KNN_REGRESSOR(X_training, X_validation,
                                                                                         Y_training, model_parameters,
                                                                                         verbose)
        elif model == 'nn':
            if model_type == 'classification':
                validation_predictions, train_predictions, trained_model = NN_CLASSIFIER(X_training, X_validation,
                                                                                         Y_training, model_parameters,
                                                                                         verbose)
            if model_type == 'regression':
                validation_predictions, train_predictions, trained_model = NN_REGRESSOR(X_training, X_validation,
                                                                                        Y_training, model_parameters,
                                                                                        verbose)

    elif callable(model):
        try:
            validation_predictions, train_predictions, trained_model = model(X_training, X_validation, Y_training)
        except ValueError:
            sys.exit("The user-defined function is not compatible with the definition.")

    else:
        sys.exit(
            "The model must be name of one of the supported models: {'gbm', 'glm', 'knn', 'nn'}\nOr user defined function.")

    return train_predictions, validation_predictions, trained_model

import pandas as pd
import numpy as np
import sys
import datetime
from models import KNN_REGRESSOR, KNN_CLASSIFIER, NN_REGRESSOR, NN_CLASSIFIER, GLM_REGRESSOR, GLM_CLASSIFIER, \
    GBM_REGRESSOR, GBM_CLASSIFIER


def complete_predicted_probabilities(predictions, all_labels, present_labels):
    
    present_probabilities = predictions
    all_probabilities = np.zeros([len(present_probabilities),len(all_labels)])
    ind = 0
    for label_number, label in enumerate(all_labels):
        if label in present_labels:
            all_probabilities[:,label_number] = present_probabilities[:,ind]
            ind += 1

        else:
            all_probabilities[:,label_number] = 0
            
    return all_probabilities


def train_evaluate(training_data, validation_data, model, model_type, model_parameters=None, labels = None, verbose=1):
    supported_models_name = ['gbm', 'glm', 'knn', 'nn']
    train_predictions = None
    validation_predictions = None
    trained_model = None
    
    
    if type(training_data) == str:
        try:
            training_data = pd.read_csv(training_data)
        except FileNotFoundError:
            sys.exit("File '{0}' does not exist.".format(training_data))
    if type(validation_data) == str:
        try:
            validation_data = pd.read_csv(validation_data)
        except FileNotFoundError:
            sys.exit("File '{0}' does not exist.".format(validation_data))

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
    
    # Labels presented in input training data
    present_labels = list(np.unique(Y_training))
    present_labels.sort()

    if (type(model) == str) and (model in supported_models_name):
        try:
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
        except Exception as ex:
            raise Exception("{0} model\n\t   {1}".format(model.upper(),ex))
            
        if (model == 'nn') and (not np.allclose(1, train_predictions.sum(axis=1))) or (not np.allclose(1, validation_predictions.sum(axis=1))):
                 raise Exception(
                     "The output predictions of the neural network model need to be probabilities "
                     "i.e. they should sum up to 1.0 over classes. But the output does not match the condition. "
                     "Revise the model parameters to solve the problem.")

        
    elif callable(model):
        try:
            train_predictions, validation_predictions, trained_model = model(X_training, X_validation, Y_training)
        except:
            raise Exception("The user-defined model is not compatible with the definition.")
        
        if (type(train_predictions) not in (np.ndarray,list)) or (type(validation_predictions) not in (np.ndarray,list)):
            raise Exception("The output predictions of the user-defined model must be of type array.")
        
        train_predictions = np.array(train_predictions)
        validation_predictions = np.array(validation_predictions)
        
        if (len(train_predictions) != len(X_training)) or (len(validation_predictions) != len(X_validation)):
            raise Exception("The output of the user-defined model has a different length from the input data.")
        
        if model_type == 'classification':
            try:
                train_predictions.shape[1]
                validation_predictions.shape[1]
            except IndexError:
                raise Exception("The output of the user_defined classification model must be an array of shape (n_samples,n_classes).")
                                    
            if ((train_predictions.shape[1] != len(present_labels)) or (validation_predictions.shape[1] != len(present_labels))):
                raise Exception("The probability predictions of the user-defined model are not compatible with the number of classes in the input data.")
            
            if (not np.allclose(1, train_predictions.sum(axis=1))) or (not np.allclose(1, validation_predictions.sum(axis=1))):
                 raise Exception(
                     "The output predictions of the user-defined model need to be probabilities "
                     "i.e. they should sum up to 1.0 over classes")

    else:
        sys.exit(
            "The model must be name of one of the supported models: {'gbm', 'glm', 'knn', 'nn'}\nOr user defined function.")
    
    # adding the zero probability for the labels which are not included in the train data and thus are 
    # not considered in the predictions
    if (model_type == 'classification') and (labels is not None):
        
        train_predictions = complete_predicted_probabilities(predictions = train_predictions,
                                                             all_labels = labels, present_labels = present_labels)
        validation_predictions = complete_predicted_probabilities(predictions = validation_predictions,
                                                                  all_labels = labels, present_labels = present_labels)
    
    return train_predictions, validation_predictions, trained_model


#####################################################################################################

def inner_train_evaluate(training_data, validation_data, model, model_type, model_parameters = None, labels = None, verbose = 0):
    
    train_predictions, validation_predictions, trained_model = train_evaluate(training_data = training_data,
                                                                              validation_data = validation_data,
                                                                              model = model, 
                                                                              model_type = model_type,
                                                                              model_parameters = model_parameters, 
                                                                              labels = labels,
                                                                              verbose = verbose)
    if model == 'gbm':
        # get the number of trees
        number_of_parameters = trained_model.n_estimators_
        
    elif model == 'glm':
        # get the number of coefficients and intercept
        if model_type == 'classification':
            number_of_parameters = trained_model.coef_.shape[0]*trained_model.coef_.shape[1]
            if not all(trained_model.intercept_ == 0):
                number_of_parameters += trained_model.intercept_.shape[0]
        if model_type == 'regression':
            number_of_parameters = trained_model.coef_.shape[0]
            if trained_model.get_params()['fit_intercept']:
                number_of_parameters += 1
                
    elif model == 'knn':
        # get the number of nearest neighbours
        number_of_parameters = trained_model.get_params()['n_neighbors']
        
    elif model == 'nn':
        # get the number of parameters
        number_of_parameters = trained_model.count_params()
        
    else:
        number_of_parameters = None
        
    return train_predictions, validation_predictions, trained_model, number_of_parameters
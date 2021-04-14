import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LogisticRegression
import statsmodels.api as sm
from sklearn import svm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from pexecute.process import ProcessLoom
import time
from sys import argv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
import os
from numpy.random import seed
seed(1)
tf.random.set_seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


####################################################### GBM: Gradient Boosting Regressor
def GBM_REGRESSOR(X_train, X_test, y_train, user_params, verbose):

    parameters = {'loss':'least_squares', 'learning_rate':0.1, 'max_iter':100, 'max_leaf_nodes':31, 'max_depth':None,
                'min_samples_leaf':20, 'l2_regularization':0.0, 'max_bins':255, #'categorical_features':None,
                'monotonic_cst':None, 'warm_start':False, 'early_stopping':'auto', 'scoring':'loss',
                'validation_fraction':0.1, 'n_iter_no_change':10, 'tol':1e-07, 'verbose':0, 'random_state':1}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]
                
    GradientBoostingRegressorObject = HistGradientBoostingRegressor(**parameters)

    GradientBoostingRegressorObject.fit(X_train, y_train)
    y_prediction = GradientBoostingRegressorObject.predict(X_test)
    y_prediction_train = GradientBoostingRegressorObject.predict(X_train)


    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), GradientBoostingRegressorObject

###################################################### GLM: Generalized Linear Model Regressor
def GLM_REGRESSOR(X_train, X_test, y_train, user_params, verbose):
    
    parameters = {'alpha':1.0, 'l1_ratio':0.5, 'fit_intercept':True, 'normalize':False, 'precompute':False,
                  'max_iter':1000, 'copy_X':True, 'tol':0.0001, 'warm_start':False, 'positive':False, 'random_state':1,
                  'selection':'cyclic'}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]
    
    GLM_Model = ElasticNet(**parameters)
    GLM_Model.fit(X_train, y_train)
    y_prediction = GLM_Model.predict(X_test)
    y_prediction_train = GLM_Model.predict(X_train)
    
    if verbose == 1:
        print('GLM coef: ', GLM_Model.coef_)

    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), GLM_Model


######################################################### KNN: K-Nearest Neighbors Regressor
def KNN_REGRESSOR(X_train, X_test, y_train, user_params, verbose):
    
    parameters = {'weights':'uniform', 'algorithm':'auto', 'leaf_size':30, 'p':2, 'metric':'minkowski',
                  'metric_params':None, 'n_jobs':None}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]
                
    # if user does not specify the K parameter or specified value is too large, the best k will be obtained using a grid search
    if (user_params is not None) and ('n_neighbors' in user_params.keys()) and (user_params['n_neighbors']<len(X_train)):
            K = user_params['n_neighbors']
    else:
        KNeighborsRegressorObject = KNeighborsRegressor()
        # Grid search over different Ks to choose the best one
        neighbors=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200])
        neighbors=neighbors[neighbors<len(X_train)*(4/5)] #4/5 of samples is used as train when cv=5
        grid_parameters = {'n_neighbors': neighbors}
        GridSearchOnKs = GridSearchCV(KNeighborsRegressorObject, grid_parameters, cv=5)
        GridSearchOnKs.fit(X_train, y_train)
        best_K = GridSearchOnKs.best_params_
        
        print("Warning: The number of neighbors for KNN algorithm is not specified or is too large for input data shape.")
        print("The number of neighbors will be set to the best number of neighbors obtained by grid search in the range [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200]")
    
        if verbose == 1:
            print('best k:', best_K['n_neighbors'])
            
        K = best_K['n_neighbors']
        
    KNN_Model = KNeighborsRegressor(n_neighbors=K, **parameters)
    KNN_Model.fit(X_train, y_train)
    y_prediction = KNN_Model.predict(X_test)
    y_prediction_train = KNN_Model.predict(X_train)

    return y_prediction, y_prediction_train, KNN_Model


####################################################### NN: Neural Network Regressor
def NN_REGRESSOR(X_train, X_test, y_train, user_params, verbose):
    
    # default parameters
    parameters = {'headen_layers_neurons':[(X_train.shape[1]) // 2 + 1], 'headen_layers_activations':[None], 'output_activation':'exponential', 'loss':'mean_squared_error',
                  'optimizer':'RMSprop', 'metrics':['mean_squared_error'],
                  'EarlyStopping_monitor':'val_loss', 'EarlyStopping_patience':30, 'batch_size':128,
                  'validation_split':0.2,'epochs':100}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]


    NeuralNetworkObject = keras.Sequential()
    NeuralNetworkObject.add(tf.keras.Input(shape=(X_train.shape[1],)))
    for layer,neurons in enumerate(parameters['headen_layers_neurons']):
        NeuralNetworkObject.add(tf.keras.layers.Dense(neurons, activation=parameters['headen_layers_activations'][layer]))
    NeuralNetworkObject.add(tf.keras.layers.Dense(1, activation=parameters['output_activation']))
    
    
    # Compile the model
    NeuralNetworkObject.compile(
        loss=parameters['loss'],
        optimizer=parameters['optimizer'],
        metrics=parameters['metrics'])

    early_stop = EarlyStopping(monitor=parameters['EarlyStopping_monitor'], patience=parameters['EarlyStopping_patience'])

    NeuralNetworkObject.fit(X_train, y_train.ravel(),
                   callbacks=[early_stop],
                   batch_size=parameters['batch_size'],
                   validation_split=parameters['validation_split'],
                   epochs=parameters['epochs'], verbose=0)
    
        
    y_prediction = NeuralNetworkObject.predict(X_test)
    y_prediction_train = NeuralNetworkObject.predict(X_train)
    
    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), NeuralNetworkObject

####################################################### GBM: Gradient Boosting Classifier

def GBM_CLASSIFIER(X_train, X_test, y_train, user_params, verbose):

    parameters = {'loss':'deviance', 'learning_rate':0.1, 'n_estimators':100, 'subsample':1.0, 'criterion':'friedman_mse',
                  'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_depth':3, 'min_impurity_decrease':0.0,
                  'min_impurity_split':None, 'init':None, 'random_state':1, 'max_features':None, 'verbose':0, 'max_leaf_nodes':None,
                  'warm_start':False, 'validation_fraction':0.1, 'n_iter_no_change':None, 'tol':0.0001, 'ccp_alpha':0.0}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]
                
    GradientBoostingclassifierObject = GradientBoostingClassifier(**parameters)

    GradientBoostingclassifierObject.fit(X_train, y_train)
    y_prediction = GradientBoostingclassifierObject.predict(X_test)
    y_prediction_train = GradientBoostingclassifierObject.predict(X_train)


    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), GradientBoostingclassifierObject


##################################################### GLM: Generalized Linear Model Classifier

def GLM_CLASSIFIER(X_train, X_test, y_train, user_params, verbose):
    
    parameters = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1,
                  'class_weight':None, 'random_state':1, 'solver':'lbfgs', 'max_iter':100, 'multi_class':'auto',
                  'verbose':0, 'warm_start':False, 'n_jobs':None, 'l1_ratio':None}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]
    
    GLM_Model = LogisticRegression(**parameters)
    GLM_Model.fit(X_train, y_train)
    y_prediction = GLM_Model.predict(X_test)
    y_prediction_train = GLM_Model.predict(X_train)
    

    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), GLM_Model

######################################################### KNN: K-Nearest Neighbors Classifier
def KNN_CLASSIFIER(X_train, X_test, y_train, user_params, verbose):
    
    parameters = {'n_neighbors':5, 'weights':'uniform', 'algorithm':'auto', 'leaf_size':30, 'p':2,
                  'metric':'minkowski', 'metric_params':None, 'n_jobs':None}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]
                
    # if user does not specify the K parameter or specified value is too large, the best k will be obtained using a grid search
    if (user_params is not None) and ('n_neighbors' in user_params.keys()) and (user_params['n_neighbors']<len(X_train)):
            K = user_params['n_neighbors']
    else:
        KNeighborsClassifierObject = KNeighborsClassifier()
        # Grid search over different Ks to choose the best one
        neighbors=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200])
        neighbors=neighbors[neighbors<len(X_train)*(4/5)] #4/5 of samples is used as train when cv=5
        grid_parameters = {'n_neighbors': neighbors}
        GridSearchOnKs = GridSearchCV(KNeighborsClassifierObject, grid_parameters, cv=5)
        GridSearchOnKs.fit(X_train, y_train)
        best_K = GridSearchOnKs.best_params_
        
        print("Warning: The number of neighbors for KNN algorithm is not specified or is too large for input data shape.")
        print("The number of neighbors will be set to the best number of neighbors obtained by grid search in the range [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,20 ,40 ,60 ,80, 100, 120, 140, 160, 180, 200]")
    
        if verbose == 1:
            print('best k:', best_K['n_neighbors'])
            
        K = best_K['n_neighbors']
        
    KNN_Model = KNeighborsClassifier(n_neighbors=K, **parameters)
    KNN_Model.fit(X_train, y_train)
    y_prediction = KNN_Model.predict(X_test)
    y_prediction_train = KNN_Model.predict(X_train)

    return y_prediction, y_prediction_train, KNN_Model

####################################################### NN: Neural Network Classifier

def NN_CLASSIFIER(X_train, X_test, y_train, user_params, verbose):
    
    
    # default parameters
    parameters = {'headen_layers_neurons':[(X_train.shape[1]) // 2 + 1], 'headen_layers_activations':[None],
                  'output_activation':'softmax', 'loss':'categorical_crossentropy',
                  'optimizer':'adam', 'metrics':['accuracy'],
                  'EarlyStopping_monitor':'val_loss', 'EarlyStopping_patience':30, 'batch_size':128,
                  'validation_split':0.2,'epochs':100}
    
    if user_params is not None:
        for key in parameters.keys():
            if key in user_params.keys():
                parameters[key] = user_params[key]
                
    encoder = LabelEncoder().fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    number_of_classes = len(encoder.classes_)
    print('le.classes_',number_of_classes)
    
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_train = np_utils.to_categorical(encoded_y_train)
    
    
    output_neurons = number_of_classes
    
    if (parameters['output_activation'] == 'sigmoid') and (number_of_classes == 2):
        output_neurons = 1
        model_y_train = encoded_y_train
    else:
        model_y_train = dummy_y_train

    
    def create_model():
        NeuralNetworkObject = keras.Sequential()
        NeuralNetworkObject.add(tf.keras.Input(shape=(X_train.shape[1],)))
        for layer,neurons in enumerate(parameters['headen_layers_neurons']):
            NeuralNetworkObject.add(tf.keras.layers.Dense(neurons, activation=parameters['headen_layers_activations'][layer]))
        NeuralNetworkObject.add(tf.keras.layers.Dense(output_neurons, activation=parameters['output_activation']))

        # Compile the model
        NeuralNetworkObject.compile(
            loss=parameters['loss'],
            optimizer=parameters['optimizer'],
            metrics=parameters['metrics'])
        
        return NeuralNetworkObject

    early_stop = EarlyStopping(monitor=parameters['EarlyStopping_monitor'], patience=parameters['EarlyStopping_patience'])
    
    
    NeuralNetworkObject = KerasClassifier(build_fn=create_model, callbacks=[early_stop],
                   batch_size=parameters['batch_size'],
                   validation_split=parameters['validation_split'],
                   epochs=parameters['epochs'], verbose=1)
    
    NeuralNetworkObject.fit(X_train, model_y_train)
    
    if number_of_classes == 2:
        y_prediction = NeuralNetworkObject.predict(X_test)
        y_prediction = encoder.inverse_transform(y_prediction)
        y_prediction_train = NeuralNetworkObject.predict(X_train)
        y_prediction_train = encoder.inverse_transform(y_prediction_train)
        
    else:
        y_prediction = NeuralNetworkObject.predict(X_test)
        y_prediction = encoder.inverse_transform(y_prediction)
        y_prediction_train = NeuralNetworkObject.predict(X_train)
        y_prediction_train = encoder.inverse_transform(y_prediction_train)
    
    return np.array(y_prediction).ravel(), np.array(y_prediction_train).ravel(), NeuralNetworkObject